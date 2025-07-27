"""
Video-to-Voice Pipeline
"""
from __future__ import annotations
import logging
from pathlib import Path
import requests, uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────
DATA = Path("/data"); DATA.mkdir(exist_ok=True, parents=True) # Ensure DATA dir exists
_FMT = "%(asctime)s %(levelname)-7s %(request_id)s | %(message)s" # Added %(request_id)s

# Custom Filter to inject request_id
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        # Check if 'request_id' is already set in the record's extra attributes
        # If not, set it to an empty string.
        # This handles cases where log records come from libraries that don't use our custom logger class
        if not hasattr(record, 'request_id'):
            record.request_id = ""
        return True

# Configure basic logging with the custom format
# Set force=True to reconfigure existing loggers
logging.basicConfig(filename=DATA / "pipeline.log",
                    level=logging.DEBUG,
                    format=_FMT,
                    force=True)

_console = logging.StreamHandler(); _console.setFormatter(logging.Formatter(_FMT))
logging.getLogger("").addHandler(_console)

# Apply the filter to all handlers of the root logger
for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIdFilter())


# Create a custom logger that can accept request_id (this is for our application's direct logs)
class CustomLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        if extra is None:
            extra = {}
        # Ensure request_id is always present in extra for our direct logs
        if 'request_id' not in extra:
            extra['request_id'] = ""
        super()._log(level, msg, args, exc_info, extra, stack_info)

logging.setLoggerClass(CustomLogger)
log = logging.getLogger("pipeline") # Use the custom logger
log.info("🚀 Pipeline service starting", extra={"request_id": ""}) # Initial message without a request_id

# ── Down-stream endpoints ─────────────────────────────────
ENDPOINTS = {
    "download":  ("downloader", 8000, "download"),
    "stt":       ("stt",        8001, "stt"),
    "translate": ("translate",  8002, "translate"),
    "tts":       ("tts",        8003, "tts"),
}

def call(name: str, payload: dict, request_id: str):
    """
    Calls a downstream service endpoint.
    Includes request_id for consistent logging.
    """
    host, port, ep = ENDPOINTS[name]
    url = f"http://{host}:{port}/{ep}"
    log.debug("→ %s  POST %s", name.upper(), url, extra={"request_id": request_id})
    try:
        resp = requests.post(url, json=payload, timeout=3600)
        if resp.status_code >= 400:
            log.error("%s %s\n%s", resp.status_code, resp.reason, resp.text, extra={"request_id": request_id})
        resp.raise_for_status() # Raise an exception for 4xx or 5xx status codes
        log.debug("← %s response received (status: %d)", name.upper(), resp.status_code, extra={"request_id": request_id})
        return resp.json()
    except requests.exceptions.Timeout:
        log.error("Timeout occurred while calling %s endpoint: %s", name.upper(), url, extra={"request_id": request_id})
        raise HTTPException(status_code=504, detail=f"Timeout calling {name} service.")
    except requests.exceptions.ConnectionError:
        log.error("Connection error while calling %s endpoint: %s", name.upper(), url, extra={"request_id": request_id})
        raise HTTPException(status_code=503, detail=f"Failed to connect to {name} service. Is it running?")
    except requests.exceptions.RequestException as e:
        # Catch any other requests-related exceptions
        log.error("Error calling %s endpoint: %s", name.upper(), e, exc_info=True, extra={"request_id": request_id})
        # If resp was obtained, use its status code, otherwise default to 500
        status_code = resp.status_code if 'resp' in locals() and resp is not None else 500
        detail = f"Error from {name} service: {e}"
        if 'resp' in locals() and resp is not None:
             detail = f"Error from {name} service (status {resp.status_code}): {resp.text}"
        raise HTTPException(status_code=status_code, detail=detail)


# ── API schema ────────────────────────────────────────────
class PipeReq(BaseModel):
    url          : str
    src_lang     : str | None = None
    tgt_lang     : str
    style        : str | None = None

class PipeResp(BaseModel):
    wav: str

app = FastAPI(title="Video-to-Voice Pipeline")

# ── Endpoint ──────────────────────────────────────────────
@app.post("/pipeline", response_model=PipeResp)
def pipeline(req: PipeReq):
    request_id = uuid.uuid4().hex # Generate a unique ID for each request
    extra_log_context = {"request_id": request_id} # Context for this request's logs

    try:
        log.info("📥 Pipeline request received: URL='%s', target_lang='%s', style='%s'",
                 req.url, req.tgt_lang, req.style if req.style else "N/A",
                 extra=extra_log_context)

        # Step 1: Download Audio
        log.info("Calling downloader service...", extra=extra_log_context)
        dl = call("download", {"url": req.url}, request_id)
        wav = dl["wav"]
        log.info("✅ Downloader returned WAV: %s", wav, extra=extra_log_context)

        # Step 2: Speech-to-Text (STT)
        stt_payload = {"wav": wav, "language": req.src_lang}
        tgt_is_en = req.tgt_lang.lower() in {"en", "eng"}

        if tgt_is_en:
            stt_payload["task"] = "translate" # Whisper can directly translate to English
            log.debug("Target language is English; setting STT task to 'translate'.", extra=extra_log_context)
        else:
            log.debug("Target language is not English; STT task will be 'transcribe'.", extra=extra_log_context)

        log.info("Calling STT service...", extra=extra_log_context)
        stt = call("stt", stt_payload, request_id)
        segments = stt.get("segments", [])
        detected_lang = stt.get("detected_src_lang")
        stt_full_text = stt.get("text", "")

        if not segments:
            log.warning("STT returned no segments for the audio.", extra=extra_log_context)
            # Decide if this should be an error or continue (e.g., empty audio)
            raise HTTPException(status_code=400, detail="STT service returned no text segments.")

        log.info("✅ STT done. Detected source language: '%s', Number of segments: %d",
                 detected_lang, len(segments), extra=extra_log_context)
        log.debug("STT full text (first 100 chars): '%s...'", stt_full_text[:100], extra=extra_log_context)

        # Step 3: Machine Translation (MT) if needed
        # Need MT if detected language is different from target AND target is NOT English
        # (because STT already handled en translation)
        need_mt = (detected_lang and detected_lang != req.tgt_lang.lower()) and not tgt_is_en

        if need_mt:
            log.info("Translation required: '%s' → '%s'", detected_lang, req.tgt_lang, extra=extra_log_context)
            mt = call("translate", {
                "text": stt_full_text, # Pass the full text for translation
                "src_lang": detected_lang,
                "tgt_lang": req.tgt_lang
            }, request_id)

            trans_text = mt.get("text", "")
            if not trans_text:
                log.warning("Translation service returned empty text.", extra=extra_log_context)
                # This might indicate an issue with translation for the given text/languages
                # You might want to raise an error or proceed with original text
                # For now, we'll proceed but log the warning.

            # Re-align translated text with segments if necessary.
            # Assuming translate service returns a single string for the whole text.
            # A more robust solution would involve segment-by-segment translation or re-alignment.
            # For now, we'll just assign the full translated text to the first segment's text
            # and clear others, or handle it based on a line-by-line split if applicable.
            # The current code assumes a direct 1:1 mapping of lines from original segments to translated lines.
            # If `mt["text"]` is a single string, this `split("\n")` logic might be problematic.
            # Let's assume `mt["text"]` can contain newlines corresponding to segment breaks.
            trans_lines = trans_text.split("\n")

            if len(trans_lines) != len(segments):
                log.warning(f"Number of translated lines ({len(trans_lines)}) does not match original segments ({len(segments)}). "
                            "Attempting best-effort text assignment to segments.", extra=extra_log_context)
                # Fallback: if line counts don't match, just use the full translated text for the first segment
                # or find a better way to re-distribute. For this example, we'll combine all.
                if trans_lines:
                    # Create a single segment with the entire translated text if segment count mismatch
                    segments = [{"text": " ".join(trans_lines), "start": segments[0]["start"] if segments else 0, "end": segments[-1]["end"] if segments else 0}]
                else:
                    segments = [] # No translation text means no segments
            else:
                segments = [
                    {**seg, "text": txt}
                    for seg, txt in zip(segments, trans_lines)
                ]
            log.info("✅ Machine Translation done. Translated text (first 100 chars): '%s...'", trans_text[:100], extra=extra_log_context)
        else:
            log.info("Machine Translation not needed.", extra=extra_log_context)
            if detected_lang and detected_lang != req.tgt_lang.lower() and tgt_is_en:
                log.info("Translation to English already handled by STT (Whisper's translate task).", extra=extra_log_context)
            else:
                log.info("Source and target languages are the same, or no specific translation task requested.", extra=extra_log_context)

        # Step 4: Text-to-Speech (TTS)
        tts_req_segments = [
            {"text": s["text"].strip(), "start": s["start"], "end": s["end"]}
            for s in segments if s["text"].strip() # Only include segments with actual text
        ]

        if not tts_req_segments:
            log.warning("No valid text segments to send to TTS after processing.", extra=extra_log_context)
            # If no segments with text, we can't perform TTS.
            raise HTTPException(status_code=400, detail="No text available for Text-to-Speech synthesis.")

        tts_req = {
            "speaker_wav": wav, # Use the original downloaded audio for speaker embedding
            "lang": req.tgt_lang,
            "segments": tts_req_segments
        }
        if req.style:
            tts_req["style"] = req.style
            log.debug("TTS request includes style: %s", req.style, extra=extra_log_context)

        log.info("Calling TTS service with %d segments...", len(tts_req_segments), extra=extra_log_context)
        tts_resp = call("tts", tts_req, request_id)

        stitched_wav = tts_resp.get("stitched")
        if not stitched_wav:
            log.error("TTS service did not return a 'stitched' WAV file path.", extra=extra_log_context)
            raise HTTPException(status_code=500, detail="TTS service failed to produce a stitched audio file.")

        log.info("✅ Pipeline finished successfully. Final output WAV: %s", stitched_wav, extra=extra_log_context)
        return {"wav": stitched_wav}

    except HTTPException as exc:
        # Re-raise HTTPExceptions directly, as they are already properly formatted with status code and detail
        log.error("Pipeline failed with HTTPException: %s", exc.detail, extra=extra_log_context)
        raise exc
    except Exception as exc:
        # Catch any other unexpected errors and log them comprehensively
        log.exception("Pipeline failed with an unhandled exception: %s", exc, extra=extra_log_context)
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred: {exc}")