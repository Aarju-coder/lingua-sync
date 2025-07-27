import logging, uuid, hashlib, gc
from pathlib import Path
from typing import List

import torch, torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torchaudio

MIN_SEC = 0.5

# ── lo                           # 2) if it's too short, pad with silence up to MIN_SEC
                if dur < MIN_SEC:
                    pad_samples = int((MIN_SEC - dur) * sr_seg)
                    wav_seg = torch.nn.functional.pad(wav_seg, (0, pad_samples))
                    torchaudio.save(segment_temp_wav, wav_seg, sr_seg)
                
                # 3) now run VAD as normal—silence will be dropped internally
                src_se, _ = se_extractor.get_se(
                    segment_temp_wav,
                    tcc,
                    vad=True
                )s too short, pad with silence up to MIN_SEC
                if dur < MIN_SEC:
                    pad_samples = int((MIN_SEC - dur) * sr_seg)
                    wav_seg = torch.nn.functional.pad(wav_seg, (0, pad_samples))
                    torchaudio.save(segment_temp_wav, wav_seg, sr_seg)
                
                # 3) now run VAD as normal—silence will be dropped internally
                src_se, _ = se_extractor.get_se(
                    segment_temp_wav,
                    tcc,
                    vad=True
                )─────────────────────────────────────
DATA = Path("/data"); DATA.mkdir(exist_ok=True, parents=True) # Ensure DATA dir exists
LOG_FILE = DATA / "tts.log"

# import os
# os.environ["HF_HUB_OFFLINE"] = "1"

# Custom Logger Class to include request_id in format
class RequestIdLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        if extra is None:
            extra = {}
        if 'request_id' not in extra:
            extra['request_id'] = "" 
        super()._log(level, msg, args, exc_info, extra, stack_info)

logging.setLoggerClass(RequestIdLogger)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s %(request_id)s | %(message)s",
    force=True,
)
# Suppress noisy loggers and specifically the 'too short audio' or 'bert' warnings
for noisy in ("numba", "urllib3", "openvoice.se_extractor", "transformers.modeling_utils"):
    logging.getLogger(noisy).setLevel(logging.WARNING) # or logging.ERROR for more aggressive suppression

log = logging.getLogger("tts")
log.setLevel(logging.DEBUG)

log.info("🚀 TTS service starting", extra={"request_id": ""})

# ── OpenVoice + Melo-TTS ─────────────────────────────────
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS as MELO_TTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_DIR = Path(__file__).parent / "OpenVoiceV2" / "converter"
log.info(f"Loading OpenVoice converter from {CKPT_DIR} on device: {DEVICE}", extra={"request_id": ""})
try:
    tcc = ToneColorConverter(str(CKPT_DIR / "config.json"), device=DEVICE)
    tcc.load_ckpt(str(CKPT_DIR / "checkpoint.pth"))
    log.info("✓ OpenVoice converter ready.", extra={"request_id": ""})
except Exception as e:
    log.critical(f"Failed to load OpenVoice converter: {e}", exc_info=True, extra={"request_id": ""})
    raise RuntimeError(f"Failed to load OpenVoice converter: {e}")

LANG_TO_KEY = {
    "en": "EN",
    "es": "ES",
    "fr": "FR",
    "zh": "ZH",
    "ja": "JP",
    "jp": "JP",
    "ko": "KR",
    "kr": "KR",
}
# SES_DIR is not strictly needed for this revised approach as we'll extract SE per segment.
# However, if you keep base speaker embeddings for some reason, ensure the path is correct.

_MELO_CACHE: dict[str, MELO_TTS] = {}      # Cache for Melo-TTS models
_CACHE_SIZE_LIMIT = 2  # Limit number of cached models

def _get_melo(language_code: str, request_id: str) -> MELO_TTS:
    """
    Retrieves or initializes a Melo-TTS model for a given language.
    Memory-efficient version with cache size limit.
    """
    if language_code not in _MELO_CACHE:
        # If cache is at limit, remove the oldest entry
        if len(_MELO_CACHE) >= _CACHE_SIZE_LIMIT:
            oldest_lang = next(iter(_MELO_CACHE))
            log.info(f"[{request_id}] Removing cached Melo-TTS model for '{oldest_lang}' to free memory", extra={"request_id": request_id})
            del _MELO_CACHE[oldest_lang]
            # Force garbage collection and clear GPU cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        log.info(f"[{request_id}] Initializing Melo-TTS model for language: {language_code}", extra={"request_id": request_id})
        try:
            # Melo-TTS expects specific language codes like 'en', 'es', 'zh', 'jp', 'kr', 'fr'
            # The LANG_TO_KEY mapping for Melo-TTS should reflect these.
            # For general English, just 'en' is common.
            melo_lang_code = LANG_TO_KEY.get(language_code.lower(), "EN")
            _MELO_CACHE[language_code] = MELO_TTS(language=melo_lang_code, device=DEVICE)
            log.info(f"[{request_id}] Melo-TTS model for '{language_code}' initialized.", extra={"request_id": request_id})
        except Exception as e:
            log.error(f"[{request_id}] Failed to initialize Melo-TTS for language '{language_code}': {e}", exc_info=True, extra={"request_id": request_id})
            raise HTTPException(status_code=500, detail=f"Failed to load TTS model for language: {language_code}")
    else:
        log.debug(f"[{request_id}] Melo-TTS model for '{language_code}' retrieved from cache.", extra={"request_id": request_id})
    return _MELO_CACHE[language_code]

# ── helpers ──────────────────────────────────────────────
def fit_len(wav: torch.Tensor, sr: int, tgt_len_samples: int, request_id: str) -> torch.Tensor:
    """
    Adjusts the length of an audio tensor to a target number of samples.
    """
    current_len_samples = len(wav)
    log.debug(f"[{request_id}] Adjusting WAV length: current={current_len_samples}, target={tgt_len_samples}", extra={"request_id": request_id})
    
    if current_len_samples == tgt_len_samples:
        log.debug(f"[{request_id}] WAV length already matches target.", extra={"request_id": request_id})
        return wav
    
    if current_len_samples > tgt_len_samples:
        log.debug(f"[{request_id}] Trimming WAV from {current_len_samples} to {tgt_len_samples} samples.", extra={"request_id": request_id})
        return wav[:tgt_len_samples]
    else: # current_len_samples < tgt_len_samples
        log.debug(f"[{request_id}] Padding WAV from {current_len_samples} to {tgt_len_samples} samples.", extra={"request_id": request_id})
        return torch.nn.functional.pad(wav, (0, tgt_len_samples - current_len_samples))


# ── schema ───────────────────────────────────────────────
class Segment(BaseModel):
    text : str
    start: float
    end  : float

class TTSBatchRequest(BaseModel):
    speaker_wav: str # Path to speaker reference WAV
    lang       : str # Language code for segments
    segments   : List[Segment] # List of text segments with timing

class TTSBatchResponse(BaseModel):
    snippets: List[str] # Paths to individual converted audio snippets
    stitched: str        # Path to the final stitched audio file

# ── FastAPI ──────────────────────────────────────────────
app = FastAPI(title="OV2 + Melo-TTS")
    
@app.on_event("startup")
async def preload_tts():
    # Force garbage collection before loading models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Only preload the default EN model to save memory
    _get_melo("en", request_id="")
    log.info("✅ Melo-TTS EN model preloaded on startup", extra={"request_id": ""})
    
    # Force another cleanup after preloading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@app.post("/tts", response_model=TTSBatchResponse)
def tts_batch(req: TTSBatchRequest):
    request_id = uuid.uuid4().hex
    extra_log_context = {"request_id": request_id}
    
    log.info("📥 TTS request received: speaker_wav='%s', lang='%s', segments=%d", 
             req.speaker_wav, req.lang, len(req.segments), extra=extra_log_context)
    log.debug("First segment: '%s'", req.segments[0].text[:50] if req.segments else "N/A", extra=extra_log_context)

    speaker_wav_path = Path(req.speaker_wav)
    if not speaker_wav_path.exists():
        log.error(f"[{request_id}] Speaker WAV file not found: {req.speaker_wav}", extra=extra_log_context)
        raise HTTPException(status_code=404, detail=f"Speaker WAV file not found: {req.speaker_wav}")
    if not speaker_wav_path.is_file():
        log.error(f"[{request_id}] Speaker WAV path is not a file: {req.speaker_wav}", extra=extra_log_context)
        raise HTTPException(status_code=400, detail=f"Speaker WAV path is not a file: {req.speaker_wav}")
    if not req.segments:
        log.warning(f"[{request_id}] No segments provided for TTS.", extra=extra_log_context)
        raise HTTPException(status_code=400, detail="No segments provided for TTS.")

    # List to keep track of all temporary files generated in this request
    temp_files_to_clean: List[Path] = []

    try:
        # Step 1: Extract target voice embedding once from the speaker_wav
        log.info(f"[{request_id}] Extracting target voice embedding from '{req.speaker_wav}'...", extra=extra_log_context)
        
        # Suppress potential warnings from se_extractor about "too short audio" if `vad=True` is problematic
        # For production, ensure your speaker_wav is of sufficient quality and length.
        tgt_se, _ = se_extractor.get_se(req.speaker_wav, tcc, vad=True) 
        log.info(f"[{request_id}] Target voice embedding extracted successfully.", extra=extra_log_context)

        melo_model = _get_melo(req.lang, request_id)
        # Assuming all Melo-TTS models have at least one speaker ID for generation
        # You might need to adjust spk_id selection if your Melo-TTS models have multiple speakers
        spk_id = next(iter(melo_model.hps.data.spk2id.values()))

        snippets: list[str] = []
        
        # Step 2: Process each segment
        for i, seg in enumerate(req.segments):
            segment_temp_wav = DATA / f"melo_segment_{uuid.uuid4().hex}.wav"
            converted_wav_out = DATA / f"converted_segment_{uuid.uuid4().hex}.wav"
            temp_files_to_clean.extend([segment_temp_wav, converted_wav_out])

            log.debug(f"[{request_id}] Processing segment {i+1}/{len(req.segments)}: Text='{seg.text[:50]}...', Start={seg.start:.2f}, End={seg.end:.2f}", extra=extra_log_context)
            
            if not seg.text.strip():
                log.warning(f"[{request_id}] Segment {i+1} has no text, skipping.", extra=extra_log_context)
                continue
            
            target_duration_seconds = seg.end - seg.start
            if target_duration_seconds <= 0:
                log.warning(f"[{request_id}] Segment {i+1} has non-positive duration ({target_duration_seconds:.2f}s), skipping.", extra=extra_log_context)
                continue

            # Generate audio for the segment text using Melo-TTS
            log.debug(f"[{request_id}] Generating Melo-TTS audio for segment {i+1} to '{segment_temp_wav}'...", extra=extra_log_context)
            try:
                melo_model.tts_to_file(seg.text, spk_id, str(segment_temp_wav), speed=1.0)
                log.debug(f"[{request_id}] Melo-TTS audio generated for segment {i+1}.", extra=extra_log_context)
            except Exception as e:
                log.error(f"[{request_id}] Failed to generate Melo-TTS audio for segment {i+1}: {e}", exc_info=True, extra=extra_log_context)
                raise HTTPException(status_code=500, detail=f"Failed to generate base audio for segment {i+1}: {e}")

            # Extract source embedding from the Melo-TTS generated audio
            log.debug(f"[{request_id}] Extracting source embedding from Melo-TTS generated audio for segment {i+1}...", extra=extra_log_context)
            try:
                # 1) load the just-synthesized clip
                wav_seg, sr_seg = torchaudio.load(segment_temp_wav)
                dur = wav_seg.shape[-1] / sr_seg

                # 2) if it’s too short, pad with silence up to MIN_SEC
                if dur < MIN_SEC:
                    pad_samples = int((MIN_SEC - dur) * sr_seg)
                    wav_seg = torch.nn.functional.pad(wav_seg, (0, pad_samples))
                    torchaudio.save(segment_temp_wav, wav_seg, sr_seg)
                
                # 3) now run VAD as normal—silence will be dropped internally
                src_se, _ = se_extractor.get_se(
                    str(segment_temp_wav),
                    tcc,
                    vad=True
                )
                log.debug(f"[{request_id}] Source embedding extracted for segment {i+1}.", extra=extra_log_context)
            except Exception as e:
                log.error(f"[{request_id}] Failed to extract source embedding for segment {i+1}: {e}", exc_info=True, extra=extra_log_context)
                raise HTTPException(status_code=500, detail=f"Failed to extract source embedding for segment {i+1}: {e}")

            # Convert voice color from the segment's source audio to the target speaker's tone
            log.debug(f"[{request_id}] Converting voice color for segment {i+1} to '{converted_wav_out}'...", extra=extra_log_context)
            try:
                tcc.convert(
                    audio_src_path = str(segment_temp_wav), # Use the Melo-TTS generated audio as source
                    src_se         = src_se,
                    tgt_se         = tgt_se,
                    output_path    = str(converted_wav_out),
                    message        = "@MyShell",
                )
                log.debug(f"[{request_id}] Voice color conversion complete for segment {i+1}.", extra=extra_log_context)
            except Exception as e:
                log.error(f"[{request_id}] Failed voice color conversion for segment {i+1}: {e}", exc_info=True, extra=extra_log_context)
                raise HTTPException(status_code=500, detail=f"Voice conversion failed for segment {i+1}: {e}")
            
            # Adjust length to match segment timing
            log.debug(f"[{request_id}] Adjusting length for segment {i+1} to {target_duration_seconds:.2f} seconds.", extra=extra_log_context)
            try:
                wav, sr = torchaudio.load(converted_wav_out)
                if wav.ndim > 1:
                    wav = wav.mean(dim=0) # Ensure mono audio
                
                target_samples = int(target_duration_seconds * sr)
                wav = fit_len(wav.squeeze(0), sr, target_samples, request_id)
                
                torchaudio.save(converted_wav_out, wav.unsqueeze(0), sr)
                log.debug(f"[{request_id}] Segment {i+1} length adjusted and saved to '{converted_wav_out}'.", extra=extra_log_context)
                snippets.append(str(converted_wav_out))
            except Exception as e:
                log.error(f"[{request_id}] Failed to adjust length or save segment {i+1}: {e}", exc_info=True, extra=extra_log_context)
                raise HTTPException(status_code=500, detail=f"Failed to process audio length for segment {i+1}: {e}")

        if not snippets:
            log.warning(f"[{request_id}] No valid snippets generated after processing all segments. Returning empty stitched path.", extra=extra_log_context)
            return {"snippets": [], "stitched": ""}
            
        # Step 3: Stitch the clips
        log.info(f"[{request_id}] Stitching {len(snippets)} audio snippets...", extra=extra_log_context)
        tensors = []
        sr = None
        for fp in snippets:
            try:
                w, _sr = torchaudio.load(fp)
                if sr is None:
                    sr = _sr
                elif sr != _sr:
                    log.warning(f"[{request_id}] Mismatch in sample rates detected: expected {sr}, got {_sr} for '{fp}'. Resampling.", extra=extra_log_context)
                    w = torchaudio.functional.resample(w, _sr, sr)
                tensors.append(w)
            except Exception as e:
                log.error(f"[{request_id}] Failed to load snippet '{fp}' for stitching: {e}", exc_info=True, extra=extra_log_context)
                continue

        if not tensors:
            log.error(f"[{request_id}] No audio tensors available to stitch.", extra=extra_log_context)
            raise HTTPException(status_code=500, detail="No audio data to stitch into a full file.")

        stitched = torch.cat(tensors, dim=1)
        stitched_path = DATA / f"{uuid.uuid4().hex}_full.wav"
        temp_files_to_clean.append(stitched_path) # Add final stitched file to cleanup list if error occurs before return
        
        try:
            torchaudio.save(stitched_path, stitched, sr)
            log.info(f"[{request_id}] All snippets stitched successfully to: {stitched_path}", extra=extra_log_context)
        except Exception as e:
            log.error(f"[{request_id}] Failed to save stitched audio to '{stitched_path}': {e}", exc_info=True, extra=extra_log_context)
            raise HTTPException(status_code=500, detail=f"Failed to save final stitched audio: {e}")

        log.info(f"[{request_id}] ✓ TTS processing complete. Generated {len(snippets)} snippets, stitched to '{stitched_path}'", extra=extra_log_context)
        return {"snippets": snippets, "stitched": str(stitched_path)}

    except HTTPException as e:
        log.error(f"[{request_id}] TTS request failed with HTTPException: {e.detail}", extra=extra_log_context)
        raise e
    except Exception as e:
        log.exception(f"[{request_id}] TTS request failed with an unhandled exception: {e}", extra=extra_log_context)
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred during TTS: {e}")

    finally:
        log.debug(f"[{request_id}] Cleaning up memory and temporary files...", extra=extra_log_context)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.debug(f"[{request_id}] CUDA cache emptied.", extra=extra_log_context)
        
        # Clean up all temporary files generated during this request
        for temp_file in temp_files_to_clean:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                    log.debug(f"[{request_id}] Deleted temporary file: '{temp_file}'", extra=extra_log_context)
                except OSError as e:
                    log.warning(f"[{request_id}] Failed to delete temporary file '{temp_file}': {e}", extra=extra_log_context)