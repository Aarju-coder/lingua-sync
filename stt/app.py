import logging, uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel
import torch

# ─── Logging & paths ──────────────────────────────────────────
DATA = Path("/data")
DATA.mkdir(exist_ok=True, parents=True) # Ensure DATA dir exists
LOG_FILE = DATA / "stt.log"

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,  # Set default logging level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(request_id)s | %(message)s", # Added %(request_id)s
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to a file
        logging.StreamHandler()         # Log to console
    ]
)

# Create a custom logger that can accept request_id
class RequestIdLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        if extra is None:
            extra = {}
        # Ensure request_id is always present, even if it's an empty string
        if 'request_id' not in extra:
            extra['request_id'] = "" 
        super()._log(level, msg, args, exc_info, extra, stack_info)

logging.setLoggerClass(RequestIdLogger)
logger = logging.getLogger(__name__) # Get a logger instance for this module

logger.info("🔊 STT service starting up", extra={"request_id": ""}) # Initial message without a request_id

app = FastAPI(title="Speech-to-Text (Whisper)")

model: WhisperModel # Declare model globally, will be initialized in startup event

@app.on_event("startup")
async def load_model():
    """
    Loads the Whisper model on application startup.
    """
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8" # Use int8 for CPU for better performance
    
    logger.info(f"Loading WhisperModel('large-v2', device='{device}', compute_type='{compute_type}')", extra={"request_id": ""})
    try:
        # Using 'large-v2' as it's a common and robust choice for 'large'
        model = WhisperModel("large-v2", device=device, compute_type=compute_type)
        logger.info("✅ Whisper model 'large-v2' loaded successfully.", extra={"request_id": ""})
    except Exception as e:
        logger.critical(f"Failed to load Whisper model: {e}", exc_info=True, extra={"request_id": ""})
        # It's critical if model fails to load, the app won't function.
        # Consider exiting or making the app unhealthy.
        raise RuntimeError(f"Failed to load Whisper model: {e}")

# ─── Pydantic I/O ─────────────────────────────────────────────
class STTReq(BaseModel):
    wav: str
    language: str | None = None
    task: str | None = None            # ("transcribe" or "translate")

# ─── Endpoint ─────────────────────────────────────────────────
@app.post("/stt")
def stt(req: STTReq):
    request_id = uuid.uuid4().hex # Generate a unique ID for each request
    extra_log_context = {"request_id": request_id} # Context for this request's logs

    logger.info("Received STT request for WAV file: '%s'", req.wav, extra=extra_log_context)
    logger.debug("Request parameters: language='%s', task='%s'", 
                 req.language if req.language else "auto-detect", 
                 req.task if req.task else "transcribe", 
                 extra=extra_log_context)

    wav_path = Path(req.wav)
    if not wav_path.exists():
        logger.error("WAV file not found at path: '%s'", req.wav, extra=extra_log_context)
        raise HTTPException(status_code=404, detail=f"WAV file not found: {req.wav}")
    if not wav_path.is_file():
        logger.error("Provided WAV path is not a file: '%s'", req.wav, extra=extra_log_context)
        raise HTTPException(status_code=400, detail=f"Provided WAV path is not a file: {req.wav}")

    try:
        logger.info("Starting transcription/translation for '%s'...", req.wav, extra=extra_log_context)
        # Ensure model is loaded before calling transcribe
        if 'model' not in globals() or model is None:
            logger.error("Whisper model not loaded. Cannot process request.", extra=extra_log_context)
            raise HTTPException(status_code=503, detail="STT service is not ready. Model not loaded.")

        # Perform transcription or translation
        seg_iter, info = model.transcribe(
            str(wav_path), # faster-whisper expects a string path or bytes
            language=req.language,
            task=req.task or "transcribe", 
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=250)
        )
        logger.info("Transcription/translation complete. Detected language: '%s', Transcription time: %.2f s", 
                    info.language, info.duration, extra=extra_log_context)

        segments = list(seg_iter)
        text_full = "".join(seg.text for seg in segments).strip()
        logger.debug("Full transcribed/translated text (first 100 chars): '%s...'", text_full[:100], extra=extra_log_context)

        # Write one plain-text file per job
        txt_path = DATA / f"{request_id}_{info.language}.txt"
        try:
            txt_path.write_text(text_full, encoding="utf-8")
            logger.info("Transcript saved to: '%s'", txt_path, extra=extra_log_context)
        except IOError as io_err:
            logger.error("Failed to write transcript to file '%s': %s", txt_path, io_err, exc_info=True, extra=extra_log_context)
            # Decide if this should stop the process or just be a warning
            # For now, we'll continue but the file might be missing.

        segs_json = [
            {"text": seg.text, "start": seg.start, "end": seg.end}
            for seg in segments
        ]
        logger.info("STT request processed successfully. Returning %d segments.", len(segs_json), extra=extra_log_context)
        
        return {
            "text": text_full,
            "txt_file": str(txt_path), 
            "detected_src_lang": info.language,
            "segments": segs_json
        }

    except Exception as e:
        logger.exception("STT processing failed for WAV file '%s': %s", req.wav, e, extra=extra_log_context)
        raise HTTPException(status_code=500, detail=f"Speech-to-Text processing failed: {e}")