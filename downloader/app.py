import logging, os, uuid, subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import yt_dlp

# ──────────── Logging Setup ────────────
LOG_FILE = "/data/downloader.log"
# Ensure the data directory exists before setting up the log file
Path("/data").mkdir(exist_ok=True, parents=True)

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,  # Set default logging level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to a file
        logging.StreamHandler()         # Log to console
    ]
)
logger = logging.getLogger(__name__) # Get a logger instance for this module

logger.info("Downloader service starting up.")

# ──────────── App & Model ────────────
app = FastAPI(title="Downloader")

DATA = Path("/data")
DATA.mkdir(exist_ok=True)
logger.info(f"Data directory ensured: {DATA}")

class DLReq(BaseModel):
    url: str
    start: int | None = None    # optional, seconds
    duration: int | None = None # optional, seconds

def _dload(url: str, start: int|None, duration: int|None, request_id: str) -> Path:
    """
    Internal function to handle the actual downloading and conversion.
    Includes request_id for tracing.
    """
    logger.info(f"[{request_id}] Starting download process for URL: {url}")
    logger.debug(f"[{request_id}] Parameters: start={start}, duration={duration}")

    vid = uuid.uuid4().hex
    tmp = DATA / f"{vid}.%(ext)s"
    
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(tmp),
        "quiet": True, # Suppress yt-dlp output to stdout/stderr
        "noplaylist": True, # Ensure only single video is downloaded
        "postprocessor_args": [],
        "logger": logger # Pass our logger to yt-dlp
    }

    if start is not None:
        opts["postprocessor_args"] += ["-ss", str(start)]
        logger.debug(f"[{request_id}] Added start time to ffmpeg args: -ss {start}")
    if duration is not None:
        opts["postprocessor_args"] += ["-t", str(duration)]
        logger.debug(f"[{request_id}] Added duration to ffmpeg args: -t {duration}")

    src = None # Initialize src to None
    try:
        logger.info(f"[{request_id}] Initializing yt-dlp download for URL: {url}")
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            src = Path(ydl.prepare_filename(info))
        logger.info(f"[{request_id}] yt-dlp download complete. Source file: {src}")
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"[{request_id}] yt-dlp Download Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {e}")
    except Exception as e:
        logger.error(f"[{request_id}] An unexpected error occurred during yt-dlp download: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during download: {e}")

    wav = DATA / f"{vid}.wav"
    logger.info(f"[{request_id}] Converting downloaded file '{src}' to WAV format: '{wav}'")
    
    try:
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)]
        logger.debug(f"[{request_id}] FFmpeg command: {' '.join(ffmpeg_cmd)}")
        
        result = subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
            text=True # Capture stdout/stderr as text
        )
        logger.debug(f"[{request_id}] FFmpeg stdout: {result.stdout.strip()}")
        logger.debug(f"[{request_id}] FFmpeg stderr: {result.stderr.strip()}")
        logger.info(f"[{request_id}] FFmpeg conversion successful for '{src}'.")
    except subprocess.CalledProcessError as e:
        logger.error(f"[{request_id}] FFmpeg failed with exit code {e.returncode}: {e.stderr.strip()}")
        if src and src.exists():
            logger.warning(f"[{request_id}] Cleaning up partially downloaded source file: {src}")
            src.unlink() # Clean up the source file even on ffmpeg failure
        raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {e.stderr.strip()}")
    except Exception as e:
        logger.error(f"[{request_id}] An unexpected error occurred during FFmpeg conversion: {e}", exc_info=True)
        if src and src.exists():
            logger.warning(f"[{request_id}] Cleaning up partially downloaded source file: {src}")
            src.unlink()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during conversion: {e}")

    if src and src.exists():
        try:
            src.unlink()
            logger.info(f"[{request_id}] Original downloaded file '{src}' deleted successfully.")
        except OSError as e:
            logger.error(f"[{request_id}] Error deleting original downloaded file '{src}': {e}")
    else:
        logger.warning(f"[{request_id}] Original downloaded file '{src}' not found for deletion (might have been deleted already or not created).")

    logger.info(f"[{request_id}] Successfully created WAV file: {wav}")
    return wav

@app.post("/download")
def download(req: DLReq):
    request_id = str(uuid.uuid4()) # Generate a unique ID for each request
    logger.info(f"[{request_id}] Received /download request for URL: {req.url}")
    logger.debug(f"[{request_id}] Request details: start={req.start}, duration={req.duration}")

    try:
        wav = _dload(req.url, req.start, req.duration, request_id)
        logger.info(f"[{request_id}] Download and conversion completed successfully. WAV file: {wav}")
        return {"wav": str(wav)}
    except HTTPException as e:
        # Re-raise HTTPExceptions directly, as they are already properly formatted
        logger.error(f"[{request_id}] Download endpoint failed with HTTPException: {e.detail}")
        raise e
    except Exception as e:
        logger.exception(f"[{request_id}] Download endpoint failed with an unhandled exception.")
        # Raise a generic HTTPException for any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")