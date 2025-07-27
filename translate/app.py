from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from pathlib import Path
import uuid, re
import logging # Import the logging module

app = FastAPI(title="Universal Translate → English")

# ──────────── Configure Logging ────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("translator.log"), # Log to a file
                        logging.StreamHandler() # Log to console
                    ])
logger = logging.getLogger(__name__)

# ──────────── Load model once ────────────
MODEL_NAME = "facebook/m2m100_418M"
logger.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
logger.info("Tokenizer loaded successfully.")

logger.info(f"Loading model: {MODEL_NAME}")
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
logger.info("Model loaded successfully.")

# grab the set of supported language codes
SUPPORTED_LANGS = set(tokenizer.lang_code_to_id.keys())
logger.info(f"Supported languages loaded: {len(SUPPORTED_LANGS)} languages.")


# ──────────── Data folder ────────────
DATA = Path("/data")
DATA.mkdir(exist_ok=True)
logger.info(f"Data directory ensured: {DATA}")


class TransReq(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str = "en"


@app.get("/languages")
def list_languages():
    """
    Return the list of language codes this service can translate between.
    """
    logger.info("Received request for supported languages.")
    return {"supported": sorted(SUPPORTED_LANGS)}


def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    logger.debug(f"Chunking text (first 50 chars): '{text[:50]}...' with max_chars={max_chars}")
    # split on sentence enders (Hindi danda, ., !, ?)
    sents = re.split(r'(?<=[।\.!?])\s*', text)
    logger.debug(f"Text split into {len(sents)} sentences.")
    chunks, buf = [], ""
    for s in sents:
        if not s:
            continue
        if buf and len(buf) + len(s) + 1 > max_chars:
            chunks.append(buf)
            logger.debug(f"Appended chunk of size {len(buf)}.")
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        chunks.append(buf)
        logger.debug(f"Appended final chunk of size {len(buf)}.")
    logger.info(f"Text chunked into {len(chunks)} pieces.")
    return chunks


@app.post("/translate")
def translate(req: TransReq):
    request_id = str(uuid.uuid4()) # Generate a unique ID for each request
    logger.info(f"[{request_id}] Translation request received: src='{req.src_lang}', tgt='{req.tgt_lang}', text_len={len(req.text)}")

    # make sure the user only requests supported codes
    if req.src_lang not in SUPPORTED_LANGS:
        logger.warning(f"[{request_id}] Unsupported source language requested: {req.src_lang}")
        raise HTTPException(400, f"Unsupported src_lang: {req.src_lang}")
    if req.tgt_lang not in SUPPORTED_LANGS:
        logger.warning(f"[{request_id}] Unsupported target language requested: {req.tgt_lang}")
        raise HTTPException(400, f"Unsupported tgt_lang: {req.tgt_lang}")

    # configure tokenizer
    tokenizer.src_lang = req.src_lang
    bos_id = tokenizer.get_lang_id(req.tgt_lang)
    logger.info(f"[{request_id}] Tokenizer configured: src_lang='{req.src_lang}', target_bos_id='{bos_id}'.")

    # chop long text into ~1k‐char pieces
    chunks = chunk_text(req.text, max_chars=1000)
    if not chunks:
        logger.warning(f"[{request_id}] No text to translate after chunking.")
        raise HTTPException(400, "No text to translate")
    logger.info(f"[{request_id}] Text divided into {len(chunks)} chunks for translation.")

    output_parts = []
    for i, chunk in enumerate(chunks):
        logger.debug(f"[{request_id}] Translating chunk {i+1}/{len(chunks)} (length: {len(chunk)}).")
        encoded = tokenizer(chunk, return_tensors="pt",
                            truncation=True, max_length=1024)
        logger.debug(f"[{request_id}] Chunk {i+1} encoded. Input IDs shape: {encoded['input_ids'].shape}.")

        generated = model.generate(
            **encoded,
            forced_bos_token_id=bos_id,
            max_length=encoded["input_ids"].shape[1] * 2
        )
        logger.debug(f"[{request_id}] Chunk {i+1} generated translation.")

        out = tokenizer.batch_decode(generated,
                                     skip_special_tokens=True)[0]
        output_parts.append(out)
        logger.debug(f"[{request_id}] Chunk {i+1} decoded. Output length: {len(out)}.")

    final = "\n".join(output_parts)
    logger.info(f"[{request_id}] All chunks translated and joined. Final translated text length: {len(final)}.")

    # write out to a file for debugging
    fn = DATA / f"{uuid.uuid4().hex}_{req.src_lang}2{req.tgt_lang}.txt"
    try:
        fn.write_text(final, encoding="utf-8")
        logger.info(f"[{request_id}] Translated text successfully written to file: {fn}")
    except IOError as e:
        logger.error(f"[{request_id}] Error writing translated text to file {fn}: {e}")
        # Consider if you want to raise an HTTPException here or just log it

    logger.info(f"[{request_id}] Translation complete.")
    return {
        "text": final,
        "txt_file": str(fn)
    }