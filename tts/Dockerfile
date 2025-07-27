FROM python:3.10-slim

# 1. system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

# 2. Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. *** NEW ***  download the full UniDic once
#     (≈ 390 MB extra in the image, 2-3 min build time)
RUN python -m unidic download

# download once – no interactive prompt, silent
RUN python - <<'PY'
import nltk, sys, pathlib, textwrap

dest = pathlib.Path("/app/nltk_data")
dest.mkdir(exist_ok=True)
pkgs = ["averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "punkt",
        "cmudict"]

for p in pkgs:
    print(f"• downloading {p} …", file=sys.stderr)
    nltk.download(p, download_dir=str(dest), quiet=True, raise_on_error=True)
PY

# make sure every python process sees it
ENV NLTK_DATA=/app/nltk_data


# 0. Make a dedicated cache dir
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV HF_DATASETS_CACHE=/app/hf_cache
RUN mkdir -p /app/hf_cache
ENV HF_HUB_OFFLINE=0
# allow online fetches for caching
ENV HF_HUB_OFFLINE=0

RUN python - <<'PY'
import torch
# pre-cache Silero-VAD
torch.hub._validate_not_a_forked_repo = lambda a,b,c: True
_ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

from transformers import (
  BertTokenizer, BertForMaskedLM,
  AutoTokenizer, AutoModel
)

# English (most common, always load)
BertTokenizer.from_pretrained("bert-base-uncased")
BertForMaskedLM.from_pretrained("bert-base-uncased")

# Load only multilingual BERT instead of individual language models to save memory
AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
AutoModel.from_pretrained("bert-base-multilingual-uncased")

import wavmark
wavmark.load_model()

from melo.api import TTS
# Load only English model at build time, others loaded on demand
print("Caching MeloTTS model for EN…")
TTS(language="EN", device="cpu")
PY

# now lock down to offline for runtime
ENV HF_HUB_OFFLINE=1

# ─────────────────────────────────────────────────────────
# Nothing else is required – MeCab now finds
# /usr/local/lib/python3.10/site-packages/unidic/dicdir/mecabrc
# on its own.  **Do NOT keep the old  `ENV MECABRC=…`  line.**
# ─────────────────────────────────────────────────────────

# 4. app code
COPY . .
EXPOSE 8003
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
