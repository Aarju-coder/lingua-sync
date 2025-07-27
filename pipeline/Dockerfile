FROM python:3.11-slim

# make sure ffmpeg & pydub deps are available
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
