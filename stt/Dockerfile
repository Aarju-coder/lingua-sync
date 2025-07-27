FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# make sure cuDNN libs are found
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# system deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install Python deps
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && pip3 install --no-cache-dir -r requirements.txt

# copy the API
COPY app.py .

# serve
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
