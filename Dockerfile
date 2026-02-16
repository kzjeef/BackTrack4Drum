FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY remove_instrument.py .

# Pre-download the default model
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs_6s')"

VOLUME ["/data/input", "/data/output"]

ENTRYPOINT ["python", "remove_instrument.py"]
