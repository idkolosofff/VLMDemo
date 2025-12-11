ARG BASE_IMAGE=pytorch/pytorch:2.2.2-cuda12.1-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app

# Install system dependencies for OpenCV if using a slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install .

VOLUME ["/cache", "/artifacts"]

EXPOSE 7860

CMD ["python", "-m", "app.main"]
