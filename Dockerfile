FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    gnupg \
    git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --upgrade pip setuptools wheel

ARG TORCH_VERSION=2.7.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
RUN python3.12 -m pip install \
    torch==${TORCH_VERSION} \
    torchvision \
    torchaudio \
    --index-url ${TORCH_INDEX_URL}

ARG SAM3_REF=main
RUN python3.12 -m pip install "sam3 @ git+https://github.com/facebookresearch/sam3.git@${SAM3_REF}"

RUN python3.12 -m pip install \
    fastapi \
    uvicorn[standard] \
    pillow \
    python-multipart

RUN useradd -m -u 10001 appuser \
    && mkdir -p /app /tmp/huggingface \
    && chown -R appuser:appuser /app /tmp/huggingface

WORKDIR /app
COPY main.py /app/main.py

USER appuser
ENV PORT=8080

EXPOSE 8080
CMD ["sh", "-c", "python3.12 -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]