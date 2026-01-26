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
    curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12 from upstream (avoids distutils issues)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.12 /tmp/get-pip.py \
    && rm -f /tmp/get-pip.py \
    && python3.12 -m pip install --upgrade pip setuptools wheel

ARG TORCH_VERSION=2.7.0
ARG TORCHVISION_VERSION=0.22.0
ARG TORCHAUDIO_VERSION=2.7.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
RUN python3.12 -m pip install \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url ${TORCH_INDEX_URL}

ARG SAM3_REF=11dec2936de97f2857c1f76b66d982d5a001155d
RUN python3.12 -m pip install "sam3 @ git+https://github.com/facebookresearch/sam3.git@${SAM3_REF}"

# Runtime deps (include missing einops)
RUN python3.12 -m pip install \
    fastapi \
    uvicorn[standard] \
    pillow \
    python-multipart \
    einops \
    decord \
    pycocotools \
    psutil \
 && python3.12 -c "import einops; print('einops OK', einops.__version__)"

RUN useradd -m -u 10001 appuser \
    && mkdir -p /app /tmp/huggingface \
    && chown -R appuser:appuser /app /tmp/huggingface

WORKDIR /app
COPY main.py /app/main.py

USER appuser
EXPOSE 8080

# Cloud Run expects the process to listen on 8080; keep it explicit
CMD ["python3.12", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
