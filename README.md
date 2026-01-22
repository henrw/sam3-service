# sam3-service

FastAPI service that exposes SAM 3 image segmentation by text prompt, ready for
Cloud Run.

## Requirements

- Access to the SAM 3 checkpoints on Hugging Face:
  https://huggingface.co/facebook/sam3
- CUDA 12.6 GPU recommended. CPU works but will be very slow.

## Build

Default (CUDA wheels):

```bash
docker build -t sam3-service .
```

CPU-only wheels (keeps the CUDA base image but installs CPU Torch):

```bash
docker build -t sam3-service \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
```

Pin a SAM3 commit or tag:

```bash
docker build -t sam3-service --build-arg SAM3_REF=<commit-or-tag> .
```

The Dockerfile defaults to a pinned commit for reproducible builds; override it
with `SAM3_REF` as needed.

## Run locally

```bash
docker run --rm -p 8080:8080 \
  -e HF_TOKEN=your_hf_token \
  -e SAM3_DEVICE=cuda \
  sam3-service
```

```bash
curl -X POST http://localhost:8080/segment/text \
  -F "image=@/path/to/image.jpg" \
  -F "prompt=a dog"
```

## Cloud Run notes

- Set `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) as a secret/env var.
- Set `SAM3_DEVICE=cuda` when deploying with a GPU; use `cpu` otherwise.
- Expect large memory requirements due to the model size.
- The Cloud Build deploy uses a Secret Manager secret (default `hf-token`) via
  `_HF_TOKEN_SECRET`/`_HF_TOKEN_SECRET_VERSION` substitutions in
  `cloudbuild.yaml`.
- The Cloud Build deploy uses stable `gcloud run deploy`. If you need larger
  ephemeral storage for model downloads, upgrade your Cloud SDK until it
  supports `--ephemeral-storage`, or bake the model weights into the image.

## Configuration

- `HUGGINGFACE_HUB_TOKEN` or `HF_TOKEN`: Hugging Face access token.
- `SAM3_DEVICE`: `auto` (default), `cuda`, or `cpu`.
- `TORCH_NUM_THREADS`: optional CPU thread cap.
- `LOG_LEVEL`: log verbosity (default `INFO`).
- `PORT`: server port (Cloud Run uses 8080).
