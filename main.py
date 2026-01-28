import logging
import os
import threading
from io import BytesIO
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pycocotools import mask as mask_utils

# NOTE: DO NOT import sam3 at module import time; it can pull in training deps and crash uvicorn import.
# We import sam3 lazily inside the background loader.

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("sam3-service")

app = FastAPI(title="sam3-service")

processor = None  # type: Optional["Sam3Processor"]
model_device: Optional[str] = None
startup_error: Optional[str] = None


def _get_hf_token() -> str:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    logger.info(
        "HUGGINGFACE_HUB_TOKEN present=%s len=%s",
        bool(token),
        len(token) if token else 0,
    )
    if not token:
        raise RuntimeError("HUGGINGFACE_HUB_TOKEN is not set")
    return token


def _set_torch_threads() -> None:
    # Optional: keep startup responsive on CPU by limiting threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    threads = os.getenv("TORCH_NUM_THREADS")
    if threads:
        try:
            torch.set_num_threads(int(threads))
        except ValueError:
            logger.warning("Invalid TORCH_NUM_THREADS=%s, ignoring.", threads)
    else:
        torch.set_num_threads(1)


def _resolve_device() -> str:
    requested = os.getenv("SAM3_DEVICE", "auto").strip().lower()
    if requested in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("SAM3_DEVICE requested CUDA but no GPU detected; using CPU.")
        return "cpu"
    if requested != "cpu":
        logger.warning("SAM3_DEVICE=%s is unsupported; using CPU.", requested)
        return "cpu"
    return "cpu"


def _load_model_background() -> None:
    """Loads the SAM3 model without blocking the web server from starting."""
    global processor, model_device, startup_error

    try:
        # Ensure HF auth is registered for gated downloads
        token = _get_hf_token()
        from huggingface_hub import login  # lazy import

        login(token=token, add_to_git_credential=False)

        _set_torch_threads()
        model_device = _resolve_device()

        # Useful boot diagnostics
        try:
            logger.info(
                "Torch: version=%s cuda.is_available=%s torch.version.cuda=%s",
                torch.__version__,
                torch.cuda.is_available(),
                getattr(torch.version, "cuda", None),
            )
            if torch.cuda.is_available():
                logger.info(
                    "CUDA device: count=%s name=%s capability=%s",
                    torch.cuda.device_count(),
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_capability(0),
                )
        except Exception:
            logger.exception("Failed to log torch/cuda diagnostics.")

        logger.info("Loading SAM3 model on %s...", model_device)

        # Lazy imports to avoid killing uvicorn at import time
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(device=model_device)
        processor = Sam3Processor(model, device=model_device)

        startup_error = None
        logger.info("SAM3 model ready.")

    except Exception as exc:
        startup_error = str(exc)
        processor = None
        logger.exception("Failed to load SAM3 model.")


@app.on_event("startup")
def startup() -> None:
    t = threading.Thread(target=_load_model_background, daemon=True)
    t.start()


# @app.get("/healthz")
def healthz():
    """
    Cloud Run startup probes require a 2xx/3xx response to pass.
    Return 200 even while loading, but include status info.
    """
    if processor is None:
        return JSONResponse(status_code=200, content={"status": "loading", "error": startup_error})
    return {"status": "ok", "device": model_device}


@app.get("/health")
def health():
    return healthz()


def _load_image(img_bytes: bytes) -> Image.Image:
    if not img_bytes:
        raise ValueError("Image is empty.")
    try:
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image file.") from exc


def _encode_masks_rle(masks: torch.Tensor) -> List[dict]:
    if masks.numel() == 0:
        return []

    if masks.dim() == 4:
        masks = masks[:, 0, :, :]

    masks_np = masks.detach().cpu().numpy().astype(np.uint8)
    if masks_np.ndim == 2:
        masks_np = masks_np[:, :, None]
    else:
        masks_np = np.transpose(masks_np, (1, 2, 0))

    rles = mask_utils.encode(np.asfortranarray(masks_np))
    if isinstance(rles, dict):
        rles = [rles]

    for rle in rles:
        counts = rle.get("counts")
        if isinstance(counts, bytes):
            rle["counts"] = counts.decode("ascii")
    return rles


def _masks_to_contours(masks: torch.Tensor) -> List[List[List[List[int]]]]:
    if masks.numel() == 0:
        return []

    if masks.dim() == 4:
        masks = masks[:, 0, :, :]

    masks_np = masks.detach().cpu().numpy().astype(np.uint8)
    contours_per_mask: List[List[List[List[int]]]] = []
    for mask in masks_np:
        if mask.max() == 0:
            contours_per_mask.append([])
            continue
        mask_u8 = (mask * 255).astype(np.uint8)
        found = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]
        mask_contours: List[List[List[int]]] = []
        for contour in contours:
            if contour.size == 0:
                continue
            points = contour.reshape(-1, 2).tolist()
            mask_contours.append([[int(x), int(y)] for x, y in points])
        contours_per_mask.append(mask_contours)

    return contours_per_mask


def _format_output(
    state: dict,
    label: Optional[str] = None,
    include_masks: bool = False,
    include_contours: bool = False,
) -> dict:
    masks = state["masks"]
    boxes = state["boxes"]
    scores = state["scores"]

    result = {
        "boxes": boxes.detach().cpu().tolist(),
        "scores": scores.detach().cpu().tolist(),
        "mask_shape": list(masks.shape),
    }
    if label is not None:
        result["label"] = label
    if include_masks:
        result["masks_rle"] = _encode_masks_rle(masks)
    if include_contours:
        result["contours"] = _masks_to_contours(masks)
    return result


def _run_segmentation(img_bytes: bytes, prompt: str) -> dict:
    if processor is None:
        raise RuntimeError("Model not loaded")

    pil = _load_image(img_bytes)
    state = processor.set_image(pil)
    out = processor.set_text_prompt(state=state, prompt=prompt)

    return _format_output(out)


def _run_segmentation_batch(
    img_bytes: bytes,
    prompts: List[str],
    include_masks: bool = False,
    include_contours: bool = False,
) -> dict:
    if processor is None:
        raise RuntimeError("Model not loaded")

    pil = _load_image(img_bytes)
    state = processor.set_image(pil)

    results = []
    for prompt in prompts:
        processor.reset_all_prompts(state)
        out = processor.set_text_prompt(state=state, prompt=prompt)
        results.append(
            _format_output(
                out,
                label=prompt,
                include_masks=include_masks,
                include_contours=include_contours,
            )
        )

    return {"results": results}


@app.post("/segment")
async def segment_text_batch(
    image: UploadFile = File(...),
    prompts: List[str] = Form(...),
    return_masks: bool = Form(False),
    return_contours: bool = Form(False),
):
    if processor is None:
        detail = "Model is not ready."
        if startup_error:
            detail = f"Model failed to load: {startup_error}"
        raise HTTPException(status_code=503, detail=detail)

    cleaned = [prompt.strip() for prompt in prompts if prompt and prompt.strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="Prompts must not be empty.")
    if len(cleaned) != len(prompts):
        raise HTTPException(status_code=400, detail="Prompts must not be empty.")

    img_bytes = await image.read()
    try:
        result = await run_in_threadpool(
            _run_segmentation_batch,
            img_bytes,
            cleaned,
            return_masks,
            return_contours,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception("Batch segmentation failed.")
        raise HTTPException(status_code=500, detail="Batch segmentation failed.")
    finally:
        await image.close()

    return result
