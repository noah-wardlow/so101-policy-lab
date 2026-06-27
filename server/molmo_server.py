#!/usr/bin/env python3
"""Thin MolmoAct2-SO100_101 inference server for the SO-101 policy lab.

Design:
  * GENERIC: speaks the policy's native convention — joint **degrees** + 2 RGB
    images + a task string in, an absolute-joint-degree action chunk out. The
    sim<->policy embodiment transform lives in ONE place on the client
    (src/policies/molmo.ts), never split across browser and server.
  * STRICT: images are required and validated; a decode failure or a missing
    camera returns HTTP 400 with a clear message — no silent base64 failures,
    no "duplicate the one image we got" band-aid.

Run on a RunPod GPU pod (>=24GB; bf16 ~16GB):
  pip install "transformers>=4.46" accelerate torch fastapi uvicorn pillow
  MODEL_ID=allenai/MolmoAct2-SO100_101 python server/molmo_server.py
"""
from __future__ import annotations

import base64
import io
import os
import threading
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "allenai/MolmoAct2-SO100_101")
NORM_TAG = os.environ.get("NORM_TAG", "so100_so101_molmoact2")
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE = torch.bfloat16 if os.environ.get("DTYPE", "bf16") == "bf16" else torch.float32
NUM_STEPS = int(os.environ.get("NUM_STEPS", "10"))  # flow-matching solver steps

app = FastAPI(title="MolmoAct2 SO-101 server")

# Allow the browser app (any origin) to POST images cross-origin.
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Any = None
_processor: Any = None
_lock = threading.Lock()


def _load() -> None:
    global _model, _processor
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"loading {MODEL_ID} (dtype={DTYPE}, device={DEVICE})…", flush=True)
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = (
        AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, trust_remote_code=True, dtype=DTYPE
        )
        .to(DEVICE)
        .eval()
    )
    print("model ready", flush=True)


def _decode(name: str, value: str) -> Image.Image:
    """Decode a data-URL or bare base64 PNG/JPEG. Raises 400 on any failure."""
    if not value:
        raise HTTPException(400, f"image '{name}' is empty")
    payload = value.split(",", 1)[1] if value.lstrip().startswith("data:") else value
    try:
        raw = base64.b64decode(payload, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 — surface the real reason
        raise HTTPException(400, f"image '{name}' failed to decode: {exc}") from exc


class InferRequest(BaseModel):
    # Camera key -> data-URL / base64. Order is preserved as sent.
    images: dict[str, str]
    state: list[float]          # 6 joint angles, POLICY degrees
    task: str
    reset: bool = False


class InferResponse(BaseModel):
    actions: list[list[float]]  # chunk of 6-D absolute joint targets, POLICY degrees


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": _model is not None, "model": MODEL_ID, "norm_tag": NORM_TAG}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    if _model is None:
        raise HTTPException(503, "model still loading")
    if len(req.images) < 2:
        raise HTTPException(400, f"expected >=2 camera images, got {list(req.images)}")
    if len(req.state) != 6:
        raise HTTPException(400, f"state must be length 6, got {len(req.state)}")

    images = [_decode(name, val) for name, val in req.images.items()]
    state = np.asarray(req.state, dtype=np.float32)

    with _lock:
        out = _model.predict_action(
            processor=_processor,
            images=images,
            task=req.task,
            state=state,
            norm_tag=NORM_TAG,
            inference_action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=NUM_STEPS,
            normalize_language=True,
        )

    actions = out.actions
    if hasattr(actions, "detach"):
        actions = actions.detach().to("cpu").float().numpy()
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise HTTPException(500, f"unexpected action shape {actions.shape}")
    return InferResponse(actions=actions.tolist())


if __name__ == "__main__":
    import uvicorn

    _load()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
