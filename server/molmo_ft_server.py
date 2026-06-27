#!/usr/bin/env python3
"""Serve a FINE-TUNED MolmoAct2 LeRobot policy (from scripts/pod_molmo_ft.sh).

Unlike molmo_server.py (which serves the stock HF checkpoint via predict_action),
this loads the LeRobot MolmoAct2Policy + its processor pipeline. After fine-tuning
on our SO-101 data, the policy speaks OUR convention directly (sim degrees, our
wrist/front cameras), so the browser sends/receives raw sim units — no embodiment
transform needed.

  CKPT=/workspace/molmo_ft/checkpoints/last/pretrained_model python molmo_ft_server.py
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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

CKPT = os.environ.get("CKPT", "/workspace/molmo_ft/checkpoints/last/pretrained_model")
DEVICE = os.environ.get("DEVICE", "cuda")
CAMERAS = os.environ.get("CAMERAS", "wrist,front").split(",")

app = FastAPI(title="MolmoAct2 fine-tuned SO-101 server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_policy: Any = None
_pre: Any = None
_post: Any = None
_lock = threading.Lock()


def _load() -> None:
    global _policy, _pre, _post
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from lerobot.policies.factory import make_pre_post_processors

    print(f"loading fine-tuned policy from {CKPT}…", flush=True)
    _policy = MolmoAct2Policy.from_pretrained(CKPT).to(DEVICE).eval()
    _pre, _post = make_pre_post_processors(_policy.config, CKPT)
    print("policy ready", flush=True)


def _decode(name: str, value: str) -> Image.Image:
    if not value:
        raise HTTPException(400, f"image '{name}' is empty")
    payload = value.split(",", 1)[1] if value.lstrip().startswith("data:") else value
    try:
        raw = base64.b64decode(payload, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"image '{name}' failed to decode: {exc}") from exc


class InferRequest(BaseModel):
    images: dict[str, str]      # camera key -> data-URL
    state: list[float]          # 6 joint angles, OUR sim degrees
    task: str
    reset: bool = False


class InferResponse(BaseModel):
    actions: list[list[float]]  # 6-D joint targets, OUR sim degrees


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": _policy is not None, "ckpt": CKPT, "cameras": CAMERAS}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    if _policy is None:
        raise HTTPException(503, "policy still loading")
    for cam in CAMERAS:
        if cam not in req.images:
            raise HTTPException(400, f"missing camera '{cam}'; got {list(req.images)}")

    def to_chw(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W

    batch: dict[str, Any] = {
        "observation.state": torch.tensor([req.state], dtype=torch.float32),
        "task": [req.task],
    }
    for cam in CAMERAS:
        batch[f"observation.images.{cam}"] = to_chw(_decode(cam, req.images[cam]))

    with _lock:
        if req.reset:
            _policy.reset()
        processed = _pre(batch)
        chunk = _policy.predict_action_chunk(processed, inference_action_mode="continuous")
        out = _post(chunk)

    arr = out.detach().to("cpu").float().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return InferResponse(actions=arr.tolist())


if __name__ == "__main__":
    import uvicorn

    _load()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8001")))
