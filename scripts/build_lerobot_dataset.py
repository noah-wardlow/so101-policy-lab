#!/usr/bin/env python3
"""Convert raw browser recordings (scripts/record.mjs output) into a LeRobotDataset
for ACT training. Only episodes the in-browser physics verifier marked successful
are written (record.mjs already filters), preserving a clean, controlled distribution.

  python scripts/build_lerobot_dataset.py \
      --raw data/raw --repo-id local/so101-pickplace --root data/lerobot --overwrite
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
CAMERAS = ["wrist", "front"]
TASK = "pick up the red cube and place it on the green target"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def image_shape(episode: Path, rows: list[dict[str, Any]], cam: str) -> tuple[int, int, int]:
    for row in rows:
        rel = row.get(cam)
        if rel and (episode / rel).exists():
            with Image.open(episode / rel) as im:
                w, h = im.convert("RGB").size
                return (h, w, 3)
    raise FileNotFoundError(f"no frames for camera {cam!r} in {episode}")


def features(shapes: dict[str, tuple[int, int, int]]) -> dict[str, dict[str, Any]]:
    feats: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES},
    }
    for cam, shape in shapes.items():
        feats[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": shape, "names": ["height", "width", "channels"],
        }
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--repo-id", default="local/so101-pickplace")
    ap.add_argument("--root", default="data/lerobot")
    ap.add_argument("--fps", type=int, default=15)  # matches recorder's ~14.3Hz effective rate
    ap.add_argument("--robot-type", default="so101")
    ap.add_argument("--cameras", default="", help="comma-separated; default = auto-detect from frames")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    raw = Path(args.raw).expanduser()
    episodes = sorted(d for d in raw.iterdir() if (d / "frames.jsonl").exists())
    if not episodes:
        raise FileNotFoundError(f"no episode_*/frames.jsonl under {raw}")
    print(f"found {len(episodes)} episodes")

    first_rows = read_jsonl(episodes[0] / "frames.jsonl")
    # Cameras: explicit --cameras, else auto-detect (any non-meta key in a row is
    # a camera image path) so 2-cam and 3-cam datasets both Just Work.
    meta_keys = {"t", "state", "action"}
    cameras = (
        [c.strip() for c in args.cameras.split(",") if c.strip()]
        if args.cameras
        else [k for k in first_rows[0] if k not in meta_keys]
    )
    print(f"cameras: {cameras}")
    shapes = {cam: image_shape(episodes[0], first_rows, cam) for cam in cameras}

    root = Path(args.root).expanduser()
    if root.exists():
        if args.overwrite:
            shutil.rmtree(root)
        else:
            raise FileExistsError(f"{root} exists; pass --overwrite")

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=features(shapes),
        root=root, robot_type=args.robot_type, use_videos=True,
        image_writer_threads=4, encoder_threads=2,
    )

    total_frames = 0
    try:
        for ep in episodes:
            rows = read_jsonl(ep / "frames.jsonl")
            for row in rows:
                frame: dict[str, Any] = {
                    "task": TASK,
                    "observation.state": np.asarray(row["state"][:6], dtype=np.float32),
                    "action": np.asarray(row["action"][:6], dtype=np.float32),
                }
                for cam in cameras:
                    with Image.open(ep / row[cam]) as im:
                        frame[f"observation.images.{cam}"] = im.convert("RGB").copy()
                ds.add_frame(frame)
                total_frames += 1
            ds.save_episode()
            print(f"  + {ep.name}: {len(rows)} frames")
        ds.finalize()
    except Exception:
        if ds.has_pending_frames():
            ds.clear_episode_buffer()
        raise

    print(json.dumps({
        "ok": True, "repo_id": args.repo_id, "root": str(root),
        "num_episodes": len(episodes), "num_frames": total_frames,
        "cameras": shapes, "fps": args.fps,
    }, indent=2))


if __name__ == "__main__":
    main()
