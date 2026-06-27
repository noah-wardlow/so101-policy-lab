#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from huggingface_hub import snapshot_download
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


DEFAULT_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def camera_name(feature_key: str) -> str:
    return feature_key.removeprefix("observation.images.")


def feature_names(feature: object | None, fallback: list[str]) -> list[str]:
    names = getattr(feature, "names", None)
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return fallback


def fallback_joint_names(action_dim: int) -> list[str]:
    if action_dim == len(DEFAULT_JOINT_NAMES):
        return DEFAULT_JOINT_NAMES
    return [f"action_{index}" for index in range(action_dim)]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().flatten().tolist()
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def processor_stats(policy_dir: Path, config_name: str) -> dict[str, dict[str, Any]]:
    try:
        pipeline = PolicyProcessorPipeline.from_pretrained(
            policy_dir,
            config_filename=f"{config_name}.json",
            local_files_only=True,
            overrides={"device_processor": {"device": "cpu"}},
        )
    except FileNotFoundError:
        return {}
    stats: dict[str, dict[str, Any]] = {}
    for step in pipeline.steps:
        step_stats = getattr(step, "stats", None)
        if isinstance(step_stats, dict):
            for key, sub_stats in step_stats.items():
                if isinstance(sub_stats, dict):
                    stats[key] = {name: to_jsonable(value) for name, value in sub_stats.items()}
    return stats


def normalization_mode(
    norm_map: dict[FeatureType, NormalizationMode],
    feature: PolicyFeature,
) -> str:
    return norm_map.get(feature.type, NormalizationMode.IDENTITY).value


def normalization_entry(
    feature: PolicyFeature,
    feature_key: str,
    stats: dict[str, dict[str, Any]],
    norm_map: dict[FeatureType, NormalizationMode],
) -> dict[str, Any] | None:
    mode = normalization_mode(norm_map, feature)
    if mode == NormalizationMode.IDENTITY.value:
        return None
    feature_stats = stats.get(feature_key)
    if not feature_stats:
        return None
    return {
        "mode": mode,
        "stats": feature_stats,
        "eps": 1e-8,
    }


class ACTOnnxWrapper(torch.nn.Module):
    def __init__(self, policy: ACTPolicy, image_features: list[str]) -> None:
        super().__init__()
        self.policy = policy
        self.image_features = image_features

    def forward(self, state: torch.Tensor, *images: torch.Tensor) -> torch.Tensor:
        batch = {"observation.state": state}
        for key, image in zip(self.image_features, images, strict=True):
            batch[key] = image
        return self.policy.predict_action_chunk(batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained LeRobot ACT policy to browser ONNX.")
    parser.add_argument("--policy", required=True, help="Local trained ACT policy directory.")
    parser.add_argument("--out", default="public/models/browser-act", help="Output directory for ONNX + policy.json.")
    parser.add_argument("--model-name", default="act.onnx")
    parser.add_argument("--fps", type=int, default=30, help="Rollout FPS for the browser policy manifest.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="Experimental: export with fp16 model and image inputs.")
    args = parser.parse_args()

    policy_dir = Path(args.policy).expanduser()
    if not policy_dir.exists():
        policy_dir = Path(snapshot_download(repo_id=args.policy))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = ACTPolicy.from_pretrained(policy_dir, local_files_only=True)
    policy.eval()
    policy.to("cpu")
    if args.fp16:
        policy.half()

    image_feature_map = dict(policy.config.image_features)
    image_features = list(image_feature_map.keys())
    state_feature = policy.config.robot_state_feature
    action_feature = policy.config.action_feature
    if not image_features:
        raise ValueError("ACT policy has no image features; browser ACT route expects camera inputs.")
    if not state_feature:
        raise ValueError("ACT policy has no observation.state feature; browser ACT route expects state input.")
    if not action_feature:
        raise ValueError("ACT policy has no action feature; browser ACT route expects action output.")
    preprocessor_stats = processor_stats(policy_dir, POLICY_PREPROCESSOR_DEFAULT_NAME)
    postprocessor_stats = processor_stats(policy_dir, POLICY_POSTPROCESSOR_DEFAULT_NAME)
    action_stats = postprocessor_stats.get("action") or preprocessor_stats.get("action")

    state_dim = state_feature.shape[0]
    action_dim = action_feature.shape[0]
    camera_shapes = [image_feature_map[key].shape for key in image_features]
    if len({tuple(shape) for shape in camera_shapes}) != 1:
        raise ValueError(f"Browser exporter expects all cameras to share one shape, got {camera_shapes}.")

    image_shape = tuple(camera_shapes[0])
    channels, height, width = image_shape
    image_dtype = torch.float16 if args.fp16 else torch.float32
    dummy_state = torch.zeros(1, state_dim, dtype=torch.float32)
    dummy_images = [
        torch.zeros(1, channels, height, width, dtype=image_dtype)
        for _ in image_features
    ]
    wrapper = ACTOnnxWrapper(policy, image_features).eval()

    input_names = ["state", *[camera_name(key) for key in image_features]]
    output_names = ["action"]
    onnx_path = out_dir / args.model_name
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_state, *dummy_images),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    manifest = {
        "model": args.model_name,
        "variants": {"fp16": args.model_name} if args.fp16 else {"fp32": args.model_name},
        "fps": args.fps,
        "joints": feature_names(action_feature, fallback_joint_names(action_dim)),
        "chunk_size": policy.config.chunk_size,
        "n_action_steps": policy.config.n_action_steps,
        "temporal_ensemble_coeff": policy.config.temporal_ensemble_coeff,
        "cameras": [camera_name(key) for key in image_features],
        "image": {
            "height": height,
            "width": width,
            "channels": channels,
            "layout": "CHW",
            "range": [0, 1],
        },
        "inputs": [
            {
                "name": "state",
                "shape": [1, state_dim],
                "dtype": "float32",
            },
            *[
                {
                    "name": camera_name(key),
                    "shape": [1, channels, height, width],
                    "dtype": "float16" if args.fp16 else "float32",
                }
                for key in image_features
            ],
        ],
        "output": {
            "name": "action",
            "shape": [1, policy.config.chunk_size, action_dim],
            "dtype": "float32",
            "units": "radians",
        },
        "normalization": {
            "inputs": {
                input_name: entry
                for input_name, entry in [
                    (
                        "state",
                        normalization_entry(
                            state_feature,
                            "observation.state",
                            preprocessor_stats,
                            policy.config.normalization_mapping,
                        ),
                    ),
                    *[
                        (
                            camera_name(key),
                            normalization_entry(
                                image_feature_map[key],
                                key,
                                preprocessor_stats,
                                policy.config.normalization_mapping,
                            ),
                        )
                        for key in image_features
                    ],
                ]
                if entry is not None
            },
            "output": normalization_entry(
                action_feature,
                "action",
                {"action": action_stats} if action_stats else {},
                policy.config.normalization_mapping,
            ),
        },
    }
    (out_dir / "policy.json").write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "policy": args.policy,
        "onnx": str(onnx_path),
        "manifest": str(out_dir / "policy.json"),
        "inputs": manifest["inputs"],
        "output": manifest["output"],
    }, indent=2))


if __name__ == "__main__":
    main()
