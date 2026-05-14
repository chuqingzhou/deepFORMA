from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class ZScorePassthroughSegmentationModel(nn.Module):
    """Demo-only model that treats positive z-scored voxels as foreground logits."""

    def __init__(self, *, scale: float = 1.0, bias: float = 0.0):
        super().__init__()
        self.scale = float(scale)
        self.bias = float(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


def load_torch_checkpoint(path: Path, device: str) -> Dict[str, Any]:
    obj = torch.load(str(path), map_location=device)
    if isinstance(obj, dict):
        return obj
    return {"state_dict": obj}


def load_model_state_dict(model: torch.nn.Module, ckpt: Dict[str, Any]) -> None:
    """
    Load a model state dict from common checkpoint formats.
    Supported keys: 'model_state', 'model_state_dict', 'state_dict'.
    """
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            model.load_state_dict(ckpt[key])
            return
    # Fallback: the checkpoint itself may already be a state_dict
    if all(isinstance(k, str) for k in ckpt.keys()):
        model.load_state_dict(ckpt)  # type: ignore[arg-type]
        return
    raise ValueError("Unsupported checkpoint format (no recognizable state dict keys).")


def infer_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_segmentation_model(model_path: Path, *, dropout: float = 0.3, prefer_cuda: bool = True) -> Tuple[torch.nn.Module, str]:
    from .transunet3d import build_model

    device = infer_device(prefer_cuda=prefer_cuda)
    ckpt = load_torch_checkpoint(model_path, device=device)

    if ckpt.get("model_type") == "demo_zscore_passthrough":
        model = ZScorePassthroughSegmentationModel(
            scale=float(ckpt.get("scale", 1.0)),
            bias=float(ckpt.get("bias", 0.0)),
        ).to(device)
        model.eval()
        return model, device

    model = build_model(dropout=dropout).to(device)
    load_model_state_dict(model, ckpt)
    model.eval()
    return model, device

