from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as cc_label


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Canonical preprocess used in this project:
    - input raw is expected to be min-max normalized to [0,1] (as stored in training/validation H5)
    - constant pad to multiple of pad_m
    - z-score on the padded volume
    """

    pad_m: int = 16
    z_eps: float = 1e-5


@dataclass(frozen=True)
class PostprocessConfig:
    """
    Canonical postprocess (instance definition):
    - threshold: prob > threshold
    - connectivity: 1 => 6-neighborhood, 2 => 18-neighborhood, 3 => 26-neighborhood (SciPy semantics)
    - min_volume: remove connected components smaller than this voxel count
    """

    threshold: float = 0.5
    connectivity: int = 1
    min_volume: int = 100


def pad_to_multiple_constant(x: np.ndarray, m: int) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {x.shape}")
    d, h, w = x.shape
    pd = (m - d % m) % m
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    x_pad = np.pad(x, ((0, pd), (0, ph), (0, pw)), mode="constant")
    return x_pad, (pd, ph, pw)


def zscore_on_padded(x_pad: np.ndarray, *, z_eps: float) -> np.ndarray:
    x_pad = x_pad.astype(np.float32, copy=False)
    mu = float(np.mean(x_pad))
    sd = float(np.std(x_pad))
    return (x_pad - mu) / (sd + float(z_eps))


def preprocess_raw_minmax_for_model(raw_minmax: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Return:
    - x_pre: preprocessed padded+zscored volume (float32), shape padded
    - orig_shape: (d0,h0,w0)
    - pads: (pd,ph,pw)
    """
    if raw_minmax.ndim != 3:
        raise ValueError(f"Expected 3D raw volume, got shape {raw_minmax.shape}")
    d0, h0, w0 = map(int, raw_minmax.shape)
    raw_pad, pads = pad_to_multiple_constant(raw_minmax.astype(np.float32, copy=False), int(cfg.pad_m))
    raw_z = zscore_on_padded(raw_pad, z_eps=float(cfg.z_eps))
    return raw_z.astype(np.float32, copy=False), (d0, h0, w0), pads


def predict_prob_from_raw_minmax(model, raw_minmax: np.ndarray, device: str, cfg: PreprocessConfig) -> np.ndarray:
    import torch

    x_pre, (d0, h0, w0), _pads = preprocess_raw_minmax_for_model(raw_minmax, cfg)
    x = torch.from_numpy(x_pre).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0, :d0, :h0, :w0].detach().cpu().numpy().astype(np.float32)
    return prob


def prob_to_binary_mask(prob: np.ndarray, *, threshold: float) -> np.ndarray:
    return (prob > float(threshold)).astype(np.uint8)


def label_connected_components(mask: np.ndarray, *, connectivity: int) -> Tuple[np.ndarray, int]:
    """
    SciPy connectivity semantics for 3D:
    - connectivity=1: 6-neighborhood
    - connectivity=2: 18-neighborhood
    - connectivity=3: 26-neighborhood
    """
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be 1(6), 2(18), or 3(26) for SciPy generate_binary_structure")
    struct = generate_binary_structure(3, int(connectivity))
    labeled, num = cc_label(mask > 0, structure=struct)
    return labeled.astype(np.uint16), int(num)


def filter_and_relabel_by_volume(labeled: np.ndarray, *, min_volume: int) -> Tuple[np.ndarray, int]:
    """Remove small components and relabel remaining components as 1..K."""
    min_volume = int(min_volume)
    if min_volume <= 1:
        # still ensure uint16 output and correct num
        num = int(np.max(labeled)) if labeled.size else 0
        return labeled.astype(np.uint16, copy=False), num
    out = np.zeros_like(labeled, dtype=np.uint16)
    next_id = 0
    max_id = int(np.max(labeled)) if labeled.size else 0
    for cid in range(1, max_id + 1):
        vox = int(np.count_nonzero(labeled == cid))
        if vox < min_volume:
            continue
        next_id += 1
        out[labeled == cid] = next_id
    return out, int(next_id)

