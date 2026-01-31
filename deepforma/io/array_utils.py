from __future__ import annotations

from typing import Tuple

import numpy as np


def min_max_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1] on finite values."""
    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr
    mn = float(np.min(arr[finite]))
    mx = float(np.max(arr[finite]))
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-8:
        return arr
    return (arr - mn) / (mx - mn)


def pad_to_multiple_reflect(x: np.ndarray, m: int) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Reflect-pad (D,H,W) array so each dim becomes a multiple of m."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {x.shape}")
    d, h, w = x.shape
    pd = (m - d % m) % m
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    x_pad = np.pad(x, ((0, pd), (0, ph), (0, pw)), mode="reflect")
    return x_pad, (pd, ph, pw)

