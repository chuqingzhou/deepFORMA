from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np


def read_h5_raw(path: Path, *, key: str = "raw") -> np.ndarray:
    """
    Read a 3D raw volume from H5.
    Expected canonical format: dataset 'raw' contains float32 min-max normalized volume in [0,1].
    """
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"H5 missing dataset '{key}': {path}")
        arr = f[key][()]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array in {path}['{key}'], got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def read_h5_raw_and_label(path: Path, *, raw_key: str = "raw", label_key: str = "label") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with h5py.File(path, "r") as f:
        if raw_key not in f:
            raise KeyError(f"H5 missing dataset '{raw_key}': {path}")
        raw = np.asarray(f[raw_key][()], dtype=np.float32)
        lbl = None
        if label_key in f:
            lbl = np.asarray(f[label_key][()], dtype=np.uint8)
    if raw.ndim != 3:
        raise ValueError(f"Expected 3D raw in {path}['{raw_key}'], got shape {raw.shape}")
    if lbl is not None and lbl.shape != raw.shape:
        raise ValueError(f"raw/label shape mismatch for {path}: raw={raw.shape} label={lbl.shape}")
    return raw, lbl

