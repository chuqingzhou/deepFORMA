#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert raw NRRD volumes to H5 files (raw-only).

Output H5 format:
  - raw: float32, min-max normalized to [0,1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import nrrd
from tqdm import tqdm


def min_max_norm(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr
    mn = float(np.min(arr[finite]))
    mx = float(np.max(arr[finite]))
    if (mx - mn) < 1e-8:
        return arr
    return (arr - mn) / (mx - mn)


def convert_raw(raw_path: Path, h5_path: Path) -> None:
    raw_data, _ = nrrd.read(str(raw_path))
    raw = min_max_norm(np.asarray(raw_data)).astype(np.float32)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert raw NRRD volumes to H5 (raw-only).")
    p.add_argument("--raw-dir", type=Path, required=True, help="Directory with raw NRRD files.")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for H5 files.")
    p.add_argument(
        "--samples",
        type=str,
        default=None,
        help="Comma-separated sample IDs. If omitted, all NRRD files in raw-dir are converted.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.samples:
        sample_ids = [s.strip() for s in args.samples.split(",") if s.strip()]
    else:
        sample_ids = [p.stem for p in sorted(raw_dir.glob("*.nrrd"))]

    if not sample_ids:
        raise RuntimeError("No NRRD samples found to convert.")

    for sid in tqdm(sample_ids, desc="Converting raw"):
        raw_path = raw_dir / f"{sid}.nrrd"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file not found: {raw_path}")
        convert_raw(raw_path, out_dir / f"{sid}.h5")


if __name__ == "__main__":
    main()

