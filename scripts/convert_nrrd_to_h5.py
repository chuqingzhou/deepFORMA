#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert NRRD pairs (raw + label) to H5 files for training.

Expected inputs:
  - <raw_dir>/<sample>.nrrd
  - <label_dir>/<sample>.nrrd

Output H5 format:
  - raw: float32, min-max normalized to [0,1]
  - label: uint8
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


def convert_pair(raw_path: Path, label_path: Path, h5_path: Path) -> None:
    raw_data, _ = nrrd.read(str(raw_path))
    label_data, _ = nrrd.read(str(label_path))

    raw = min_max_norm(np.asarray(raw_data)).astype(np.float32)
    lbl = np.asarray(label_data).astype(np.uint8)

    if raw.shape != lbl.shape:
        raise ValueError(f"Shape mismatch for {raw_path.stem}: raw={raw.shape}, label={lbl.shape}")

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("label", data=lbl, compression="gzip")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert NRRD raw/label pairs to H5 format.")
    p.add_argument("--raw-dir", type=Path, required=True, help="Directory with raw NRRD files.")
    p.add_argument("--label-dir", type=Path, required=True, help="Directory with label NRRD files.")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for H5 files.")
    p.add_argument(
        "--samples",
        type=str,
        default=None,
        help="Comma-separated sample IDs. If omitted, all matching NRRD pairs are converted.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    label_dir: Path = args.label_dir
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.samples:
        sample_ids = [s.strip() for s in args.samples.split(",") if s.strip()]
    else:
        sample_ids = []
        for raw_file in sorted(raw_dir.glob("*.nrrd")):
            label_file = label_dir / raw_file.name
            if label_file.exists():
                sample_ids.append(raw_file.stem)

    if not sample_ids:
        raise RuntimeError("No samples found to convert.")

    for sid in tqdm(sample_ids, desc="Converting"):
        raw_path = raw_dir / f"{sid}.nrrd"
        label_path = label_dir / f"{sid}.nrrd"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file not found: {raw_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        convert_pair(raw_path, label_path, out_dir / f"{sid}.h5")


if __name__ == "__main__":
    main()

