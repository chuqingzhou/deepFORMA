#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the FORMA atlas database from raw MRI NRRD volumes.

This script performs:
- NRRD loading (raw MRI volume)
- Min-max normalization to [0, 1] ("raw_norm")
- Segmentation inference using a CNN-Transformer hybrid model
- 26-neighborhood connected component labeling
- Background z-scoring ("bgz") based on background voxels (outside predicted mask)
- Per-component H5 export (cropped bbox)
- Atlas export (Excel/CSV) with nine metrics (raw_norm and bgz variants)

Notes:
- The script does not ship raw data or model weights; provide paths via CLI arguments.
- Designed for reproducibility and code release (no hard-coded absolute paths).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as connected_components

from deepforma.features.nine_metrics import MetricConfig, compute_nine_metrics
from deepforma.io.array_utils import min_max_norm, pad_to_multiple_reflect
from deepforma.io.nrrd_io import read_nrrd_with_spacing


LOG = logging.getLogger("deepforma.build_database")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def predict_prob_and_mask(model, raw_norm: np.ndarray, device: str, pad_m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Inference: reflect-pad to a multiple of pad_m; no z-score; sigmoid threshold at 0.5."""
    import torch

    d0, h0, w0 = raw_norm.shape
    raw_pad, _pads = pad_to_multiple_reflect(raw_norm, pad_m)
    x = torch.from_numpy(raw_pad).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)
        prob = prob[:, :, :d0, :h0, :w0]
        prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)
    mask = (prob_np > 0.5).astype(np.uint8)
    return prob_np, mask


def label_cc(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """26-neighborhood connected components for 3D volumes."""
    struct = generate_binary_structure(3, 3)
    labeled, num = connected_components(mask > 0, struct)
    return labeled.astype(np.uint16), int(num)


def compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    coords = np.where(mask)
    return int(coords[0].min()), int(coords[0].max()) + 1, int(coords[1].min()), int(coords[1].max()) + 1, int(coords[2].min()), int(coords[2].max()) + 1


def estimate_background(raw_norm: np.ndarray, mask: np.ndarray, clip_low: float, clip_high: float) -> Tuple[float, float, float, float, int]:
    bg = raw_norm[mask == 0]
    bg = bg[np.isfinite(bg)]
    if bg.size == 0:
        return 0.0, 1.0, float("nan"), float("nan"), 0
    lo, hi = np.percentile(bg, [clip_low, clip_high])
    bg2 = bg[(bg >= lo) & (bg <= hi)]
    if bg2.size < 10:
        bg2 = bg
    mu = float(np.mean(bg2))
    sd = float(np.std(bg2))
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    return mu, sd, float(lo), float(hi), int(bg2.size)


def save_connected_h5(
    path: Path,
    raw_norm: np.ndarray,
    raw_bgz: np.ndarray,
    prob: np.ndarray,
    mask: np.ndarray,
    label: np.ndarray,
    *,
    bg_mean: float,
    bg_std: float,
    bg_clip_low_pct: float,
    bg_clip_high_pct: float,
    bg_clip_low_value: float,
    bg_clip_high_value: float,
    bg_n: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("raw", data=raw_norm.astype(np.float32), compression="gzip")
        f.create_dataset("raw_bgz", data=raw_bgz.astype(np.float32), compression="gzip")
        f.create_dataset("prediction_prob", data=prob.astype(np.float32), compression="gzip")
        f.create_dataset("prediction", data=mask.astype(np.uint8), compression="gzip")
        f.create_dataset("label", data=label.astype(np.uint16), compression="gzip")
        # Connectivity semantics:
        # - Connected components use SciPy full 3D neighborhood: generate_binary_structure(3,3) => 26-neighborhood.
        # - Feature computation is performed per connected component mask.
        f.attrs["cc_connectivity"] = 26
        f.attrs["cc_connectivity_scipy_structure"] = "generate_binary_structure(3, 3)"
        f.attrs["feature_connectivity"] = 26
        f.attrs["bg_mean_rawnorm"] = float(bg_mean)
        f.attrs["bg_std_rawnorm"] = float(bg_std)
        f.attrs["bg_clip_low_pct"] = float(bg_clip_low_pct)
        f.attrs["bg_clip_high_pct"] = float(bg_clip_high_pct)
        f.attrs["bg_clip_low_value"] = float(bg_clip_low_value) if np.isfinite(bg_clip_low_value) else np.nan
        f.attrs["bg_clip_high_value"] = float(bg_clip_high_value) if np.isfinite(bg_clip_high_value) else np.nan
        f.attrs["bg_n"] = int(bg_n)


def load_connected_h5(path: Path):
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        raw_bgz = f["raw_bgz"][:]
        prob = f["prediction_prob"][:]
        pred = f["prediction"][:]
        label = f["label"][:]
        attrs = dict(f.attrs)
    return raw, raw_bgz, prob, pred, label, attrs


def save_component_h5(
    path: Path,
    raw_norm: np.ndarray,
    raw_bgz: np.ndarray,
    prob: np.ndarray,
    comp_mask: np.ndarray,
    bbox: Tuple[int, int, int, int, int, int],
    *,
    sample_id: str,
    connect_id: int,
) -> None:
    z0, z1, y0, y1, x0, x1 = bbox
    raw_crop = raw_norm[z0:z1, y0:y1, x0:x1].astype(np.float32)
    bgz_crop = raw_bgz[z0:z1, y0:y1, x0:x1].astype(np.float32)
    prob_crop = prob[z0:z1, y0:y1, x0:x1].astype(np.float32)
    mask_crop = comp_mask[z0:z1, y0:y1, x0:x1].astype(np.uint8)
    pred_crop = (mask_crop > 0).astype(np.uint8)
    prob_crop = (prob_crop * pred_crop).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("raw", data=raw_crop, compression="gzip")
        f.create_dataset("raw_bgz", data=bgz_crop, compression="gzip")
        f.create_dataset("prediction", data=pred_crop, compression="gzip")
        f.create_dataset("prediction_prob", data=prob_crop, compression="gzip")
        f.attrs["cc_connectivity"] = 26
        f.attrs["cc_connectivity_scipy_structure"] = "generate_binary_structure(3, 3)"
        f.attrs["feature_connectivity"] = 26
        f.attrs["scan_id"] = str(sample_id)
        f.attrs["connect_id"] = int(connect_id)
        f.attrs["bbox_min_z"] = int(z0)
        f.attrs["bbox_max_z"] = int(z1)
        f.attrs["bbox_min_y"] = int(y0)
        f.attrs["bbox_max_y"] = int(y1)
        f.attrs["bbox_min_x"] = int(x0)
        f.attrs["bbox_max_x"] = int(x1)
        f.attrs["shape"] = raw_crop.shape


def append_rows_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def _atlas_xlsx_name(tag: str | None) -> str:
    """Project-level output filename: DeepFORMA_atlas_database.xlsx or DeepFORMA_atlas_database_{tag}.xlsx"""
    base = "DeepFORMA_atlas_database"
    return f"{base}_{tag}.xlsx" if tag else f"{base}.xlsx"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]

    p = argparse.ArgumentParser(description="Build FORMA atlas database from raw MRI NRRD volumes.")
    p.add_argument("--root", type=Path, default=repo_root, help="Repository root (default: inferred from script path).")
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing raw NRRD volumes. Required unless --demo.",
    )
    p.add_argument(
        "--atlas-existing",
        type=Path,
        default=None,
        help="Existing connected-id atlas Excel. Required unless --demo.",
    )
    p.add_argument("--model-path", type=Path, default=None, help="Path to model checkpoint (.pt). Required unless --demo.")
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root directory. Required unless --demo.",
    )
    p.add_argument("--demo", action="store_true", help="Run a minimal demo without real data or model weights.")
    p.add_argument("--tag", type=str, default=None, help="Optional tag for output filename (e.g. paper2026) to distinguish versions.")

    p.add_argument("--spacing", nargs=3, type=float, default=[0.16, 0.16, 0.3], help="Voxel spacing (dz dy dx) in mm.")
    p.add_argument("--min-size", type=int, default=10, help="Minimum component size (voxels).")
    p.add_argument("--bbox-pad", type=int, default=2, help="Padding for bbox used in surface area estimation.")
    p.add_argument("--pad-m", type=int, default=16, help="Reflect padding multiple for inference.")
    p.add_argument("--dropout", type=float, default=0.3, help="Model dropout used when building the architecture.")

    p.add_argument("--bg-clip", nargs=2, type=float, default=[1, 99], help="Background percentile clipping (low high).")
    p.add_argument("--resume-csv", type=Path, default=None, help="Optional path to partial CSV for resumable runs.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    root: Path = args.root
    raw_dir: Path | None = args.raw_dir
    atlas_existing: Path | None = args.atlas_existing
    out_root: Path | None = args.out_root
    if args.demo and out_root is None:
        out_root = root / "demo_output"
    if out_root is None:
        raise ValueError("--out-root is required. For demo mode, use: --demo --out-root <dir>")
    out_atlas_dir = out_root / "atlas"
    out_connected_dir = out_root / "predictions_connected"
    out_wells_dir = out_root / "wells_h5"
    atlas_xlsx_name = _atlas_xlsx_name(args.tag)

    out_atlas_dir.mkdir(parents=True, exist_ok=True)
    out_connected_dir.mkdir(parents=True, exist_ok=True)
    out_wells_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        # Demo mode does not require external data or weights.
        # It generates a synthetic 3D volume and a trivial "prediction" via thresholding.
        out_root.mkdir(parents=True, exist_ok=True)
        out_atlas_dir.mkdir(parents=True, exist_ok=True)
        out_connected_dir.mkdir(parents=True, exist_ok=True)
        out_wells_dir.mkdir(parents=True, exist_ok=True)

        sample = "DEMO-001"
        spacing = (0.16, 0.16, 0.3)
        cfg = MetricConfig(spacing=spacing, min_size_voxels=int(args.min_size), bbox_pad=int(args.bbox_pad))

        # Synthetic volume: a bright ellipsoid in the center.
        d, h, w = 48, 64, 64
        zz, yy, xx = np.meshgrid(
            np.linspace(-1, 1, d), np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij"
        )
        raw = np.exp(-((zz / 0.6) ** 2 + (yy / 0.5) ** 2 + (xx / 0.5) ** 2)).astype(np.float32)
        raw_norm = min_max_norm(raw)
        prob = raw_norm.astype(np.float32)
        pred = (prob > 0.35).astype(np.uint8)
        labeled, num = label_cc(pred)
        bg_mean, bg_std, bg_lo, bg_hi, bg_n = estimate_background(raw_norm, pred, clip_low=float(args.bg_clip[0]), clip_high=float(args.bg_clip[1]))
        raw_bgz = (raw_norm - bg_mean) / (bg_std + 1e-8)

        connected_h5 = out_connected_dir / f"{sample}_connected.h5"
        save_connected_h5(
            connected_h5,
            raw_norm,
            raw_bgz,
            prob,
            pred,
            labeled,
            bg_mean=bg_mean,
            bg_std=bg_std,
            bg_clip_low_pct=float(args.bg_clip[0]),
            bg_clip_high_pct=float(args.bg_clip[1]),
            bg_clip_low_value=bg_lo,
            bg_clip_high_value=bg_hi,
            bg_n=bg_n,
        )
        # Save spacing for methods clarity
        with h5py.File(connected_h5, "a") as f:
            f.attrs["spacing_d"] = float(spacing[0])
            f.attrs["spacing_h"] = float(spacing[1])
            f.attrs["spacing_w"] = float(spacing[2])

        rows: List[Dict] = []
        for cid in range(1, int(num) + 1):
            comp_mask = labeled == cid
            vox = int(np.count_nonzero(comp_mask))
            if vox < cfg.min_size_voxels:
                continue
            z0, z1, y0, y1, x0, x1 = compute_bbox(comp_mask)
            bbox = (z0, z1, y0, y1, x0, x1)

            comp_h5 = out_wells_dir / f"{sample}-C{cid}.h5"
            save_component_h5(
                comp_h5,
                raw_norm,
                raw_bgz,
                prob,
                comp_mask,
                bbox,
                sample_id=sample,
                connect_id=cid,
            )
            with h5py.File(comp_h5, "a") as f:
                f.attrs["spacing_d"] = float(spacing[0])
                f.attrs["spacing_h"] = float(spacing[1])
                f.attrs["spacing_w"] = float(spacing[2])

            m_norm = compute_nine_metrics(raw_norm, comp_mask, cfg=cfg)
            m_bgz = compute_nine_metrics(raw_bgz, comp_mask, cfg=cfg)
            rec: Dict = {
                "Raw_Data_ID": sample,
                "Connect_ID": int(cid),
                "organoid_volume_voxels": int(vox),
                "bbox_min_z": int(z0),
                "bbox_max_z": int(z1),
                "bbox_min_y": int(y0),
                "bbox_max_y": int(y1),
                "bbox_min_x": int(x0),
                "bbox_max_x": int(x1),
                "shape_z": int(z1 - z0),
                "shape_y": int(y1 - y0),
                "shape_x": int(x1 - x0),
            }
            for k, v in m_norm.items():
                rec[f"{k}_rawnorm"] = v
            for k, v in m_bgz.items():
                rec[f"{k}_bgz"] = v
            rows.append(rec)

        partial_csv = args.resume_csv if args.resume_csv is not None else (out_atlas_dir / "_atlas_rows_partial.csv")
        pd.DataFrame(rows).to_csv(partial_csv, index=False)
        out_xlsx = out_atlas_dir / atlas_xlsx_name
        pd.DataFrame(rows).to_excel(out_xlsx, index=False)
        LOG.info("Demo completed. Outputs root: %s", out_root)
        return

    if raw_dir is None or atlas_existing is None:
        raise ValueError(
            "When not using --demo, --raw-dir and --atlas-existing are required. See --help for usage."
        )
    if args.model_path is None:
        raise ValueError("--model-path is required unless --demo is set.")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw NRRD directory not found: {raw_dir}")
    if not atlas_existing.exists():
        raise FileNotFoundError(f"Existing atlas not found: {atlas_existing}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    LOG.info("Loading sample list from existing atlas: %s", atlas_existing)
    old = pd.read_excel(atlas_existing)
    if "Raw_Data_ID" not in old.columns:
        raise ValueError("Existing atlas must contain a 'Raw_Data_ID' column.")
    old["Raw_Data_ID"] = old["Raw_Data_ID"].astype(str)

    # Keep non-feature metadata columns for each sample (first row per sample).
    drop_cols = {
        "Connect_ID",
        "organoid_volume_voxels",
        "bbox_min_z", "bbox_max_z", "bbox_min_y", "bbox_max_y", "bbox_min_x", "bbox_max_x",
        "shape_z", "shape_y", "shape_x",
    }
    meta_cols = [c for c in old.columns if c not in drop_cols]
    meta_by_sample = old.groupby("Raw_Data_ID")[meta_cols].first().to_dict(orient="index")
    samples = sorted(meta_by_sample.keys())
    LOG.info("Samples: %d", len(samples))

    partial_csv = args.resume_csv if args.resume_csv is not None else (out_atlas_dir / "_atlas_rows_partial.csv")
    done_samples = set()
    if partial_csv.exists():
        try:
            df_done = pd.read_csv(partial_csv)
            if "Raw_Data_ID" in df_done.columns:
                done_samples = set(df_done["Raw_Data_ID"].astype(str).unique().tolist())
            LOG.info("Resume: found partial rows for %d samples.", len(done_samples))
        except Exception as e:
            LOG.warning("Resume: failed to read partial CSV (%s). Will rebuild.", e)
            done_samples = set()

    # Import torch only when not running in demo mode.
    from deepforma.model.checkpoint import load_segmentation_model

    model, device = load_segmentation_model(args.model_path, dropout=args.dropout, prefer_cuda=True)
    LOG.info("Model loaded on device: %s", device)

    default_spacing = (float(args.spacing[0]), float(args.spacing[1]), float(args.spacing[2]))
    clip_low, clip_high = float(args.bg_clip[0]), float(args.bg_clip[1])

    for i, sample in enumerate(samples, 1):
        if sample in done_samples:
            continue

        nrrd_path = raw_dir / f"{sample}.nrrd"
        connected_h5 = out_connected_dir / f"{sample}_connected.h5"

        if connected_h5.exists():
            raw_norm, raw_bgz, prob, pred, labeled, attrs = load_connected_h5(connected_h5)
            num = int(np.max(labeled))
            bg_mean = float(attrs.get("bg_mean_rawnorm", 0.0))
            bg_std = float(attrs.get("bg_std_rawnorm", 1.0))
        else:
            if not nrrd_path.exists():
                LOG.warning("[%d/%d] Missing NRRD, skip: %s", i, len(samples), nrrd_path.name)
                done_samples.add(sample)
                continue

            raw, spacing, header = read_nrrd_with_spacing(nrrd_path, default_spacing=default_spacing)
            raw_norm = min_max_norm(raw)
            prob, pred = predict_prob_and_mask(model, raw_norm, device, pad_m=int(args.pad_m))
            labeled, num = label_cc(pred)
            bg_mean, bg_std, bg_lo, bg_hi, bg_n = estimate_background(raw_norm, pred, clip_low=clip_low, clip_high=clip_high)
            raw_bgz = (raw_norm - bg_mean) / (bg_std + 1e-8)

            save_connected_h5(
                connected_h5,
                raw_norm,
                raw_bgz,
                prob,
                pred,
                labeled,
                bg_mean=bg_mean,
                bg_std=bg_std,
                bg_clip_low_pct=clip_low,
                bg_clip_high_pct=clip_high,
                bg_clip_low_value=bg_lo,
                bg_clip_high_value=bg_hi,
                bg_n=bg_n,
            )
            with h5py.File(connected_h5, "a") as f:
                f.attrs["spacing_d"] = float(spacing[0])
                f.attrs["spacing_h"] = float(spacing[1])
                f.attrs["spacing_w"] = float(spacing[2])
                f.attrs["spacing_source"] = "nrrd_header_or_default"

        if int(num) == 0:
            LOG.info("[%d/%d] %s: no connected components.", i, len(samples), sample)
            done_samples.add(sample)
            continue

        # Use header-derived spacing if available; otherwise fall back to CLI defaults.
        if connected_h5.exists():
            with h5py.File(connected_h5, "r") as f:
                spacing = (
                    float(f.attrs.get("spacing_d", default_spacing[0])),
                    float(f.attrs.get("spacing_h", default_spacing[1])),
                    float(f.attrs.get("spacing_w", default_spacing[2])),
                )
        else:
            spacing = default_spacing

        cfg = MetricConfig(spacing=spacing, min_size_voxels=int(args.min_size), bbox_pad=int(args.bbox_pad))
        meta = meta_by_sample.get(sample, {})
        rows: List[Dict] = []
        kept = 0

        for cid in range(1, num + 1):
            comp_mask = (labeled == cid)
            vox = int(np.count_nonzero(comp_mask))
            if vox < cfg.min_size_voxels:
                continue

            z0, z1, y0, y1, x0, x1 = compute_bbox(comp_mask)
            bbox = (z0, z1, y0, y1, x0, x1)

            comp_h5 = out_wells_dir / f"{sample}-C{cid}.h5"
            if not comp_h5.exists():
                save_component_h5(
                    comp_h5,
                    raw_norm,
                    raw_bgz,
                    prob,
                    comp_mask,
                    bbox,
                    sample_id=sample,
                    connect_id=cid,
                )
                with h5py.File(comp_h5, "a") as f:
                    f.attrs["spacing_d"] = float(spacing[0])
                    f.attrs["spacing_h"] = float(spacing[1])
                    f.attrs["spacing_w"] = float(spacing[2])

            m_norm = compute_nine_metrics(raw_norm, comp_mask, cfg=cfg)
            m_bgz = compute_nine_metrics(raw_bgz, comp_mask, cfg=cfg)

            record: Dict = {
                "Raw_Data_ID": sample,
                "Connect_ID": int(cid),
                "organoid_volume_voxels": int(vox),
                "bbox_min_z": int(z0),
                "bbox_max_z": int(z1),
                "bbox_min_y": int(y0),
                "bbox_max_y": int(y1),
                "bbox_min_x": int(x0),
                "bbox_max_x": int(x1),
                "shape_z": int(z1 - z0),
                "shape_y": int(y1 - y0),
                "shape_x": int(x1 - x0),
                "bg_mean_rawnorm": float(bg_mean),
                "bg_std_rawnorm": float(bg_std),
                **meta,
            }

            for k, v in m_norm.items():
                record[f"{k}_rawnorm"] = v
            for k, v in m_bgz.items():
                record[f"{k}_bgz"] = v

            rows.append(record)
            kept += 1

        LOG.info("[%d/%d] %s: components=%d kept=%d", i, len(samples), sample, int(num), kept)
        append_rows_csv(partial_csv, rows)
        done_samples.add(sample)

    # Final export
    if len(done_samples) == len(samples) and partial_csv.exists():
        df = pd.read_csv(partial_csv)
        if df.empty:
            raise RuntimeError("Partial CSV is empty; cannot export the final atlas.")
        out_xlsx = out_atlas_dir / atlas_xlsx_name
        df.to_excel(out_xlsx, index=False)
        LOG.info("Saved atlas: %s (rows=%d, cols=%d)", out_xlsx, len(df), len(df.columns))

    LOG.info("Progress: %d/%d samples have atlas rows in %s", len(done_samples), len(samples), partial_csv)
    LOG.info("Outputs root: %s", out_root)


if __name__ == "__main__":
    main()
