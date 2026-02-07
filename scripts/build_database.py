#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the FORMA atlas database (canonical v1.0).

Canonical pipeline (matches final minv100 atlas):
- Input raw source: H5 files storing min-max normalized volume in dataset 'raw' (float32, [0,1])
- Model preprocess: constant pad to multiple of pad_m, then z-score on padded volume
- Binary mask: sigmoid(logits) > threshold
- Instance definition: 3D connected components with SciPy connectivity=1 (6-neighborhood)
- Filter: remove components with voxel_count < min_volume (default 100)
- Background z-scoring (bgz): compute mean/std on background voxels (mask==0) after percentile clipping (default 1~99)
- Metrics: exported per component for both raw_norm and raw_bgz with the same column names as
  FORMA_Atlas_data0201_final_canonical_h5_minv100.xlsx

Outputs (under --out-root):
- predictions_connected_<tag>/{sample}_connected.h5
- wells_h5_<tag>/{sample}-C{connect_id}.h5
- atlas/FORMA_Atlas_v1.0.xlsx
- atlas/_atlas_rows_partial_<tag>.csv (resumable)

Notes:
- This release does not ship raw data or model weights.
- Provide paths via CLI; avoid hard-coded absolute paths.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

from deepforma.features.nine_metrics import MetricConfig, compute_component_metrics
from deepforma.io.h5_io import read_h5_raw
from deepforma.model.checkpoint import load_segmentation_model
from deepforma.segmentation.canonical import (
    PostprocessConfig,
    PreprocessConfig,
    filter_and_relabel_by_volume,
    label_connected_components,
    predict_prob_from_raw_minmax,
    prob_to_binary_mask,
)


LOG = logging.getLogger("deepforma.build_database")


META_EXCLUDE_COLS = {
    "Connect_ID",
    "organoid_volume_voxels",
    "bbox_min_z",
    "bbox_max_z",
    "bbox_min_y",
    "bbox_max_y",
    "bbox_min_x",
    "bbox_max_x",
    "shape_z",
    "shape_y",
    "shape_x",
}


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    z, y, x = np.where(mask)
    return int(z.min()), int(z.max()) + 1, int(y.min()), int(y.max()) + 1, int(x.min()), int(x.max()) + 1


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
    post_cfg: PostprocessConfig,
    pre_cfg: PreprocessConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("raw", data=raw_norm.astype(np.float32), compression="gzip")
        f.create_dataset("raw_bgz", data=raw_bgz.astype(np.float32), compression="gzip")
        f.create_dataset("prediction_prob", data=prob.astype(np.float32), compression="gzip")
        f.create_dataset("prediction", data=mask.astype(np.uint8), compression="gzip")
        f.create_dataset("label", data=label.astype(np.uint16), compression="gzip")
        # record canonical params
        f.attrs["pre_pad_m"] = int(pre_cfg.pad_m)
        f.attrs["pre_z_eps"] = float(pre_cfg.z_eps)
        f.attrs["post_threshold"] = float(post_cfg.threshold)
        f.attrs["post_connectivity"] = int(post_cfg.connectivity)  # SciPy semantics (1/2/3)
        f.attrs["post_min_volume"] = int(post_cfg.min_volume)
        f.attrs["bg_mean_rawnorm"] = float(bg_mean)
        f.attrs["bg_std_rawnorm"] = float(bg_std)
        f.attrs["bg_clip_low_pct"] = float(bg_clip_low_pct)
        f.attrs["bg_clip_high_pct"] = float(bg_clip_high_pct)
        f.attrs["bg_clip_low_value"] = float(bg_clip_low_value) if np.isfinite(bg_clip_low_value) else np.nan
        f.attrs["bg_clip_high_value"] = float(bg_clip_high_value) if np.isfinite(bg_clip_high_value) else np.nan
        f.attrs["bg_n"] = int(bg_n)


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
    post_cfg: PostprocessConfig,
    pre_cfg: PreprocessConfig,
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
        f.attrs["scan_id"] = str(sample_id)
        f.attrs["connect_id"] = int(connect_id)
        f.attrs["bbox_min_z"] = int(z0)
        f.attrs["bbox_max_z"] = int(z1)
        f.attrs["bbox_min_y"] = int(y0)
        f.attrs["bbox_max_y"] = int(y1)
        f.attrs["bbox_min_x"] = int(x0)
        f.attrs["bbox_max_x"] = int(x1)
        f.attrs["shape"] = raw_crop.shape
        f.attrs["pre_pad_m"] = int(pre_cfg.pad_m)
        f.attrs["pre_z_eps"] = float(pre_cfg.z_eps)
        f.attrs["post_threshold"] = float(post_cfg.threshold)
        f.attrs["post_connectivity"] = int(post_cfg.connectivity)
        f.attrs["post_min_volume"] = int(post_cfg.min_volume)


def append_rows_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FORMA atlas database (canonical v1.0).")
    p.add_argument("--h5-raw-dir", type=Path, required=True, help="Directory containing H5 raw files: <sample>.h5 with dataset 'raw'.")
    p.add_argument("--atlas-existing", type=Path, required=True, help="Existing atlas Excel with Raw_Data_ID and metadata columns.")
    p.add_argument("--model-path", type=Path, required=True, help="Segmentation model checkpoint (.pt).")
    p.add_argument("--out-root", type=Path, required=True, help="Output root directory.")
    p.add_argument("--tag", type=str, default="canonical_h5_minv100", help="Output tag used in folder names.")
    p.add_argument("--out-atlas-name", type=str, default="FORMA_Atlas_v1.0.xlsx", help="Output atlas Excel filename under <out-root>/atlas/.")

    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--connectivity", type=int, default=1, choices=[1, 2, 3], help="SciPy connectivity: 1=6,2=18,3=26.")
    p.add_argument("--min-volume", type=int, default=100, help="Minimum component size in voxels.")
    p.add_argument("--pad-m", type=int, default=16, help="Constant pad multiple.")
    p.add_argument("--z-eps", type=float, default=1e-5, help="Z-score epsilon.")
    p.add_argument("--bg-clip", nargs=2, type=float, default=[1, 99], help="Background percentile clipping (low high).")
    p.add_argument("--spacing", nargs=3, type=float, default=[0.16, 0.16, 0.3], help="Voxel spacing (dz dy dx) in mm.")
    p.add_argument("--min-size-metrics", type=int, default=10, help="Minimum voxels for metric computation.")
    p.add_argument("--bbox-pad", type=int, default=2, help="Padding for surface area marching cubes bbox.")

    p.add_argument("--resume-csv", type=Path, default=None, help="Optional path to partial CSV for resumable runs.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    h5_raw_dir: Path = args.h5_raw_dir
    atlas_existing: Path = args.atlas_existing
    out_root: Path = args.out_root
    tag: str = str(args.tag).strip()
    out_atlas_name: str = str(args.out_atlas_name).strip()

    out_atlas_dir = out_root / "atlas"
    out_connected_dir = out_root / f"predictions_connected_{tag}"
    out_wells_dir = out_root / f"wells_h5_{tag}"
    out_partial_csv = out_atlas_dir / f"_atlas_rows_partial_{tag}.csv"
    out_xlsx = out_atlas_dir / out_atlas_name

    out_atlas_dir.mkdir(parents=True, exist_ok=True)
    out_connected_dir.mkdir(parents=True, exist_ok=True)
    out_wells_dir.mkdir(parents=True, exist_ok=True)

    # sample list + metadata
    old = pd.read_excel(atlas_existing)
    old["Raw_Data_ID"] = old["Raw_Data_ID"].astype(str).str.strip()
    meta_cols = [c for c in old.columns if c not in META_EXCLUDE_COLS and c != "Raw_Data_ID"]
    meta_by_sample = old.groupby("Raw_Data_ID")[meta_cols].first().to_dict(orient="index")
    samples = sorted(meta_by_sample.keys())
    LOG.info("Samples to build: %d (from %s)", len(samples), atlas_existing)

    done_samples = set()
    resume_csv = args.resume_csv or out_partial_csv
    if resume_csv.exists():
        try:
            df_done = pd.read_csv(resume_csv)
            if "Raw_Data_ID" in df_done.columns:
                done_samples = set(df_done["Raw_Data_ID"].astype(str).unique().tolist())
            LOG.info("Resume: found partial rows for %d samples", len(done_samples))
        except Exception as e:
            LOG.warning("Resume: failed to read partial CSV (%s): %s", resume_csv, e)
            done_samples = set()

    model, device = load_segmentation_model(args.model_path, dropout=0.3, prefer_cuda=True)
    LOG.info("Device: %s", device)

    pre_cfg = PreprocessConfig(pad_m=int(args.pad_m), z_eps=float(args.z_eps))
    post_cfg = PostprocessConfig(threshold=float(args.threshold), connectivity=int(args.connectivity), min_volume=int(args.min_volume))
    clip_low, clip_high = float(args.bg_clip[0]), float(args.bg_clip[1])
    metric_cfg = MetricConfig(
        spacing=(float(args.spacing[0]), float(args.spacing[1]), float(args.spacing[2])),
        min_size_voxels=int(args.min_size_metrics),
        bbox_pad=int(args.bbox_pad),
    )

    # main loop
    for i, sample in enumerate(samples, 1):
        if sample in done_samples:
            continue
        h5_path = h5_raw_dir / f"{sample}.h5"
        if not h5_path.exists():
            LOG.warning("[%d/%d] SKIP missing H5: %s", i, len(samples), h5_path)
            continue

        raw_norm = read_h5_raw(h5_path, key="raw")
        prob = predict_prob_from_raw_minmax(model, raw_norm, device, pre_cfg)
        mask = prob_to_binary_mask(prob, threshold=post_cfg.threshold)
        labeled0, _num0 = label_connected_components(mask, connectivity=post_cfg.connectivity)
        labeled, num = filter_and_relabel_by_volume(labeled0, min_volume=post_cfg.min_volume)

        bg_mean, bg_std, bg_lo, bg_hi, bg_n = estimate_background(raw_norm, mask, clip_low=clip_low, clip_high=clip_high)
        raw_bgz = (raw_norm - bg_mean) / (bg_std + 1e-8)

        connected_h5 = out_connected_dir / f"{sample}_connected.h5"
        save_connected_h5(
            connected_h5,
            raw_norm,
            raw_bgz,
            prob,
            mask,
            labeled,
            bg_mean=bg_mean,
            bg_std=bg_std,
            bg_clip_low_pct=clip_low,
            bg_clip_high_pct=clip_high,
            bg_clip_low_value=bg_lo,
            bg_clip_high_value=bg_hi,
            bg_n=bg_n,
            post_cfg=post_cfg,
            pre_cfg=pre_cfg,
        )

        meta = meta_by_sample.get(sample, {})
        rows: List[Dict] = []
        built = 0
        for cid in range(1, int(num) + 1):
            organ_mask = (labeled == cid)
            vox = int(np.count_nonzero(organ_mask))
            if vox <= 0:
                continue

            z0, z1, y0, y1, x0, x1 = compute_bbox(organ_mask)
            rec: Dict = {
                "Raw_Data_ID": sample,
                "Connect_ID": int(cid),
                "organoid_count": 1,
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
                "bg_clip_low": int(clip_low),
                "bg_clip_high": int(clip_high),
                **meta,
                "bg_mean_rawnorm": float(bg_mean),
                "bg_std_rawnorm": float(bg_std),
            }

            m_raw = compute_component_metrics(raw_norm, organ_mask, cfg=metric_cfg)
            m_bgz = compute_component_metrics(raw_bgz, organ_mask, cfg=metric_cfg)

            # write with suffixes matching the final atlas columns
            for k, v in m_raw.items():
                rec[f"{k}_rawnorm"] = v
            for k, v in m_bgz.items():
                rec[f"{k}_bgz"] = v

            # per-component H5 crop
            comp_path = out_wells_dir / f"{sample}-C{cid}.h5"
            if not comp_path.exists():
                save_component_h5(
                    comp_path,
                    raw_norm,
                    raw_bgz,
                    prob,
                    organ_mask,
                    (z0, z1, y0, y1, x0, x1),
                    sample_id=sample,
                    connect_id=int(cid),
                    post_cfg=post_cfg,
                    pre_cfg=pre_cfg,
                )

            rows.append(rec)
            built += 1

        LOG.info("[%d/%d] %s: cc=%d rows=%d", i, len(samples), sample, int(num), built)
        append_rows_csv(out_partial_csv, rows)
        done_samples.add(sample)

    # export final xlsx
    df = pd.read_csv(out_partial_csv) if out_partial_csv.exists() else pd.DataFrame()
    if df.empty:
        raise RuntimeError("No rows generated; cannot export atlas.")

    # enforce column order to match the published v1.0 atlas
    wanted_cols = [
        "Raw_Data_ID",
        "Connect_ID",
        "organoid_count",
        "organoid_volume_voxels",
        "bbox_min_z",
        "bbox_max_z",
        "bbox_min_y",
        "bbox_max_y",
        "bbox_min_x",
        "bbox_max_x",
        "shape_z",
        "shape_y",
        "shape_x",
        "bg_clip_low",
        "bg_clip_high",
        *[c for c in old.columns if c not in {"Raw_Data_ID"} and c in df.columns and c not in META_EXCLUDE_COLS],
        "bg_mean_rawnorm",
        "bg_std_rawnorm",
    ]
    # add metric columns (rawnorm then bgz) in the exact names used by the final atlas
    metric_order = [
        "voxel_count",
        "volume",
        "surface_area",
        "sav_ratio",
        "sphericity",
        "intensity_mean",
        "intensity_std",
        "intensity_median",
        "intensity_min",
        "intensity_max",
        "intensity_range",
        "intensity_q25",
        "intensity_q75",
        "intensity_iqr",
        "intensity_skewness",
        "intensity_kurtosis",
        "intensity_cv",
        "outer_20",
        "inner_20",
        "outer_40",
        "inner_40",
        "outer_60",
        "inner_60",
        "outer_80",
        "inner_80",
        "radial_intensity_slope",
        "inner_outer_20_ratio",
        "inner_20_mean",
        "outer_20_mean",
        "uniformity_entropy",
    ]
    wanted_cols += [f"{m}_rawnorm" for m in metric_order]
    wanted_cols += [f"{m}_bgz" for m in metric_order]

    # keep only existing columns, append any extras at end
    cols_in = [c for c in wanted_cols if c in df.columns]
    extras = [c for c in df.columns if c not in cols_in]
    out_df = df[cols_in + extras].copy()

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(out_xlsx, index=False)
    LOG.info("Saved atlas: %s rows=%d cols=%d", out_xlsx, len(out_df), len(out_df.columns))
    LOG.info("Outputs root: %s", out_root)


if __name__ == "__main__":
    main()

