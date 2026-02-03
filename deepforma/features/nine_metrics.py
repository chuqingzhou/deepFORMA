from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import linregress
from skimage import measure

N_SHELLS = 20


NINE_METRICS = [
    "volume",
    "sphericity",
    "sav_ratio",
    "radial_intensity_slope",
    "inner_outer_20_ratio",
    "intensity_cv",
    "intensity_mean",
    "outer_20_mean",
    "inner_20_mean",
]


@dataclass(frozen=True)
class MetricConfig:
    spacing: Tuple[float, float, float] = (0.16, 0.16, 0.3)  # (dz, dy, dx), in mm
    min_size_voxels: int = 10
    bbox_pad: int = 2  # mitigate surface truncation for marching cubes
    eps: float = 1e-8


def _calculate_sphericity(volume: float, surface_area: float, eps: float) -> float:
    # Standard definition:
    # sphericity = (pi^(1/3) * (6V)^(2/3)) / A
    if not np.isfinite(volume) or not np.isfinite(surface_area):
        return float("nan")
    if volume <= 0 or surface_area <= 0:
        return float("nan")
    numerator = (np.pi ** (1.0 / 3.0)) * ((6.0 * volume) ** (2.0 / 3.0))
    if numerator <= eps:
        return float("nan")
    return float(numerator / surface_area)


def _padded_bbox(mask: np.ndarray, pad: int) -> Tuple[int, int, int, int, int, int]:
    coords = np.where(mask)
    z0, z1 = int(coords[0].min()), int(coords[0].max()) + 1
    y0, y1 = int(coords[1].min()), int(coords[1].max()) + 1
    x0, x1 = int(coords[2].min()), int(coords[2].max()) + 1
    z0 = max(0, z0 - pad)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    z1 = min(mask.shape[0], z1 + pad)
    y1 = min(mask.shape[1], y1 + pad)
    x1 = min(mask.shape[2], x1 + pad)
    return z0, z1, y0, y1, x0, x1


def _surface_area_from_mask(mask: np.ndarray, spacing: Tuple[float, float, float], bbox_pad: int) -> float:
    if mask.sum() == 0:
        return float("nan")
    z0, z1, y0, y1, x0, x1 = _padded_bbox(mask, bbox_pad)
    crop = mask[z0:z1, y0:y1, x0:x1].astype(np.float32)
    try:
        verts, faces, *_ = measure.marching_cubes(crop, level=0.5, spacing=spacing)
        return float(measure.mesh_surface_area(verts, faces))
    except (ValueError, RuntimeError):
        return float("nan")


def _inner_outer_20_means(
    img: np.ndarray,
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    eps: float,
) -> Tuple[float, float, float]:
    """
    Inner/outer 20% intensity means based on distance transform.

    - outer_20_mean: mean intensity for voxels with distance <= P20(dist)
    - inner_20_mean: mean intensity for voxels with distance >= P80(dist)
    - inner_outer_20_ratio: inner_20_mean / (outer_20_mean + eps)
    """
    if mask.sum() == 0:
        return float("nan"), float("nan"), float("nan")

    dist_map = distance_transform_edt(mask.astype(bool), sampling=spacing)
    idx = np.where(mask)
    dist = dist_map[idx]
    inten = img[idx]
    if dist.size == 0:
        return float("nan"), float("nan"), float("nan")

    thr_outer = np.percentile(dist, 20)
    thr_inner = np.percentile(dist, 80)
    outer_sel = dist <= thr_outer
    inner_sel = dist >= thr_inner

    outer_mean = float(np.mean(inten[outer_sel])) if np.any(outer_sel) else float("nan")
    inner_mean = float(np.mean(inten[inner_sel])) if np.any(inner_sel) else float("nan")
    ratio = float(inner_mean / (outer_mean + eps)) if np.isfinite(inner_mean) and np.isfinite(outer_mean) else float("nan")
    return outer_mean, inner_mean, ratio


def _compute_radial_profile(
    img: np.ndarray,
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    n_shells: int = N_SHELLS,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compute radial intensity profile: (normalized_distance, shell_means)."""
    if mask.sum() < 10:
        return None
    dist_map = distance_transform_edt(mask.astype(bool), sampling=spacing)
    idx = np.where(mask)
    dist = dist_map[idx]
    inten = img[idx]
    if dist.size == 0:
        return None
    max_dist = dist.max()
    if max_dist < 1e-8:
        return None
    norm_dist = dist / max_dist
    shell_edges = np.linspace(0, 1, n_shells + 1)
    shell_means = []
    shell_centers = []
    for i in range(n_shells):
        lo, hi = shell_edges[i], shell_edges[i + 1]
        mask_shell = (norm_dist >= lo) & (norm_dist < hi)
        if i == n_shells - 1:
            mask_shell = (norm_dist >= lo) & (norm_dist <= hi)
        if np.any(mask_shell):
            shell_means.append(float(np.mean(inten[mask_shell])))
            shell_centers.append((lo + hi) / 2)
        else:
            shell_means.append(np.nan)
            shell_centers.append((lo + hi) / 2)
    x = np.array(shell_centers)
    y = np.array(shell_means)
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return None
    return x[valid], y[valid]


def _compute_ris_linregress(
    img: np.ndarray,
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    n_shells: int = N_SHELLS,
) -> float:
    """RIS = linear regression slope (intensity vs normalized radial distance)."""
    result = _compute_radial_profile(img, mask, spacing, n_shells)
    if result is None:
        return float("nan")
    x, y = result
    if len(x) < 2:
        return float("nan")
    slope, *_ = linregress(x, y)
    return float(slope)


def compute_nine_metrics(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    cfg: MetricConfig = MetricConfig(),
) -> Dict[str, float]:
    """
    Compute nine quantitative organoid metrics for a single connected component.

    Inputs:
    - img: 3D image (e.g., raw normalized or background z-scored), shape (D,H,W)
    - mask: 3D boolean/0-1 mask for this component, same shape as img
    """
    if img.shape != mask.shape:
        raise ValueError(f"img/mask shape mismatch: {img.shape} vs {mask.shape}")

    voxels = int(np.count_nonzero(mask))
    if voxels < cfg.min_size_voxels:
        return {k: float("nan") for k in NINE_METRICS}

    dz, dy, dx = cfg.spacing
    voxel_volume = float(dz * dy * dx)
    volume = float(voxels) * voxel_volume

    surface_area = _surface_area_from_mask(mask, cfg.spacing, cfg.bbox_pad)
    sav_ratio = float(surface_area / volume) if np.isfinite(surface_area) and volume > cfg.eps else float("nan")
    sphericity = _calculate_sphericity(volume, surface_area, cfg.eps)

    intensities = img[mask.astype(bool)]
    if intensities.size == 0:
        intensity_mean = float("nan")
        intensity_cv = float("nan")
    else:
        intensity_mean = float(np.mean(intensities))
        intensity_std = float(np.std(intensities))
        intensity_cv = float(intensity_std / (intensity_mean + cfg.eps))

    outer_20_mean, inner_20_mean, inner_outer_20_ratio = _inner_outer_20_means(
        img, mask, cfg.spacing, cfg.eps
    )

    # RIS = linear regression slope on all shells (intensity vs normalized radial distance).
    # x: normalized distance (0=edge, 1=center), y: mean intensity per shell.
    radial_intensity_slope = _compute_ris_linregress(img, mask, cfg.spacing)

    return {
        "volume": float(volume),
        "sphericity": float(sphericity),
        "sav_ratio": float(sav_ratio),
        "radial_intensity_slope": float(radial_intensity_slope),
        "inner_outer_20_ratio": float(inner_outer_20_ratio),
        "intensity_cv": float(intensity_cv),
        "intensity_mean": float(intensity_mean),
        "outer_20_mean": float(outer_20_mean),
        "inner_20_mean": float(inner_20_mean),
    }

