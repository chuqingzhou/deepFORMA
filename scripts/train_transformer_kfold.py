#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train TransUNet3D (CNN-Transformer hybrid) with optional K-fold cross-validation.

This is a compact, release-friendly training script intended for reproducibility.
It trains on H5 files with datasets:
  - raw: float32 volume, shape (D,H,W)
  - label: uint8/bool mask, shape (D,H,W)

Key options:
  - --splits: a text file that lists sample filenames (Train + Val). Only those files are used.
  - --kfold / --n-folds: run K-fold CV over the selected file list.
  - --augment: basic 3D augmentations (flips/rotations/intensity jitter).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from deepforma.model.transunet3d import build_model

LOG = logging.getLogger("deepforma.train_transformer_kfold")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def load_splits_txt(path: Path) -> List[str]:
    """
    Parse a splits.txt with sections:
      Train:
      xxx.h5
      ...
      Val:
      yyy.h5
    Returns combined list (Train + Val) in file order.
    """
    names: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.endswith(":"):
            continue
        names.append(s)
    return names


def pad_to_multiple(arr: np.ndarray, m: int) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    d, h, w = arr.shape
    pd = (m - d % m) % m
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    out = np.pad(arr, ((0, pd), (0, ph), (0, pw)), mode="constant")
    return out, (d, h, w)


@dataclass(frozen=True)
class AugmentConfig:
    prob: float = 0.5


def apply_augment(raw: np.ndarray, lbl: np.ndarray, cfg: AugmentConfig) -> Tuple[np.ndarray, np.ndarray]:
    # Random flip
    if np.random.rand() < cfg.prob:
        axis = int(np.random.randint(0, 3))
        raw = np.flip(raw, axis=axis).copy()
        lbl = np.flip(lbl, axis=axis).copy()

    # Random 90-degree rotation in (D,H) and (H,W) planes
    if np.random.rand() < cfg.prob:
        k = int(np.random.randint(0, 4))
        if k:
            raw = np.rot90(raw, k=k, axes=(0, 1)).copy()
            lbl = np.rot90(lbl, k=k, axes=(0, 1)).copy()

    if np.random.rand() < cfg.prob:
        k = int(np.random.randint(0, 4))
        if k:
            raw = np.rot90(raw, k=k, axes=(1, 2)).copy()
            lbl = np.rot90(lbl, k=k, axes=(1, 2)).copy()

    # Intensity jitter (raw only)
    if np.random.rand() < cfg.prob:
        scale = float(np.random.uniform(0.85, 1.15))
        raw = raw * scale

    if np.random.rand() < cfg.prob:
        shift = float(np.random.uniform(-0.1, 0.1) * (raw.std() + 1e-8))
        raw = raw + shift

    if np.random.rand() < cfg.prob:
        noise = np.random.normal(0, 0.05 * (raw.std() + 1e-8), size=raw.shape)
        raw = raw + noise

    return raw, lbl


class H5SegDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[Path],
        *,
        pad_m: int = 16,
        norm: str = "z",
        augment: bool = False,
        augment_prob: float = 0.5,
    ):
        self.paths = list(paths)
        self.pad_m = int(pad_m)
        self.norm = str(norm)
        self.augment = bool(augment)
        self.aug_cfg = AugmentConfig(prob=float(augment_prob))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        with h5py.File(p, "r") as f:
            raw = f["raw"][()]
            lbl = f["label"][()].astype(np.float32)
        if raw.shape != lbl.shape:
            raise ValueError(f"raw/label shape mismatch in {p.name}: {raw.shape} vs {lbl.shape}")

        raw, orig = pad_to_multiple(raw, self.pad_m)
        lbl, _ = pad_to_multiple(lbl, self.pad_m)

        if self.augment:
            raw, lbl = apply_augment(raw, lbl, self.aug_cfg)

        if self.norm == "z":
            mu = float(raw.mean())
            sd = float(raw.std() + 1e-5)
            raw = (raw - mu) / sd
        elif self.norm == "minmax":
            mn = float(raw.min())
            mx = float(raw.max())
            raw = (raw - mn) / (mx - mn + 1e-5)
        else:
            raise ValueError("norm must be 'z' or 'minmax'")

        raw = raw[None]  # (1,D,H,W)
        lbl = lbl[None]
        return torch.from_numpy(raw).float(), torch.from_numpy(lbl).float(), orig, p.name


def dice_coefficient_from_logits(logits: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum()
    return (2.0 * inter + eps) / (pred.sum() + target.sum() + eps)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    den = prob.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (den + eps)


def make_folds(n: int, k: int, seed: int) -> List[np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return [a.astype(int) for a in np.array_split(idx, k)]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    bce_weight: float,
) -> Tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    loss_sum = 0.0
    dice_sum = 0.0
    n_batches = 0

    for raw, lbl, orig_shape, _name in loader:
        raw = raw.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            logits = model(raw)
            loss = bce_weight * bce(logits, lbl) + (1.0 - bce_weight) * soft_dice_loss(logits, lbl)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        # Crop padding for metric computation (assumes batch_size=1 for simplicity).
        d0, h0, w0 = orig_shape[0]
        logits_c = logits[:, :, :d0, :h0, :w0]
        lbl_c = lbl[:, :, :d0, :h0, :w0]
        dice = float(dice_coefficient_from_logits(logits_c, lbl_c).detach().cpu())

        loss_sum += float(loss.detach().cpu())
        dice_sum += dice
        n_batches += 1

    if n_batches == 0:
        return float("nan"), float("nan")
    return loss_sum / n_batches, dice_sum / n_batches


def train_one_fold(
    train_paths: Sequence[Path],
    val_paths: Sequence[Path],
    *,
    out_dir: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    pad_m: int,
    norm: str,
    augment: bool,
    batch_size: int,
    num_workers: int,
    early_stopping: int,
    seed: int,
) -> Tuple[float, int]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(dropout=dropout).to(device)

    train_ds = H5SegDataset(train_paths, pad_m=pad_m, norm=norm, augment=augment, augment_prob=0.5)
    val_ds = H5SegDataset(val_paths, pad_m=pad_m, norm=norm, augment=False)

    # This code supports batch_size>1, but metric cropping is implemented for batch_size=1.
    # For reproducibility and simplicity, default batch_size is 1.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"
    best_ckpt = out_dir / "best_transformer.pt"

    best_dice = -1.0
    best_epoch = -1
    patience = 0

    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_dice", "val_loss", "val_dice"])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            train_loss, train_dice = run_epoch(model, train_loader, optimizer=optimizer, device=device, bce_weight=0.5)
            val_loss, val_dice = run_epoch(model, val_loader, optimizer=None, device=device, bce_weight=0.5)

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                }
            )
            f.flush()

            LOG.info(
                "epoch=%d | train loss=%.4f dice=%.4f | val loss=%.4f dice=%.4f",
                epoch,
                train_loss,
                train_dice,
                val_loss,
                val_dice,
            )

            if val_dice > best_dice:
                best_dice = float(val_dice)
                best_epoch = int(epoch)
                patience = 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_dice": best_dice}, best_ckpt)
            else:
                patience += 1
                if early_stopping > 0 and patience >= early_stopping:
                    LOG.info("Early stopping triggered (patience=%d).", patience)
                    break

    return best_dice, best_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TransUNet3D with optional K-fold cross-validation.")
    p.add_argument("data_dir", type=Path, help="Directory with training H5 files.")
    p.add_argument("--splits", type=Path, default=None, help="splits.txt listing sample filenames to use (Train+Val).")
    p.add_argument("--kfold", action="store_true", help="Enable K-fold cross-validation.")
    p.add_argument("--n-folds", type=int, default=5, help="Number of folds for K-fold.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    p.add_argument("--epochs", type=int, default=400, help="Max epochs.")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay.")
    p.add_argument("--dropout", type=float, default=0.3, help="Model dropout.")
    p.add_argument("--early-stopping", type=int, default=100, help="Early stopping patience (0 disables).")

    p.add_argument("--pad-m", type=int, default=16, help="Pad each axis to a multiple of this value.")
    p.add_argument("--norm", type=str, default="z", choices=["z", "minmax"], help="Normalization method.")
    p.add_argument("--augment", action="store_true", help="Enable training-time augmentation.")

    p.add_argument("--batch-size", type=int, default=1, help="Batch size (default 1 for stability).")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")

    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: runs/<timestamp>).")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    all_h5 = sorted(data_dir.glob("*.h5"))
    if args.splits is not None:
        if not args.splits.exists():
            raise FileNotFoundError(f"splits file not found: {args.splits}")
        names = load_splits_txt(args.splits)
        file_map = {p.name: p for p in all_h5}
        missing = [n for n in names if n not in file_map]
        if missing:
            raise FileNotFoundError(f"Some files from splits are missing in data_dir: {missing[:10]}")
        files = [file_map[n] for n in names]
        LOG.info("Using %d files from splits.txt", len(files))
    else:
        files = all_h5
        LOG.info("Using all %d H5 files in data_dir", len(files))

    if not files:
        raise RuntimeError("No H5 files found for training.")

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = args.out_dir if args.out_dir is not None else (Path.cwd() / "runs" / run_id)
    out_root.mkdir(parents=True, exist_ok=True)

    # Save the file list for reproducibility
    (out_root / "files_used.txt").write_text("\n".join([p.name for p in files]) + "\n", encoding="utf-8")

    # Ensure deterministic shuffling for fold assignment
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.kfold:
        k = int(args.n_folds)
        if k < 2:
            raise ValueError("--n-folds must be >= 2 when --kfold is enabled.")
        folds = make_folds(len(files), k, seed=int(args.seed))

        summary_rows = []
        for fold_idx in range(k):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx], axis=0)
            train_paths = [files[i] for i in train_idx.tolist()]
            val_paths = [files[i] for i in val_idx.tolist()]

            fold_dir = out_root / f"fold{fold_idx + 1}"
            LOG.info("Fold %d/%d | train=%d val=%d | out=%s", fold_idx + 1, k, len(train_paths), len(val_paths), fold_dir)

            best_dice, best_epoch = train_one_fold(
                train_paths,
                val_paths,
                out_dir=fold_dir,
                epochs=int(args.epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                dropout=float(args.dropout),
                pad_m=int(args.pad_m),
                norm=str(args.norm),
                augment=bool(args.augment),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                early_stopping=int(args.early_stopping),
                seed=int(args.seed),
            )
            summary_rows.append({"fold": fold_idx + 1, "best_val_dice": best_dice, "best_epoch": best_epoch})

        df = pd.DataFrame(summary_rows)
        df.to_csv(out_root / "kfold_summary.csv", index=False)
        LOG.info("Saved K-fold summary: %s", out_root / "kfold_summary.csv")
    else:
        # Single split: use 80/20 split deterministically
        n = len(files)
        idx = np.arange(n)
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(idx)
        n_val = max(1, math.ceil(0.2 * n))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        train_paths = [files[i] for i in train_idx.tolist()]
        val_paths = [files[i] for i in val_idx.tolist()]

        LOG.info("Single split | train=%d val=%d", len(train_paths), len(val_paths))
        best_dice, best_epoch = train_one_fold(
            train_paths,
            val_paths,
            out_dir=out_root,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            dropout=float(args.dropout),
            pad_m=int(args.pad_m),
            norm=str(args.norm),
            augment=bool(args.augment),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            early_stopping=int(args.early_stopping),
            seed=int(args.seed),
        )
        LOG.info("Best val dice=%.4f at epoch=%d", best_dice, best_epoch)


if __name__ == "__main__":
    # Lazy import to avoid making pandas a hard requirement unless training is invoked.
    import pandas as pd  # noqa: E402

    main()

