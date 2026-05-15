# `deepforma/model`: segmentation network and checkpoint loading

This package implements a **3D TransUNet-style hybrid (CNN + Transformer)** for organoid segmentation, plus helpers to load `.pt` weights at inference time. Training and atlas-build **entry scripts** live under `scripts/` at the repository root; this directory contains **model definitions and loading only**.

---

## 1. Files and roles

| File | Role |
|------|------|
| **`transunet3d.py`** | Defines **`TransUNet3D`**: `CNNEncoder` for multi-scale features → **`PatchEmbedding3D`** to tokenize the bottleneck map → stacked **`TransformerBlock`** layers (multi-head self-attention + MLP) → **`CNNDecoder`** with skip connections and upsampling → final **`Conv3d(..., 1, 1)`** logits (foreground/background interpretation is handled upstream/downstream). |
| **`checkpoint.py`** | **`load_segmentation_model`**: reads a `torch` checkpoint dict, finds common keys (`state_dict`, `model_state`, …), and loads weights into **`build_model()`**. Also supports a smoke-test-only **`demo_zscore_passthrough`** stub (see `scripts/create_demo_assets.py`). |
| **`__init__.py`** | Re-exports the public API (`TransUNet3D`, `build_model`, `load_segmentation_model`). |

Default hyper-parameters are fixed in **`build_model()`** and should match the training script (`embed_dim=768`, `num_heads=12`, `num_layers=6`, `patch_size=2`, `in_channels=1`, …). If you change width/depth, the saved checkpoint **must** match this definition or `load_state_dict` will fail.

---

## 2. Training (how this folder is used)

- **Entry script:** `scripts/train_transformer_kfold.py` (`--help` for CLI).
- **Typical flow:** prepare H5/NRRD (or project-specific) volumes → configure folds and hyper-parameters → build the same **`TransUNet3D`** → train/validate → save **`state_dict`** (or project keys such as `model_state`) to `.pt`.
- **This directory:** supplies **forward modules only**; augmentations, losses, optimizers, logging, and distributed setup live in **training scripts or other packages**.

Released weights (e.g. **`best_transformer.pt`** on the restricted Zenodo data record) are passed to **`scripts/build_database.py --model-path`**.

---

## 3. Inference / atlas build (how this folder is used)

- **Entry script:** `scripts/build_database.py` (plus preprocessing and connected-component utilities in the same pipeline).
- **Typical flow:** load volumes → **`checkpoint.load_segmentation_model`** → obtain **`nn.Module`** and device → `model.eval()` and **`torch.no_grad()`** forward → logits/probabilities → thresholding, labeling, per-well H5 export → feature extraction and spreadsheet export.
- **This directory:** instantiates the **same `TransUNet3D` topology** as training and **aligns tensors by parameter name**.

If the checkpoint contains **`model_type: demo_zscore_passthrough`**, **`TransUNet3D` is not built**; the stub is for pipeline checks only, **not** for scientific segmentation.

---

## 4. Tensor flow (high level)

1. **Input:** single-channel 3D tensor, shape roughly `(B, 1, D, H, W)` (exact sizing depends on upstream padding / tiling).
2. **CNN encoder:** four stages with downsampling; bottleneck tensor plus three skip tensors.
3. **Patch embedding:** non-overlapping **`Conv3d` with `kernel_size = stride = patch_size`** on the bottleneck, flattened to tokens `(B, N, embed_dim)` with grid shape `(D', H', W')`.
4. **Transformer:** by default **six** `TransformerBlock` layers (pre-`LayerNorm` + residual on attention and MLP branches).
5. **CNN decoder:** project tokens back to 3D feature maps, merge skips, upsample, **1×1 conv** → **single-channel logits**.

Loss, sigmoid/thresholding, and morphological post-processing are **not** hard-coded at the end of `TransUNet3D.forward`; they belong to training or inference scripts.

---

## 5. Relation to the rest of the repository

- **Metrics / phenotyping:** `deepforma/features/` (e.g. `nine_metrics.py`), after masks are fixed.
- **I/O:** `deepforma/io/` (H5, NRRD, …).
- **Figure-only scripts:** this repository is the **core software**; journal figure pipelines may live elsewhere.

Chinese translation of this note: `deepforma/model/README.cn.md`.
