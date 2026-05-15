# deepFORMA

Code-only release for reproducible organoid MRI segmentation and FORMA database construction.

This repository provides:
- A CNN-Transformer hybrid segmentation model (`deepforma.model.TransUNet3D`)
- A canonical atlas builder (v1.0.1) for organoid atlas construction (`scripts/build_database.py`)
- Nine primary MRI-derived feature classes defined in `docs/metrics.md`, together with auxiliary per-component statistics exported in the atlas table for reproducibility

## Code availability scope

This repository is the core DeepFORMA software release. It includes model training, inference, connected-component demultiplexing, atlas construction, and feature extraction code.

Additional model architecture and checkpoint-loading notes: `deepforma/model/README.md`. (Chinese: `deepforma/model/README.cn.md`.)

## Installation

Create a fresh environment and install dependencies.

1) Install PyTorch (choose CPU-only or CUDA build for your platform).

2) Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

3) Install this package in editable mode:

```bash
pip install -e .
```

## Smoke tests (minimal runnable checks)

### Smoke test 1: import

```bash
python -c "import deepforma; print(deepforma.__version__)"
```

### Smoke test 2: script CLI

```bash
python scripts/build_database.py --help
python scripts/create_demo_assets.py --help
python scripts/train_transformer_kfold.py --help
python scripts/convert_nrrd_to_h5.py --help
python scripts/convert_nrrd_to_h5_raw.py --help
```

## Runnable demo

The repository includes these runnable entry points under `scripts/`: `build_database.py`, `convert_nrrd_to_h5.py`, `convert_nrrd_to_h5_raw.py`, `create_demo_assets.py`, `run_demo.sh`, and `train_transformer_kfold.py`.

The demo creates a small synthetic H5 volume, a minimal atlas metadata Excel file, and a demo-only checkpoint. It then runs the same `scripts/build_database.py` entry point used for the atlas export:

```bash
bash scripts/run_demo.sh
```

On a CPU-only laptop, the synthetic demo is expected to finish within approximately **1–3 minutes**; runtime may vary by hardware (GPU builds are typically faster for full-scale data, but this demo is small).

Expected outputs:
- `demo/output/predictions_connected_demo/DEMO001_connected.h5`
- `demo/output/wells_h5_demo/DEMO001-C1.h5`
- `demo/output/atlas/_atlas_rows_partial_demo.csv`
- `demo/output/atlas/FORMA_Atlas_demo.xlsx`

The demo checkpoint is intentionally synthetic and only verifies that the software pipeline runs end-to-end. It is not a trained segmentation model and must not be used for scientific inference.

To run the same workflow with study data, replace the demo paths with the released model checkpoint, raw H5 directory, and atlas metadata file:

```bash
python scripts/build_database.py \
  --model-path /ABS/PATH/TO/best_transformer.pt \
  --h5-raw-dir /ABS/PATH/TO/H5_RAW_DIR \
  --atlas-existing /ABS/PATH/TO/FORMA_Atlas_data0124_connect_id.xlsx \
  --out-root /ABS/PATH/TO/OUTPUT_DIR \
  --tag canonical_h5_minv100 \
  --out-atlas-name FORMA_Atlas_v1.0.1.xlsx
```

## Build the database

You need:
- Raw MRI H5 volumes: `<H5_RAW_DIR>/<Raw_Data_ID>.h5` with dataset `raw` (float32, min-max normalized to [0,1])
- An existing atlas Excel with a `Raw_Data_ID` column (used to define the sample list and carry metadata columns)
- A trained model checkpoint `.pt`

If you only have raw NRRD volumes (no labels), convert them to H5 raw first:

```bash
python scripts/convert_nrrd_to_h5_raw.py \
  --raw-dir /ABS/PATH/TO/RAW_NRRD_DIR \
  --output-dir /ABS/PATH/TO/H5_RAW_DIR
```

Example:

```bash
python scripts/build_database.py \
  --model-path /ABS/PATH/TO/best_transformer.pt \
  --h5-raw-dir /ABS/PATH/TO/H5_RAW_DIR \
  --atlas-existing /ABS/PATH/TO/FORMA_Atlas_data0124_connect_id.xlsx \
  --out-root /ABS/PATH/TO/OUTPUT_DIR \
  --tag canonical_h5_minv100 \
  --out-atlas-name FORMA_Atlas_v1.0.1.xlsx
```

Canonical defaults (v1.0):
- constant pad + z-score for model input
- connectivity=1 (6-neighborhood)
- min_volume=100
- bg_clip=1 99

Outputs:
- `<out-root>/predictions_connected_<tag>/<sample>_connected.h5`
- `<out-root>/wells_h5_<tag>/<sample>-C<id>.h5`
- `<out-root>/atlas/_atlas_rows_partial_<tag>.csv` (resumable)
- `<out-root>/atlas/FORMA_Atlas_v1.0.1.xlsx` (final export; name configurable via `--out-atlas-name`)

## Feature definitions (nine primary classes)

See `docs/metrics.md`.

## Training (optional)

The training script uses PyTorch and is optional. Install PyTorch first, then:

```bash
pip install -r requirements-train.txt
```

See `scripts/train_transformer_kfold.py --help` for usage.

## Data and model weights

This repository does not ship unrestricted raw MRI volumes or trained weights as part of the Git tree. Versioned downloads are archived on Zenodo:

- **Software** (repository release **v1.0.1**; Zenodo concept / landing): https://doi.org/10.5281/zenodo.20184803  
  For journal **Code availability**, cite the **Version DOI** displayed on Zenodo for the upload created from GitHub **release v1.0.1** once you archive that tag (Springer Nature guidance: a GitHub URL alone is not a sufficient permanent identifier). Update the software DOI in your bibliography if Zenodo mints a new version-specific DOI for that upload.
- **Restricted supporting data** (raw MRI subset, segmentation annotations, scan-level metadata, and files such as `best_transformer.pt` / `metadata_0425.xlsx`; access per record terms): https://doi.org/10.5281/zenodo.19406546

The data record is restricted-access. Access will be granted for peer review, editorial assessment, and non-commercial academic research, subject to the terms listed on the record. The trained segmentation checkpoint is provided in this record as `best_transformer.pt`, and the accompanying scan-level metadata file is `metadata_0425.xlsx`. After access is granted, use `best_transformer.pt` as `--model-path` and `metadata_0425.xlsx`, or the released metadata spreadsheet with a `Raw_Data_ID` column, as `--atlas-existing`.

For reproducible use, record the following for any downloaded artifacts (use the **Zenodo Version DOI** for software that matches the Git tag you used, e.g. **v1.0.1**; data DOI `10.5281/zenodo.19406546` as applicable):
- DOI or access URL
- File name and version
- SHA256 checksum
- Access date
- License or usage restrictions

The code has been tested with:
- Python >= 3.9
- `numpy==1.26.*`
- `scipy==1.11.*`
- `scikit-image==0.22.*`
- `torch==2.2.*` installed separately for the user's CPU/CUDA platform

## Citation

If you use this software, cite the **Zenodo software archive** (DOI below). When you use the restricted materials, cite the **supporting data record** as well.

Zenodo software archive:

```text
Zhou, C., Ren, X., Gao, S., Jiang, Y., Tian, X., Zhao, J., Fu, M.,
Wang, Q., Zhao, L., Wang, Q., Guo, W., Ni, P., & Li, T. (2026).
deepFORMA v1.0.1: organoid MRI segmentation, demultiplexing, and FORMA atlas construction
[Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20195208
```

The supporting restricted data record is:

```text
Zhou, C., Ren, X., Gao, S., Jiang, Y., Tian, X., Zhao, J., Fu, M.,
Wang, Q., Zhao, L., Wang, Q., Guo, W., Ni, P., & Li, T. (2026).
Raw MRI subset, segmentation annotations, and scan-level metadata supporting
DeepFORMA (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19406546
```
