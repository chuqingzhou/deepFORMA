# deepFORMA

Code-only release for reproducible organoid MRI segmentation and FORMA database construction.

This repository provides:
- A CNN-Transformer hybrid segmentation model (`deepforma.model.TransUNet3D`)
- A canonical atlas builder (v1.0) for organoid atlas construction (`scripts/build_database.py`)
- Extraction of nine quantitative organoid metrics (morphology, intensity, spatial distribution) used by the atlas export

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
python scripts/train_transformer_kfold.py --help
python scripts/convert_nrrd_to_h5.py --help
python scripts/convert_nrrd_to_h5_raw.py --help
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
  --out-atlas-name FORMA_Atlas_v1.0.xlsx
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
- `<out-root>/atlas/FORMA_Atlas_v1.0.xlsx` (final export; name configurable via `--out-atlas-name`)

## Nine metrics definition

See `docs/metrics.md`.

## Training (optional)

The training script uses PyTorch and is optional. Install PyTorch first, then:

```bash
pip install -r requirements-train.txt
```

See `scripts/train_transformer_kfold.py --help` for usage.

## Data and model weights

This code release does not include raw data or model weights.

The code has been tested with:
- Python >= 3.9
- NumPy 1.26.*, SciPy 1.11.*, scikit-image 0.22.*
- PyTorch 2.2.* (install separately for your platform; CPU-only or CUDA)

If the dataset/model weights are distributed via a repository (e.g., Zenodo/OSF) or controlled access, provide:
- Download link / DOI
- Checksum (e.g., SHA256)
- Access request instructions (if applicable)
