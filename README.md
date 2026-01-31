# deepFORMA

Code-only release for reproducible organoid MRI segmentation and FORMA database construction.

This repository provides:
- A CNN-Transformer hybrid segmentation model (`deepforma.model.TransUNet3D`)
- A database builder for the Figure 5/6 definition (`scripts/build_database_final_fig56.py`)
- Extraction of nine quantitative organoid metrics (morphology, intensity, spatial distribution)

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
python scripts/build_database_final_fig56.py --help
python scripts/train_transformer_kfold.py --help
python scripts/convert_nrrd_to_h5.py --help
```

### Smoke test 3: demo (no data/weights required)

```bash
python scripts/build_database_final_fig56.py --demo --out-root demo_output
```

## Build the database (Figure 5/6 definition)

You need:
- Raw MRI NRRD volumes: `<RAW_DIR>/<Raw_Data_ID>.nrrd`
- An existing atlas Excel with a `Raw_Data_ID` column (used to define the sample list and carry metadata columns)
- A trained model checkpoint `.pt`

Example:

```bash
python scripts/build_database_final_fig56.py \
  --model-path /ABS/PATH/TO/best_transformer.pt \
  --raw-dir /ABS/PATH/TO/RAW_NRRD_DIR \
  --atlas-existing /ABS/PATH/TO/FORMA_Atlas_data0124_connect_id.xlsx \
  --out-root /ABS/PATH/TO/OUTPUT_DIR
```

Outputs:
- `<out-root>/predictions_connected/<sample>_connected.h5`
- `<out-root>/wells_h5/<sample>-C<id>.h5`
- `<out-root>/atlas/_atlas_rows_partial.csv` (resumable)
- `<out-root>/atlas/FORMA_Atlas_database_final_fig56.xlsx` (final export when completed)

## Nine metrics definition

See `docs/metrics.md`.

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

