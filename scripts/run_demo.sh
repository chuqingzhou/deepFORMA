#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEMO_DIR="${1:-demo}"

python scripts/create_demo_assets.py --out-dir "$DEMO_DIR"

python scripts/build_database.py \
  --model-path "$DEMO_DIR/model/demo_zscore_passthrough.pt" \
  --h5-raw-dir "$DEMO_DIR/h5_raw" \
  --atlas-existing "$DEMO_DIR/atlas_existing.xlsx" \
  --out-root "$DEMO_DIR/output" \
  --tag demo \
  --out-atlas-name FORMA_Atlas_demo.xlsx \
  --min-volume 20 \
  --min-size-metrics 5

python - "$DEMO_DIR" <<'PY'
from pathlib import Path
import sys

demo_dir = Path(sys.argv[1])
expected = [
    demo_dir / "output" / "predictions_connected_demo" / "DEMO001_connected.h5",
    demo_dir / "output" / "wells_h5_demo" / "DEMO001-C1.h5",
    demo_dir / "output" / "atlas" / "_atlas_rows_partial_demo.csv",
    demo_dir / "output" / "atlas" / "FORMA_Atlas_demo.xlsx",
]
missing = [str(path) for path in expected if not path.exists()]
if missing:
    raise SystemExit("Demo failed; missing expected outputs:\n" + "\n".join(missing))

print("Demo completed successfully. Key outputs:")
for path in expected:
    print(f"  {path}")
PY
