from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def spacing_from_nrrd_header(header: Dict[str, Any], default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Extract voxel spacing (dz, dy, dx) from a NRRD header.

    The returned order follows the array axis order used throughout this project: (D, H, W).

    Supported header fields:
    - 'space directions': a list/array of 3 direction vectors
      Spacing is computed as the Euclidean norm of each vector.
    - 'spacings': a list/array of 3 scalars
    """
    # 1) space directions
    if "space directions" in header:
        dirs = header["space directions"]
        try:
            spacings = []
            for d in dirs:
                if d is None:
                    spacings.append(float("nan"))
                    continue
                if isinstance(d, (list, tuple, np.ndarray)):
                    v = np.asarray(d, dtype=float)
                    spacings.append(float(np.linalg.norm(v)))
                else:
                    spacings.append(float(d))
            if len(spacings) >= 3 and all(np.isfinite(spacings[:3])):
                return float(spacings[0]), float(spacings[1]), float(spacings[2])
        except Exception:
            pass

    # 2) spacings
    if "spacings" in header:
        try:
            s = header["spacings"]
            s = [float(x) for x in s]
            if len(s) >= 3 and all(np.isfinite(s[:3])):
                return float(s[0]), float(s[1]), float(s[2])
        except Exception:
            pass

    return default

