from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import nrrd

from .spacing import spacing_from_nrrd_header


def read_nrrd(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a NRRD file and return (array, header)."""
    arr, header = nrrd.read(str(path))
    return np.asarray(arr), dict(header)


def read_nrrd_with_spacing(
    path: Path,
    *,
    default_spacing: Tuple[float, float, float] = (0.16, 0.16, 0.3),
) -> Tuple[np.ndarray, Tuple[float, float, float], Dict[str, Any]]:
    """
    Read a NRRD file and return (array, spacing, header).

    Spacing order follows this project's axis convention: (D, H, W) = (z, y, x).
    If the header does not contain spacing information, `default_spacing` is returned.
    """
    arr, header = read_nrrd(path)
    spacing = spacing_from_nrrd_header(header, default=default_spacing)
    return arr, spacing, header

