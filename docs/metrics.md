# Nine quantitative organoid metrics

This document defines the nine primary MRI-derived organoid feature classes implemented in `deepforma/features/nine_metrics.py`. Metrics are computed per segmented organoid instance (one connected component) and are exported for both min-max-normalized intensities (`raw_norm`) and background z-scored intensities (`bgz`) using identical definitions.

The atlas export also includes auxiliary per-component statistics, such as voxel counts, surface area, intensity summary statistics, additional radial zones, and entropy, to support reproducibility and downstream quality control.

## Inputs

- **Image** (`img`): 3D volume with shape (D, H, W)
  - **`raw_norm`**: per-volume min–max normalized intensity, scaled to \([0, 1]\)
  - **`bgz`**: background z-scored intensity computed from `raw_norm` as \((raw\_norm - \mu_{\mathrm{bg}}) / (\sigma_{\mathrm{bg}} + \varepsilon)\)
- **Mask** (`mask`): 3D binary mask for a single organoid instance (one connected component), same shape as `img`
- **Spacing** (`spacing`): voxel spacing \((dz, dy, dx)\) in mm  
  Used for physical unit conversion (volume), distance transform, and surface/volume-derived metrics.

## Instance definition (connectivity)

- Organoid instances are defined by 3D connected-component labeling of the binary segmentation mask.
- By default, connected components use **6-neighborhood connectivity** (SciPy `generate_binary_structure(3, 1)`), corresponding to connectivity = 1 (options: 2/3 for 18/26-neighborhood).
- Feature extraction is computed **per connected component** mask. Metrics that rely on distance-to-surface (e.g., `outer_20_mean`, `inner_20_mean`, `radial_intensity_slope`) use an Euclidean distance transform computed **within each component mask** with physical spacing.

## Metric list

All metrics are computed per connected component (one organoid instance).

### 1) `volume`

- Voxel volume:

$$
V_{\mathrm{voxel}} = dz \cdot dy \cdot dx
$$

- Component voxel count: \(N\)
- Volume (physical units):

$$
V = N \cdot V_{\mathrm{voxel}}
$$

### 2) `sav_ratio`

- Surface area \(A\) is estimated from a triangular surface mesh reconstructed from the binary component mask using marching cubes (with a small bounding-box padding to reduce truncation artifacts).
- Surface area-to-volume ratio:

$$
SA/V = \frac{A}{V}
$$

### 3) `sphericity`

- Sphericity (voxel-based compactness) is defined as:

$$
\phi = \frac{\pi^{1/3}(6V)^{2/3}}{A}
$$
- For continuous, smooth solids, \(\phi \le 1\) with equality for a perfect sphere. In this pipeline, \(V\) is computed from voxel counts and voxel spacing, whereas \(A\) is estimated from a mesh reconstructed from the discretized binary mask (marching cubes). Due to discretization and mesh approximation—particularly under anisotropic sampling—\(A\) can be slightly underestimated, which can yield \(\phi > 1\) as a numerical artifact rather than a geometric violation. We therefore interpret \(\phi\) as a voxel-based compactness metric.
  - *(Optional; include only if this is what you actually do in figures/tables)* For visualization, values above 1 may be capped at 1.

### 4) `intensity_mean`

- Mean voxel intensity within the component mask:

$$
\mu = \mathrm{mean}\{I(\mathbf{r}) \mid \mathbf{r} \in \text{mask}\}
$$

### 5) `intensity_cv`

- Coefficient of variation (CV) of voxel intensities within the component mask:

$$
CV = \frac{\sigma}{\mu + \epsilon}
$$
  where \(\mu\) and \(\sigma\) are the mean and standard deviation of in-mask intensities, and \(\epsilon\) is a small constant to avoid division by zero.

### 6) `outer_20_mean`

- Compute the Euclidean distance transform (EDT) **inside** the component mask using physical spacing. Let \(d(\mathbf{r})\) denote the distance-to-surface for voxel \(\mathbf{r}\) within the mask.
- Define the outer compartment threshold as the 20th percentile of EDT values over all in-mask voxels:

$$
t_{\mathrm{out}} = P_{20}\left(\{d(\mathbf{r}) \mid \mathbf{r} \in \text{mask}\}\right)
$$

- Outer 20% mean intensity is the mean intensity of voxels satisfying \(d(\mathbf{r}) \le t_{\mathrm{out}}\):  

$$
\mathrm{outer\_20\_mean} = \mathrm{mean}\{I(\mathbf{r}) \mid \mathbf{r}\in\text{mask},\ d(\mathbf{r}) \le t_{\mathrm{out}}\}
$$

### 7) `inner_20_mean`

- Define the inner compartment threshold as the 80th percentile of EDT values over all in-mask voxels:

$$
t_{\mathrm{in}} = P_{80}\left(\{d(\mathbf{r}) \mid \mathbf{r} \in \text{mask}\}\right)
$$

- Inner 20% mean intensity is the mean intensity of voxels satisfying \(d(\mathbf{r}) \ge t_{\mathrm{in}}\):  

$$
\mathrm{inner\_20\_mean} = \mathrm{mean}\{I(\mathbf{r}) \mid \mathbf{r}\in\text{mask},\ d(\mathbf{r}) \ge t_{\mathrm{in}}\}
$$

### 8) `inner_outer_20_ratio`

- Ratio of inner to outer mean intensity:

$$
\mathrm{inner\_outer\_20\_ratio} = \frac{\mathrm{inner\_20\_mean}}{\mathrm{outer\_20\_mean} + \epsilon}
$$

### 9) `radial_intensity_slope`

- Using the EDT values \(d(\mathbf{r})\) computed inside the mask, define a normalized radial coordinate:

$$
\tilde{d}(\mathbf{r}) = \frac{d(\mathbf{r})}{\max_{\mathbf{r}\in\text{mask}} d(\mathbf{r})}
$$

  so that \(\tilde{d}=0\) is near the surface and \(\tilde{d}=1\) is near the center.
- Partition \(\tilde{d}\) into \(K\) shells (default \(K=20\)) and compute the mean intensity per shell. Empty shells are recorded as `NaN`.
- The radial intensity slope (RIS) is the slope from a linear regression of shell-wise mean intensity \(y\) against shell-center positions \(x\) in normalized depth:

$$
\mathrm{RIS} = \mathrm{slope}\left(\mathrm{linregress}(x, y)\right)
$$

## Notes

- The same definitions are applied to both `raw_norm` and `bgz`.
- Metrics are returned as `NaN` when the component is smaller than `min_size_voxels` (implementation safeguard against unstable estimates in very small instances).
