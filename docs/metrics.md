# Nine quantitative organoid metrics

This document specifies the nine quantitative metrics implemented in `deepforma/features/nine_metrics.py`.

## Inputs

- **Image** (`img`): 3D volume, shape (D, H, W)
  - `raw_norm`: min-max normalized intensity in \([0, 1]\)
  - `bgz`: background z-scored intensity \((raw_norm - \mu_{bg}) / \sigma_{bg}\)
- **Mask** (`mask`): 3D binary mask for a single connected component (one organoid), same shape as `img`
- **Spacing** (`spacing`): voxel spacing \((dz, dy, dx)\) in mm
  - Used for distance transform and surface/volume-related metrics

## Connectivity

- Connected component labeling uses **26-neighborhood connectivity** (SciPy `generate_binary_structure(3, 3)`).
- Feature extraction is computed **per connected component** mask and does not depend on an additional connectivity definition.

## Metric list

All metrics are computed per connected component.

1) `volume`

- Voxel volume: \(V_{voxel} = dz \cdot dy \cdot dx\)
- Component voxel count: \(N\)
- **Volume**: \(V = N \cdot V_{voxel}\)

2) `sav_ratio`

- Surface area \(A\) is estimated via marching cubes on the component mask (with a small bbox padding to reduce truncation).
- **SA/V ratio**: \(SA/V = A / V\)

3) `sphericity`

- **Sphericity**:
\[
\phi = \frac{\pi^{1/3}(6V)^{2/3}}{A}
\]

4) `intensity_mean`

- Mean of voxel intensities within the component mask.

5) `intensity_cv`

- Coefficient of variation:
\[
CV = \sigma / (\mu + \epsilon)
\]

6) `outer_20_mean`

- Compute distance transform inside the component mask (with physical spacing).
- Let \(d\) be distances of all voxels inside the mask.
- Define threshold \(t_{out} = P_{20}(d)\).
- **Outer 20% mean**: mean intensity of voxels with \(d \le t_{out}\).

7) `inner_20_mean`

- Define threshold \(t_{in} = P_{80}(d)\).
- **Inner 20% mean**: mean intensity of voxels with \(d \ge t_{in}\).

8) `inner_outer_20_ratio`

- **Ratio**: \(inner\_20\_mean / (outer\_20\_mean + \epsilon)\)

9) `radial_intensity_slope`

- Linear regression slope of intensity vs normalized radial distance (0=edge, 1=center).
- The organoid is divided into N shells (default 20) by normalized distance from surface; mean intensity per shell is computed.
- **RIS**: slope from `scipy.stats.linregress(x, y)` where x = shell center positions (normalized distance), y = shell mean intensities.

## Notes

- The same metric definitions are applied to both `raw_norm` and `bgz`.
- The implementation returns `NaN` for metrics when the component is smaller than `min_size_voxels`.

