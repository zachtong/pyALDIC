# STAQ-DIC-GUI

**S**elf-adaptive **T**uning **A**ugmented Lagrangian **Q**uadtree Digital Image Correlation

A MATLAB toolkit for full-field displacement and strain measurement using DIC with adaptive quadtree mesh refinement and RBF-based smoothing.

## Features

- Augmented Lagrangian DIC (AL-DIC) formulation for robust displacement tracking
- Adaptive quadtree mesh with automatic refinement near high-gradient regions
- RBF (Radial Basis Function) smoothing for displacement and strain fields
- Incremental and accumulative reference frame modes
- Optional POD-GPR (Proper Orthogonal Decomposition - Gaussian Process Regression) for denoising
- Full strain analysis: engineering strains, principal strains, von Mises strain, max shear
- Stress computation via linear elasticity (plane stress / plane strain)

## Requirements

- MATLAB R2020b or later
- Image Processing Toolbox
- A C/C++ compiler supported by MATLAB (for MEX compilation)

## Quick Start

1. **Compile MEX file** (first time only):
   ```matlab
   mex -O ba_interp2_spline.cpp
   ```

2. **Run the main script**:
   ```matlab
   main_aldic
   ```
   Follow the GUI prompts to select reference/deformed images, define the ROI, and configure DIC parameters.

3. **Run synthetic tests** (optional):
   ```matlab
   test_aldic_synthetic
   ```
   This generates synthetic speckle images and validates DIC accuracy across 5 test cases (zero displacement, translation, affine deformation, annular mask, and shear).

## Directory Structure

```
STAQ-DIC-GUI/
├── main_aldic.m              Main entry point (GUI-driven)
├── test_aldic_synthetic.m    Automated synthetic test suite
├── config/                   Default parameter configuration
├── io/                       Image I/O and preprocessing
├── mesh/                     Quadtree mesh generation and refinement
├── solver/                   Integer search, ICGN, ADMM solvers
├── strain/                   Strain computation and RBF smoothing
├── plotting/                 Visualization and figure export
├── third_party/              External dependencies (gridfit, rbfinterp, etc.)
├── ba_interp2_spline.cpp     MEX source for bicubic interpolation
└── licences/                 License information
```

## Configuration

All DIC parameters have sensible defaults in `config/dicpara_default.m`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `winsizeMin` | 8 | Minimum quadtree element size (pixels) |
| `winsize` | 32 | Initial subset window size |
| `winstepsize` | 4 | Grid step size |
| `SizeOfFFTSearchRegion` | 120 | FFT search region for initial guess |
| `tol` | 1e-4 | ICGN convergence tolerance |
| `maxIter` | 50 | Maximum ICGN iterations |
| `alpha` | 100 | ADMM penalty parameter |
| `referenceMode` | 'incremental' | Reference frame mode |

## Third-Party Dependencies

The following external packages are bundled in `third_party/`:

- **gridfit** - Surface fitting on a grid (John D'Errico)
- **inpaint_nans** - Fill missing data by interpolation (John D'Errico)
- **regularizeNd** - N-dimensional regularization
- **rbfinterp** - Radial basis function interpolation
- **findpeak** - Peak finding for cross-correlation

Optional: [export_fig](https://github.com/altmany/export_fig) for high-quality PDF figure export (place in `plotting/export_fig-d966721/`).

## References

- Yang, J., & Bhattacharya, K. (2019). Augmented Lagrangian Digital Image Correlation. *Experimental Mechanics*, 59, 187-205.
- Yang, J., & Bhattacharya, K. (2021). Combining image compression with digital image correlation. *Experimental Mechanics*, 61, 469-489.

## License

See `licences/license.txt` for details.

## Author

Originally developed by Jin Yang (Caltech/UW-Madison).
Customized and restructured by Zach Tong (UT Austin).
