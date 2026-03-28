# STAQ-DIC-GUI

**S**patio**T**emporally **A**daptive **Q**uadtree Mesh Digital Image Correlation

A MATLAB toolkit for full-field displacement and strain measurement using DIC with adaptive quadtree mesh refinement and RBF-based smoothing. Includes a programmatic GUI for interactive analysis.

## Features

- Augmented Lagrangian DIC (AL-DIC) formulation for robust displacement tracking
- Adaptive quadtree mesh with automatic refinement near boundaries and high-gradient regions
- RBF (Radial Basis Function) smoothing for displacement and strain fields
- Programmatic GUI (`gui_aldic`) with composable region tools (rectangle, polygon, circle, cut polygon, import masks), parameter editing, solver mode selection (ALDIC / Local DIC only), natural sort, progress tracking, and full-image overlay visualization with configurable colormap and transparency
- CLI entry point (`main_aldic`) for scripting and batch processing
- Incremental and accumulative reference frame modes
- Optional POD-GPR prediction for multi-frame sequences
- Full strain analysis: engineering strains, principal strains, von Mises strain, max shear

## Requirements

- MATLAB R2020b or later
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox (for `knnsearch`, `pdist2`)
- Curve Fitting Toolbox (for ADMM beta optimization)
- A C/C++ compiler supported by MATLAB (for MEX compilation of `ba_interp2_spline.cpp`)

## Quick Start

### GUI mode (recommended)

```matlab
gui_aldic
```

1. Click **Browse Folder** to load images.
2. Define a region: **Draw Rect**, **Draw Poly**, **Add Circle**, or **Import Masks**. All operations are additive (stackable). Use **Cut Polygon** to subtract areas. **Clear** resets everything.
3. Adjust parameters (subset size, grid spacing, etc.). Uncheck **Compute Strain** for a displacement-only run.
4. Click **Run DIC**. Results overlay on the full original image with configurable colormap, transparency, and color range.
5. Use **Compute Strain** button post-hoc to add strain fields to displacement-only results. Save via **Save Results**.

### CLI mode

```matlab
main_aldic
```

Follow the interactive prompts to select images, define ROI, and configure parameters.

### Programmatic / batch mode

```matlab
DICpara = struct('winsize', 40, 'winstepsize', 16, 'winsizeMin', 8, ...
    'SizeOfFFTSearchRegion', 10, 'showPlots', false);
DICpara.gridxyROIRange.gridx = [30, 226];
DICpara.gridxyROIRange.gridy = [30, 226];
DICpara = dicpara_default(DICpara);

[file_name, Img, DICpara] = read_images(DICpara, file_name_cell, Img_cell);
[~, ImgMask] = read_masks(DICpara, mask_file_name_cell, ImgMask_cell);
results = run_aldic(DICpara, file_name, Img, ImgMask);
```

### Synthetic tests

```matlab
test_aldic_synthetic
```

Generates synthetic speckle images and validates DIC accuracy across 10 test cases (zero displacement, translation, affine, annular mask, shear, large deformation, multi-frame incremental/accumulative, local-only DIC, pure rotation).

## Directory Structure

```
STAQ-DIC-GUI/
├── gui_aldic.m               GUI entry point (programmatic uifigure)
├── run_aldic.m               Core DIC pipeline (callable function)
├── main_aldic.m              CLI entry point (thin interactive wrapper)
├── test_aldic_synthetic.m    Automated synthetic test suite (10 cases)
├── test_case6_profile.m      Performance profiling on real experimental data
├── config/                   Parameter defaults (dicpara_default.m)
├── io/                       Image I/O and preprocessing
│   ├── read_images.m         Load images (interactive or programmatic)
│   ├── read_masks.m          Load mask images
│   ├── normalize_img.m       Image normalization
│   └── img_gradient.m        Image gradient computation
├── mesh/                     Quadtree mesh generation and refinement
│   ├── mesh_setup.m          Initial uniform mesh
│   ├── generate_mesh.m       Quadtree refinement with hanging-node handling
│   ├── qrefine_r.m           Red-refinement of quadrilateral elements
│   ├── mark_edge.m           Mark elements for refinement
│   ├── mark_inside.m         Identify elements inside/outside mask
│   ├── provide_geometric_data.m
│   └── precompute_node_regions.m  One-time mask→node region mapping for smoothing
├── solver/                   DIC solvers
│   ├── integer_search.m      FFT-based integer displacement search
│   ├── local_icgn.m          Inverse Compositional Gauss-Newton (IC-GN)
│   ├── subpb1_solver.m       ADMM Subproblem 1 (local)
│   ├── subpb2_solver.m       ADMM Subproblem 2 (global FEM)
│   ├── init_disp.m           Initialize displacement field
│   ├── remove_outliers.m     Outlier detection and removal
│   └── por_gpr.m             POD-GPR displacement prediction
├── strain/                   Strain computation and smoothing
│   ├── apply_strain_type.m   Vectorized strain type conversion (infinitesimal/Eulerian/Green-Lagrangian)
│   ├── compute_strain.m      Legacy strain dispatcher (plane-fit/FD, retained as alternative)
│   ├── comp_def_grad.m       Local weighted LSQ deformation gradient (used by compute_strain)
│   ├── global_nodal_strain_fem.m  FEM-based nodal strain (primary method, O(nEle))
│   ├── global_nodal_strain_rbf.m  RBF-based global strain
│   ├── plane_fit.m           Local plane-fit strain method
│   ├── smooth_disp_rbf.m     Sparse Gaussian displacement smoothing
│   ├── smooth_strain_rbf.m   Sparse Gaussian strain smoothing
│   └── smooth_field_sparse.m Core sparse Gaussian kernel smoother
├── plotting/                 Visualization functions and colormaps
├── third_party/              External dependencies
│   ├── ba_interp2_spline.cpp MEX source for bicubic spline interpolation
│   ├── rbfinterp/            Radial basis function interpolation
│   ├── gridfit.m             Surface fitting (John D'Errico)
│   ├── inpaint_nans.m        NaN interpolation (John D'Errico)
│   ├── regularizeNd.m        N-dimensional regularization
│   ├── findpeak.m            Peak finding for cross-correlation
│   └── coolwarm.m            Coolwarm colormap
├── licences/                 License information
└── staq-dic-python/          Python port (NumPy/SciPy/Numba)
    ├── src/staq_dic/         Core algorithm: IC-GN, ADMM, FFT search, mesh
    ├── tests/                218+ tests (unit, integration, cross-validation)
    └── scripts/              Benchmarks, profiling, diagnostics
```

## Architecture

```
gui_aldic.m ──┐
              ├──> run_aldic(DICpara, file_name, Img, ImgMask)
main_aldic.m ─┘         │
                         ├── Section 3: integer_search → mesh_setup → generate_mesh
                         ├── Section 4: local_icgn (IC-GN Subproblem 1)
                         ├── Section 5: subpb2_solver (FEM Subproblem 2)
                         ├── Section 6: ADMM iterations (Subpb1 ↔ Subpb2)
                         ├── Section 7: Convergence check
                         └── Section 8: global_nodal_strain_fem + apply_strain_type → ResultStrain
```

Both `gui_aldic` and `main_aldic` call the same `run_aldic()` function. The GUI adds `ProgressFcn` / `StopFcn` / `ComputeStrain` callbacks and sets `showPlots=false` (all visualization is handled by the GUI's results viewer with full-image overlay). `run_aldic` also supports `ExportIntermediates` / `ExportDir` for saving checkpoint `.mat` files at each pipeline section (used for Python port cross-validation).

## Key Parameters

All parameters are defined in `config/dicpara_default.m` with inline documentation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `winsize` | 40 | Subset (window) size in pixels. No power-of-2 restriction. |
| `winstepsize` | 16 | Grid spacing in pixels. **Must be a power of 2.** |
| `winsizeMin` | 8 | Minimum quadtree element size. **Must be a power of 2.** |
| `SizeOfFFTSearchRegion` | 10 | FFT search region for initial guess (pixels) |
| `mu` | 1e-3 | ADMM penalty parameter |
| `alpha` | 0 | Regularization in global subproblem |
| `ADMM_maxIter` | 3 | Maximum ADMM outer iterations |
| `tol` | 1e-2 | IC-GN convergence tolerance |
| `referenceMode` | 'incremental' | `'incremental'` or `'accumulative'` |
| `StrainType` | 0 | 0=infinitesimal, 1=Eulerian-Almansi, 2=Green-Lagrangian |

> **Why power of 2?** Quadtree refinement (`qrefine_r.m`) halves element sizes by computing midpoints `(a+b)/2`. Non-power-of-2 step sizes eventually produce fractional pixel coordinates, which crash `mark_edge.m` when used as array indices.

## References

- Yang, J., & Bhattacharya, K. (2019). Augmented Lagrangian Digital Image Correlation. *Experimental Mechanics*, 59, 187-205.
- Yang, J., & Bhattacharya, K. (2021). Combining image compression with digital image correlation. *Experimental Mechanics*, 61, 469-489.

## License

See `licences/license.txt` for details.

## Author

Originally developed by Jin Yang (Caltech/UW-Madison).
Customized and restructured by Zach Tong (UT Austin).
