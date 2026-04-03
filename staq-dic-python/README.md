# STAQ-DIC Python

Python port of **STAQ-DIC**: Augmented Lagrangian Digital Image Correlation with adaptive quadtree mesh.

## Features

- **IC-GN solver** — Inverse Compositional Gauss-Newton with 6-DOF (deformation gradient + displacement) and 2-DOF (displacement only) modes. Three-tier backend: Numba prange (multi-core), batch NumPy, sequential fallback.
- **FFT initial search** — Direct NCC (`cv2.matchTemplate`) and pyramid NCC with sub-pixel quadratic refinement.
- **ADMM global-local iteration** — Subproblem 1 (local IC-GN) + Subproblem 2 (global FEM regularization with Q8 elements, sparse LU/PCG solver). Beta auto-tuning via grid search.
- **Adaptive quadtree mesh** — Refinement policies: mask boundary, ROI edge, brush region, manual selection, posterior error criterion.
- **Window splitting** — Masked subset IC-GN: gradients from raw image, connected-component center mask, Hessian conditioning check.
- **Frame scheduling** — Accumulative, incremental, and custom DAG modes with cumulative displacement composition.
- **Mask warping** — Iterative inverse mapping for automatic mask update when changing reference frames. Topology-preserving with fragment cleanup.
- **Outlier detection & fill** — Statistical criterion + k-NN inverse-distance RBF interpolation.
- **Strain computation** — FEM-based nodal strain with configurable smoothing.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.10. Dependencies: NumPy, SciPy, OpenCV, Numba, scikit-image.

## Project Structure

```
src/staq_dic/
├── core/           Pipeline, config, data structures, frame scheduling
├── io/             Image I/O and utilities
├── mesh/           Quadtree mesh generation, refinement criteria, edge marking
│   └── criteria/   Refinement criteria (mask boundary, ROI edge, brush, manual, posterior error)
├── solver/         IC-GN, ADMM (Subpb1/Subpb2), FFT search, FEM assembly, Numba kernels
├── strain/         Strain computation, deformation gradient, smoothing
└── utils/          Interpolation, outlier detection, mask warping, validation

tests/              37 test files (unit, integration, mesh, solver, strain)
scripts/            31 benchmark and report scripts
reports/            Generated PDF reports
```

~10,100 lines of algorithm code.

## Quick Start

```python
from staq_dic.core.config import DICConfig
from staq_dic.core.pipeline import run_pipeline

config = DICConfig(
    winstepsize=16,
    winsize=32,
    n_admm_iter=3,
    reference_mode="accumulative",
)
results = run_pipeline(images, mask, config)
```

## Testing

```bash
# Run all tests
pytest

# Run with parallel workers
pytest -n auto

# Run specific test module
pytest tests/test_solver/test_icgn_solver.py
```

## Accuracy

| Test Case | Displacement RMSE | Strain RMSE |
|-----------|------------------|-------------|
| Rigid translation (2.5px) | < 0.03 px | < 0.01 |
| Affine (2% strain) | < 0.05 px | < 0.02 |
| Rotation (2°) | < 0.05 px | < 0.08 |
| Large deformation (10%) | < 1.0 px | < 0.05 |

High-resolution (1024², step=4, ~56k nodes): AL-DIC achieves 0.004 px RMSE, 60-78% improvement over local DIC for large deformation.

## Performance

| Config | Nodes | Total Time |
|--------|-------|------------|
| 256², step=16 | 225 | ~0.07s |
| 256², step=8 | 961 | ~0.26s |
| 256², step=4 | 3,969 | ~1.3s |
| 1024², step=4 | ~56,000 | ~6.5s |

Numba JIT, post-warmup. First run adds ~0.5s for compilation.

## License

MIT
