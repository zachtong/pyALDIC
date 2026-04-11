<p align="center">
  <!-- TODO: Create banner (1280x640px, dark background + UI screenshot + logo)
       Save to docs/images/banner.png or banner.svg -->
  <!-- <img src="docs/images/banner.png" alt="pyALDIC Banner" width="100%"/> -->
</p>

<h1 align="center">pyALDIC</h1>

<p align="center">
  <b>Augmented Lagrangian Digital Image Correlation in Python</b><br/>
  Full-field displacement and strain measurement with adaptive mesh refinement,<br/>
  ADMM global–local optimization, and a built-in desktop GUI.
</p>

<p align="center">
  <a href="https://github.com/zachtong/pyALDIC/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/zachtong/pyALDIC/ci.yml?style=flat-square&label=CI" alt="CI"/></a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/GUI-PySide6-41cd52?style=flat-square" alt="PySide6"/>
  <img src="https://img.shields.io/badge/License-BSD--3--Clause-22c55e?style=flat-square" alt="License"/>
  <a href="https://doi.org/10.5281/zenodo.19521071"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19521071-blue?style=flat-square" alt="DOI"/></a>
  <a href="https://pypi.org/project/al-dic/"><img src="https://img.shields.io/pypi/v/al-dic?style=flat-square&label=PyPI" alt="PyPI"/></a>
</p>

---

## Why pyALDIC?

Standard subset-based DIC (IC-GN) solves each node independently — accurate for small deformations, but struggles with large displacement gradients, discontinuities, and noisy images. pyALDIC uses an **Augmented Lagrangian (ADMM)** framework that couples local IC-GN subproblems with a global FEM regularizer, producing smoother, more accurate fields while maintaining sub-pixel precision.

### Comparison with Existing Tools

|  | **pyALDIC** | Ncorr | DICe | VIC-2D | MatchID |
|---|---|---|---|---|---|
| **Algorithm** | ADMM global–local | Subset (IC-GN) | Subset + Global | Subset (proprietary) | Subset (proprietary) |
| **Regularization** | FEM Q8 global | — | — | — | — |
| **Adaptive mesh** | Quadtree | — | — | — | — |
| **Mask handling** | Auto warp + window splitting | Manual | Manual | GUI masks | GUI masks |
| **Platform** | Python (cross-platform) | MATLAB | C++ | Windows | Windows |
| **Cost** | Free (BSD-3) | Free (MATLAB req.) | Free | $5K–50K+ | $5K–30K+ |
| **Open source** | Yes | Yes | Yes | No | No |

<!-- TODO: Fill in accuracy comparison data from benchmark tests -->

---

## Key Features

### User-Friendly GUI

A complete desktop application built with PySide6. Three-column layout with image list, ROI tools, and parameter controls on the left — interactive zoom/pan canvas in the center — run controls, field overlay, and console log on the right. Load images, draw ROIs, configure parameters, run DIC, and visualize results — all without writing a single line of code.

<p align="center">
  <!-- TODO: Full-window screenshot of the GUI with data loaded and displacement overlay -->
  <!-- <img src="docs/images/feature-gui.png" alt="pyALDIC GUI" width="90%"/> -->
  <i>Desktop GUI — demo coming soon</i>
</p>

### Adaptive Spatial Refinement

Quadtree mesh refinement with 5 built-in criteria: mask boundary, ROI edge, brush region, manual selection, and posterior error. Concentrates computational effort where it matters — near boundaries, discontinuities, and high-gradient regions.

<p align="center">
  <!-- TODO: Screenshot or GIF showing quadtree refinement near a crack/mask boundary -->
  <!-- <img src="docs/images/feature-adaptive-mesh.gif" alt="Adaptive Mesh Refinement" width="80%"/> -->
  <i>Adaptive quadtree mesh — demo coming soon</i>
</p>

### Dual Solver: Local DIC + AL-DIC

Run traditional local IC-GN (fast, independent nodes) or full AL-DIC with ADMM global–local coupling (regularized, smoother). Switch between modes with a single parameter — same GUI, same workflow.

<p align="center">
  <!-- TODO: Side-by-side comparison: local DIC vs AL-DIC displacement field -->
  <!-- <img src="docs/images/feature-local-vs-aldic.png" alt="Local DIC vs AL-DIC" width="80%"/> -->
  <i>Local DIC vs AL-DIC comparison — demo coming soon</i>
</p>

### Dual Tracking Modes

**Accumulative mode** — every frame compared to the first reference (best for small, monotonic deformation). **Incremental mode** — each frame compared to the previous (handles large cumulative deformation with automatic displacement composition and mask warping).

<p align="center">
  <!-- TODO: GIF showing incremental tracking through a multi-frame sequence -->
  <!-- <img src="docs/images/feature-tracking-modes.gif" alt="Tracking Modes" width="80%"/> -->
  <i>Accumulative vs incremental tracking — demo coming soon</i>
</p>

### Window Splitting (Masked Subsets)

Near mask boundaries, standard square subsets include invalid pixels. pyALDIC automatically detects partially masked subsets, splits them using connected-component analysis, and solves IC-GN on the valid region only — with Hessian conditioning checks to ensure reliability.

<p align="center">
  <!-- TODO: Visualization of window splitting near a mask boundary -->
  <!-- <img src="docs/images/feature-window-splitting.png" alt="Window Splitting" width="80%"/> -->
  <i>Window splitting near mask boundaries — demo coming soon</i>
</p>

### Visualization & Export

Full-field displacement and strain overlay with configurable colormaps, alpha blending, and deformed configuration display. Export to MATLAB `.mat`, NumPy `.npz`, CSV, PNG field maps, animated GIF/MP4, and PDF reports.

<p align="center">
  <!-- TODO: Screenshot of the GUI with displacement overlay and export dialog -->
  <!-- <img src="docs/images/feature-visualization.png" alt="Visualization & Export" width="80%"/> -->
  <i>GUI visualization and export — demo coming soon</i>
</p>

---

## Quick Start

### Installation

```bash
git clone https://github.com/zachtong/pyALDIC.git
cd pyALDIC
pip install -e ".[dev]"
```

Requires Python >= 3.10. Dependencies: NumPy, SciPy, OpenCV, Numba, scikit-image, PySide6.

### Launch GUI

```bash
al-dic
# or
python -m al_dic
```

### Programmatic API

```python
from al_dic.core.config import dicpara_default
from al_dic.core.pipeline import run_aldic
from al_dic.io.io_utils import load_images, load_masks

images = load_images("path/to/images", pattern="*.tif")
masks = load_masks("path/to/masks", pattern="*.tif")

para = dicpara_default(winsize=32, winstepsize=16)
result = run_aldic(para, images, masks)
```

---

## Accuracy

| Test Case | Displacement RMSE | Strain RMSE |
|-----------|------------------|-------------|
| Rigid translation (2.5 px) | < 0.03 px | < 0.01 |
| Affine (2% strain) | < 0.05 px | < 0.02 |
| Rotation (2°) | < 0.05 px | < 0.08 |
| Large deformation (10%) | < 1.0 px | < 0.05 |

High-resolution (1024², step=4, ~56k nodes): AL-DIC achieves **0.004 px RMSE**, 60–78% improvement over local DIC for large deformation.

## Performance

| Config | Nodes | Total Time |
|--------|-------|------------|
| 256², step=16 | 225 | ~0.07 s |
| 256², step=8 | 961 | ~0.26 s |
| 256², step=4 | 3,969 | ~1.3 s |
| 1024², step=4 | ~56,000 | ~6.5 s |

Numba JIT, post-warmup. First run adds ~0.5 s for compilation.

---

<details>
<summary><b>Project Structure</b></summary>

```
src/al_dic/
├── core/           Pipeline, config, data structures, frame scheduling
├── gui/            PySide6 GUI application
│   ├── controllers/  Image, ROI, pipeline, visualization controllers
│   ├── dialogs/      Batch import, export dialogs
│   ├── panels/       Canvas area, left/right sidebars
│   └── widgets/      Image list, parameter panel, ROI toolbar, frame nav
├── io/             Image I/O and utilities
├── mesh/           Quadtree mesh generation, refinement criteria
│   └── criteria/   Mask boundary, ROI edge, brush region, manual selection
├── solver/         IC-GN, ADMM (Subpb1/Subpb2), FFT search, FEM assembly
├── strain/         Strain computation, deformation gradient, smoothing
└── utils/          Interpolation, outlier detection, mask warping

tests/              86 test files, 800+ tests
```

</details>

<details>
<summary><b>Testing</b></summary>

```bash
# Run all tests
pytest

# Run with parallel workers
pytest -n auto

# Run specific module
pytest tests/test_solver/test_icgn_solver.py
```

</details>

---

## Citation

If you use pyALDIC in your research, please cite the software and the accompanying paper:

```bibtex
@software{tong2026pyaldic_software,
  author = {Tong, Zixiang},
  title  = {pyALDIC: Augmented Lagrangian Digital Image Correlation in Python},
  year   = {2026},
  doi    = {10.5281/zenodo.19521071},
  url    = {https://github.com/zachtong/pyALDIC}
}

%% JOSS paper (in preparation — will be updated upon publication)
@article{tong2026pyaldic,
  author  = {Tong, Zixiang and Yang, Jin},
  title   = {pyALDIC: A Python Package for Augmented Lagrangian
             Digital Image Correlation},
  journal = {Journal of Open Source Software},
  year    = {2026},
  note    = {in preparation}
}
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- Based on the [AL-DIC](https://github.com/jyang526843/2D_ALDIC) MATLAB implementation by Jin Yang
- Developed at **The University of Texas at Austin**

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
