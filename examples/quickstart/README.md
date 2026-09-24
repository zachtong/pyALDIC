# pyALDIC — quick-start example images

Small real-experiment speckle-image sequences so you can run pyALDIC right after
cloning the repository. Each folder is one experiment: `image_0000.png` is the
undeformed reference and `image_0001.png`–`image_0004.png` are successive loaded
frames.

**These images are everything you need.** They are downsampled crops of larger
experiments, chosen to cover a few typical displacement fields, and all three
examples below run on what is in this folder — there is nothing further to
download. The full-resolution originals are not published.

To run one: launch the GUI (`al-dic`), open the folder, draw a region of interest
on the reference frame, set the parameters below, and press **Run**. The initial
guess defaults to **FFT**, which is the recommended setting for all three cases.

## Examples and recommended settings

### `01_tension_without_holes`
Uniaxial tension of an aluminum dog-bone specimen — a smooth uniaxial-tension field.
- Subset size **31 px**, step **16 px**, search range **20 px**
- Solver: **AL-DIC** (or **Local DIC** for a faster first run); initial guess: **FFT**

### `02_tension_with_holes`
Uniaxial tension of a perforated aluminum specimen — tension with stress
concentration around the holes.
- Subset size **31 px**, step **16 px**, search range **20 px**
- Solver: **AL-DIC**; initial guess: **FFT**
- Cut the two holes out of the ROI (ROI → *Cut*) so the automatic masked-subset
  window splitting handles the hole boundaries.

### `03_rotation`
Rigid-body rotation of a speckled plate — a rotation field whose displacement
grows with distance from the center.
- **Workflow: Incremental mode (required).** The total rotation across the
  sequence is too large for single-reference (Accumulative) correlation, which
  would decorrelate. Incremental mode updates the reference every frame, so each
  step only tracks the small rotation relative to the previous frame, and the
  displacements are accumulated.
- Subset size **41 px**, step **16 px**, search range **40 px** — the search
  range must cover the per-step edge motion, radius × sin(per-step angle); 40 px
  is generous for this sequence.
- Solver: **AL-DIC** (the global step enforces the rotation compatibility);
  initial guess: **FFT**

*The values above are recommended starting points; fine-tune the subset and
search range for your region of interest.*
