# pyALDIC User Guide

LaTeX sources for the user-facing walkthrough. Compiled PDFs are not
committed (to avoid binary bloat); you build them locally.

Two documents in this directory:

- **`user-guide.tex`** — the full, multi-section walkthrough.
  Produces a ~15-page PDF with detailed explanations of every panel,
  parameter, and failure mode.
- **`quick-guide.tex`** — a 2-page, two-column reference sheet. Same
  content, condensed. First thing most users should read.

Both documents must stay in sync. See the paired memory note for
the update policy when user-visible features change.

## Files

```
user-guide.tex        full walkthrough (loads preamble + all sections)
quick-guide.tex       two-page reference sheet (self-contained)
preamble.tex          package imports, colors, fonts, callout boxes
sections/
  01-overview.tex
  02-launching.tex
  03-loading-images.tex
  04-workflow-type.tex
  05-region-of-interest.tex
  06-parameters.tex
  07-advanced.tex
  08-canvas.tex
  09-running.tex
  10-viewing-results.tex
  11-strain-processing.tex
  12-export.tex
  13-session.tex
  14-troubleshooting.tex
figures/              screenshots (currently empty, TODO)
```

## Compiling

Requires any modern LaTeX distribution (TeX Live, MiKTeX, MacTeX).
Packages used: `geometry`, `fancyhdr`, `graphicx`, `xcolor`,
`hyperref`, `tcolorbox`, `listings`, `enumitem`, `booktabs`.

From the `docs/user-guide/` directory:

```bash
# Full guide (two passes for TOC)
pdflatex user-guide.tex
pdflatex user-guide.tex

# Quick two-page reference (one pass suffices — no TOC)
pdflatex quick-guide.tex

# or xelatex / lualatex — all work
xelatex user-guide.tex
xelatex user-guide.tex

# latexmk handles passes + cleanup automatically
latexmk -pdf user-guide.tex quick-guide.tex
latexmk -c   # clean up aux files after
```

Output: `user-guide.pdf` and `quick-guide.pdf`.

## Screenshots (TODO)

The `figures/` directory is intentionally empty. When adding
screenshots, place them as `figures/<section>-<what>.png` and
reference from the relevant section with:

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\textwidth]{04-workflow-type-panel.png}
  \caption{The Workflow Type section at the top of the left sidebar.}
  \label{fig:workflow-type-panel}
\end{figure}
```

## Maintaining the guide

Any change in pyALDIC that affects what a user sees --- new button,
renamed label, new dialog, changed default, moved section --- must
be reflected in this guide in the same commit (or immediately
after). The list of sections above maps directly to GUI panels, so
find the section whose topic your change touches and update it
there.
