# pyALDIC User Guide

LaTeX sources for the user-facing walkthrough. Compiled PDF is not
committed (to avoid binary bloat); you build it locally.

## Files

```
user-guide.tex        main document (loads preamble + all sections)
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
# pdflatex (two passes for TOC)
pdflatex user-guide.tex
pdflatex user-guide.tex

# or xelatex / lualatex — all work
xelatex user-guide.tex
xelatex user-guide.tex

# latexmk (runs whatever is needed, cleans intermediate files)
latexmk -pdf user-guide.tex
latexmk -c   # clean up aux files after
```

Output: `user-guide.pdf`.

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
