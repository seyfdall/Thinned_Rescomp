# Paper Plots

This folder contains interactive, publication-oriented analysis of reservoir-computing runs. The grid search and HDF5 aggregation happen in the parent workflow; this folder contains selected inputs, saved exemplar runs, and figures for deeper analysis.

## Contents

* `paper_plots.ipynb` is the main notebook. It loads exemplar bundles, examines reservoir states and effective connectivity, and calls plotting helpers from `utils/paper_visualization.py`.
* `parameters.json` stores named parameter configurations used by the notebook.
* `data/bundle_*/` contains saved exemplar runs. Artifact arrays are stored as `.npy` files, summary attributes are stored in `mean_attrs.json`, and optional raw datasets use the `dataset_*.npy` naming convention.
* `data/effective_connectivity_*.npy` and matching `.json` files contain effective-connectivity results and summaries for selected network sizes and mean degrees.
* The root PNG files are example response figures.

## Loading A Bundle

Run notebook cells from the repository's `Thinned_Rescomp` directory so the relative imports and paths resolve. The shared loaders are in `utils/file_io.py`:

```python
from utils.file_io import load_exemplar_bundle

bundle = load_exemplar_bundle(
    "paper_plots/data/bundle_average_spectral_radius_good_vpt_good_diversity_good_consistency_parameters"
)
artifacts = bundle["artifacts"]
mean_attrs = bundle["mean_attrs"]
```

Use `load_exemplar_bundle(path, load_datasets=True)` when the optional `dataset_*.npy` files are needed. The named bundle helpers in `file_io.py` can also load the standard `data/bundle_<parameter_set_name>/` layout.

## Plotting

`utils/paper_visualization.py` contains the publication plotting API. It can configure TeX styling, generate metric heatmaps and correlation plots, compare metrics across `rho` or `p_thin`, and visualize reservoir processing and aggregation. Plot functions accept an optional output path and create parent directories when needed.

The older command-line workflow in `utils/visualization.py` reads the HDF5 files produced by the grid search and writes aggregate plots under `paper_plots/plots/`. It is invoked from the repository root through `scripts/visualization.sh`.

## Reproducibility Notes

* Keep the parameter-set names and `rho`/`p_thin` grids aligned with the files used to produce the source results.
* The plotting module enables TeX rendering for publication figures, so a working LaTeX installation may be required when using `configure_paper_style()`.
* Large arrays and generated plots are intentionally kept separate from the simulation code; avoid committing new bulk outputs unless they are needed as documented examples or analysis inputs.