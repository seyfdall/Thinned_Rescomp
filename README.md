# Thinned_Rescomp
This repository is designed to run analyses on Reservoir Computer simulations in parallel.  It builds off of the rescomp package designed here: https://github.com/djpasseyjr/rescomp.  

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Repository Layout](#repository-layout)
- [Utilities](#utilities)
- [Paper Plots](#paper-plots)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)

## Installation
Steps for downloading the Github Repo and setting up a working conda environment.

```bash
git clone https://github.com/seyfdall/Thinned_Rescomp.git
cd Thinned_Rescomp
conda env create -f environment.yml
conda activate reservoir
```

As of December 2024, the Rescomp package does not store the states of the system, so for now we've added slight modifications to the Rescomp Package Rescomp.py file.  
Update the 'update_tikhanov_factors' function to look like this so the diversity metrics on the states can be calculated.  To accomplish this I cloned the package
down into a directory on the same level as the Thinned_Rescomp directory.
```code
def update_tikhanov_factors(self, t, U):
    """ Drive the reservoir with the u and collect state information into
        self.Rhat and self.Yhat
        Parameters
        t (1 dim array): array of time values
        U (array): for each i, U[i, :] produces the state of the target system
            at time t[i]
    """
    # The i + batchsize + 1 ending adds one timestep of overlap to provide
    # the initial condition for the next batch. Overlap is removed after
    # the internal states are generated
    idxs = [(i, i + self.batchsize + 1) for i in range(0, len(t), self.batchsize)]
    #Prevent length-1 segment at the end
    if len(t)-idxs[-1][0] <= 1:
        idxs = idxs[:-1]
        idxs[-1] = (idxs[-1][0], len(t))
    # Set initial condition for reservoir nodes
    r0 = self.initial_condition(U[0, :])
    for start, end in idxs:
        ti = t[start:end]
        Ui = U[start:end, :]
        states = self.internal_state_response(ti, Ui, r0)
        if self.states is None:
            self.states = states
        else:
            self.states = np.vstack((self.states, states))
        # Get next initial condition and trim overlap
        states, r0 = states[:-1, :], states[-1, :]
        # Update Rhat and Yhat
        self.Rhat += states.T @ states
        self.Yhat += Ui[:-1, :].T @ states
    self.r0 = r0
```

Most of the filepaths from the import statements will also need to be changed to fit your current system.  The main ones will be in:
* `utils/driver.py` (path to rescomp package)
* `utils/helper.py` (path to rescomp package)
* `utils/visualization.py` (results_path and save_path variables)
* `main.py` (path to utils folder, and results_path variable)

The scripts will also need to be updated to your current system (change pathing, email, processor #, etc.)
* `scripts/simulations_array.sh`
* `scripts/visualization.sh`

May need to occassionally redownload this or move it off the autodelete system.

## Usage
Once this is done, you should be good to run it on the supercomputer.  We have slurm scripts setup in the scripts/ directory to use.
To run the gridsearch script, simply type the following in the terminal:

```bash
sh scripts/submit.sh -s
```

This will take the values from `scripts/vars.txt` and run a new gridsearch for each of the given lines.

This should generate a decent amount of initial test data to run the visualization script on to see what you're working with:

```bash
sh scripts/submit.sh -v
```

Preliminary results will be stored in a results folder.

To create new parameter or rho_p_thin sets, look over some of the functionality available in `scratch.ipynb`. The active parameter and `rho`/`p_thin` sets are JSON files in `utils/param_sets/` and `utils/rho_p_thin_sets/`; a few CSV files are retained for legacy/reference use. Feel free to create more and run gridsearches on them by updating `scripts/vars.txt`.

## Repository Layout

* `main.py` is the command-line entry point for one grid-search job. It combines a parameter set with a `rho`/`p_thin` set and sends the selected combination to the driver.
* `scripts/` contains the Slurm submission and job scripts. `scripts/vars.txt` defines the combinations submitted by `scripts/submit.sh`.
* `utils/` contains the reusable simulation, metric, file I/O, and plotting code. See [Utilities](#utilities) below.
* `paper_plots/` contains notebook-based analysis, publication figures, and saved analysis artifacts. See [Paper Plots](#paper-plots) and its [folder README](paper_plots/README.md).
* `results/` is the runtime output location used by the analysis workflow. Large result files may instead be written under the configured scratch or `nobackup` path.

## Utilities

The modules in `utils/` are normally run from the repository root, with the local `rescomp` package available beside this repository.

* `helper.py` provides argument parsing, parameter-grid construction, system-orbit generation, Erdos-Renyi network construction, spectral-radius scaling, and edge thinning.
* `driver.py` runs repeated random reservoir draws for one `rho`/`p_thin` combination. It handles time limits and recoverable numerical errors while writing each successful draw to an HDF5 group.
* `reservoir_workflows.py` owns the single-reservoir workflow. It creates and thins a network, runs replicas, trains and predicts, computes VPT, diversity, consistency, and graph metrics, and optionally returns states and other artifacts.
* `metrics.py` implements prediction, graph, diversity, and consistency metrics used by the workflows.
* `file_io.py` contains HDF5 handlers, metric aggregation helpers, and save/load helpers for notebook-focused exemplar bundles. `get_average_system_metrics()` reconstructs a `rho` by `p_thin` metric grid from HDF5 files.
* `visualization.py` is the command-line post-processing script used by `scripts/visualization.sh`. It reads averaged HDF5 metrics and writes heatmaps, correlation plots, diameter plots, and line plots under `paper_plots/plots/`.
* `paper_visualization.py` contains the more publication-oriented plotting functions used by the paper-analysis notebook. It supports publication styling, metric heatmaps, correlation plots, reservoir-state figures, and aggregation/error plots.
* `param_sets/` stores reservoir parameter sets. JSON is the active format consumed by the code; a few CSV files are retained for legacy/reference use.
* `rho_p_thin_sets/` stores JSON grids of `rho` and `p_thin` values. These names are passed to the command line with `-r`/`--rho-p-thin-set`.
* `debug.py` contains small timing helpers used while profiling or debugging runs.

## Paper Plots

The `paper_plots/` folder is a separate analysis and figure-generation workspace. It is not required to run the grid search itself.

* `paper_plots.ipynb` is the main interactive notebook for loading saved bundles, exploring reservoir states and effective connectivity, and generating paper figures.
* `parameters.json` contains named parameter configurations used by the notebook analyses.
* `data/` contains reusable inputs and outputs: effective-connectivity arrays/JSON summaries and `bundle_*` directories containing `.npy` artifacts plus optional `mean_attrs.json` and dataset files.
* The PNG files in the folder root are example figures for consistent, inconsistent, and constant reservoir responses.
* `utils/file_io.py` provides `load_exemplar_bundle()` and related helpers for reading these bundles. `utils/paper_visualization.py` provides the publication plotting functions.

For the artifact naming conventions and a focused notebook workflow, see [`paper_plots/README.md`](paper_plots/README.md).

For more information see:
* https://acme.byu.edu/00000180-6d94-d2d1-ade4-6ff4c7cf0001/mpi (for a decent walkthrough of mpi basic principles)
* https://rc.byu.edu/wiki/?id=Slurm (for more information on Slurm scripts - the site in general is good if you're operating on the BYU supercomputer)
