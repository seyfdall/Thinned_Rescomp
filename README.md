# Thinned_Rescomp
This repository is designed to run analyses on Reservoir Computer simulations in parallel.  It builds off of the rescomp package designed here: https://github.com/djpasseyjr/rescomp.  

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
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
* utils/driver.py (path to rescomp package)
* utils/helper.py (path to rescomp package)
* utils/visualization.py (results_path variable)
* main.py (path to utils folder, and results_path variable)

The scripts will also need to be updated to your current system (change pathing, email, processor #, etc.)
* scripts/mpi_search.sh
* scripts/mpi_test.sh
* scripts/visualization.sh

## Usage
Once this is done, you should be good to run it on the supercomputer.  We have slurm scripts setup in the scripts/ directory to use.
To run the test script, simply type the following in the terminal:

```bash
sbatch scripts/mpitest.sh
```

This should generate a decent amount of initial test data to run the visualization script on to see what you're working with:

```bash
sbatch scripts/visualization.sh
```

Preliminary results will be stored in a results/ folder in the directory.  Once that is working run the more major script:
scripts/mpi_search.sh

For more information see:
* https://acme.byu.edu/00000180-6d94-d2d1-ade4-6ff4c7cf0001/mpi (for a decent walkthrough of mpi basic principles)
* https://rc.byu.edu/wiki/?id=Slurm (for more information on Slurm scripts - the site in general is good if you're operating on the BYU supercomputer)
