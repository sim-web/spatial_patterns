# spatial_patterns

Code for the simulations in Simon N Weber & Henning Sprekeler: 'Learning place cells, grid cells and invariances with excitatory and inhibitory plasticity'. eLife 2018.

Please cite the paper if you use any part of this package.

The package is made available under the GNU General Public License v3.0.

Copyright 2018 Simon N. Weber

# Getting started

Clone this repository with

`git clone --recursive https://github.com/sim-web/spatial_patterns`

The recursive flag is necessary to download the content of the git
submodules `general_utils` and `gridscore`

Try an example experiment in `experiment.py`.
To this end make sure that `experiment.py` can import from `spatial_patterns`.
So either add the directory `spatial_patterns` to your `PYTHONPATH` or move `experiment.py` one step up with resepect to `spatial_patterns`.

Select a predefined set of parameters, e.g., `parameters.params_test_2d` and run the script. A plot of the time evolution of the grid pattern should appear.
You find the predefined parameter sets in `parameters.py`.
You can also define your own set of parameters there.

# Dependencies

* Python 3, NumPy, SciPy, matplotlib


