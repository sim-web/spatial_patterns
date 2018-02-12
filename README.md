# spatial_patterns

Code for the simulations in Simon N Weber & Henning Sprekeler: 'Learning place cells, grid cells and invariances with excitatory and inhibitory plasticity'. eLife 2018.

Please cite the paper if you use any part of this package.

The package is made available under the GNU General Public License v3.0.

Copyright 2018 Simon N. Weber

# Getting started

Clone this repository using

`git clone --recursive https://github.com/sim-web/spatial_patterns`

The recursive flag is necessary to download the content of the git
submodules `general_utils` and `gridscore`.

For all the imports to work, make sure that the parent directory of `spatial_patterns` is part of your `PYTHONPATH`.
If it isn't, use the following command in your terminal to add it:

`export PYTHONPATH="${PYTHONPATH}:<path to parent directory of spatial_patterns>"`

Example:
So say you cloned `spatial_patterns` to
`/Users/joe/workspace/spatial_patterns`,
then use

`export PYTHONPATH="${PYTHONPATH}:/Users/joe/workspace"`

If you want `/Users/joe/workspace` to pe permantly part of your python path, add the export line above to your .bashrc (Linux) or .bash_profile (Mac).

You find example experiments in `experiment.py`.

Go there and select a predefined set of parameters, e.g., `parameters.params_test_2d` for a quick test of 2 dimensional simulations.

Run it.

A plot of the time evolution of the grid pattern should appear.

You find the predefined parameter sets in `parameters.py`.
You can also define your own set of parameters there.
The given parameter sets are identical to those used in the paper.
See the parameter tables in the paper to find more interesting parameter combinations.

Contact me over GitHub if you encounter problems.

# Dependencies

* Python 3, NumPy, SciPy, matplotlib


