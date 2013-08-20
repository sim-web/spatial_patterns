# open the tablefile
from snep.configuration import config
config['network_type'] = 'empty'
import snep.utils
import utils
import general_utils.arrays
import plotting
import matplotlib.pyplot as plt
import time
import numpy as np
tables = snep.utils.make_tables_from_path('/Users/simonweber/localfiles/itb_experiments/learning_grids/')
# tables = snep.utils.make_tables_from_path('/Users/simonweber/localfiles/itb_experiments/learning_grids/2013-08-19-19h55m42s/experiment.h5')

tables.open_file(False)
tables.initialize()

for psp in tables.iter_paramspace_pts():
	for t in ['exc', 'inh']:
		all_weights = tables.get_raw_data(psp, t + '_weights')
		sparse_weights = general_utils.arrays.sparsify_two_dim_array_along_axis_1(all_weights, 1000)
		tables.add_computed(paramspace_pt=psp, all_data={t + '_weights_sparse': sparse_weights})