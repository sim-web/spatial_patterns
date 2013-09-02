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

date_dir = '2013-08-23-18h47m27s'
tables = snep.utils.make_tables_from_path('/Users/simonweber/localfiles/itb_experiments/learning_grids/' + date_dir + '/experiment.h5')

tables.open_file(False)
tables.initialize()

print tables.get_raw_data()
for psp in tables.paramspace_pts():
	all_weights = tables.get_raw_data()
	print all_weights
	# for t in ['exc', 'inh']:

	# 	all_weights = tables.get_raw_data(psp, t + '_weights')
	# 	sparse_weights = general_utils.arrays.sparsify_two_dim_array_along_axis_1(
	# 						all_weights, 100)
	# 	tables.add_computed(
	# 		paramspace_pt=psp, all_data={t + '_weights_sparse': sparse_weights})