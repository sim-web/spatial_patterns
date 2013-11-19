# open the tablefile
from snep.configuration import config
config['network_type'] = 'empty'
import snep.utils
import utils
import plotting
import matplotlib.pyplot as plt
import time
import numpy as np
tables = snep.utils.make_tables_from_path('/Users/simonweber/localfiles/itb_experiments/learning_grids/2013-08-20-11h48m27s/experiment.h5')
# tables = snep.utils.make_tables_from_path('/Users/simonweber/localfiles/itb_experiments/learning_grids/2013-08-19-19h55m42s/experiment.h5')

tables.open_file(False)
tables.initialize()
my_all_data = {'test_group': {'group_array': np.arange(3)}, 'test_array': np.random.random(4)}
for psp in tables.iter_paramspace_pts():
	tables.add_computed(paramspace_pt=psp, all_data=my_all_data)