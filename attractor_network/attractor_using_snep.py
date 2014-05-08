# import pdb; pdb.set_trace()
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import time
import functools
import tables
import attractor
import learning_grids.initialization

path = os.path.expanduser('~/localfiles/itb_experiments/attractor_network/')

from snep.configuration import config
# config['multiproc'] = False
config['network_type'] = 'empty'

def main():
	from snep.utils import Parameter, ParameterArray
	from snep.experiment import Experiment

	# Note that runnet gets assigned to a function "run"
	exp = Experiment(path, runnet=run)
	tables = exp.tables

	beta = 3./(13**2)
	gamma = 1.05 * beta
	param_ranges = {
		'sim': 
			{
			'factor_GABA':ParameterArray([1]),
			},
	}

	params = {
		# 'visual': 'none',
		'sim':
			{
			'sidelength': 64,
			'beta': beta,
			'gamma': gamma,
			'external_current': 1.0,
			'dt': 0.5,
			'tau': 10.0,
			'simulation_time': 3000.0,
			'every_nth_step': 100,
			},
	}
	tables.add_parameter_ranges(param_ranges)
	tables.add_parameters(params)
	# Note: maybe change population to empty string
	# linked_params_tuples = [
	# 	('inh', 'sigma'),
	# 	('inh', 'init_weight'),
	# 	('inh', 'init_weight_spreading')]
	# tables.link_parameter_ranges(linked_params_tuples)
	exp.process()

def run(params, all_network_objects, monitor_objs):
	a = attractor.Attractor(params)
	my_rawdata = a.run()
	# rawdata is a dictionary of dictionaries (arbitrarily nested) with
	# keys (strings) and values (arrays or deeper dictionaries)
	# snep creates a group for each dictionary key and finally an array for
	# the deepest value. you can do this for raw_data or computed whenever you wish
	# rawdata = {'raw_data': my_rawdata}
	rawdata = {'raw_data': my_rawdata}
	return rawdata

if __name__ == '__main__':
	# cProfile.run('main()', 'profile_optimized_test')
	# pstats.Stats('profile').sort_stats('cumulative').print_stats(200)
	tables = main()
