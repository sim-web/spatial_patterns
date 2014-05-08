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

path = os.path.expanduser('~/localfiles/itb_experiments/learning_grids/')

from snep.configuration import config
# config['multiproc'] = False
config['network_type'] = 'empty'

def main():
	from snep.utils import Parameter, ParameterArray
	from snep.experiment import Experiment

	# Note that runnet gets assigned to a function "run"
	exp = Experiment(path, runnet=run)
	tables = exp.tables

	param_ranges = {
		'sim': 
			{
			# 'seed_centers':ParameterArray([4]),
			},
	}

	params = {
		# 'visual': 'none',
		'sim':
			{
			'sidelength': 64,
			'factr_GABA': 1.0,
			'external_current': 1.0,
			'dt': 0.5,
			'tau': 10.0,
			'simulation_time': 100.0,
			'every_nth_step': 20.0,
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
	attractor = attractor.Attractor(params)
	my_rawdata = attractor.run()
	# rawdata is a dictionary of dictionaries (arbitrarily nested) with
	# keys (strings) and values (arrays or deeper dictionaries)
	# snep creates a group for each dictionary key and finally an array for
	# the deepest value. you can do this for raw_data or computed whenever you wish
	rawdata = {'raw_data': my_rawdata}
	return rawdata

if __name__ == '__main__':
	# cProfile.run('main()', 'profile_optimized_test')
	# pstats.Stats('profile').sort_stats('cumulative').print_stats(200)
	tables = main()
