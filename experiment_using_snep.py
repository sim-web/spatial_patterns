# import pdb; pdb.set_trace()
import numpy as np
import os
import matplotlib as mpl
mpl.use('PDF')
import initialization
import plotting
import animating
import observables
import matplotlib.pyplot as plt
import math
import time
import output
import utils
# import tables

# sys.path.insert(1, os.path.expanduser('~/.local/lib/python2.7/site-packages/'))
path = os.path.expanduser('~/localfiles/itb_experiments/learning_grids/')

from snep.configuration import config
config['network_type'] = 'empty'

def main():
	from snep.utils import Parameter, ParameterArray
	from snep.experiment import Experiment

	# Note that runnet gets assigned to a function "run"
	exp = Experiment(path,runnet=run, postproc=postproc)
	tables = exp.tables

	target_rate = 5.0
	n_exc = 10000
	n_inh = 10000
	boxlength = 1.0
   	# For string arrays you need the list to start with the longest string
   	# you can automatically achieve this using .sort(key=len, reverse=True)
   	# motion = ['persistent', 'diffusive']
   	# motion.sort(key=len, reverse=True)
   	init_weight_noise = [0.05]
   # Note: Maybe you don't need to use Parameter() if you don't have units
	param_ranges = {
		'exc':
			{
			'eta':ParameterArray([1e-9]),
			'sigma':ParameterArray([0.05]),
			'init_weight_noise':ParameterArray(init_weight_noise),
			},
		'inh': 
			{
			'eta':ParameterArray([2e-6]),
			'sigma':ParameterArray([0.2]),
			'init_weight_noise':ParameterArray(init_weight_noise),
			},
		'sim': 
			{
			'seed_trajectory':ParameterArray([1, 2]),
			'seed_network':ParameterArray([1, 2]),
			},
		# 'exc':
		# 	{
		# 	'sigma':ParameterArray([0.05, 0.07])
		# 	},
		# 'inh':
		# 	{
		# 	'sigma':ParameterArray([0.15, 0.2, 0.3])
		# 	},
		# 'sim':
		# 	{
		# 	'velocity':ParameterArray([0.01, 0.])
		# 	},
		# 'sim':
		# 	{
		# 	'simulation_time':ParameterArray([1000000, 1])
		# 	}
		# 'sim':
		# 	{
		# 	'motion':ParameterArray(motion)
		# 	}
	}
	
	params = {
		'sim':
			{
			'dimensions': 2,
			'boxtype': 'line',
			'boxlength': boxlength,
			'diff_const': 0.01,
			'every_nth_step': 1,
			'every_nth_step_weights': 2000,
			'seed_trajectory': 2,
			'seed_network': 2,
			'simulation_time': 1e6,
			'dt': 1.0,
			'initial_x': boxlength / 2.0,
			'initial_y': boxlength / 2.0,
			'velocity': 0.01,
			'persistence_length': 0.5,
			'motion': 'persistent',
			'boundary_conditions': 'billiard',	
			},
		'out':
			{
			'target_rate': target_rate,
			'normalization': 'quadratic_multiplicative'
			},
		'exc':
			{
			'eta': 1e-8,
			'sigma': 0.05,
			'n': n_exc,
			'init_weight':ParameterArray(20. * target_rate / n_exc),
			'init_weight_noise': 0.05,
			},
		'inh':
			{
			'eta': 2e-5,
			'sigma': 0.3,
			'n': n_inh,
			'init_weight':ParameterArray(5. * target_rate / n_inh),
			'init_weight_noise': 0.05,				
			}
	}

	tables.add_parameter_ranges(param_ranges)
	tables.add_parameters(params)

	# Note: maybe change population to empty string
	linked_params_tuples_1 = [
		('exc', 'eta'),
		('inh', 'eta')]
	tables.link_parameter_ranges(linked_params_tuples_1)

	linked_params_tuples_2 = [
		('exc', 'init_weight_noise'),
		('inh', 'init_weight_noise')]
	tables.link_parameter_ranges(linked_params_tuples_2)

	# memory_usage = 
	# print "Estimated memory usage by synaptic weights alone: " 
	exp.process()
	
	# Working on a table after it has been stored to disk
	# Note: here snep still knows which table you're working on
	# if that weren't the case you use make_tables_from_path(path) (in utils.py)

	# # open the tablefile
	# tables.open_file(True)
	# # iterate over all the paramspasepoints
	# for psp in tables.iter_paramspace_pts():
	#     # raw0 = tables.get_raw_data(psp,'something')
	#     # raw1 = tables.get_raw_data(psp,'something/simonsucks')
	#     # com = tables.get_computed(psp)
	#     # Note: a psp is a dictionary like in params (I think)
	#     # You can specify a path (here 'exc_sigmas') if you just want this 
	#     # specific part of it
	#     raw0 = tables.get_raw_data(psp, 'exc_sigmas')
	# print raw0
	# tables.close_file()

# Code from Owen:
	# tables.open_file(readonly)
	# for psp in tables.iter_paramspace_pts():
	#   tables.get_raw_data(psp)

# def run(params, all_network_objects, monitor_objs):
#     rawdata = {'raw_data':{'something':{'simonsucks':np.arange(50),
#                                         'owenrules':np.arange(5)
#                                         }}, 
#                'computed':{'poop':np.arange(20.)}}
#     return rawdata

def run(params, all_network_objects, monitor_objs):
	# my_params = {}

	# Construct old style params file
	# for k, v in params.iteritems():
	# 	my_params[k[1]] = v
	# print params

	# for d in params:
	# 	if isinstance(params[d], dict):
	# 		for k in params[d]:
	# 			my_params[k] = params[d][k]
	
	# rat = initialization.Rat(my_params)
	rat = initialization.Rat(params)
	my_rawdata = rat.run()
	# rawdata is a dictionary of dictionaries (arbitrarily nested) with
	# keys (strings) and values (arrays or deeper dictionaries)
	# snep creates a group for each dictionary key and finally an array for
	# the deepest value. you can do this for raw_data or computed whenever you wish
	rawdata = {'raw_data': my_rawdata}
	return rawdata

def postproc(params, rawdata):
	# test_dict = {'test': np.arange(7)}
	# computed = {'computed': test_dict}
	# rawdata.update(computed)
# for psp in tables.iter_paramspace_pts():
#     for t in ['exc', 'inh']:
#         all_weights = tables.get_raw_data(psp, t + '_weights')
#         sparse_weights = general_utils.arrays.sparsify_two_dim_array_along_axis_1(all_weights, 1000)
#         tables.add_computed(paramspace_pt=psp, all_data={t + '_weights_sparse': sparse_weights})
	return rawdata

if __name__ == '__main__':
	tables = main()
