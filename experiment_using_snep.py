# import pdb; pdb.set_trace()
import numpy as np
import os
import matplotlib as mpl
mpl.use('TkAgg')
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
	n_exc = 100
	n_inh = 100
   	boxlength = 1.0

   # Note: Maybe you don't need to use Parameter() if you don't have units
	param_ranges = {
		'exc':
			{
			'eta':ParameterArray([0.00000001, 0.0000001])
			},
		'inh': 
			{
			'eta':ParameterArray([0.00002, 0.0002])
			}
	}
	
	params = {
		'sim':
			{
			'dimensions': 1,
			'boxtype': 'line',
			'boxlength': boxlength,
			'diff_const': 0.01,
			'every_nth_step': 10,
			# 'seed': 1,
			'simulation_time': 10000.0,
			'dt': 1.0,
			'initial_x': boxlength / 2.0,
			'initial_y': boxlength / 2.0,
			'velocity': 0.01,
			'persistence_length': 0.1,
			'motion': 'diffusive',
			'boundary_conditions': 'reflective',		
			},
		'out':
			{
			'target_rate': target_rate,
			'normalization': 'quadratic_multiplicative'
			},
		'exc':
			{
			'sigma': 0.03,
			'n': n_exc,
			'init_weight':ParameterArray(20. * target_rate / n_exc),
			'init_weight_noise': 0.05,
			},
		'inh':
			{
			'sigma': 0.1,
			'n': n_inh,
			'init_weight':ParameterArray(5. * target_rate / n_inh),
			'init_weight_noise': 0.05,				
			}
	}

	tables.add_parameter_ranges(param_ranges)
	tables.add_parameters(params)

	# Note: maybe change population to empty string
	linked_params_tuples = [
		('population', 'exc', 'eta'),
		('population', 'inh', 'eta')
	]

	tables.link_parameter_ranges(linked_params_tuples)

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
	my_params = {}

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
	my_rawdata = rat.run(output=True)
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
