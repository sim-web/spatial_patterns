# import pdb; pdb.set_trace()
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import initialization
import animating
import observables
import matplotlib.pyplot as plt
import math
import time
# import output
import plotting
import utils
import plot
import functools
# from memory_profiler import profile

# import cProfile    
# import pstats
# import tables

# sys.path.insert(1, os.path.expanduser('~/.local/lib/python2.7/site-packages/'))
path = os.path.expanduser('~/localfiles/itb_experiments/learning_grids/')

from snep.configuration import config
# config['multiproc'] = False
config['network_type'] = 'empty'

def get_fixed_point_initial_weights(dimensions, radius, weight_overlap,
		target_rate, init_weight_exc, sigma_exc, sigma_inh, n_exc, n_inh,
		sigma_exc_x=None, sigma_exc_y=None, sigma_inh_x=None, sigma_inh_y=None):
	"""Initial inhibitory weights chosen s.t. firing rate = target rate

	From the analytics we know which combination of initial excitatory 
	and inhibitory weights leads to an overall output rate of the 
	target rate.
	Note: it is crucial to link the corresponding parameters
	
	Parameters
	----------
	dimensions : int
	sigma_exc : float or ndarray
	sigma_inh : float or ndarray
		`sigma_exc` and `sigma_inh` must be of same shape

	Returns
	-------
	output : float or ndarray
		Values for the initial inhibitory weights
	"""

	if dimensions == 1:
		init_weight_inh = ( (init_weight_exc * n_exc * sigma_exc
						- 2*(radius+weight_overlap)*target_rate / np.sqrt(2. * np.pi))
						/ (n_inh * sigma_inh) )

	elif dimensions == 2:
		# Check for asymmetry
		if sigma_exc_x is None:
			sigma_exc_x, sigma_exc_y = sigma_exc, sigma_exc
		if sigma_inh_x is None:
			sigma_inh_x, sigma_inh_y = sigma_inh, sigma_inh
		init_weight_inh = ((init_weight_exc * n_exc * sigma_exc_x * sigma_exc_y
						- (2*(radius+weight_overlap))**2 * target_rate / (2. * np.pi))
						/ (n_inh * sigma_inh_x * sigma_inh_y) )
	return init_weight_inh

simulation_time = 1e7
def main():
	from snep.utils import Parameter, ParameterArray
	from snep.experiment import Experiment

	# Note that runnet gets assigned to a function "run"
	exp = Experiment(path,runnet=run, postproc=postproc)
	tables = exp.tables

	dimensions = 2
	target_rate = 1.0
	# n_exc = 1000
	# n_inh = 1000
	# radius = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
	radius = 0.5
	eta_inh = 1e-3 / (2*radius)
	eta_exc = 1e-4 / (2*radius)
	# simulation_time = 8*radius*radius*10**5
	# We want 100 fields on length 1
	# length = 2*radius + 2*overlap
	# n = 100 * (2*radius + 2*overlap)

	# sigma_exc = np.array([0.05, 0.07, 0.04, 0.03])
	# sigma_inh = np.array([0.15, 0.15, 0.12, 0.1])
	sigma_exc = np.array([0.04])
	sigma_inh = np.array([0.12])
	weight_overlap = 2.*sigma_inh
	n = int(100 * (2*radius + 2*weight_overlap))
	n = 5000
	n_exc, n_inh = n, n

	init_weight_exc = 1.0
	init_weight_inh = get_fixed_point_initial_weights(
		dimensions, radius, weight_overlap, target_rate, init_weight_exc,
		sigma_exc, sigma_inh, n_exc, n_inh)

	init_weight_spreading_norm = 2.
	# For string arrays you need the list to start with the longest string
	# you can automatically achieve this using .sort(key=len, reverse=True)
	# motion = ['persistent', 'diffusive']
	# motion.sort(key=len, reverse=True)
	boxtype = ['linear', 'circular']
	boxtype.sort(key=len, reverse=True)
	# init_weight_noise = [0, 0.05, 0.1, 0.5, 0.99999]
   # Note: Maybe you don't need to use Parameter() if you don't have units
	param_ranges = {
		'exc':
			{
			# 'sigma_noise':ParameterArray([0.1]),
			# 'number_desired':ParameterArray(n),
			# 'fields_per_synapse':ParameterArray([1, 4, 8]),
			# 'weight_overlap':ParameterArray(weight_overlap),
			# 'sigma_x':ParameterArray([0.05, 0.1, 0.2]),
			# 'sigma_y':ParameterArray([0.05]),
			# 'eta':ParameterArray([1e-6, 1e-5]),
			'sigma':ParameterArray(sigma_exc),
			# 'sigma_spreading':ParameterArray([1e-4, 1e-3, 1e-2, 1e-1]),
			# 'init_weight':ParameterArray(init_weight_exc),
			# 'init_weight_spreading':ParameterArray(init_weight_exc/1.5),
			},
		'inh': 
			{
			# 'sigma_x':ParameterArray([1.5, 0.2, 0.04, 0.2, 0.15, 0.15]),
			# 'sigma_y':ParameterArray([0.04, 0.04, 1.5, 1.5, 0.04, 1.5]),
			# 'eta':ParameterArray([1e-2, 1e-3]),
			'init_weight':ParameterArray(init_weight_inh),
			# 'number_desired':ParameterArray(n),
			# 'fields_per_synapse':ParameterArray([1, 4, 8]),
			# 'weight_overlap':ParameterArray(weight_overlap),
			# 'sigma_noise':ParameterArray([0.1]),
			# 'eta':ParameterArray([1e-5, 1e-4]),
			# 'sigma_spreading':ParameterArray([1e-4, 1e-3, 1e-2, 1e-1]),
			'sigma':ParameterArray(sigma_inh),
			# 'init_weight_spreading':ParameterArray(init_weight_inh/init_weight_spreading_norm),
			},
		'sim': 
			{
			'input_space_resolution':ParameterArray(sigma_exc/10.),
			'weight_overlap':ParameterArray(weight_overlap),
			# 'symmetric_centers':ParameterArray([False, True]),
			# 'seed_centers':ParameterArray([1]),
			# 'radius':ParameterArray(radius),
			# 'gaussians_with_height_one':ParameterArray([False, True]),
			# 'weight_lateral':ParameterArray(
			# 	[0.5, 1.0, 2.0, 4.0]),
			# 'output_neurons':ParameterArray([3, 4]),
			# 'seed_trajectory':ParameterArray([1, 2]),
			# 'initial_x':ParameterArray([-radius/1.42, -radius/5.3, radius/1.08]),
			# 'seed_init_weights':ParameterArray([5, 6]),
			# 'lateral_inhibition':ParameterArray([False]),
			# 'motion':ParameterArray(['persistent', 'diffusive']),
			# 'dt':ParameterArray([0.1, 0.01]),
			# 'tau':ParameterArray([0.1, 0.2, 0.4]),
			# 'boxtype':ParameterArray(boxtype),
			# 'boundary_conditions':ParameterArray(['reflective', 'periodic'])
			},
		'out':
			{
			# 'normalization':ParameterArray(['quadratic_multiplicative',
			# 	'quadratic_multiplicative_lateral_inhibition']),
			}

	}
	
	params = {
		'visual': 'figure',
		'sim':
			{
			# If -1, the input rates will be determined for the current position
			# in each time step, # Take something smaller than the smallest
			# Gaussian (by a factor of 10 maybe)
			'input_space_resolution': sigma_exc[0]/10.,
			'spacing': 51,
			'equilibration_steps': 10000,
			'gaussians_with_height_one': True,
			'stationary_rat': False,
			'same_centers': False,
			'first_center_at_zero': False,
			'lateral_inhibition': False,
			'output_neurons': 1,
			'weight_lateral': 0.0,
			'tau': 10.,
			'symmetric_centers': True,
			'dimensions': dimensions,
			'boxtype': 'linear',
			'radius': radius,
			'diff_const': 0.01,
			'every_nth_step': simulation_time/10,
			'every_nth_step_weights': simulation_time/10,
			'seed_trajectory': 3,
			'seed_init_weights': 3,
			'seed_centers': 3,
			'simulation_time': simulation_time,
			'dt': 1.0,
			'initial_x': 0.1,
			'initial_y': 0.2,
			# 'velocity': 3e-4,
			'velocity': 1e-2,
			'persistence_length': radius,
			# 'motion': 'persistent_semiperiodic',
			'motion': 'persistent',
			# 'boundary_conditions': 'periodic',
			},
		'out':
			{
			'target_rate': target_rate,
			'normalization': 'quadratic_multiplicative',
			},
		'exc':
			{
			'weight_overlap': weight_overlap,
			'eta': eta_exc,
			'sigma': sigma_exc[0],
			'sigma_spreading': 0.0,
			'sigma_distribution': 'uniform',
			'sigma_x': 0.05,
			'sigma_y': 0.05,
			'number_desired': n_exc,
			'fields_per_synapse': 1,
			'init_weight':init_weight_exc,
			'init_weight_spreading': 0.8,
			# 'init_weight_spreading': 0.0,		

			'init_weight_distribution': 'uniform',
			},
		'inh':
			{
			'weight_overlap': weight_overlap,
			'eta': eta_inh,
			'sigma': 0.1,
			# 'sigma_spreading': {'stdev': 0.01, 'left': 0.01, 'right': 0.199},
			'sigma_spreading': 0.0,
			'sigma_distribution': 'uniform',
			'sigma_x': 0.1,
			'sigma_y': 0.1,
			'number_desired': n_inh,
			'fields_per_synapse': 1,
			'init_weight':0.56,
			'init_weight_spreading': 0.8,
			# 'init_weight_spreading': 0.0,		

			'init_weight_distribution': 'uniform',
			}
	}

	tables.add_parameter_ranges(param_ranges)
	tables.add_parameters(params)

	# Note: maybe change population to empty string
	linked_params_tuples = [
		('inh', 'sigma'),
		('inh', 'init_weight'),
		# ('inh', 'init_weight_spreading'),
		('exc', 'sigma'),
		('sim', 'weight_overlap'),
		('sim', 'input_space_resolution')]
	tables.link_parameter_ranges(linked_params_tuples)

	# linked_params_tuples = [
	# 	('exc', 'fields_per_synapse'),
	# 	('inh', 'fields_per_synapse')]
	# tables.link_parameter_ranges(linked_params_tuples)

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
	file_name = os.path.basename(os.path.dirname(params['results_file']))
	save_dir = os.path.join(os.path.dirname(os.path.dirname(params['results_file'])), 'visuals')
	
	if params['visual'] == 'figure':
		file_type = '.pdf'
		file_full = file_name + file_type
		save_path = os.path.join(save_dir, file_full)
		try:
			os.mkdir(save_dir)
		except OSError:
			pass
		plot_class = plotting.Plot(params=params, rawdata=rawdata['raw_data'])
		fig = plt.figure()
		function_kwargs = [
				# ('plot_output_rates_from_equation',
				# 	{'time': 0, 'spacing': 601, 'from_file': False}),
				# # ('plot_output_rates_from_equation',
				# # 	{'time': 1e3, 'spacing': 401, 'from_file': False}),
				# # ('plot_output_rates_from_equation',
				# # 	{'time': 5e3, 'spacing': 401, 'from_file': False}),
				('plot_output_rates_from_equation', {'time': 0., 'from_file': True}),
				('plot_output_rates_from_equation', {'time': simulation_time/4., 'from_file': True}),
				('plot_output_rates_from_equation', {'time': simulation_time/2., 'from_file': True}),
				('plot_output_rates_from_equation', {'time': simulation_time, 'from_file': True}),
				# ('plot_output_rates_from_equation',
				# 	{'time': 0, 'spacing': 601, 'from_file': False}),
				# ('output_rate_heat_map', {'from_file': True, 'end_time': -1})
			]
		plot_list = [functools.partial(getattr(plot_class, f), **kwargs) for f, kwargs in function_kwargs]
		plotting.plot_list(fig, plot_list)
		plt.savefig(save_path, dpi=170, bbox_inches='tight', pad_inches=0.02)

	if params['visual'] == 'video':
		file_type = '.mp4'
		file_full = file_name + file_type
		save_path = os.path.join(save_dir, file_full)
		try:
			os.mkdir(save_dir)
		except OSError:
			pass
		animation_class = animating.Animation(params, rawdata['raw_data'],
			start_time=0, end_time=params['sim']['simulation_time'],
			step_factor=1, take_weight_steps=True)
		ani = getattr(animation_class, 'animate_output_rates')
		ani(save_path=save_path, interval=50)
		plt.show()

	# # Clear figure and close windows
	# plt.clf()
	# plt.close()

	# test_dict = {'test': np.arange(7)}
	# computed = {'computed': test_dict}
	# rawdata.update(computed)
# for psp in tables.iter_paramspace_pts():
#     for t in ['exc', 'inh']:
#         all_weights = tables.get_raw_data(psp, t + '_weights')
#         sparse_weights = general_utils.arrays.sparsify_two_dim_array_along_axis_1(all_weights, 1000)
#         tables.add_computed(paramspace_pt=psp, all_data={t + '_weights_sparse': sparse_weights})
	# exp_dir = os.path.dirname(os.path.dirname(params['results_file']))
	return rawdata

if __name__ == '__main__':
	# cProfile.run('main()', 'profile2')
	# pstats.Stats('profile').sort_stats('cumulative').print_stats(20)
	tables = main()
