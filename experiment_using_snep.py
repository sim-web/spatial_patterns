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
import scipy.special as sps
# import output
import plotting
import utils
import plot
import functools
import general_utils.arrays
import add_computed

# from memory_profiler import profile

# import cProfile
# import pstats
# import tables

# sys.path.insert(1, os.path.expanduser('~/.local/lib/python2.7/site-packages/'))
path = os.path.expanduser('~/localfiles/itb_experiments/learning_grids/')

from snep.configuration import config
# config['multiproc'] = False
config['network_type'] = 'empty'

simulation_time = 4e6
def main():
	from snep.utils import Parameter, ParameterArray, ParametersNamed, flatten_params_to_point
	from snep.experiment import Experiment


	dimensions = 2
	periodicity = 'none'

	if periodicity == 'none':
		boxtype = ['linear']
		motion = 'persistent'
		tuning_function = 'gaussian'
	elif periodicity == 'semiperiodic':
		boxtype = ['linear']
		motion = 'persistent_semiperiodic'
		tuning_function = 'von_mises'
	elif periodicity == 'periodic':
		boxtype = ['linear']
		motion = 'persistent_periodic'
		tuning_function = 'periodic'

	boxtype.sort(key=len, reverse=True)

	# sigma_distribution = 'gamma_with_cut_off'
	sigma_distribution = 'uniform'

	target_rate = 1.0
	# radius = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
	radius = 0.5
	eta_inh = 2e-4 / (2*radius)
	eta_exc = 2e-5 / (2*radius)

	sigma_exc = np.array([
						[0.05, 0.05],
						# [0.05, 0.05],
						# [0.05, 0.05],
						# [0.05, 0.05],
						# [0.05, 0.05],
						# [0.05, 0.05],
						# [0.05, 0.05],
						# [0.05, 0.05],
						# [0.05, 0.05],
						# [0.05, 0.05],
						# [0.05, 0.05],
						# [0.05, 0.05],
						# [0.05, 0.05],
						[0.05, 0.05],
						[0.05, 0.05],
						[0.05, 0.05],
						[0.05, 0.05],
						[0.05, 0.05],
						])


	sigma_inh = np.array([
						[0.50, 0.049],
						[0.70, 0.049],
						[0.90, 0.049],
						[0.50, 0.50],
						[0.70, 0.70],
						[0.90, 0.90],
						# [0.20, 0.10],
						# [0.10, 0.20],
						# [2.00, 0.10],
						# [0.10, 2.00],
						# [2.00, 0.20],
						# [0.20, 2.00],
						# [0.049, 0.049],
						# [0.10, 0.10],
						# [0.20, 0.20],
						# [2.0, 2.0],
						# [0.10, 0.049],
						# [0.049, 0.10],
						# [0.20, 0.049],
						# [0.049, 0.20],
						# [2.0, 0.049],
						# [0.049, 2.0],
						])

	number_per_dimension_exc = np.array([70, 70])
	number_per_dimension_inh = np.array([35, 35])

	# sinh = np.arange(0.08, 0.4, 0.02)
	# sexc = np.tile(0.03, len(sinh))
	# sigma_inh = np.atleast_2d(sinh).T.copy()
	# sigma_exc = np.atleast_2d(sexc).T.copy()

	input_space_resolution = sigma_exc/6.

	def get_ParametersNamed(a):
		l = []
		for x in a:
			l.append((str(x).replace(' ', '_'), ParameterArray(x)))
		return ParametersNamed(l)

	gaussian_process = False
	if gaussian_process:
		# init_weight_exc = 1.0 / 22.
		init_weight_exc = 1.0
		symmetric_centers = False
	else:
		init_weight_exc = 1.0
		symmetric_centers = True

	# For string arrays you need the list to start with the longest string
	# you can automatically achieve this using .sort(key=len, reverse=True)
	# motion = ['persistent', 'diffusive']
	# motion.sort(key=len, reverse=True)
   # Note: Maybe you don't need to use Parameter() if you don't have units
	param_ranges = {
		'exc':
			{
			# 'sigma_noise':ParameterArray([0.1]),
			# 'fields_per_synapse':ParameterArray([1, 16, 32]),
			# 'eta':ParameterArray(eta_exc * np.array([1, 2, 4])),
			# 'fields_per_synapse':ParameterArray([20]),
			'sigma':get_ParametersNamed(sigma_exc),
			# 'number_per_dimension':get_ParametersNamed(number_per_dimension_exc)
			# 'sigma_spreading':ParameterArray([1e-4, 1e-3, 1e-2, 1e-1]),
			# 'init_weight':ParameterArray(np.array([1.0, 2.0, 4.0])),
			# 'init_weight_spreading':ParameterArray(init_weight_exc/1.5),
			},
		'inh':
			{
			# 'weight_factor':ParameterArray(1 + 2*np.array([20]) /
			# float(n_inh)),
			# 'fields_per_synapse':ParameterArray([20]),
			'sigma':get_ParametersNamed(sigma_inh),
			# 'number_per_dimension':get_ParametersNamed(number_per_dimension_inh)
			# 'number_per_dimension':get_ParametersNamed(
			# 		np.array(
			# 		[number_per_dimension_exc/1,
			# 		 number_per_dimension_exc/2,
			# 		 number_per_dimension_exc/4])),
			# 'number_fraction':ParameterArray([1., 1./2, 1./3, 1./4]),
			# 'init_weight':ParameterArray(init_weight_inh),
			# 'fields_per_synapse':ParameterArray([1, 16, 32]),
			# 'sigma_noise':ParameterArray([0.1]),
			# 'eta':ParameterArray(eta_inh * np.array([1, 2, 4])),
			# 'sigma_spreading':ParameterArray([1e-4, 1e-3, 1e-2, 1e-1]),
			# 'sigma':ParameterArray(sigma_inh),
			# 'init_weight_spreading':ParameterArray(init_weight_inh/init_weight_spreading_norm),
			# 'gaussian_height':ParameterArray(np.sqrt([1, 2, 4, 8]))
			},
		'sim':
			{
			'input_space_resolution':get_ParametersNamed(input_space_resolution),
			# 'tuning_function':ParameterArray(['von_mises', 'periodic', 'gaussian']),
			# 'tuning_function':ParameterArray(['von_mises']),
			# 'input_normalization':ParameterArray(['rates_sum', 'none']),
			# 'input_normalization':ParameterArray(['rates_sum']),
			# 'symmetric_centers':ParameterArray([False, True]),
			# 'gaussian_process_rescale':ParameterArray([True, False]),
			'seed_centers':ParameterArray(np.arange(3)),
			# 'gaussian_process':ParameterArray([True, False]),
			# 'seed_init_weights':ParameterArray(np.arange(2)),
			# 'seed_sigmas':ParameterArray(np.arange(40)),
			# 'weight_lateral':ParameterArray(
			# 	[0.5, 1.0, 2.0, 4.0]),
			# 'output_neurons':ParameterArray([3, 4]),
			# 'seed_trajectory':ParameterArray(np.arange(3)),
			# 'initial_x':ParameterArray([-radius/1.42, -radius/5.3, radius/1.08]),
			# 'seed_init_weights':ParameterArray([1, 2]),
			# 'lateral_inhibition':ParameterArray([False]),
			# 'motion':ParameterArray(['persistent_semiperiodic', 'persistent_periodic', 'persistent']),
			# 'motion':ParameterArray(['persistent_semiperiodic']),
			# 'dt':ParameterArray([0.1, 0.01]),
			# 'tau':ParameterArray([0.1, 0.2, 0.4]),
			# 'boxtype':ParameterArray(['circular', 'linear']),
			# 'boundary_conditions':ParameterArray(['reflective', 'periodic'])
			},
		'out':
			{
			# 'normalization':ParameterArray(['quadratic_multiplicative',
			# 	'quadratic_multiplicative_lateral_inhibition']),
			}

	}
	if dimensions > 1:
		# compute = ['grid_score_1d', 'watson_u2']
		compute = ['grid_score_2d']
		compute = []
	else:
		compute = []
	params = {
		'visual': 'figure',
		'compute': ParameterArray(compute),
		'sim':
			{
			'input_normalization': 'none',
			'tuning_function': tuning_function,
			'save_n_input_rates': 3,
			'gaussian_process': gaussian_process,
			'gaussian_process_rescale': True,
			'take_fixed_point_weights': True,
			'discretize_space': True,
			# Take something smaller than the smallest
			# Gaussian (by a factor of 10 maybe)
			'input_space_resolution': ParameterArray(np.amin(sigma_exc, axis=1)/10.),
			'spacing': 51,
			'equilibration_steps': 10000,
			# 'gaussians_with_height_one': True,
			'stationary_rat': False,
			'same_centers': False,
			'first_center_at_zero': False,
			'lateral_inhibition': False,
			'output_neurons': 1,
			'weight_lateral': 0.0,
			'tau': 10.,
			'symmetric_centers': symmetric_centers,
			'dimensions': dimensions,
			'boxtype': 'linear',
			'radius': radius,
			'diff_const': 0.01,
			'every_nth_step': simulation_time/4,
			'every_nth_step_weights': simulation_time/4,
			'seed_trajectory': 1,
			'seed_init_weights': 1,
			'seed_centers': 1,
			'seed_sigmas': 1,
			'simulation_time': simulation_time,
			'dt': 1.0,
			'initial_x': 0.1,
			'initial_y': 0.2,
			'initial_z': 0.15,
			# 'velocity': 3e-4,
			'velocity': 1e-2,
			'persistence_length': radius,
			'motion': motion,
			# 'boundary_conditions': 'periodic',
			},
		'out':
			{
			'target_rate': target_rate,
			'normalization': 'quadratic_multiplicative',
			},
		'exc':
			{
			'center_overlap_factor': 3.,
			'number_per_dimension': ParameterArray(number_per_dimension_exc),
			'distortion': 'half_spacing',
			# 'distortion': 0.0,
			'eta': eta_exc,
			'sigma': sigma_exc[0,0],
			'sigma_spreading': ParameterArray([0.0, 0.0, 0.0][:dimensions]),
			# 'sigma_spreading': ParameterArray([0.03, 1e-5, 1e-5][:dimensions]),
			# 'sigma_distribution': ParameterArray(['uniform', 'uniform', 'uniform'][:dimensions]),
			'sigma_distribution': ParameterArray([sigma_distribution,
						sigma_distribution, sigma_distribution][:dimensions]),		
			# 'sigma_x': 0.05,
			# 'sigma_y': 0.05,
			'fields_per_synapse': 1,
			'init_weight':init_weight_exc,
			'init_weight_spreading': 5e-2,
			'init_weight_distribution': 'uniform',
			'gaussian_height': 1,
			},
		'inh':
			{
			'center_overlap_factor': 3.,
			'weight_factor': 1.0,
			'number_per_dimension': ParameterArray(number_per_dimension_inh),
			'distortion': 'half_spacing',
			# 'distortion': 0.0,
			'eta': eta_inh,
			'sigma': sigma_inh[0,0],
			# 'sigma_spreading': {'stdev': 0.01, 'left': 0.01, 'right': 0.199},
			'sigma_spreading': ParameterArray([0.0, 0.0, 0.0][:dimensions]),
			# 'sigma_spreading': ParameterArray([0.03, 0.4, 1e-5][:dimensions]),
			'sigma_distribution': ParameterArray(['uniform', 'uniform', 'uniform'][:dimensions]),
			'sigma_distribution': ParameterArray([sigma_distribution,
						sigma_distribution, sigma_distribution][:dimensions]),		
			# 'sigma_y': 0.1,
			'fields_per_synapse': 1,
			'init_weight': 1.0,
			'init_weight_spreading': 5e-2,
			'init_weight_distribution': 'uniform',
			'gaussian_height': 1,
			}
	}

	# Decide which parameters should be part of the directory name
	# For parameters that depend on each other it makes sense to only
	# take the primary one and unlist the others
	# CAUTION: if you remove too much, you might get file of identical name
	# which lead to overwriting. Only the last one will remain.
	unlisted = [('sim', 'input_space_resolution'),
				('inh', 'fields_per_synapse')
				]
	# Create list of all the parameter ranges
	listed = [l for l in flatten_params_to_point(param_ranges) if l not in unlisted]
	# Reverse to get seeds in the end
	listed = listed[::-1]
	custom_order = [('exc', 'sigma'), ('inh', 'sigma')]
	listed = general_utils.arrays.custom_order_for_some_elements(listed,
																custom_order)
	results_map = {p:i for i,p in enumerate([l for l in listed if l in
									flatten_params_to_point(param_ranges)])}
	results_map.update({p:-1 for p in [l for l in unlisted if l in
									flatten_params_to_point(param_ranges)]})

	# Note that runnet gets assigned to a function "run"
	exp = Experiment(path,runnet=run, postproc=postproc,
						results_coord_map=results_map)
	tables = exp.tables


	tables.add_parameter_ranges(param_ranges)
	tables.add_parameters(params)

	# Note: maybe change population to empty string
	linked_params_tuples = [
		('inh', 'sigma'),
		# ('inh', 'sigma_y'),
		# ('inh', 'init_weight'),
		('exc', 'sigma'),
		# ('exc', 'sigma_y'),
		('sim', 'input_space_resolution'),
		]
	tables.link_parameter_ranges(linked_params_tuples)

	# linked_params_tuples = [
	# 	('exc', 'number_per_dimension'),
	# 	('inh', 'number_per_dimension'),
	# ]
	# tables.link_parameter_ranges(linked_params_tuples)

	# linked_params_tuples = [
	# 	('exc', 'eta'),
	# 	('inh', 'eta')]
	# tables.link_parameter_ranges(linked_params_tuples)

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
	file_name = os.path.basename(params['subprocdir'])
	save_dir = os.path.join(os.path.dirname(params['subprocdir']), 'visuals')

	######################################
	##########	Create Figures	##########
	######################################
	if params['visual'] == 'figure':
		file_type = '.pdf'

		try:
			os.mkdir(save_dir)
		except OSError:
			pass
		plot_class = plotting.Plot(params=params, rawdata=rawdata['raw_data'])

		function_kwargs_list =\
			[
				### Figure 1 ###
				[
					# ('input_norm', {'ylim': [0, 2]}),

					('plot_output_rates_from_equation',
						{'time': 0., 'from_file': True}),
					('plot_correlogram',
						{'time': 0, 'from_file': True, 'mode': 'same'}),
					('plot_output_rates_from_equation',
						{'time': simulation_time/4., 'from_file': True}),
					('plot_correlogram',
						{'time': simulation_time/4., 'from_file': True, 'mode': 'same'}),
					('plot_output_rates_from_equation',
						{'time': simulation_time/2., 'from_file': True}),
					('plot_correlogram',
						{'time': simulation_time/2., 'from_file': True, 'mode': 'same'}),
					('plot_output_rates_from_equation',
						{'time': simulation_time, 'from_file': True}),
					('plot_correlogram',
						{'time': simulation_time, 'from_file': True, 'mode': 'same'}),
				],
				### End of Figure 1 ###
				### Figure 2 ###
				# [
				# 	('trajectory_with_firing', {'start_frame': 0, 'end_frame':0.5e4}),
				# 	('trajectory_with_firing', {'start_frame': 0, 'end_frame':1e4}),
				# 	('trajectory_with_firing', {'start_frame': 0, 'end_frame':2e4}),
				# 	('trajectory_with_firing', {'start_frame': 0, 'end_frame':3e4}),
				# ]
				### End of Figure 2 ###
			]
		# Plot the figures
		for n, function_kwargs in enumerate(function_kwargs_list):
			fig = plt.figure()
			plot_list = [functools.partial(getattr(plot_class, f), **kwargs)
						for f, kwargs in function_kwargs]
			plotting.plot_list(fig, plot_list)
			file_full = str(n) + file_name + file_type
			save_path = os.path.join(save_dir, file_full)
			plt.savefig(save_path, dpi=170, bbox_inches='tight', pad_inches=0.02)

	######################################
	##########	Create Videos	##########
	######################################
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

	######################################
	##########	Add to computed	##########
	######################################
	if params['compute'].size != 0:
		all_data = {}
		add_comp = add_computed.Add_computed(
						params=params, rawdata=rawdata['raw_data'])
		for f in params['compute']:
			all_data.update(getattr(add_comp, f)())
		
		print all_data
		rawdata.update({'computed': all_data})

	# rawdata.update({'computed': {'test1': np.random.random(10), 'test2': np.random.random(10)}})

	# # Clear figure and close windows
	# plt.clf()
	# plt.close()


	# rawdata.update(computed)
# for psp in tables.iter_paramspace_pts():
#     for t in ['exc', 'inh']:
#         all_weights = tables.get_raw_data(psp, t + '_weights')
#         sparse_weights = general_utils.arrays.sparsify_two_dim_array_along_axis_1(all_weights, 1000)
#         tables.add_computed(paramspace_pt=psp, all_data={t + '_weights_sparse': sparse_weights})
	# exp_dir = os.path.dirname(os.path.dirname(params['results_file']))
	return rawdata

if __name__ == '__main__':
	# cProfile.run('main()', 'profile_same_4th')
	# pstats.Stats('profile_off').sort_stats('cumulative').print_stats(20)
	tables = main()
