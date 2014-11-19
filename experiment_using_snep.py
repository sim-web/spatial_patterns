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

simulation_time = 2e5
def main():
	from snep.utils import Parameter, ParameterArray, ParametersNamed, flatten_params_to_point
	from snep.experiment import Experiment


	dimensions = 1
	von_mises = False

	if von_mises:
		# number_per_dimension = np.array([70, 20, 7])[:dimensions]
		number_per_dimension = np.array([60, 60, 20])[:dimensions]
		boxtype = ['linear']
		motion = 'persistent_semiperiodic'
	else:
		number_per_dimension = np.array([2000, 20, 4])[:dimensions]
		# boxtype = ['linear', 'circular']
		boxtype = ['linear']
		motion = 'persistent'
	boxtype.sort(key=len, reverse=True)

	# sigma_distribution = 'gamma_with_cut_off'
	sigma_distribution = 'uniform'
	# number_per_dimension_exc=number_per_dimension_inh=number_per_dimension
	number_per_dimension_exc = number_per_dimension
	number_per_dimension_inh = number_per_dimension
	# n = np.prod(number_per_dimension)
	n_exc, n_inh = np.prod(number_per_dimension_exc), np.prod(number_per_dimension_inh)

	target_rate = 1.0
	# n_exc = 1000
	# n_inh = 1000
	# radius = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
	radius = 1.0
	eta_exc = 2e-5 / (2*radius)
	eta_inh = 2e-4 / (2*radius)
	# simulation_time = 8*radius*radius*10**5
	# We want 100 fields on length 1
	# length = 2*radius + 2*overlap
	# n = 100 * (2*radius + 2*overlap)

	sigma_exc = np.array([
						# [0.15, 0.1],
						# [0.4, 0.4],
						# [0.1, 0.1, 0.2],
						# [0.05, 0.2],
						# [0.05, 0.2],
						# [0.09, 0.15],
						# [0.05, 0.7],
						# [0.05, 0.07],
						# [0.07, 0.07],
						# [0.06],
						# [0.03],
						# [0.2, 0.4],
						# [0.05, 0.2],
						# [0.05, 0.2],
						# [0.11, 0.4],
						# [0.11, 0.4],
						# [0.12, 0.5],
						# [0.15, 0.2],
						# [0.10, 0.15],
						# [0.105, 0.15],
						# [0.05, 0.05],
						[0.03],
						# [0.04],
						# [0.05],
						# [0.065, 0.065, 0.2],
						# [0.070, 0.070, 0.2],
						# [0.15, 0.15, 0.2],
						])

	sigma_inh = np.array([
						# [0.12, 0.2],
						# [0.12, 1.5],
						# [0.12, 0.6],
						# [0.12, 0.6],
						# [0.2, 0.04],
						# [1.5, 1.5],
						# [0.20],
						# [0.38],
						# [0.14, 0.7],
						# [0.10, 0.10],
						# [0.12, 1.5],
						# [1.5, 0.3],
						# [0.11, 0.7],
						# [0.12, 0.6],
						# [0.12, 0.7],
						# [0.12, 0.7],
						# [0.12, 1.5],
						[0.1],
						# [0.12],
						# [0.15],
						# [0.12, 0.12, 1.5],
						# [0.12, 0.12, 1.5],
						])

	# sinh = np.arange(0.2, 0.4, 0.02)
	# sexc = np.tile(0.03, len(sinh))
	# sigma_inh = np.atleast_2d(sinh).T.copy()
	# sigma_exc = np.atleast_2d(sexc).T.copy()


	# print sigma_inh.shape
	# sigma_inh = np.arange(0.08, 0.4, 0.02)

	center_overlap_exc = 3 * sigma_exc
	center_overlap_inh = 3 * sigma_inh
	if von_mises:
		# No center overlap for periodic dimension!
		center_overlap_exc[:, -1] = 0.
		center_overlap_inh[:, -1] = 0.

	input_space_resolution = sigma_exc/8.

	def get_ParametersNamed(a):
		l = []
		for x in a:
			l.append((str(x).replace(' ', '_'), ParameterArray(x)))
		return ParametersNamed(l)


	init_weight_exc = 1.0
	# For string arrays you need the list to start with the longest string
	# you can automatically achieve this using .sort(key=len, reverse=True)
	# motion = ['persistent', 'diffusive']
	# motion.sort(key=len, reverse=True)
   # Note: Maybe you don't need to use Parameter() if you don't have units
	param_ranges = {
		'exc':
			{
			# 'sigma_noise':ParameterArray([0.1]),
			# 'number_desired':ParameterArray(n),
			# 'fields_per_synapse':ParameterArray([1]),
			# 'fields_per_synapse':ParameterArray([1, 2, 4, 8, 16, 32]),
			# 'center_overlap':ParameterArray(center_overlap),
			# 'sigma_x':ParameterArray([0.05, 0.1, 0.2]),
			# 'sigma_y':ParameterArray([0.05]),
			# 'eta':ParameterArray([4e-6, 4e-7]),
			# 'sigma_x':ParameterArray(sigma_exc_x),
			# 'sigma_y':ParameterArray(sigma_exc_y),
			'sigma':get_ParametersNamed(sigma_exc),
			'center_overlap':get_ParametersNamed(center_overlap_exc),
			# 'center_overlap_x':ParameterArray(center_overlap_exc_x),
			# 'center_overlap_y':ParameterArray(center_overlap_exc_y),
			# 'sigma_spreading':ParameterArray([1e-4, 1e-3, 1e-2, 1e-1]),
			# 'init_weight':ParameterArray(init_weight_exc),
			# 'init_weight_spreading':ParameterArray(init_weight_exc/1.5),
			},
		'inh':
			{
			# 'sigma_x':ParameterArray(sigma_inh_x),
			# 'sigma_y':ParameterArray(sigma_inh_y),
			'sigma':get_ParametersNamed(sigma_inh),
			# 'eta':ParameterArray([4e-5, 4e-6]),
			# 'init_weight':ParameterArray(init_weight_inh),
			# 'center_overlap_x':ParameterArray(center_overlap_inh_x),
			# 'center_overlap_y':ParameterArray(center_overlap_inh_y),
			# 'center_overlap':ParametersNamed(
			# 								[
			# 								(str(center_overlap_inh), ParameterArray(center_overlap_inh)),
			# 								# ('y', ParameterArray(np.array([1, 2])))
			# 								]
			# 								),
			'center_overlap':get_ParametersNamed(center_overlap_inh),
			# 'number_desired':ParameterArray(n),
			# 'fields_per_synapse':ParameterArray([1]),
			# 'fields_per_synapse':ParameterArray([1, 2, 4, 8, 16, 32]),
			# 'center_overlap':ParameterArray(center_overlap),
			# 'sigma_noise':ParameterArray([0.1]),
			# 'eta':ParameterArray([1e-5, 1e-4]),
			# 'sigma_spreading':ParameterArray([1e-4, 1e-3, 1e-2, 1e-1]),
			# 'sigma':ParameterArray(sigma_inh),
			# 'init_weight_spreading':ParameterArray(init_weight_inh/init_weight_spreading_norm),
			},
		'sim':
			{
			'input_space_resolution':get_ParametersNamed(input_space_resolution),
			# 'symmetric_centers':ParameterArray([False, True]),
			# 'seed_centers':ParameterArray(np.arange(3)),
			'seed_centers':ParameterArray([4])
			# 'seed_sigmas':ParameterArray(np.arange(40)),
			# 'radius':ParameterArray(radius),
			# 'weight_lateral':ParameterArray(
			# 	[0.5, 1.0, 2.0, 4.0]),
			# 'output_neurons':ParameterArray([3, 4]),
			# 'seed_trajectory':ParameterArray([1, 2]),
			# 'initial_x':ParameterArray([-radius/1.42, -radius/5.3, radius/1.08]),
			# 'seed_init_weights':ParameterArray([1, 2]),
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
	if dimensions > 1:
		# compute = ['grid_score_1d', 'watson_u2']
		compute = []
	else:
		compute = []
	params = {
		'visual': 'figure',
		'compute': ParameterArray(compute),
		'sim':
			{
			'gaussian_process': True,
			'take_fixed_point_weights': False,
			'discretize_space': True,
			'von_mises': von_mises,
			# Take something smaller than the smallest
			# Gaussian (by a factor of 10 maybe)
			'input_space_resolution': ParameterArray(np.amin(sigma_exc, axis=1)/10.),
			'spacing': 201,
			'equilibration_steps': 10000,
			# 'gaussians_with_height_one': True,
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
			'every_nth_step': simulation_time/100,
			'every_nth_step_weights': simulation_time/100,
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
			'number_per_dimension': ParameterArray(number_per_dimension_exc),
			'number_desired': n_exc,
			# 'distortion': np.sqrt(radius**2 * np.pi/ n_inh),
			'distortion':ParameterArray(radius/number_per_dimension_exc),
			# 'distortion': 0.0,
			# 'center_overlap_x':ParameterArray(center_overlap_exc_x),
			# 'center_overlap_y':ParameterArray(center_overlap_exc_y),
			'center_overlap':ParameterArray(center_overlap_exc),
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
			},
		'inh':
			{
			'number_per_dimension': ParameterArray(number_per_dimension_inh),
			'number_desired': n_inh,
			# 'distortion': np.sqrt(radius**2 * np.pi/ n_inh),
			'distortion':ParameterArray(radius/number_per_dimension_inh),
			# 'distortion': 0.0,
			# 'center_overlap_x':ParameterArray(center_overlap_inh_x),
			# 'center_overlap_y':ParameterArray(center_overlap_inh_y),
			'center_overlap':ParameterArray(center_overlap_inh),
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
			}
	}

	# Decide which parameters should be part of the directory name
	# For parameters that depend on each other it makes sense to only
	# take the primary one and unlist the others
	# CAUTION: if you remove too much, you might get file of identical name
	# which lead to overwriting. Only the last one will remain.
	unlisted = [('exc','center_overlap'), ('inh','center_overlap'),
				('sim', 'input_space_resolution'),
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
		('exc', 'center_overlap'),
		('inh', 'center_overlap'),
		('exc', 'sigma'),
		# ('exc', 'sigma_y'),
		('sim', 'input_space_resolution'),
		]
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
	file_name = os.path.basename(params['subprocdir'])
	save_dir = os.path.join(os.path.dirname(params['subprocdir']), 'visuals')

	######################################
	##########	Create Figures	##########
	######################################
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
				('plot_output_rates_from_equation',
					{'time': 0., 'from_file': True}),
				('plot_output_rates_from_equation',
					{'time': simulation_time/4., 'from_file': True}),
				('plot_output_rates_from_equation',
					{'time': simulation_time/2., 'from_file': True}),
				('plot_output_rates_from_equation',
					{'time': simulation_time, 'from_file': True}),
				# ('plot_output_rates_from_equation',
				# 	{'time': 0, 'spacing': 601, 'from_file': False}),
				# ('output_rate_heat_map',
				#	{'from_file': True, 'end_time': simulation_time})
			]
		plot_list = [functools.partial(getattr(plot_class, f), **kwargs)
						for f, kwargs in function_kwargs]
		plotting.plot_list(fig, plot_list)
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
