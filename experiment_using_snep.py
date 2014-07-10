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
# from memory_profiler import profile

# import cProfile    
# import pstats
# import tables

# sys.path.insert(1, os.path.expanduser('~/.local/lib/python2.7/site-packages/'))
path = os.path.expanduser('~/localfiles/itb_experiments/learning_grids/')

from snep.configuration import config
# config['multiproc'] = False
config['network_type'] = 'empty'

def get_fixed_point_initial_weights(dimensions, radius, center_overlap_exc,
		center_overlap_inh,
		target_rate, init_weight_exc, n_exc, n_inh, 
		sigma_exc=None, sigma_inh=None, von_mises=False):
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
	von_mises : bool
		If True it is assumed that the bell curves are periodic in y direction

	Returns
	-------
	output : float or ndarray
		Values for the initial inhibitory weights
	"""

	limit_exc = radius + center_overlap_exc
	limit_inh = radius + center_overlap_inh
	if dimensions == 1:
		init_weight_inh = ( (n_exc * init_weight_exc  * sigma_exc[:,0] / limit_exc[:,0]
						- target_rate*np.sqrt(2/np.pi))
						/ (n_inh * sigma_inh[:,0] / limit_inh[:,0]) )

	elif dimensions == 2:
		if not von_mises:
			init_weight_inh = (
						(n_exc * init_weight_exc * sigma_exc[:,0] * sigma_exc[:,1]
							/ (limit_exc[:,0]*limit_exc[:,1]) 
							- 2 * target_rate / np.pi)
							/ (n_inh * sigma_inh[:,0] * sigma_inh[:,1]
							/ (limit_inh[:,0]*limit_inh[:,1]))
							)
		else:
			scaled_kappa_exc = (limit_exc[:,1] / (np.pi*sigma_exc[:,1]))**2
			scaled_kappa_inh = (limit_inh[:,1] / (np.pi*sigma_inh[:,1]))**2
			init_weight_inh = (
					(n_exc * init_weight_exc * sigma_exc[:,0] * sps.iv(0, scaled_kappa_exc)
						/ (limit_exc[:,0] * np.exp(scaled_kappa_exc))
						- np.sqrt(2/np.pi) * target_rate)
						/ (n_inh * sigma_inh[:,0] * sps.iv(0, scaled_kappa_inh)
							/ (limit_inh[:,0] * np.exp(scaled_kappa_inh)))
							)
	return init_weight_inh


simulation_time = 1e7
def main():
	from snep.utils import Parameter, ParameterArray, ParametersNamed, flatten_params_to_point
	from snep.experiment import Experiment


	dimensions = 2
	von_mises = True
	if von_mises:
		motion = 'persistent_semiperiodic'
	else:
		motion = 'persistent'
	target_rate = 1.0
	# n_exc = 1000
	# n_inh = 1000
	# radius = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
	radius = 0.5
	eta_inh = 1e-4 / (2*radius)
	eta_exc = 1e-5 / (2*radius)
	# simulation_time = 8*radius*radius*10**5
	# We want 100 fields on length 1
	# length = 2*radius + 2*overlap
	# n = 100 * (2*radius + 2*overlap)

	sigma_exc = np.array([
						# [0.025, 0.025],
						[0.15, 0.1],
						# [0.035, 0.035],
						# [0.040, 0.040],
						# [0.045, 0.045],
						# [0.050, 0.050],
						# [0.05, 0.07],
						# [0.06, 0.05],
						# [0.07, 0.05],
						])

	sigma_inh = np.array([
						# [0.10, 0.10],
						# [0.10, 0.10],
						# [0.10, 0.10],
						# [0.10, 0.10],
						# [0.10, 0.10],
						[0.15, 1.5],
						# [0.15, 1.5],
						# [0.15, 1.5],
						# [0.15, 1.5],
						# [0.15, 1.5],
						])

	# We don't want weight overlap in y direction if this direction is
	# periodic
	if von_mises:
		center_overlap_exc = np.array([3., 0.]) * sigma_exc
		center_overlap_inh = np.array([3., 0.]) * sigma_inh
	else:
		center_overlap_exc = 3 * sigma_exc
		center_overlap_inh = 3 * sigma_inh

	def get_ParametersNamed(a):
		l = []
		for x in a:
			l.append((str(x).replace(' ', '_'), ParameterArray(x)))
		return ParametersNamed(l)

	# n = 5000
	n_x = 70
	n_y = 20
	n = n_x * n_y
	n_exc, n_inh = n, n
	n_exc_x, n_exc_y, n_inh_x, n_inh_y = n_x, n_y, n_x, n_y

	init_weight_exc = 1.0
	# init_weight_inh = get_fixed_point_initial_weights(
	# 	dimensions, radius, center_overlap, target_rate, init_weight_exc,
	# 	sigma_exc, sigma_inh, n_exc, n_inh)
	init_weight_inh = get_fixed_point_initial_weights(
		dimensions=dimensions, radius=radius, 
		center_overlap_exc=center_overlap_exc,
		center_overlap_inh=center_overlap_inh,
		sigma_exc=sigma_exc, sigma_inh=sigma_inh,
		target_rate=target_rate, init_weight_exc=init_weight_exc,
		n_exc=n_exc, n_inh=n_inh, von_mises=von_mises)

	# init_weight_inh = np.zeros_like(init_weight_inh)
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
			# 'center_overlap':ParameterArray(center_overlap),
			# 'sigma_x':ParameterArray([0.05, 0.1, 0.2]),
			# 'sigma_y':ParameterArray([0.05]),
			# 'eta':ParameterArray([1e-6, 1e-5]),
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
			# 'eta':ParameterArray([1e-2, 1e-3]),
			'init_weight':ParameterArray(init_weight_inh),
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
			# 'fields_per_synapse':ParameterArray([1, 4, 8]),
			# 'center_overlap':ParameterArray(center_overlap),
			# 'sigma_noise':ParameterArray([0.1]),
			# 'eta':ParameterArray([1e-5, 1e-4]),
			# 'sigma_spreading':ParameterArray([1e-4, 1e-3, 1e-2, 1e-1]),
			# 'sigma':ParameterArray(sigma_inh),
			# 'init_weight_spreading':ParameterArray(init_weight_inh/init_weight_spreading_norm),
			},
		'sim': 
			{
			'input_space_resolution':ParameterArray(np.amin(sigma_exc, axis=1) / 10.),
			# 'symmetric_centers':ParameterArray([False, True]),
			'seed_centers':ParameterArray(np.arange(3)),
			# 'radius':ParameterArray(radius),
			# 'gaussians_with_height_one':ParameterArray([False, True]),
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
	
	params = {
		'visual': 'figure',
		'sim':
			{
			# If -1, the input rates will be determined for the current position
			# in each time step, # Take something smaller than the smallest
			# Gaussian (by a factor of 10 maybe)
			'input_space_resolution': np.amin(sigma_exc, axis=1)[0]/10.,
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
			'seed_trajectory': 1,
			'seed_init_weights': 1,
			'seed_centers': 1,
			'simulation_time': simulation_time,
			'dt': 1.0,
			'initial_x': 0.1,
			'initial_y': 0.2,
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
			'n_x': n_exc_x,
			'n_y': n_exc_y,
			'number_desired': n_exc,
			# 'distortion': np.sqrt(radius**2 * np.pi/ n_inh),
			'distortion':ParameterArray(radius/np.array([n_exc_x, n_exc_y])),
			# 'distortion': 0.0,
			# 'center_overlap_x':ParameterArray(center_overlap_exc_x),
			# 'center_overlap_y':ParameterArray(center_overlap_exc_y),
			'center_overlap':ParameterArray(center_overlap_exc),
			'eta': eta_exc,
			'sigma': sigma_exc[0,0],
			'sigma_spreading': 0.0,
			'sigma_distribution': 'uniform',
			# 'sigma_x': 0.05,
			# 'sigma_y': 0.05,
			'fields_per_synapse': 1,
			'init_weight':init_weight_exc,
			'init_weight_spreading': 0.05,
			'init_weight_distribution': 'uniform',
			},
		'inh':
			{
			'n_x': n_inh_x,
			'n_y': n_inh_y,
			'number_desired': n_inh,
			# 'distortion': np.sqrt(radius**2 * np.pi/ n_inh),
			'distortion':ParameterArray(radius/np.array([n_inh_x, n_inh_y])),
			# 'distortion': 0.0,
			# 'center_overlap_x':ParameterArray(center_overlap_inh_x),
			# 'center_overlap_y':ParameterArray(center_overlap_inh_y),
			'center_overlap':ParameterArray(center_overlap_inh),
			'eta': eta_inh,
			'sigma': sigma_inh[0,0],
			# 'sigma_spreading': {'stdev': 0.01, 'left': 0.01, 'right': 0.199},
			'sigma_spreading': 0.0,
			'sigma_distribution': 'uniform',
			# 'sigma_x': 0.1,
			# 'sigma_y': 0.1,
			'fields_per_synapse': 1,
			'init_weight': 0.56,
			'init_weight_spreading': 0.05,
			'init_weight_distribution': 'uniform',
			}
	}

	listed = [('exc','sigma'), ('inh','sigma'), ('sim','boxtype'),
				('sim', 'seed_centers')]
	unlisted = [('exc','center_overlap'), ('inh','center_overlap'),
				('inh','init_weight'), ('sim', 'input_space_resolution')]

	results_map = {p:i for i,p in enumerate([l for l in listed if l in flatten_params_to_point(param_ranges)])}
	print results_map
	results_map.update({p:-1 for p in [l for l in unlisted if l in flatten_params_to_point(param_ranges)]})
	print results_map

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
		('inh', 'init_weight'),
		('exc', 'center_overlap'),
		# ('exc', 'center_overlap_y'),
		('inh', 'center_overlap'),
		# ('inh', 'center_overlap_y'),
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
