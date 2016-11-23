# import pdb; pdb.set_trace()
import os

from fabric.state import env
import numpy as np
import matplotlib as mpl
# import paramiko
# paramiko.util.log_to_file("filename.log")

from snep.experiment import Experiment

mpl.use('Agg')
import initialization
import matplotlib.pyplot as plt
import plotting
import add_computed
import utils
import functools
from snep.configuration import config

# from memory_profiler import profile
# import cProfile
# import pstats
# import tables

# Set to False if you always want to run locally
config['cluster'] = config.run_on_cluster()
env['user'] = 'weber'
timeout = None


def run_task_sleep(params, taskdir, tempdir):
	# os.mkdir(taskdir) # if you want to put something in the taskdir,
	# you must create it first
	rat = initialization.Rat(params)
	rawdata = rat.run()
	# rawdata is a dictionary of dictionaries (arbitrarily nested) with
	# keys (strings) and values (arrays or deeper dictionaries)
	# snep creates a group for each dictionary key and finally an array for
	# the deepest value. you can do this for raw_data or computed
	# whenever you wish
	results = {'raw_data': rawdata}
	######################################
	##########	Add to computed	##########
	######################################
	# compute = [('grid_score_2d', dict(type='hexagonal')),
	# 		   ('grid_score_2d', dict(type='quadratic')),
	# 		   ('grid_axes_angles', {})]
	# compute = [('mean_inter_peak_distance', {})]
	compute = None
	if compute:
		all_data = {}
		add_comp = add_computed.Add_computed(
			params=params, rawdata=results['raw_data'])
		for c in compute:
			all_data.update(getattr(add_comp, c[0])(**c[1]))
		results.update({'computed': all_data})

	file_name = os.path.basename(taskdir)
	save_dir = os.path.join(os.path.dirname(taskdir), 'visuals')

	if params['visual'] == 'figure':
		file_type = '.png'

		try:
			os.mkdir(save_dir)
		except OSError:
			pass
		plot_class = plotting.Plot(params=params, rawdata=results['raw_data'])

		# trajectory_with_firing_kwargs = {'start_frame': 0}
		every_nth_step = params['sim']['every_nth_step']
		sim_time = params['sim']['simulation_time']
		function_kwargs_list = (
			# [
			# 	[
			# 		('plot_output_rates_from_equation',
			# 		 dict(time=t, from_file=True, n_cumulative=None)),
			# 		('plot_correlogram',
			# 		 dict(time=t, from_file=True, mode='same',
			# 			method='Weber', n_cumulative=None))
			# 	]
			# 	for t in np.arange(0,
			# 		sim_time + every_nth_step, every_nth_step)
			# ]

			[
				# [(
				# 	'trajectory_with_firing',
				# 		dict(start_frame=0, end_frame=sim_time,
				#   firing_indicator='none_but_z_component', small_dt=None,
				#   symbol_size=8, show_title=True, colormap='viridis',
				# 			   max_rate_for_colormap=6.0)
				# )]
				### Figure 1 ###
				[
					(
					'plot_output_rates_from_equation',
						dict(time=t, from_file=True, subdimension=params['subdimension'])
					)
					# for t in sim_time * np.array([0, 1/4., 1/2., 1])
					for t in sim_time * np.linspace(0, 1, 4)
				],
				### Figure 2 ###
				# [
				# 	(
				# 	'input_current',
				# 		dict(time=t, spacing=201)
				# 	)
				# 	# for t in sim_time * np.array([0, 1/4., 1/2., 1])
				# 	for t in sim_time * np.linspace(0, 1, 4)
				# ],
				# [
				# 	(
				# 	'plot_correlogram',
				# 		dict(time=t, from_file=True, mode='same',
				# 			 subdimension=params['subdimension'],
				# 			 method=None)
				# 	)
				# 	# for t in sim_time * np.array([0, 1/4., 1/2., 1])
				# 	for t in sim_time * np.linspace(0, 1, 4)
				# ],
				### Figure 2 ###
				# [
				# 	(
				# 		'output_rate_heat_map',
				# 		{'from_file': True, 'end_time': sim_time,
				# 		'publishable': True}),
				# ],
				### Head direction ###
				# [
				# 	(
				# 	'plot_head_direction_polar',
				# 		dict(time=t, from_file=True)
				# 	)
				# 	# for t in sim_time * np.array([0, 1/4., 1/2., 1])
				# 	for t in sim_time * np.linspace(0, 1, 4)
				# ],
				# ### Figure 3 ###
				# [
				# 	(
				# 	'trajectory_with_firing',
				# 	dict(start_frame=0,  end_frame=simulation_time/i)
				# 	)
				# 	for i in [4, 3, 2, 1]
				# ]
				# ### End of Figure 3 ###
			]
		)

		# Plot the figures
		for n, function_kwargs in enumerate(function_kwargs_list):
			fig = plt.figure()
			plot_list = [functools.partial(getattr(plot_class, f), **kwargs)
						 for f, kwargs in function_kwargs]
			plotting.plot_list(fig, plot_list)
			file_full = str(n) + file_name + file_type
			save_path = os.path.join(save_dir, file_full)
			plt.savefig(save_path, dpi=170, bbox_inches='tight',
						pad_inches=0.02)

	###########################################################################
	####################### Clear stuff to save memory #######################
	###########################################################################
	if params['to_clear'] == 'weights_output_rate_grid_gp_extrema_centers':
		key_lists = [['exc', 'weights'], ['inh', 'weights'],
					 ['output_rate_grid'],
					 ['exc', 'gp_min'], ['inh', 'gp_min'],
					 ['exc', 'gp_max'], ['inh', 'gp_max'],
					 ['exc', 'centers'], ['inh', 'centers'],
					]
	elif params['to_clear'] == 'weights_gp_extrema_centers':
		key_lists = [['exc', 'weights'], ['inh', 'weights'],
					 ['exc', 'gp_min'], ['inh', 'gp_min'],
					 ['exc', 'gp_max'], ['inh', 'gp_max'],
					 ['exc', 'centers'], ['inh', 'centers'],
					]
	elif params['to_clear'] == 'weights_gp_extrema_centers':
		key_lists = [['exc', 'weights'], ['inh', 'weights'],
					 ['exc', 'gp_min'], ['inh', 'gp_min'],
					 ['exc', 'gp_max'], ['inh', 'gp_max'],
					]

	else:
		key_lists = [[]]
	# Arrays that are None are not written to disk
	utils.set_values_to_none(results['raw_data'], key_lists)

	return results


class JobInfoExperiment(Experiment):
	run_task = staticmethod(run_task_sleep)

	def _prepare_tasks(self):
		from snep.utils import ParameterArray, ParametersNamed
		short_test_run = False
		# Note: 18e4 corresponds to 60 minutes
		# time_factor = 10
		simulation_time = 4e6
		np.random.seed(1)
		n_simulations = 4
		dimensions = 1
		fields_per_synapse = np.array([1, 5, 10, 20, 40])
		number_per_dimension_exc = np.array([1600]) * 4
		number_per_dimension_inh = np.array([1600])

		if short_test_run:
			simulation_time = 18e2
			n_simulations = 1
			number_per_dimension_exc = np.array([7, 7])
			number_per_dimension_inh = np.array([3, 3])


		every_nth_step = simulation_time / 4
		every_nth_step_weights = simulation_time / 4
		random_sample_x = np.random.random_sample(n_simulations)
		random_sample_y = np.random.random_sample(n_simulations)


		if dimensions == 3:
			periodicity = 'semiperiodic'
		else:
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

		# motion = 'sargolini_data'
		boxtype.sort(key=len, reverse=True)
		sigma_distribution = 'uniform'

		target_rate = 1.0
		radius = 1.0
		eta_exc = 0.00001
		eta_inh = 0.0001

		# sinh = np.arange(0.08, 0.36, 0.02)
		# sexc = np.tile(0.03, len(sinh))
		# sigma_inh = np.atleast_2d(sinh).T.copy()
		# sigma_exc = np.atleast_2d(sexc).T.copy()

		sigma_exc = np.array([
			[0.03],
		])

		sigma_inh = np.array([
			[0.10],
		])

		input_space_resolution = sigma_exc / 8.

		def get_ParametersNamed(a):
			l = []
			for x in a:
				l.append((str(x).replace(' ', '_'), ParameterArray(x)))
			return ParametersNamed(l)

		gaussian_process = False
		if gaussian_process:
			init_weight_exc = 1.0
			symmetric_centers = False
			tuning_function = 'gaussian_process'
		else:
			init_weight_exc = 1.0
			symmetric_centers = True

		# learning_rate_factor = [1.0, 0.5, 0.1, 0.05]
		### Use this if you want all center seeds (default) ###
		seed_centers = np.arange(n_simulations)
		# seed_centers = np.array([3])
		### Specify selected center seeds
		# Interesting seed selection for 180 minutes
		# seed_centers = np.array([140, 124, 105, 141, 442])
		# seed_centers = np.array([442])
		# Interesting seed selection for 600 minutes 1/3 max learning rate
  		# seed_centers = np.array([0, 1, 4, 5, 6, 8, 9, 11, 19, 22, 190])
		# OLD 600 minutes slow learning selection
		# [20, 21, 33, 296, 316, 393, 394, 419, 420, 421]
		# Interesting seed selection for GRF learning rate 0.5
		# seed_centers = np.array([51, 52, 165, 258, 297, 343])
		# Interesting seed selection for 100 fps, learning rate 0.03
		# seed_centers = np.array([9, 28, 92, 317, 319, 324, 333, 334])
		# Interesting seed selection for 500 fps, learning rate 0.003
		# seed_centers = np.array([12, 47, 93, 104, 142, 203, 228, 267])
		# Interesting seed selection for GRF, sigma_inh 0.1
		# seed_centers = np.array([1, 2, 3, 27, 83, 144, 241, 287, 320, 358, 385, 413])
		# seed_centers = np.array([144, 241, 287])
		# seed_centers = np.array([3, 27, 83, 320, 385])

		# init_weight_exc_array = np.array([1.0, 2.0, 4.0, 8.0])
		# weight_factor = (
		# 	(1 +
		# 	 (80 / np.prod(number_per_dimension_exc))
		# 	 * np.array([0.8, 1.0, 1.2]))
		# )
		# weight_factor = np.array([1.033])
		# For string arrays you need the list to start with the longest string
		# you can automatically achieve this using .sort(key=len, reverse=True)
		# motion = ['persistent', 'diffusive']
		# motion.sort(key=len, reverse=True)
		# Note: Maybe you don't need to use Parameter() if you don't have units
		param_ranges = {
			'exc':
				{
					'sigma': get_ParametersNamed(sigma_exc),
					# 'eta': ParameterArray(eta_exc * np.array(learning_rate_factor))
					# 'init_weight': ParameterArray(init_weight_exc_array),
					'fields_per_synapse': ParameterArray(fields_per_synapse),
				},
			'inh':
				{
					# 'gp_stretch_factor': ParameterArray(sigma_exc/sigma_inh),
					'sigma': get_ParametersNamed(sigma_inh),
					# 'weight_factor': ParameterArray(weight_factor),
					# float(number_per_dimension_inh[0])),
					# 'eta': ParameterArray(eta_inh * np.array(learning_rate_factor))
					'fields_per_synapse': ParameterArray(fields_per_synapse),
				},
			'sim':
				{
					# 'head_direction_sigma': ParameterArray(np.array([np.pi])),
					'input_space_resolution': get_ParametersNamed(
						input_space_resolution),
					'seed_centers': ParameterArray(seed_centers),
					# 'seed_init_weights': ParameterArray(seed_centers),
					'initial_x': ParameterArray(
						(2 * radius * random_sample_x - radius)[seed_centers]),
					'initial_y': ParameterArray(
						(2 * radius * random_sample_y - radius)[seed_centers]),
					# 'initial_x':ParameterArray([-radius/1.3, radius/5.1]),
				},
			'out':
				{
					# 'normalization':ParameterArray(['quadratic_multiplicative',
					# 	'quadratic_multiplicative_lateral_inhibition']),
					# 'normalization': ParameterArray([
					# 								'inactive',
					# 								'linear_multiplicative',
					# 								'quadratic_multiplicative',
					# 								'linear_substractive'])
				}

		}

		self.tables.coord_map = {
			# ('sim', 'head_direction_sigma'): 3,
			('sim', 'initial_x'): -1,
			('sim', 'initial_y'): -1,
			('sim', 'input_space_resolution'): -1,
			('sim', 'seed_centers'): 0,
			('exc', 'sigma'): 1,
			('inh', 'sigma'): 2,
			('exc', 'fields_per_synapse'): 3,
			('inh', 'fields_per_synapse'): 4,
			# ('sim', 'seed_init_weights'): 3,
			# ('exc', 'init_weight'): 3,
			# ('inh', 'weight_factor'): 4,
			# ('out', 'normalization'): 3,
			# ('inh', 'eta'): 3,
			# ('inh', 'eta'): -1,
			# ('inh', 'weight_factor'): 3,
			# ('inh', 'gp_stretch_factor'): 4,
			# ('sim', 'initial_x'): 3,
		}

		params = {
			'visual': 'figure',
			'subdimension': 'none',
			# 'visual': 'none',
			# 'to_clear': 'weights_output_rate_grid_gp_extrema_centers',
			# 'to_clear': 'weights_gp_extrema_centers',
			# 'to_clear': 'weights_gp_extrema',
			'to_clear': 'none',
			'sim':
				{
					'head_direction_sigma': np.pi / 6.,
					'input_normalization': 'figure',
					'tuning_function': tuning_function,
					'save_n_input_rates': False,
					'gaussian_process': gaussian_process,
					'gaussian_process_rescale': 'fixed_mean',
					'take_fixed_point_weights': True,
					'discretize_space': True,
					# Take something smaller than the smallest
					# Gaussian (by a factor of 10 maybe)
					'input_space_resolution': ParameterArray(
						np.amin(sigma_exc, axis=1) / 10.),
					'spacing': 101,
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
					'store_twoSigma2': True,
					'dimensions': dimensions,
					'boxtype': 'linear',
					'radius': radius,
					'diff_const': 0.01,
					'every_nth_step': every_nth_step,
					'every_nth_step_weights': every_nth_step_weights,
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
					'fixed_convolution_dx': False,
					# 'boundary_conditions': 'periodic',
				},
			'out':
				{
					'target_rate': target_rate,
					'normalization': 'quadratic_multiplicative',
				},
			'exc':
				{
					'save_n_input_rates': number_per_dimension_exc[0],
					# 'save_n_input_rates': 3,
					# 'gp_stretch_factor': np.sqrt(2*np.pi*sigma_exc[0][0]**2)/(2*radius),
					'gp_stretch_factor': 1.0,
					# 'gp_extremum': ParameterArray(np.array([-dabei 1., 1]) * 0.15),
					'gp_extremum': 'none',
					'center_overlap_factor': 3.,
					'number_per_dimension': ParameterArray(
						number_per_dimension_exc),
					'distortion': 'half_spacing',
					# 'distortion':ParameterArray(radius/number_per_dimension_exc),
					# 'distortion': 0.0,
					'eta': eta_exc,
					'sigma': sigma_exc[0, 0],
					'sigma_spreading': ParameterArray(
						[0.0, 0.0, 0.0][:dimensions]),
					# 'sigma_spreading': ParameterArray([0.03, 1e-5, 1e-5][:dimensions]),
					# 'sigma_distribution': ParameterArray(['uniform', 'uniform', 'uniform'][:dimensions]),
					'sigma_distribution': ParameterArray([sigma_distribution,
														  sigma_distribution,
														  sigma_distribution][
														 :dimensions]),
					# 'sigma_x': 0.05,
					# 'sigma_y': 0.05,
					'fields_per_synapse': 1,
					'init_weight': init_weight_exc,
					'init_weight_spreading': 5e-2,
					'init_weight_distribution': 'uniform',
					'gaussian_height': 1,
					'untuned': False,
				},
			'inh':
				{
					'save_n_input_rates': number_per_dimension_inh[0],
					# 'save_n_input_rates': 3,
					# 'gp_stretch_factor': np.sqrt(2*np.pi*sigma_inh[0][0]**2)/(2*radius),
					'gp_stretch_factor': 1.0,
					# 'gp_extremum': ParameterArray(np.array([-1., 1]) * 0.12),
					'gp_extremum': 'none',
					'center_overlap_factor': 3.,
					'weight_factor': 1,
					'number_per_dimension': ParameterArray(
						number_per_dimension_inh),
					'distortion': 'half_spacing',
					# 'distortion':ParameterArray(radius/number_per_dimension_inh),
					# 'distortion': 0.0,
					'eta': eta_inh,
					'sigma': sigma_inh[0, 0],
					# 'sigma_spreading': {'stdev': 0.01, 'left': 0.01, 'right': 0.199},
					'sigma_spreading': ParameterArray(
						[0.0, 0.0, 0.0][:dimensions]),
					# 'sigma_spreading': ParameterArray([0.03, 0.4, 1e-5][:dimensions]),
					# 'sigma_distribution': ParameterArray(['uniform', 'uniform', 'uniform'][:dimensions]),
					'sigma_distribution': ParameterArray([sigma_distribution,
														  sigma_distribution,
														  sigma_distribution][
														 :dimensions]),
					# 'sigma_y': 0.1,
					'fields_per_synapse': 1,
					'init_weight': 1.0,
					'init_weight_spreading': 5e-2,
					'init_weight_distribution': 'uniform',
					'gaussian_height': 1,
					'untuned': False,
				}
		}

		# # Decide which parameters should be part of the directory name
		# # For parameters that depend on each other it makes sense to only
		# # take the primary one and unlist the others
		# # CAUTION: if you remove too much, you might get file of identical name
		# # which lead to overwriting. Only the last one will remain.
		# unlisted = [('sim', 'input_space_resolution'),
		# 			('inh', 'fields_per_synapse'),
		# 			('sim', 'initial_x'),
		# 			('sim', 'initial_y'),
		# 			]
		# # Create list of all the parameter ranges
		# listed = [l for l in flatten_params_to_point(param_ranges) if l not in unlisted]
		# # Reverse to get seeds in the end
		# listed = listed[::-1]
		# custom_order = [('exc', 'sigma'), ('inh', 'sigma')]
		# listed = general_utils.arrays.custom_order_for_some_elements(listed,
		# 															custom_order)
		# results_map = {p:i for i,p in enumerate([l for l in listed if l in
		# 								flatten_params_to_point(param_ranges)])}
		# results_map.update({p:-1 for p in [l for l in unlisted if l in
		# 								flatten_params_to_point(param_ranges)]})

		# Note that runnet gets assigned to a function "run"
		# exp = Experiment(path,runnet=run, postproc=postproc,
		# 					results_coord_map=results_map)
		# tables = exp.tables

		self.tables.add_parameter_ranges(param_ranges)
		self.tables.add_parameters(params)

		# Note: maybe change population to empty string
		linked_params_tuples = [
			('inh', 'sigma'),
			# ('inh', 'gp_stretch_factor'),
			# ('inh', 'sigma_y'),
			# ('inh', 'init_weight'),
			('exc', 'sigma'),
			# ('exc', 'sigma_y'),
			('sim', 'input_space_resolution'),
		]
		self.tables.link_parameter_ranges(linked_params_tuples)

		linked_params_tuples = [
			('sim', 'seed_centers'),
			# ('sim', 'seed_init_weights'),
			('sim', 'initial_x'),
			('sim', 'initial_y'),
		]
		self.tables.link_parameter_ranges(linked_params_tuples)

		linked_params_tuples = [
			('exc', 'fields_per_synapse'),
			('inh', 'fields_per_synapse'),
		]
		self.tables.link_parameter_ranges(linked_params_tuples)



if __name__ == '__main__':
	from snep.parallel2 import run

	'''
	IMPORTANT: Only include code here that can be run repeatedly,
	because this will be run once in the parent process, and then
	once for every worker process.vi
	'''
	ji_kwargs = dict(root_dir=os.path.expanduser(
		'~/experiments/'))
	job_info = run(JobInfoExperiment, ji_kwargs, job_time=timeout, mem_per_task=20,
				   delete_tmp=True)
