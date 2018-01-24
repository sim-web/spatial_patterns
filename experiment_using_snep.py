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


def get_learning_rates(defined=None, exc=None, inh=None):
    d = {
        '10hrs_1fps': (40e-4 / 60., 160e-4 / 60.),
        'room_switch_1fps': (40e-4 / 30., 160e-4 / 30.),
        '10hrs_20fps': (0.15 * 40e-4 / 60., 0.15 * 160e-4 / 60.),
        '10hrs_100fps': (0.03 * 40e-4 / 60., 0.03 * 160e-4 / 60.),
    }
    if defined:
        r_exc, r_inh = d[defined]
    else:
        r_exc, r_inh = exc, inh
    return r_exc, r_inh

def run_task_sleep(params, taskdir, tempdir):
	"""
	Run the task

	Parameters
	----------
	params : dict
		Contains all the simulations parameters
	taskdir : str
	tempdir : str

	Returns
	-------
	results : dictionary
		This dictionary will be used to create an .h5 file.
		Most importantly all the raw data is stored in the dictionary
		element with key 'raw_data'.
		Data that is obtained from post processing of the raw data stored
		under the key 'computed'.
	"""
	# t_compare = 7 * 18e4
	# t_half = 18e5 / 2
	# t_reference = t_half
	### For test run
	# t_reference = 0
	# t_compare = 0
	# t_half = 36e2
	# The dictionary is now created automatically
	# os.mkdir(taskdir)
	###########################################################################
	############################## Run the code ###############################
	###########################################################################
	# The code should return all the rawdata as a nested dictionary whose
	# final leaves are arrays
	# See initialization.py for the run function
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
	# 		   # ('correlation_with_reference_grid', dict(
	# 			#    t_reference=t_reference)),
	# # 		   ('grid_score_2d', dict(type='quadratic')),
	# # 		   ('grid_axes_angles', {})
	# 		   ]
	compute = None
	if compute:
		all_data = {}
		add_comp = add_computed.Add_computed(
			params=params, rawdata=results['raw_data'])
		for c in compute:
			all_data.update(getattr(add_comp, c[0])(**c[1]))
		results.update({'computed': all_data})
	else:
		results['computed'] = None
	###########################################################################
	############################# Create visuals #############################
	###########################################################################
	# Store them in the same directory where the .h5 file is stored
	file_name = os.path.basename(taskdir)
	save_dir = os.path.join(os.path.dirname(taskdir), 'visuals')
	if params['visual'] == 'figure':
		file_type = '.png'
		try:
			os.mkdir(save_dir)
		except OSError:
			pass
		# See also plotting.py
		plot_class = plotting.Plot(params=params, rawdata=results['raw_data'],
								   computed=results['computed'])
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
						dict(time=t, from_file=True, subdimension=params[
							'subdimension'])
					)
					# for t in sim_time * np.array([0, 1/4., 1/2., 1])
					# for t in sim_time * np.linspace(0, 1, 4)
					# for t in np.floor(sim_time / 8. * np.linspace(0, 8, 9))
					for t in sim_time * np.linspace(0, 1, 4)
				],
				### Figure 2 ###
				# [
				# 	('spikemap_from_ratemap',
				# 	 dict(n=2000, time_l=18e5/4, time_r=18e5/2)),
				# 	('spikemap_from_ratemap',
				# 	 dict(n=2000, time=18e5)),
				# ],
				# [
				# 	('input_tuning', dict(populations=['exc'], neuron=20)),
				# 	('input_tuning', dict(populations=['exc'], neuron=35)),
				# 	('input_tuning', dict(populations=['exc'], neuron=63)),
				# 	('input_tuning', dict(populations=['exc'], neuron=11)),
				# 	('input_tuning', dict(populations=['exc'], neuron=53)),
				# 	('input_tuning', dict(populations=['inh'], neuron=0)),
				# ]
				# [
				# 	# ('weight_evolution',
				# 	#  dict(syn_type='exc', weight_sparsification=1)),
				# 	(
				# 		'trajectory_with_firing',
				# 		dict(start_frame=0, end_frame=sim_time / 4)
				# 	),
				# 	(
				# 		'trajectory_with_firing',
				# 		dict(start_frame=0, end_frame=sim_time
				# 													/ 2)
				# 	),
				# 	(
				# 		'trajectory_with_firing',
				# 		dict(start_frame=sim_time / 2 + 1, end_frame=sim_time)
				# 	)
				# ],
				# [
				# 	('plot_output_rates_from_equation',
				# 	 dict(time=t_reference, from_file=True, spacing=51)),
				# 	('plot_output_rates_from_equation',
				# 	 dict(time=t_compare, from_file=True, spacing=51)),
				# 	('plot_time_evolution',
				# 	 dict(observable='grid_score', data=True,
				# 		  vlines=[t_half, t_compare])),
				# 	('time_evolution_of_grid_correlation',
				# 	 dict(t_reference=t_reference,
				# 		  vlines=[t_compare])),
				# ],
				[
					(
					'plot_correlogram',
						dict(time=t, from_file=True, mode='same',
							 subdimension=params['subdimension'],
							 method='langston')
					)
					# for t in sim_time * np.array([0, 1/4., 1/2., 1])
					for t in sim_time * np.linspace(0, 1, 4)
				],
				### Figure 2 ###
				# [
				# 	(
				# 		'output_rate_heat_map',
				# 		{'from_file': True, 'end_time': sim_time,
				# 		'publishable': False}),
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
	# Currently very basic implementation
	# Use a string as a simulation parameter
	# This string says which parts of the rawdata should not be stored
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
	elif params['to_clear'] == 'weights_gp_extrema':
		key_lists = [['exc', 'weights'], ['inh', 'weights'],
					 ['exc', 'gp_min'], ['inh', 'gp_min'],
					 ['exc', 'gp_max'], ['inh', 'gp_max'],
					]
	# It nothing is specified everything is stored
	else:
		key_lists = [[]]
	# Arrays that are None are not written to disk
	utils.set_values_to_none(results['raw_data'], key_lists)
	return results


class JobInfoExperiment(Experiment):
	# Use the run_task_sleep function that you specified above
	run_task = staticmethod(run_task_sleep)

	def _prepare_tasks(self):
		"""
		Define all the parameters and parameter ranges.

		Here we define all the simulation parameters. Either as single
		values or as arrays.
		An array defines a set of parameters. If a set is specified, the
		single default values are ignored.
		By default the simulator runs the entire Cartesian product of
		all possible parameter combinations.
		If parameters should be varied conjointly, i.e., if the should be
		linked, this can be done using link_parameter_ranges. See below
		for examples.

		Lines that I use repeatadly are sometimes just comments.
		"""
		from snep.utils import ParameterArray, ParametersNamed
		short_test_run = False
		# Note: 18e4 corresponds to 60 minutes
		simulation_time = 10 * 18e4
		np.random.seed(1)
		n_simulations = 2
		dimensions = 2
		number_per_dimension_exc = np.array([70, 70])
		number_per_dimension_inh = np.array([35, 35])

		fields_per_synapse = np.array([50])
		explore_all_time = False
		boxside_switch_time = False
		# normalization = ['quadratic_multiplicative']
		simulation_time_divisor = 4

		if short_test_run:
			simulation_time = 18e2
			n_simulations = 1
			number_per_dimension_exc = np.array([7, 7])
			number_per_dimension_inh = np.array([5, 5])
			# boxside_switch_time = simulation_time / 4
			# explore_all_time = simulation_time / 2
			explore_all_time = False
			boxside_switch_time = False
			simulation_time_divisor = 4

		n_exc_total = np.prod(number_per_dimension_exc)
		n_inh_total = np.prod(number_per_dimension_inh)
		every_nth_step = simulation_time / simulation_time_divisor
		every_nth_step_weights = simulation_time / simulation_time_divisor
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
			# tuning_function = 'grid'
		elif periodicity == 'semiperiodic':
			boxtype = ['linear']
			motion = 'persistent_semiperiodic'
			tuning_function = 'von_mises'
		elif periodicity == 'periodic':
			boxtype = ['linear']
			motion = 'persistent_periodic'
			tuning_function = 'periodic'

		motion = 'sargolini_data'
		# motion = 'persistent_in_half_of_arena'
		boxtype.sort(key=len, reverse=True)
		sigma_distribution = 'uniform'

		target_rate = 1.0
		radius = 0.5
		velocity = 1e-2
		dt = 1.0
		limit = radius - velocity * dt

		eta_exc, eta_inh = get_learning_rates(defined='10hrs_1fps')
		eta_inh /= 50

		exc_gaussian_height = 1

		inh_gaussian_height = 0.5
		real_inh_gaussian_height = inh_gaussian_height

		sigma_exc = np.array([
			[0.05, 0.05],
			[0.05, 0.05],
			[0.05, 0.05],
			[0.05, 0.05],
		])

		sigma_inh = np.array([
			[0.10, 0.10],
			[0.11, 0.11],
			[0.12, 0.12],
			[0.15, 0.15],
		])

		input_space_resolution = sigma_exc / 4.

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
			init_weight_exc = 2.0
			symmetric_centers = True

		# learning_rate_factor = [0.001, 0.01]
		# learning_rate_factor = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
		# learning_rate_factor = [1, 10]

		### Use this if you want all center seeds (default) ###
		seed_centers = np.arange(n_simulations)
		# seed_centers = np.array([16, 24])
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
		# seed_centers = np.array([9, 28])
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
		########################################################################
		########################### Parameters ranges ##########################
		########################################################################
		# For ranges of paremeters that are just numbers, we use
		# ParameterArray(array)
		# For ranges of parameters that are themselves arrays, we use
		# get_ParametersNamed(array)
		# For string arrays you need to starte the list with the longest string
		# You can automatically achieve this using .sort(key=len, reverse=True)

		# You don't need to use Parameter() if you don't have units

		# motion = ['persistent', 'diffusive']
		# motion.sort(key=len, reverse=True)
		param_ranges = {
			'exc':
				{
					'sigma': get_ParametersNamed(sigma_exc),
					# 'eta': ParameterArray(eta_exc * np.array(
					# 	learning_rate_factor)),
					# 'init_weight': ParameterArray(init_weight_exc_array),
					# 'fields_per_synapse': ParameterArray(fields_per_synapse),
				},
			'inh':
				{
					# 'gp_stretch_factor': ParameterArray(sigma_exc/sigma_inh),
					'sigma': get_ParametersNamed(sigma_inh),
					# 'weight_factor': ParameterArray(np.array([0.2, 0.3, 0.4])),
					# float(number_per_dimension_inh[0])),
					# 'eta': ParameterArray(eta_inh * np.array(
					# 	learning_rate_factor)),
					'fields_per_synapse': ParameterArray(fields_per_synapse),
					'gaussian_height': ParameterArray([1]),
					'real_gaussian_height': ParameterArray([1]),
				},
			'sim':
				{
					# 'head_direction_sigma': ParameterArray(np.array([np.pi])),
					'input_space_resolution': get_ParametersNamed(
						input_space_resolution),
					'seed_centers': ParameterArray(seed_centers),
					'seed_init_weights': ParameterArray(seed_centers),
					'seed_motion': ParameterArray(seed_centers),
					# 'room_coherence': ParameterArray([0, 0.25, 0.5, 0.75, 1]),
					# 'alpha_room2': ParameterArray([0, 0.25, 0.5, 1.0]),
					# 'room_switch_method': ParameterArray(room_switch_method),
					'initial_x': ParameterArray(
						(2 * limit * random_sample_x - limit)[seed_centers]),
					'initial_y': ParameterArray(
						(2 * limit * random_sample_y - limit)[seed_centers]),
					# 'initial_x':ParameterArray([-radius/1.3, radius/5.1]),

				},
			# 'out':
				# {
				# 	'normalization':ParameterArray(normalization),
				# 	# 'normalization': ParameterArray([
				# 	# 								'inactive',
				# 	# 								'linear_multiplicative',
				# 	# 								'quadratic_multiplicative',
				# 	# 								'linear_substractive'])
				# }

		}

		########################################################################
		############################ Coordinate Map ############################
		########################################################################
		# Here you can specifiy the name of groups within the .h5 file
		# Typically the group name coniststs of all parameters that are
		# varied.
		# Often this is not desireable for linked parameters.
		# If for example you vary parameter_1 and parameter_2 together in a
		# specific way, it is sufficient to use only parameter_1 for naming.
		# Negative values do not show up in the final name
		# Positive values determine the order
		# Note: I also use the name for plotting, so the coordinate map
		# determines the name of the visuals
		self.tables.coord_map = {
			# ('sim', 'head_direction_sigma'): 3,
			('sim', 'initial_x'): -1,
			('sim', 'initial_y'): -1,
			('sim', 'input_space_resolution'): -1,
			('sim', 'seed_centers'): 0,
			('exc', 'sigma'): -1,
			('inh', 'sigma'): 1,
			# ('sim', 'alpha_room2'): 3,
			('sim', 'seed_init_weights'): -1,
			('sim', 'seed_motion'): -1,
			('sim', 'seed_motion'): -1,
			# ('exc', 'fields_per_synapse'): 2,
			('inh', 'fields_per_synapse'): 2,
			# ('sim', 'room_switch_method'): 1,
			# ('out', 'normalization'): 3

			('inh', 'gaussian_height'): 3,
			('inh', 'real_gaussian_height'): 4,
			# ('exc', 'eta'): 3,
			# ('inh', 'eta'): 3,
			# ('inh', 'weight_factor'): 3,
			# ('inh', 'gp_stretch_factor'): 4,
			# ('sim', 'initial_x'): 3,
		}

		########################################################################
		############################## Parameters
		# ##############################
		########################################################################
		# Here we define single parameters for the simulation
		# The structure is:
		# 'exc' / 'inh': For excitatoyr and inhibitory synapses
		# 'sim': For main simulation parameters
		# 'out':  For parameters that have to do with the output neurons
		params = {
			'visual': 'figure',
			# 'subdimension': 'space',
			'subdimension': 'none',
			# 'visual': 'none',
			# 'to_clear': 'weights_output_rate_grid_gp_extrema_centers',
			# 'to_clear': 'weights_gp_extrema_centers',
			# 'to_clear': 'weights_gp_extrema',
			'to_clear': 'none',
			'sim':
				{
					'scale_exc_weights_with_input_rate_variance': False,
					'boxside_independent_centers': False,
					# The boxside in which the rat learns first, for the
					# boxside switch experiments.
					'boxside_initial_side': 'left',
					# Time at which the rat can explore the entire arena
					# Set to False, if no 'curtain up' experiment is conducted.
					'explore_all_time': explore_all_time,
					# Time at which the rat should switch to the right side
					# of the box on move only in the right side.
					# Set to False, if no 'curtain up' experiment is conducted.
					'boxside_switch_time': boxside_switch_time,
					# We typically do not start in room2, so default is False
					'in_room2': False,
					# Correlation
					'alpha_room1': 1,
					'alpha_room2': 0.5,
					# 'room_switch_method': 'all_inputs_correlated',
					'room_switch_method': 'some_inputs_identical',
					'room_switch_time': False,
					# 'room_switch_time': room_switch_time,
					'head_direction_sigma': np.pi / 6.,
					'input_normalization': 'none',
					'tuning_function': tuning_function,
					'save_n_input_rates': 3,
					'gaussian_process': gaussian_process,
					'gaussian_process_rescale': 'fixed_mean',
					'take_fixed_point_weights': True,
					'discretize_space': True,
					# Take something smaller than the smallest
					# Gaussian (by a factor of 10 maybe)
					'input_space_resolution': ParameterArray(
						np.amin(sigma_exc, axis=1) / 10.),
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
					'store_twoSigma2': False,
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
					# A seed of 0 corresponds to the old default trajectory, 
					# for Sargolini data. Note that the motion seed used to 
					# be called seed_sargolini.
					'seed_motion': 0,
					'simulation_time': simulation_time,
					'dt': dt,
					'initial_x': 0.1,
					'initial_y': 0.2,
					'initial_z': 0.15,
					# 'velocity': 3e-4,
					'velocity': velocity,
					'persistence_length': radius,
					'motion': motion,
					'fixed_convolution_dx': False,
					# 'boundary_conditions': 'periodic',
				},
			'out':
				{
					'target_rate': target_rate,
					'normalization': 'quadratic_multiplicative',
					# 'normalization': 'inactive',
					# 'normalization': 'quadratic_multiplicative_boxside'
				},
			'exc':
				{
					'grid_input_sidelength': 10,
					'grid_input_spacing_noise': 6*sigma_exc[0][0] / 6,
					# 'save_n_input_rates': np.prod(number_per_dimension_exc),
					'save_n_input_rates': 3,
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
					'gaussian_height': exc_gaussian_height,
					'real_gaussian_height': exc_gaussian_height,
					'untuned': False,
				},
			'inh':
				{
					'grid_input_sidelength': 1,
					'grid_input_spacing_noise': 0.,
					# 'eta_factor': 2,
					# 'save_n_input_rates': np.prod(number_per_dimension_inh),
					'save_n_input_rates': 3,
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
					'gaussian_height': inh_gaussian_height,
					'real_gaussian_height': real_inh_gaussian_height,
					'untuned': False,
				}
		}

		self.tables.add_parameter_ranges(param_ranges)
		self.tables.add_parameters(params)

		#######################################################################
		########################## Linked Parameters ##########################
		#######################################################################
		# Here we link parameter ranges
		# For example the excitatory and inhibitory tuning width is varied
		# together

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
			('sim', 'seed_init_weights'),
			('sim', 'seed_motion'),
			('sim', 'initial_x'),
			('sim', 'initial_y'),
		]
		self.tables.link_parameter_ranges(linked_params_tuples)

		# linked_params_tuples = [
		# 	('exc', 'fields_per_synapse'),
		# 	('inh', 'fields_per_synapse'),
		# ]
		# self.tables.link_parameter_ranges(linked_params_tuples)



if __name__ == '__main__':
	from snep.parallel2 import run

	'''
	IMPORTANT: Only include code here that can be run repeatedly,
	because this will be run once in the parent process, and then
	once for every worker process.vi
	'''
	ji_kwargs = dict(root_dir=os.path.expanduser(
		'~/experiments/'))
	# job_time is typically None
	# mem_per_task is given in GB. Whenever it is exceeded, the simulation
	# is aborted with a memory error
	# delete_tmp should be True to delete all temporary files and save storage
	job_info = run(JobInfoExperiment, ji_kwargs, job_time=timeout,
				   mem_per_task=6, delete_tmp=True)
