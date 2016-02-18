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
# from memory_profiler import profile
# import cProfile
# import pstats
# import tables

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
	# 		   ('grid_score_2d', dict(type='quadratic'))]
	compute = [('mean_inter_peak_distance', {})]
	# compute = None
	# ('grid_score_2d', dict(type='quadratic'))]
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
			[
				[
					('plot_output_rates_from_equation',
					 dict(time=t, from_file=True, n_cumulative=None)),
					# ('plot_correlogram',
					#  dict(time=t, from_file=True, mode='same',
					# 	method='sargolini', n_cumulative=10))
				]
				for t in np.arange(0,
					sim_time + every_nth_step, every_nth_step)
				]
			# [
			# 	### Figure 1 ###
			# 	[
			# 		(
			# 		'plot_output_rates_from_equation',
			# 			dict(time=t, from_file=True)
			# 		)
			# 		for t in simulation_time * np.array([0, 1/4., 1/2., 1])
			# 	],
			# 	### Figure 2 ###
			# 	[
			# 		(
			# 		'plot_correlogram',
			# 			dict(time=t, from_file=True, mode='same',
			# 				 method='sargolini')
			# 		)
			# 		for t in simulation_time * np.array([0, 1/4., 1/2., 1])
			# 	],
			# 	### Figure 3 ###
			# 	[
			# 		(
			# 		'trajectory_with_firing',
			# 		dict(start_frame=0,  end_frame=simulation_time/i)
			# 		)
			# 		for i in [4, 3, 2, 1]
			# 	]
			# 	### End of Figure 2 ###
			# ]
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
	if params['to_clear'] == 'weights_and_output_rate_grid':
		key_lists = [['exc', 'weights'], ['inh', 'weights'],
					 ['output_rate_grid']]
		# Arrays that are None are not written to disk
		utils.set_values_to_none(results['raw_data'], key_lists)

	return results


class JobInfoExperiment(Experiment):
	run_task = staticmethod(run_task_sleep)

	def _prepare_tasks(self):
		from snep.utils import ParameterArray, ParametersNamed

		simulation_time = 4*4e7
		every_nth_step = simulation_time / 4
		np.random.seed(1)
		n_simulations = 2
		random_sample_x = np.random.random_sample(n_simulations)
		random_sample_y = np.random.random_sample(n_simulations)
		dimensions = 1
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
		radius = 4 * 5.0
		# eta_inh = 8e-3 / (2*radius * 10. * 5.5)
		# eta_exc = 8e-4 / (2*radius * 10. * 22)
		eta_inh = 5e-4 / (2*radius * 4.)
		eta_exc = 5e-5 / (2*radius * 13.)

		# sigma_exc = np.array([
		# 	[0.03],
		# ])
		#
		# sigma_inh = np.array([
		# 	[0.12],
		# ])

		# number_per_dimension_exc = np.array([70, 70]) / 5
		# number_per_dimension_inh = np.array([35, 35]) / 5

		number_per_dimension_exc = np.array([2000]) * 5 * 4
		number_per_dimension_inh = np.array([2000]) * 5 * 4


		sinh = np.arange(0.08, 0.36, 0.04)
		sexc = np.tile(0.03, len(sinh))
		sigma_inh = np.atleast_2d(sinh).T.copy()
		sigma_exc = np.atleast_2d(sexc).T.copy()

		input_space_resolution = sigma_exc / 8.

		def get_ParametersNamed(a):
			l = []
			for x in a:
				l.append((str(x).replace(' ', '_'), ParameterArray(x)))
			return ParametersNamed(l)

		gaussian_process = True
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
					'sigma': get_ParametersNamed(sigma_exc),
				},
			'inh':
				{
					# 'gp_stretch_factor': ParameterArray(sigma_exc/sigma_inh),
					'sigma': get_ParametersNamed(sigma_inh),
					'weight_factor':ParameterArray(1 + 2.*np.array([10]) / np.prod(number_per_dimension_inh)),
				},
			'sim':
				{
					'input_space_resolution': get_ParametersNamed(
						input_space_resolution),
					'seed_centers': ParameterArray(np.arange(n_simulations)),
					'initial_x': ParameterArray(
						2 * radius * random_sample_x - radius),
					'initial_y': ParameterArray(
						2 * radius * random_sample_y - radius),
					# 'initial_x':ParameterArray([-radius/1.3, radius/5.1]),
				},
			'out':
				{
					# 'normalization':ParameterArray(['quadratic_multiplicative',
					# 	'quadratic_multiplicative_lateral_inhibition']),
				}

		}

		self.tables.coord_map = {
			('sim', 'initial_x'): -1,
			('sim', 'initial_y'): -1,
			('sim', 'input_space_resolution'): -1,
			('sim', 'seed_centers'): 0,
			('exc', 'sigma'): 1,
			('inh', 'sigma'): 2,
			('inh', 'weight_factor'): 3,
			# ('inh', 'gp_stretch_factor'): 4,
			# ('sim', 'initial_x'): 3,
		}

		params = {
			'visual': 'figure',
			# 'to_clear': 'weights_and_output_rate_grid',
			'to_clear': 'none',
			'sim':
				{
					'input_normalization': 'figure',
					'tuning_function': tuning_function,
					'save_n_input_rates': 3,
					'gaussian_process': gaussian_process,
					'gaussian_process_rescale': True,
					'take_fixed_point_weights': True,
					'discretize_space': True,
					# Take something smaller than the smallest
					# Gaussian (by a factor of 10 maybe)
					'input_space_resolution': ParameterArray(
						np.amin(sigma_exc, axis=1) / 10.),
					'spacing': 2001,
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
					'every_nth_step': every_nth_step,
					'every_nth_step_weights': every_nth_step,
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
					'gp_stretch_factor': 1.0,
					'gp_extremum': ParameterArray(np.array([-1., 1]) * 0.15),
					# 'gp_extremum': 'none',
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
				},
			'inh':
				{
					'gp_stretch_factor': 1.0,
					'gp_extremum': ParameterArray(np.array([-1., 1]) * 0.12),
					# 'gp_extremum': 'none',
					'center_overlap_factor': 3.,
					'weight_factor': 1.0,
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
			('sim', 'initial_x'),
			('sim', 'initial_y'),
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
	job_info = run(JobInfoExperiment, ji_kwargs, job_time=timeout, mem_per_task=6,
				   delete_tmp=True)
