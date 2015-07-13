# open the tablefile
from snep.configuration import config
config['network_type'] = 'empty'
import snep.utils
import utils
import matplotlib as mpl
mpl.use('Agg')
import plotting
import animating
import matplotlib.pyplot as plt
import time
import general_utils.arrays
import general_utils.misc
import general_utils.snep_plotting
import numpy as np
import string
import os
# import IPython

# Set font sizes in general and for all legends
# Use fontsize 42 for firing rate maps and correlograms, then change their
# height to 164pt to have the rate map of the same size as the examples
# mpl.rc('font', size=18)
mpl.rc('font', size=12)
# mpl.rc('legend', fontsize=18)
# If you comment this out, then everything works, but in matplotlib fonts
# mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
# mpl.rc('text', usetex=True)


def animate_psps(tables, paramspace_points,
	animation_function, start_time, end_time, step_factor=1, save_path=False, interval=50, take_weight_steps=False):
	"""
	Animate (several) paramspace points

	_________
	Arguments:
	See also definition of plot_psps
	- animation_function: string with the name of animation defined in animating.py
	"""
	for n, psp in enumerate(psps):
		print n
		print psp
		params = tables.as_dictionary(psp, True)
		try:
			rawdata = tables.get_raw_data(psp)
		except:
			continue
		animation = animating.Animation(params, rawdata, start_time=start_time, end_time=end_time, step_factor=step_factor, take_weight_steps=take_weight_steps)
		ani = getattr(animation, animation_function)
		if save_path:
			# remove the uno string
			save_path_full = os.path.join(save_path, string.replace(str(psp), ' uno', '') + '.mp4')
		else:
			save_path_full = False
		ani(save_path=save_path_full, interval=interval)


def get_path_tables_psps(date_dir):
	path = os.path.join(
		os.path.expanduser('~/localfiles/itb_experiments/learning_grids/'),
		date_dir, 'experiment.h5')
	tables = snep.utils.make_tables_from_path(path)
	tables.open_file(False)
	tables.initialize()
	print tables
	psps = tables.paramspace_pts()
	return path, tables, psps

######################################################
##########	Decide what should be plotted	##########
######################################################
# function_kwargs is a list of tuples of strings (the function names)
# and dictionaries (the function parameters as keys and values)
t0 = 0.
t1 = 5e6
t2 = 2e7
# t2 = 2e7
# t3 = 1e8
t3 = 24e6
t_hm = 5e4
t = 16e7
# t4 = 24e6
# t2 = 40e5
# t1 = 120e6
# t1 = 80e6
# t1 = 100e6
# t=10e7
method = 'Weber'
type = 'exc'
# t2 = 1e7
# Neurons for conjunctive and grid cells
# neurons = [23223, 51203, 35316, 23233]
# Neurons for head direction cell
# neurons =[5212, 9845, 9885, 6212]

function_kwargs = [
	##########################################################################
	##############################   New Plots  ##############################
	##########################################################################

	('input_tuning', {'neuron': 0, 'populations': ['exc']}),
	('input_tuning', {'neuron': 0, 'populations': ['inh']}),
	('input_tuning', {'neuron': 1, 'populations': ['exc']}),
	('input_tuning', {'neuron': 1, 'populations': ['inh']}),
	#
	# ('plot_grid_spacing_vs_parameter',
	# 		{	'from_file': True,
	# 			'varied_parameter': ('inh', 'sigma'),
	# 			'parameter_range': np.linspace(0.08, 0.38, 201),
	# 			# 'varied_parameter': ('inh', 'gaussian_height'),
	# 			# 'parameter_range': np.sqrt(np.linspace(1, 8, 201)) * np.array([1]),
	# 			# 'varied_parameter': ('inh', 'number_per_dimension'),
	# 			# 'parameter_range': np.linspace(1, 8, 201) * np.array([200]),
	# 			# 'varied_parameter': ('inh', 'eta'),
	# 			# 'parameter_range': np.linspace(1, 8, 201) * 2e-3 / (2*3),
	# 			'plot_mean_inter_peak_distance': True,
	# 			'computed_data': False,
	# 			'publishable': False}),

	# ('trajectory_with_firing', {'start_frame': 0, 'end_frame':0.5e4, 'symbol_size': 20}),
	# ('trajectory_with_firing', {'start_frame': 0, 'end_frame':1e4, 'symbol_size': 20}),
	# ('trajectory_with_firing', {'start_frame': 0, 'end_frame':2e4, 'symbol_size': 20}),
	# ('trajectory_with_firing', {'start_frame': 0, 'end_frame':3e4, 'symbol_size': 20}),
	# ('trajectory_with_firing', {'start_frame': 2e4, 'end_frame':3e4, 'symbol_size': 8,
	# 							'firing_indicator': 'spikes'}),

	# ('trajectory_with_firing', {'start_frame': 0, 'end_frame':1e4}),
	# ('trajectory_with_firing', {'start_frame': 0, 'end_frame':2e4}),
	# ('trajectory_with_firing', {'start_frame': 0, 'end_frame':3e4}),

	# ('plot_output_rates_from_equation', {'time': 2e4, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 2.25e4, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 2.5e4, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 2.75e4, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 3e4, 'from_file': True}),
	# ('fields', {'neuron': 2000, 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['exc'], 'publishable': False}),
	# ('plot_grid_spacing_vs_parameter',
	# 		{	'from_file': True,
	# 			'varied_parameter': ('inh', 'sigma'),
	# 			# 'parameter_range': None,
	# 			'parameter_range': np.linspace(0.08, 0.38, 201),
	# 			'plot_mean_inter_peak_distance': True,
	# 			'computed_data': False}),

	# ('grid_spacing_vs_n_inh_fraction', {'time': 4e7}),
	# ('position_distribution', {'start_time':0, 'end_time': 12e7, 'bins': 101}),
	# ('position_distribution', {'start_time':0, 'end_time': 8e4, 'bins': 101}),
	# ('position_distribution', {'start_time':0, 'end_time': 8e5, 'bins': 101}),
	# ('position_distribution', {'start_time':0, 'end_time': 8e6, 'bins': 101}),
	# ('position_distribution', {'start_time':4e6, 'end_time': 8e6, 'bins': 101}),
	# ('position_distribution', {'start_time':0, 'end_time': 1e5}),
	# ('position_distribution', {'start_time':5e4, 'end_time': 1e5}),
	# ('plot_time_evolution', {'observable': 'grid_score'}),
	# ('plot_output_rates_from_equation', {'time': 1e4, 'from_file': True}),
	# ('plot_correlogram', {'time': t2, 'from_file': True, 'mode': 'same', 'method': method, 'publishable': False}),
	# ('fields', {'neuron': 0, 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['exc'], 'publishable': True}),
	# ('fields', {'neuron': 1, 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['exc'], 'publishable': True}),
	# ('fields', {'neuron': 0, 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['inh'], 'publishable': True}),
	# ('fields', {'neuron': 1, 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['inh'
	# ], 'publishable': True}),

	# ('input_current', {'time': 0, 'spacing':401, 'populations': ['exc', 'inh'],
	# 				   'xlim': [-2.0, 2.0]}),
	# ('input_current', {'time': 0, 'spacing':401, 'populations': ['inh']}),
	# ('input_norm', {'populations': ['exc']}),
	# ('input_norm', {'ylim': [0, 2], 'populations': ['inh']})
	# ('plot_output_rates_from_equation',
	# 	{'time': 0., 'from_file': True}),
	# ('plot_correlogram',
	#  	{'time': 0, 'from_file': True, 'mode': 'same'}),
	# ('plot_output_rates_from_equation',
	# 	{'time': t1/4., 'from_file': True}),
	# ('plot_correlogram',
	#  	{'time': t1/4., 'from_file': True, 'mode': 'same'}),
	# ('plot_output_rates_from_equation',
	# 	{'time': t1/2., 'from_file': True}),
	# ('plot_correlogram',
	#  	{'time': t1/2., 'from_file': True, 'mode': 'same'}),
	# ('plot_output_rates_from_equation',
	# 	{'time': t1, 'from_file': True}),
	# ('plot_correlogram',
	#  	{'time': t1, 'from_file': True, 'mode': 'same'}),

	###########################################################################
	######################## Grid Spacing VS Parameter ########################
	###########################################################################
	# ('plot_grid_spacing_vs_parameter',
	# 		{	'from_file': True,
	# 			'parameter_name': 'sigma_exc',
	# 			'parameter_range': np.linspace(0.012, 0.047, 200),
	# 			# 'parameter_range': np.linspace(0.015, 0.055, 200),
	# 			'plot_mean_inter_peak_distance': True})
	# ('plot_grid_spacing_vs_parameter',
	# 		{	'from_file': True,
	# 			'parameter_name': 'sigma_inh',
	# 			'parameter_range': np.linspace(0.08, 0.38, 201),
	# 			# 'parameter_range': np.linspace(0.08, 0.36, 201),
	# 			# 'parameter_range': np.linspace(0.015, 0.055, 200),
	# 			'plot_mean_inter_peak_distance': True,
	# 		'computed_data': False})

	# ('plot_grid_spacing_vs_parameter',
	# 		{	'from_file': True,
	# 			'parameter_name': 'n_inh',
	# 			'parameter_range': np.linspace(50, 1000, 201),
	# 			'plot_mean_inter_peak_distance': True,
	# 			'computed_data': False})

	# ('input_tuning', {'neuron': 0, 'populations': [type]}),
	# ('input_tuning', {'neuron': 1, 'populations': [type]}),
	# ('input_tuning', {'neuron': 2, 'populations': [type]}),

	##########################################################################
	########################## Figure with Heat Map ##########################
	##########################################################################
	# ('fields', {'neuron': 100, 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['exc'], 'publishable': True}),
	# ('fields', {'neuron': 200, 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['inh'], 'publishable': True}),
	# ('input_tuning', {'neuron': 0, 'populations': ['exc'], 'publishable':
	# 	True}),
	# ('input_tuning', {'neuron': 53, 'populations': ['inh'], 'publishable':
	# 	True}),
	# ('plot_output_rates_from_equation', {'time':  0, 'from_file': True,
	# 									 'maximal_rate': False,
	# 									 'publishable': True}),
	# ('output_rate_heat_map', {'from_file': True, 'end_time': t_hm,
	# 						  'publishable': True}),
	# ('plot_output_rates_from_equation', {'time':  t_hm, 'from_file': True,
	# 								 'maximal_rate': False,
	# 								 'publishable': True}),

	##########################################################################
	#################### Figure gridspacing vs sigma_inh #####################
	##########################################################################
	# NOTE: Use this for plotting from place field inputs
	# ('plot_output_rates_from_equation', {'time':  4e7, 'from_file': False, 'spacing': 2001,
	# 									 'maximal_rate': False,
	# 									 'publishable': True}),
	# ('plot_output_rates_from_equation', {'time':  4e7, 'from_file': False, 'spacing': 2001,
	# 									 'maximal_rate': False,
	# 									 'publishable': True}),
	# NOTE: Use this for plotting from GP inputs
	# ('plot_output_rates_from_equation', {'time':  4e7, 'from_file': True,
	# 									 'maximal_rate': False,
	# 									 'publishable': True}),
	# ('plot_output_rates_from_equation', {'time':  4e7, 'from_file': True,
	# 									 'maximal_rate': False,
	# 									 'publishable': True}),
	# ('plot_grid_spacing_vs_parameter',
	# 		{	'from_file': True,
	# 			'varied_parameter': ('inh', 'sigma'),
	# 			'parameter_range': np.linspace(0.08, 0.30, 201),
	# 			# 'parameter_range': np.linspace(0.08, 0.36, 201),
	# 			# 'parameter_range': np.linspace(0.015, 0.055, 200),
	# 			'plot_mean_inter_peak_distance': True,
	# 			'computed_data': False}),


	# ('plot_grid_spacing_vs_parameter',
	# 		{	'from_file': True,
	# 			'parameter_name': 'sigma_inh',
	# 			'parameter_range': None,
	# 			# 'parameter_range': np.linspace(0.08, 0.36, 201),
	# 			# 'parameter_range': np.linspace(0.015, 0.055, 200),
	# 			'plot_mean_inter_peak_distance': True,
	# 			'computed_data': False}),

	##########################################################################
	################################ Figure 2 ################################
	##########################################################################
	# ('fields', {'neuron': 2960, 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['exc'], 'publishable': True}),
	# ('fields', {'neuron': 1300, 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['exc'], 'publishable': True}),
	# ('fields', {'neuron': 1510, 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['inh'], 'publishable': True}),
	# ('fields', {'neuron': 2270, 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['inh'], 'publishable': True}),
	# ('plot_output_rates_from_equation', {'time': 0, 'from_file': True, 'maximal_rate': False, 'publishable': True, 'show_colorbar': False, 'show_title': False}),
	# ### Plotting the correlogram at time zero requires you to use
	# # a different plotting function in the plotting.py file
	# ('plot_correlogram', {'time': 0, 'from_file': True, 'mode': 'same', 'method': None, 'publishable': True}),
	# ('plot_output_rates_from_equation', {'time': t 1, 'from_file': True, 'maximal_rate': False, 'publishable': True, 'show_colorbar': False, 'show_title': False}),
	# ('plot_correlogram', {'time': t1, 'from_file': True, 'mode': 'same', 'method': None, 'publishable': True}),

	##########################################################################
	################################ Figure 3 ################################
	##########################################################################
	# Make sure you choose the correct neurons array above !!!
	# ('fields', {'neuron': neurons[0], 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['exc'], 'publishable': True}),
	# ('fields', {'neuron': neurons[1], 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['exc'], 'publishable': True}),
	# ('fields', {'neuron': neurons[2], 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['inh'], 'publishable': True}),
	# ('fields', {'neuron': neurons[3], 'show_each_field': False, 'show_sum': True,
	# 			'populations': ['inh'], 'publishable': True}),
	# ('fields_polar', {'syn_type': 'exc', 'neuron': 1234, 'publishable': True}),
	# ('fields_polar', {'syn_type': 'exc', 'neuron': 3523, 'publishable': True}),
	# ('fields_polar', {'syn_type': 'inh', 'neuron': 6234, 'publishable': True}),
	# ('fields_polar', {'syn_type': 'inh', 'neuron': 7233, 'publishable': True}),
	# ('plot_output_rates_from_equation', {'time': t3, 'from_file': True,
	# 									 'maximal_rate': False,
	# 									 'subdimension': 'space',
	# 									 'publishable': True}),
	# ('plot_correlogram', {'time': t3, 'from_file': True, 'mode': 'same',
	# 					  'method': method, 'publishable': True}),
	# ('plot_head_direction_polar', {'time': t3, 'from_file': True,
	# 							   'publishable': True}),

	##########################################################################
	##################### END of Cosyne abstract figures #####################
	##########################################################################
	]

if __name__ == '__main__':
	# date_dir = '2014-08-05-11h01m40s_grid_spacing_vs_sigma_inh'
	# date_dir = '2014-11-24-14h08m24s_gridspacing_vs_sigmainh_GP_input_NEW'
	# date_dir = '2014-11-05-14h50m34s_new_grids'
	# date_dir = '2014-11-06-14h45m37s_16_and_32_fps_smaller_learning_rate'
	# date_dir = '2014-11-25-18h14m49s_place_cells_32_fps'
	# date_dir = '2014-11-05-18h49m20s_inverted_exc_and_inh_width'
	# date_dir = '2015-02-18-20h50m42s_band_cells_32_fps'
	# date_dir = '2014-08-08-10h08m10s_3D_grid_and_conjunctive'
	# date_dir = '2015-03-09-12h22m55s_grid_score_stability_periodic_3_times_longer_3_times_slower_learning'
	# date_dir = '2015-03-10-17h14m00s_periodic_semiperiodic_nonperiodic'
	# date_dir = '2015-03-06-18h08m56s_grid_score_stability_periodic'
	# date_dir = '2015-03-09-12h22m55s_grid_score_stability_periodic_3_times_longer_3_times_slower_learning'
	# date_dir = '2015-04-23-15h17m05s_grid_spacing_vs_n_inh_fraction'
	# date_dir = '2015-05-19-17h18m02s_grid_spacing_vs_sigma_inh_TEST'
	# date_dir = '2015-05-19-17h13m54s_grid_spacing_vs_sigma_inh_Nexc_2000_Ninh_2000'
	# date_dir = '2015-05-19-18h23m24s_grid_spacing_vs_sigma_inh_Nexc_2000_Ninh_500'
	# date_dir = '2015-05-20-10h37m13s'
	# date_dir = '2015-07-01-17h53m22s_grid_spacing_VS_eta_inh'
	# date_dir = '2015-07-03-12h52m25s'
	# date_dir = '2015-07-02-15h08m01s_grid_spacing_VS_n_inh'
	# date_dir = '2015-07-04-10h57m42s_grid_spacing_VS_gaussian_height_inh'
	# date_dir = '2015-07-07-21h00m49s'
	# date_dir = '2014-11-20-21h29m41s_heat_map_GP_shorter_time'
	# date_dir = '2015-07-03-12h52m25s_fast_grid_trajectory'
	# date_dir = '2015-07-12-20h38m03s_trajectory_with_firing_video'
	date_dir = '2015-07-13-11h37m20s'


	path, tables, psps = get_path_tables_psps(date_dir)
	save_path = False
	save_path = os.path.join(os.path.dirname(path), 'visuals')
	try:
		os.mkdir(save_path)
	except OSError:
		pass

	all_psps = psps
	# fields_per_synapse = [1, 2, 4, 8, 16, 32]
	# for fps in fields_per_synapse:
	# sigma_exc_x = [0.08, 0.11, 0.15]
	# sigma_exc_x = [0.1]
	# sigma_inh_y = [0.7]
	# sigma_exc = [[0.12, 0.45], [0.11, 0.4], [0.11, 0.4], [0.12, 0.5]]
	# sigma_exc = [[0.05, 0.2], [0.05, 0.2]]
	# sigma_inh = [[0.12, 0.7], [0.11, 0.7],[0.12, 0.6],[0.12, 0.7]]
	# sigma_inh =	[[0.10, 0.8],[0.10, 0.9]]
	# for se, si in zip(sigma_exc, sigma_inh):
	# sigmaI_range = np.arange(0.08, 0.32, 0.02)
	# sigmaI_range = np.arange(0.2, 0.4, 0.02)


	psps = [p for p in all_psps
			# if p[('exc', 'number_per_dimension')].quantity >= 820
			# if p[('inh', 'weight_factor')].quantity < 1.025

			# and p[('exc', 'sigma')].quantity[0] == 0.05
			# if p[('sim', 'seed_centers')].quantity == 0
			# and general_utils.misc.approx_equal([('sim', 'initial_x')].quantity, -0.45045, tolerance=0.001)
	# 		# and p[('sim', 'output_neurons')].quantity == 8
	# 		# and p[('sim', 'dt')].quantity == 0.01s
	# 			if p[('sim', 'initial_x')].quantity > 5
	# 			and p[('sim', 'symmetric_centers')].quantity == True
	# 			if np.array_equal(p[('exc', 'sigma')].quantity, se)
	# 			and np.array_equal(p[('inh', 'sigma')].quantity, si)
	# 			if p[('exc', 'sigma')].quantity[0] == se
	# 			and p[('inh', 'sigma')].quantity[1] == si
	# 		# and p[('sim', 'symmetric_centers')].quantity == False
	# 		# or p[('inh', 'sigma')].quantity == 0.08
	# 		if p[('inh', 'sigma')].quantity < 0.38
			# and  p[('exc', 'sigma')].quantity <= 0.055
			# and p[('sim', 'boxtype')].quantity == 'linear'
			# if np.array_equal(p[('exc', 'sigma')].quantity, [0.06, 0.08])
			# if p[('sim', 'initial_x')].quantity > 0.6
			# if (p[('sim', 'seed_centers')].quantity == 0)
			# if (p[('sim', 'seed_centers')].quantity == 3)
			# and p[('exc', 'fields_per_synapse')].quantity == 32
			# and p[('inh', 'sigma')].quantity < 0.31
			# and (p[('inh', 'sigma')].quantity == sigmaI_r ange[0] or p[('inh', 'sigma')].quantity == sigmaI_range[5])
			# and p[('sim', 'boxtype')].quantity == 'linear'
			# and p[('sim', 'symmetric_centers')].quantity == True
			# and p[('sim', 'initial_x')].quantity > 0
			]

	prefix = general_utils.plotting.get_prefix(function_kwargs)
	# prefix = ''

	general_utils.snep_plotting.plot_psps(
				tables, psps, project_name='learning_grids', save_path=save_path,
				 psps_in_same_figure=False, function_kwargs=function_kwargs,
				 prefix=prefix, automatic_arrangement=True, file_type='png')

	# Note: interval should be <= 300, otherwise the videos are green
	# animate_psps(tables, psps, 'animate_positions', 0.0, 3e2, interval=50, save_path=save_path)
	# animate_psps(tables, psps, 'animate_output_rates', 0.0, 1e6, interval=50, save_path=save_path, take_weight_steps=True)

	# # # t2 = time.time()
	# tables.close_file()
	# plt.show()
	# print 'Plotting took % seconds' % (t2 - t1)