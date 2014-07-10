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
import general_utils.snep_plotting
import numpy as np
import string
import os
# import IPython

# Set font sizes in general and for all legends
mpl.rc('font', size=16)
mpl.rc('legend', fontsize=16)
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
	tables.open_file(True)
	print tables
	psps = tables.paramspace_pts()
	# psps = [p for p in tables.paramspace_pts()
	# 		# if p[('sim', 'output_neurons')].quantity == 2
	# 		# and p[('sim', 'weight_lateral')].quantity == 4.0
	# 		# and p[('sim', 'output_neurons')].quantity == 8
	# 		# and p[('sim', 'dt')].quantity == 0.01
	# 		if p[('sim', 'seed_centers')].quantity == 5
	# 		and np.array_equal(p[('exc', 'sigma')].quantity, [0.05, 0.05])
	# 		# and p[('inh', 'fields_per_synapse')].quantity == 8
	# 		# and p[('sim', 'symmetric_centers')].quantity == False
	# 		# or p[('inh', 'sigma')].quantity == 0.08
	# 		# and p[('exc', 'sigma')].quantity < 0.059
	# 		# if p[('inh', 'sigma')].quantity <= 0.2
	# 		# and  p[('exc', 'sigma')].quantity <= 0.055
	# 		# # # if p[('sim', 'boxtype')].quantity == 'linear'
	# 		# and p[('sim', 'seed_init_weights')].quantity == 3
	# 		# and p[('sim', 'initial_x')].quantity < -2
	# 		]
	return path, tables, psps

######################################################
##########	Decide what should be plotted	##########
######################################################
# function_kwargs is a list of tuples of strings (the function names)
# and dictionaries (the function parameters as keys and values)
t1 = 9.5e6
t2 = 10e6
function_kwargs = [
	# ('plot_output_rates_from_equation',
	# 	{'time': 0, 'from_file': True}),
	# ('plot_output_rates_from_equation',
	# 	{'time': -1, 'from_file': True}),
	# ('weights_vs_centers',
	# 	{'time': 0}),
	# ('weights_vs_centers',
	# 	{'time': 0, 'syn_type': 'inh'}),

	# ('weights_vs_centers',
	# 	{'time': -1}),
	# ('weights_vs_centers',
	# 	{'time': -1, 'syn_type': 'inh'}),
	# # ('plot_output_rates_from_equation', {'time': 1e3, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 0e6, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 7.5e7, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 8e7, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 8.5e7, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 9e7, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 9.5e7, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 1e7, 'from_file': True}),

	# ('plot_output_rates_from_equation', {'time': 9e6, 'from_file': False, 'spacing': 601}),
	# ('plot_output_rates_from_equation', {'time': 10e6, 'from_file': False, 'spacing': 601}),
	# ('weight_evolution', {'syn_type': 'exc'}),
	# ('weight_evolution', {'syn_type': 'inh'}),

	# ('fields_times_weights', {'time': 150e4, 'syn_type': 'inh'}),
	# ('plot_output_rates_from_equation', {'time': t1, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': t2, 'from_file': True}),
	# ('plot_grids_linear', {'time': t1, 'from_file': True}),	
	# ('plot_grids_linear', {'time': t2, 'from_file': True}),	
	# ('plot_head_direction_polar', {'time': t1, 'from_file': True}),
	# ('plot_head_direction_polar', {'time': t2 , 'from_file': True}),

	# ('plot_polar', {'time': 9e6, 'from_file': True}),
	# ('plot_polar', {'time': 10e6, 'from_file': True}),
	('plot_correlogram', {'time': 1e7, 'from_file': True, 'mode': 'same', 'method': 'Weber'}),
	# ('plot_grid_spacing_vs_parameter',
	# 		{	'from_file': True,
	# 			'parameter_name': 'sigma_exc',
	# 			'parameter_range': np.linspace(0.012, 0.047, 200),
	# 			# 'parameter_range': np.linspace(0.015, 0.055, 200),
	# 			'plot_mean_inter_peak_distance': True})
	# ('output_rate_heat_map', {'from_file': True, 'start_time': 8000.0, 'end_time': -1})
	# ('output_rate_heat_map', {'from_file': True, 'start_time': 0, 'end_time': -1, 'maximal_rate': 12.0})
	]

if __name__ == '__main__':
	path, tables, psps = get_path_tables_psps(
		'2014-07-10-10h41m43s_different_exc_widths')
	save_path = False
	save_path = os.path.join(os.path.dirname(path), 'visuals')

	try:
		os.mkdir(save_path)
	except OSError:
		pass
	general_utils.snep_plotting.plot_psps(
		tables, psps, project_name='learning_grids', save_path=save_path,
		 psps_in_same_figure=False, function_kwargs=function_kwargs, prefix='cor')

	# Note: interval should be <= 300, otherwise the videos are green
	# animate_psps(tables, psps, 'animate_positions', 0.0, 3e2, interval=50, save_path=save_path)
	# animate_psps(tables, psps, 'animate_output_rates', 0.0, 1e6, interval=50, save_path=save_path, take_weight_steps=True)

	# # # t2 = time.time()
	# tables.close_file()
	# plt.show()
	# print 'Plotting took %f seconds' % (t2 - t1)