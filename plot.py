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
mpl.rc('font', size=18)
mpl.rc('legend', fontsize=18)
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
	# # psps = [p for p in tables.paramspace_pts()
	# # 		# if p[('sim', 'output_neurons')].quantity == 2
	# # 		# and p[('sim', 'weight_lateral')].quantity == 4.0
	# # 		# and p[('sim', 'output_neurons')].quantity == 8
	# # 		# and p[('sim', 'dt')].quantity == 0.01
	# # 		# if p[('exc', 'sigma')].quantity > 0.019
	# # 		# and p[('exc', 'sigma')].quantity < 0.059
	# # 		# and p[('exc', 'sigma')].quantity <= 0.04
	# # 		# # # if p[('sim', 'boxtype')].quantity == 'linear'
	# # 		# and p[('sim', 'seed_init_weights')].quantity == 3
	# # 		# and p[('sim', 'initial_x')].quantity < 0.0
	# # 		]
	return path, tables, psps 

path, tables, psps = get_path_tables_psps(
	'2014-03-22-19h11m25s_grid_spacing_vs_sigma_inh')
save_path = False
save_path = os.path.join(os.path.dirname(path), 'visuals')

try:
	os.mkdir(save_path)
except OSError:
	pass
general_utils.snep_plotting.plot_psps(
	tables, psps, project_name='learning_grids', save_path=save_path,
	 psps_in_same_figure=True)

# Note: interval should be <= 300, otherwise the videos are green
# animate_psps(tables, psps, 'animate_positions', 0.0, 3e2, interval=50, save_path=save_path)
# animate_psps(tables, psps, 'animate_output_rates', 0.0, 1e6, interval=50, save_path=save_path, take_weight_steps=True)

# # # t2 = time.time()
# tables.close_file()
plt.show()
# print 'Plotting took %f seconds' % (t2 - t1)