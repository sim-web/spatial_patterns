# open the tablefile
from snep.configuration import config
config['network_type'] = 'empty'
import snep.utils
import utils
import matplotlib as mpl
mpl.use('TkAgg')
import plotting
import animating
import matplotlib.pyplot as plt
import time
import general_utils.arrays
import numpy as np
import string
import os
# import IPython


date_dir = '2013-10-24-13h55m05s'

tables = snep.utils.make_tables_from_path(
	'/Users/simonweber/localfiles/itb_experiments/learning_grids/' 
	+ date_dir 
	+ '/experiment.h5')
t0 = time.time()
tables.open_file(True)
print tables

psps = tables.paramspace_pts()
# psps = [p for p in tables.paramspace_pts() 
# 		if p[('inh', 'eta')].quantity == 2e-6
# 		and p[('sim', 'velocity')].quantity == 0.1]
# psps = [p for p in tables.paramspace_pts() 
# 		if p[('inh', 'weight_overlap')].quantity == 0.0
# 		# if p[('inh', 'sigma')].quantity == 0.2
# 		# and p[('exc', 'sigma')].quantity == 0.05
# 		# and p[('sim', 'boxtype')].quantity == 'linear'
# 		and p[('exc', 'weight_overlap')].quantity == 0.0
# 		]

def get_plot_list(plot_class):
	"""
	Returns a list of plots

	----------
	Arguments:
	- plot_class: class which contains the plot functions
	"""
	plot_list = [
		# lambda: [plot.fields_times_weights(syn_type='exc'), 
		#           plot.weights_vs_centers(syn_type='exc')],
		# lambda: [plot.fields_times_weights(syn_type='inh'), 
		#           plot.weights_vs_centers(syn_type='inh')],
		# lambda: plot.weights_vs_centers(syn_type='exc', time=-1), 
		# # lambda: plot.weights_vs_centers(syn_type='exc'),    
		# lambda: plot_class.weight_evolution(syn_type='exc', time_sparsification=1, weight_sparsification=5),
		# lambda: plot_class.weight_evolution(syn_type='inh', time_sparsification=1, weight_sparsification=5),
		# lambda: plot_class.spike_map(small_dt=0.00000001, start_frame=0, end_frame=1e4),
		# lambda: plot_class.spike_map(small_dt=0.01, start_frame=2e4, end_frame=4e4),
		# lambda: plot_class.spike_map(small_dt=0.01, start_frame=10e4, end_frame=12e4),		
		# lambda: plot_class.spike_map(small_dt=0.01, start_frame=98e4, end_frame=100e4),
		# lambda: plot_class.spike_map(small_dt=0.01, start_frame=0, end_frame=2e4),
		# lambda: plot_class.spike_map(small_dt=0.02, start_frame=8e4, end_frame=10e4),
		# lambda: plot_class.spike_map(small_dt=0.01, start_frame=8e4, end_frame=10e4),

		# # # #lambda: plot.weight_evolution(syn_type='inh', time_sparsification=10, weight_sparsification=1000),
		# # # # lambda: plot.output_rate_distribution(start_time=(params['simulation_time']-10000)/params['every_nth_step']),
		# lambda: plot_class.plot_output_rates_from_equation(frame=200, spacing=201, fill=False),
		# lambda: plot_class.plot_output_rates_from_equation(frame=0, spacing=201, fill=False),
		# lambda: plot_class.plot_output_rates_from_equation(frame=50, spacing=201, fill=False),
		# lambda: plot_class.plot_output_rates_from_equation(frame=100, spacing=201, fill=False),
		# lambda: plot_class.plot_output_rates_from_equation(frame=0, spacing=201, fill=False),
		# lambda: plot_class.plot_output_rates_from_equation(frame=150, spacing=201, fill=False),
		# lambda: plot_class.plot_output_rates_from_equation(frame=300, spacing=201, fill=False),
		# lambda: plot_class.plot_output_rates_from_equation(frame=-1, spacing=201, fill=False),

		# lambda:   plot.output_rates_vs_position(start_time=(params['simulation_time']-9000000)/params['every_nth_step']),
		# lambda: plot_class.output_rates_vs_position(start_frame=90000, clipping=True),
		]
	return plot_list

def plot_psps(tables, paramspace_points, save_path=False):
	"""
	Plot (several) paramspace points

	----------
	Arguments:
	- tables: tables from an hdf5 file
	- paramspace_points: list of paramspace_points (dictionaries)
	"""
	for n, psp in enumerate(psps):
		print n
		print psp
		params = tables.as_dictionary(psp, True)
		rawdata = tables.get_raw_data(psp)
		plot = plotting.Plot(params, rawdata)
		fig = plt.figure(str(psp))
		plot_list = get_plot_list(plot)
		plotting.plot_list(fig, plot_list)
		if save_path:
			save_path_full = os.path.join(save_path, string.replace(str(psp), ' uno', '') + '.png')
			# plt.savefig(save_path_full, dpi=170, bbox_inches='tight', pad_inches=0.1)
			plt.savefig(save_path_full, dpi=170, bbox_inches='tight', pad_inches=0.1)
		# Clear figure and close windows
		else:
			plt.show()
		plt.clf()
		plt.close()
	# if animation:
	#   animation = animating.Animation(params, rawdata, start_time=0.0, end_time=1000.0, step_factor=1)
	#   save_path = '/Users/simonweber/Desktop/2e-8_persistent.mp4'
	#   animation.animate_output_rates(save_path=save_path, interval=50)
	#   animation.animate_positions(save_path=False, interval=50)

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


t1 = time.time()
save_path = False
save_path = os.path.join('/Users/simonweber/localfiles/itb_experiments/learning_grids',
				date_dir, 'visuals')
try:
	os.mkdir(save_path)
except OSError:
	pass

plot_psps(tables, psps, save_path=save_path)
# # Note: interval should be <= 300, otherwise the videos are green
# animate_psps(tables, psps, 'animate_positions', 0.0, e2, interval=50, save_path=save_path)
# animate_psps(tables, psps, 'animate_output_rates', 0.0, 1e6, interval=50, save_path=save_path, take_weight_steps=True)

t2 = time.time()
tables.close_file()
plt.show()
print 'Plotting took %f seconds' % (t2 - t1)