# open the tablefile
from snep.configuration import config
# config['network_type'] = 'empty'
import snep.utils
# import utils
import matplotlib as mpl

mpl.use('Agg')
# import plotting
import animating
# import matplotlib.pyplot as plt
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
				 animation_function, start_time, end_time, step_factor=1,
				 save_path=False, interval=50, take_weight_steps=False):
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
		animation = animating.Animation(params, rawdata, start_time=start_time,
										end_time=end_time,
										step_factor=step_factor,
										take_weight_steps=take_weight_steps)
		ani = getattr(animation, animation_function)
		if save_path:
			# remove the uno string
			save_path_full = os.path.join(save_path,
										  string.replace(str(psp), ' uno',
														 '') + '.mp4')
		else:
			save_path_full = False
		ani(save_path=save_path_full, interval=intervall)


def get_path_tables_psps(date_dir):
	path = general_utils.snep_plotting.get_path_to_hdf_file(date_dir)
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
t1 = 1e7
t2 = 2e7
# t2 = 2e7
# t3 = 1e8
t3 = -1
t_hm = 5e4
# t = 16e7
t = 18e5 * 5
t_switch_experiments = 36e5
# t4 = 24e6
# t2 = 40e5
# t1 = 120e6
# t1 = 80e6
# t1 = 100e6
# t=10e7
method = None
type = 'exc'
# t2 = 1e7
# Neurons for conjunctive and grid cells
# neurons = [23223, 51203, 35316, 23233]
# Neurons for head direction cell
# neurons =[5212, 9845, 9885, 6212]
plot_individual = False
gridscore_norm = 'all_neighbor_pairs'
colorbar_range = np.array([0, 1])
# gridscore_norm = None
t_reference = 4.9 * 36e4
t_compare = 8 * 36e4
conversion_factor = 36e3
function_kwargs = [
	##########################################################################
	##############################   New Plots  ##############################
	##########################################################################
	('correlation_in_regions', dict(region_size=)),
	# ('grid_score_histogram', dict(end_frame=-1,type='hexagonal',
	# 							  methods=['sargolini'],
	# 							  n_cumulative=[1],
	# 							  from_computed_full=True)),
	# ('grid_axes_angles_histogram', dict(end_frame=-1, from_computed_full=True,
	# 									minimum_grid_score=0.7)),
	# ('peak_locations', dict(time=-1, minimum_grid_score=0.7)),
	# ('output_rate_heat_map', {'from_file': True, 'end_time': 1e6,
	# 						  'publishable': True}),
	# ('weight_evolution', dict(number_of_synapses=200)),
	# ('weight_evolution', dict(number_of_synapses=200, syn_type='inh', title=False))
	# ('plot_correlogram', {'time': 0, 'from_file': True, 'method': 'sargolini',
	# 					  'mode': 'same',
	# 					  'show_grid_axes': False,
	# 					  'correlogram_of': None}),
	# ('spikemap_from_ratemap',
	#  dict(frame=0, n=2000)),
	# ('plot_correlogram', {'time': -1, 'from_file': True, 'method': 'sargolini',
	# 					  'mode': 'same',
	# 					  'show_grid_axes': False,
	# 					  'correlogram_of': None}),
	# ('plot_correlogram', {'time': -1, 'from_file': True, 'method': 'sargolini',
	# 					  'mode': 'same',
	# 					  'show_grid_axes': False,
	# 					  'correlogram_of': None}),
	# ('plot_correlogram', {'time': -1, 'from_file': True, 'method': 'langston',
	# 					  'mode': 'same',
	# 					  'show_grid_axes': False,
	# 					  'correlogram_of': None}),
	# ('spikemap_from_ratemap',
	# 	dict(frame=-1, n=300, noise=0.04, gridscore_norm=gridscore_norm,
	# 			 colorbar_range=colorbar_range)),
	# ('spikemap_from_ratemap',
	# 	dict(frame=-1, n=1000, noise=0.0, gridscore_norm=gridscore_norm,
	# 			 colorbar_range=colorbar_range,
	# 		 std_threshold=0.52359877559829882)),
	# ('spikemap_from_ratemap',
	#  dict(frame=-1, n=1000, noise=0.0, gridscore_norm=None,
	# 	  colorbar_range=colorbar_range,
	# 	  std_threshold=0.52359877559829882)),
	# ('spikemap_from_ratemap',
	# 	dict(frame=-1, n=1000, noise=0.05, gridscore_norm=gridscore_norm,
	# 	  colorbar_range=colorbar_range,
	# 		 std_threshold=0.52359877559829882)),
	# ('spikemap_from_ratemap',
	#  dict(frame=-1, n=1000, noise=0.05, gridscore_norm=None,
	# 	  colorbar_range=colorbar_range,
	# 	  std_threshold=0.52359877559829882)),
	# ('spikemap_from_ratemap',
	#  dict(frame=-1, n=1000, noise=0.10, gridscore_norm=gridscore_norm,
	# 	  colorbar_range=colorbar_range,
	# 	  std_threshold=0.52359877559829882)),
	# ('spikemap_from_ratemap',
	#  dict(frame=-1, n=1000, noise=0.10, gridscore_norm=None,
	# 	  colorbar_range=colorbar_range,
	# 	  std_threshold=0.52359877559829882)),
	# ('trajectory_with_firing',
	#  	dict(
	# 		# start_frame=51e4, end_frame=54e4,
	# 		start_frame=0, end_frame=800,
	# 		firing_indicator='spikes',
	# )),
	# ('trajectory_with_firing',
	#  dict(
	# 	 # start_frame=51e4, end_frame=54e4,
	# 	 start_frame=9e4, end_frame=12e4,
	# 	 firing_indicator='spikes',
	#  )),
	# ('trajectory_with_firing',
	#  dict(
	# 	 # start_frame=51e4, end_frame=54e4,
	# 	 start_frame=18e4, end_frame=21e4,
	# 	 firing_indicator='spikes',
	#  )),
	# ('trajectory_with_firing',
	#  dict(
	# 	 # start_frame=51e4, end_frame=54e4,
	# 	 start_frame=51e4, end_frame=54e4,
	# 	 firing_indicator='spikes',
	#  )),
	# ('spikemap_from_ratemap',
	#  dict(frame=-1, n=1500, noise=0.0, gridscore_norm=None,
	# 	  colorbar_range=colorbar_range, std_threshold=None)),
	# ('spikemap_from_ratemap',
	#  dict(frame=-1, n=1500, noise=0.0, gridscore_norm=gridscore_norm,
	# 	  colorbar_range=colorbar_range, std_threshold=None)),
	# ('spikemap_from_ratemap',
	#  dict(frame=-1, n=1500, noise=0.0, gridscore_norm=None,
	# 	  colorbar_range=colorbar_range, std_threshold=0.52359877559829882)),
	# ('spikemap_from_ratemap',
	# 	dict(frame=-1, n=1500, noise=0.0, gridscore_norm=gridscore_norm,
	# 	  colorbar_range=colorbar_range, std_threshold=0.52359877559829882)),
	# ('gridscore_vs_nspikes', dict(frame=-1,
	# 							  nspikes=np.array([50, 100, 200, 400, 800]),
	# 							  noises=np.linspace(0.01, 0.01001, 25),
	# 							  gridscore_norm=None,
	# 							  legend=False)),
	# ('gridscore_vs_nspikes', dict(frame=-1,
	# 							  nspikes=np.array([50, 100, 200, 400, 800]),
	# 							  noises=np.linspace(0.05, 0.04001, 25),
	# 							  gridscore_norm=None,
	# 							  legend=False)),
	# ('gridscore_vs_nspikes', dict(frame=-1,
	# 						nspikes=np.array([50, 100, 200, 400, 800]),
	# 						noises=np.linspace(0.01, 0.01001, 25),
	# 						gridscore_norm='all_neighbor_pairs',
	# 							  legend=False)),
	# ('gridscore_vs_nspikes', dict(frame=-1,
	# 							  nspikes=np.array([50, 100, 200, 400, 800]),
	# 							  noises=np.linspace(0.05, 0.04001, 25),
	# 							  gridscore_norm='all_neighbor_pairs',
	# 							  legend=False)),
	# ('spikemap_from_ratemap',
	# 	 	dict(frame=-1, n=1000, noise=0.5)),
	# ('spikemap_from_ratemap',
	#  		dict(frame=-1, n=400)),
	# ('spikemap_from_ratemap',
	# 		dict(frame=-1, n=800)),
	# ('distance_histogram_from_ratemap',
	#  			dict(frame=-1, n=2000, neighborhood_size=0.1,
	# 				 cut_off_position=0.1)),
	# ('input_current',
	# 			dict(time=-1, populations=['exc', 'inh'],
	# 				 from_file=True)),
	# ('input_current',
	# 			dict(time=-1, populations=['inh'],
	# 				 from_file=True)),
	# ('plot_correlogram', {'time': -1, 'from_file': True, 'method': 'sargolini',
	# 					  'mode': 'same',
	# 					  'show_grid_axes': True,
	# 					  'correlogram_of': None}),
	# ('plot_head_direction_polar', dict(time=-1, from_file=True,
	# 								   show_watson_U2=False,
	# 								   hd_tuning_title=True)),
	# ('output_rate_heat_map',
	# 					{'from_file': True, 'end_time': 12e4,
	# 					'publishable': False})
	# ('plot_correlogram', {'time': -1, 'from_file': True, 'method': None,
	# 					  'mode': 'same',
	# 					  'show_grid_axes': False,
	# 					  'correlogram_of': 'input_current_exc'}),
	# ('plot_correlogram', {'time': -1, 'from_file': True, 'method': None,
	# 					  'mode': 'same',
	# 					  'show_grid_axes': False,
	# 					  'correlogram_of': 'input_current_inh'}),
	# ('plot_correlogram', {'time': -1, 'from_file': True, 'method': 'sargolini',
	# 					  'mode': 'same',
	# 					  'show_grid_axes': False,
	# 					  'inner_square': False}),
	# ('input_current',
	# 			dict(time=-1, from_file=True, populations=['inh'],
	# 				 colormap='viridis')),

	# ('plot_output_rates_from_equation', dict(
	# 			time=29*conversion_factor, from_file=True,
	# 			spacing=51)),
	# ('plot_correlogram', {'time': 29*conversion_factor, 'from_file': True,
	# 					  'method': 'langston',
	# 					  'mode': 'same',
	# 					  'show_grid_axes': False,
	# 					  'inner_square': False}),
	# ('plot_output_rates_from_equation', dict(
	# 	time=82 * conversion_factor, from_file=True,
	# 	spacing=51)),
	# ('plot_correlogram', {'time': 82 * conversion_factor, 'from_file': True,
	# 					  'method': 'langston',
	# 					  'mode': 'same',
	# 					  'show_grid_axes': False,
	# 					  'inner_square': False}),

	# ('plot_output_rates_from_equation',
	#  dict(time=1 * 36e4, from_file=True, spacing=51)),
	# ('plot_output_rates_from_equation',
	#  dict(time=2 * 36e4, from_file=True, spacing=51)),
	# ('plot_output_rates_from_equation',
	#  dict(time=3 * 36e4, from_file=True, spacing=51)),
	# ('plot_output_rates_from_equation',
	#  dict(time=4 * 36e4, from_file=True, spacing=51)),
	# ('plot_output_rates_from_equation',
	#  dict(time=5 * 36e4, from_file=True, spacing=51)),
	# ('plot_output_rates_from_equation',
	#  dict(time=6 * 36e4, from_file=True, spacing=51)),
	# ('plot_output_rates_from_equation',
	#  dict(time=7 * 36e4, from_file=True, spacing=51)),
	# ('plot_output_rates_from_equation',
	#  dict(time=8 * 36e4, from_file=True, spacing=51)),
	# ('plot_output_rates_from_equation',
	#  dict(time=9 * 36e4, from_file=True, spacing=51)),
	# ('plot_output_rates_from_equation',
	#  dict(time=10 * 36e4, from_file=True, spacing=51)),

	# ('plot_correlogram',
	#  dict(time=0 * 36e4, from_file=True, spacing=51, method='langston',
	# 	  mode='same')),
	# ('plot_correlogram',
	#  dict(time=1 * 36e4, from_file=True, spacing=51, method='langston',
	# 	  mode='same')),
	# ('plot_correlogram',
	#  dict(time=2 * 36e4, from_file=True, spacing=51, method='langston',
	# 	  mode='same')),
	# ('plot_correlogram',
	#  dict(time=3 * 36e4, from_file=True, spacing=51, method='langston',
	# 	  mode='same')),
	# ('plot_output_rates_from_equation',
	#  dict(time=4 * 36e4, from_file=True, spacing=51)),
	# ('plot_output_rates_from_equation',
	#  dict(time=5 * 36e4, from_file=True, spacing=51)),
	# ('plot_output_rates_from_equation',
	#  dict(time=6 * 36e4, from_file=True, spacing=51)),
	# ('plot_correlogram',
	#  dict(time=7 * 36e4, from_file=True, spacing=51, method='langston',
	# 	  mode='same')),
	# ('plot_correlogram',
	#  dict(time=8 * 36e4, from_file=True, spacing=51, method='langston',
	# 	  mode='same')),
	# ('plot_correlogram',
	#  dict(time=4.9 * 36e4, from_file=True, spacing=51, method='langston',
	# 	  mode='same')),

	# ('plot_output_rates_from_equation',
	#  dict(time=t_reference, from_file=True, spacing=51)),
	# ('plot_output_rates_from_equation',
	#  dict(time=t_compare, from_file=True, spacing=51)),
	#
	# ('plot_time_evolution', dict(observable='grid_score', data=True,
	# 							 vlines=[18e5, t_compare])),
	# ('time_evolution_of_grid_correlation', dict(t_reference=t_reference,
	# 											vlines=[t_compare])),


	# ('input_tuning', dict(populations=['exc'], neuron=0)),
	# ('input_tuning', dict(populations=['inh'], neuron=0)),
	# ('input_tuning', dict(populations=['exc'], neuron=1)),
	# ('input_tuning', dict(populations=['inh'], neuron=1)),
	# ('input_tuning', dict(populations=['exc'], neuron=2)),
	# ('input_tuning', dict(populations=['inh'], neuron=2)),


	# ('plot_head_direction_polar', dict(time=0, from_file=True,
	# 								   show_watson_U2=True)),
	# ('plot_head_direction_polar', dict(time=-1, from_file=True,
	# 								   show_watson_U2=True)),
	# ('input_tuning', dict(populations=['exc'], neuron=0, subdimension='space')),
	# ('input_tuning', dict(populations=['exc'], neuron=1, subdimension='space')),
	# ('input_tuning', dict(populations=['exc'], neuron=2, subdimension='space')),
	# ('input_tuning', dict(populations=['inh'], neuron=0, subdimension='space')),
	# ('input_tuning', dict(populations=['inh'], neuron=1, subdimension='space')),
	# ('input_tuning', dict(populations=['inh'], neuron=2, subdimension='space')),
	# ('input_tuning_polar', dict(populations=['exc'], neuron=0)),
	# ('input_tuning_polar', dict(populations=['exc'], neuron=1)),
	# ('input_tuning_polar', dict(populations=['exc'], neuron=2)),
	# ('input_tuning_polar', dict(populations=['inh'], neuron=0)),
	# ('input_tuning_polar', dict(populations=['inh'], neuron=1)),
	# ('input_tuning_polar', dict(populations=['inh'], neuron=2)),
	# ('input_tuning', dict(populations=['exc'], neuron=1, subdimension='head_direction')),
	# ('input_tuning', dict(populations=['exc'], neuron=2, subdimension='head_direction')),
	# ('input_tuning', dict(populations=['inh'], neuron=0, subdimension='head_direction')),
	# ('input_tuning', dict(populations=['inh'], neuron=1, subdimension='head_direction')),
	# ('input_tuning', dict(populations=['inh'], neuron=2, subdimension='head_direction')),
	# ('input_tuning_extrema_distribution', {}),
	# ('trajectory_with_firing', {'start_frame': 0, 'end_frame': 3e4}),
	# ('trajectory_with_firing', {'start_frame': 0, 'end_frame': 9e4}),
	# ('trajectory_with_firing', {'start_frame': 0, 'end_frame': 18e4}),
	# ('trajectory_with_firing', {'start_frame': 36e4, 'end_frame': 54e4}),
	# ('trajectory_with_firing', {'start_frame': 0, 'end_frame':18e4}),
	# ('trajectory_with_firing', {'start_frame': 0, 'end_frame':3e4*6}),
	# ('mean_grid_score_time_evolution', dict(methods=['sargolini'],
	# 										n_cumulative=['1'])),
	# ('grid_score_histogram', dict(type='hexagonal',
	# 							  methods=['sargolini'],
	# 							  n_cumulative=[1],
	# 							  from_computed_full=False)),
	# ('grid_score_histogram', dict(type='quadratic',
	# 							  methods=['Weber', 'sargolini', 'sargolinis_extended'])),

	# ('grid_score_evolution_and_histogram', dict(type='hexagonal',
	# 											# end_frame=-1,
	# 											methods=['sargolini'],
	# 											n_cumulative=[1, 3],
	# 											from_computed_full=True,
	# 											)),

	# ('mean_output_rate_time_evolution', {}),

	# ('grid_score_time_correlation', {}),
	# ('mean_grid_score_time_evolution', dict(end_frame=200,
	# 										n_individual_plots=1,
	# 										methods=['sargolini'],
	# 										# figsize=(12, 5),
	# 										type='hexagonal',
	# 										row_index=0)),
	# ('mean_grid_score_time_evolution', dict(end_frame=200,
	# 										n_individual_plots=1,
	# 										methods=['sargolini'],
	# 										# figsize=(12, 5),
	# 										type='quadratic',
	# 										row_index=1)),
	# ('grid_score_hexagonal_and_quadratic', {})

	# ('plot_correlogram', {'time': 0, 'from_file': True, 'method': None,
	# 					  'mode': 'same', 'publishable': False}),
	# ('plot_correlogram', {'time': 158e4, 'from_file': True, 'method': 'sargolini',
	# 					  'mode': 'same', 'publishable': False, 'n_cumulative': 10}),
	# ('plot_correlogram', dict(time=158e4, from_file=True, method='sargolini',
	# 						  mode='same', publishable=False, n_cumulative=10,
	# 						  type='quadratic')),
	#

	# ('plot_correlogram', dict(time=77e4, from_file=True, method='sargolini',
	# 						  mode='same', publishable=False, n_cumulative=10,
	# 						  type='quadratic')),
	#
	# ('plot_correlogram', {'time': 2.99e6, 'from_file': True, 'method': 'sargolini',
	# 					  'mode': 'same', 'publishable': False, 'n_cumulative': 10}),
	# ('plot_correlogram', dict(time=2.99e6, from_file=True, method='sargolini',
	# 						  mode='same', publishable=False, n_cumulative=10,
	# 						  type='quadratic')),
	# ('grid_score_vs_time', {'t_start': 0, 't_end': 3e4, 'method': 'sargolini_extended',
	# 						'plot_individual': plot_individual, 'n_cumulative': 10}),
	# ('grid_score_vs_time', {'t_start': 0, 't_end': 3e4, 'method': 'sargolini',
	# 						'plot_individual': plot_individual}),
	# ('grid_score_vs_time', {'t_start': 0, 't_end': 3e4, 'method': 'sargolini_extended',
	# 						'plot_individual': plot_individual}),
	# 						'n_cumulative': 10}),
	# ('plot_correlogram', {'time': 0, 'from_file': True, 'method': 'sargolini',
	# 					  'mode': 'same', 'publishable': False, 'n_cumulative': 1}),
	# ('plot_correlogram', {'time': 1e6, 'from_file': True, 'method': 'sargolini',
	# 					  'mode': 'same', 'publishable': False, 'n_cumulative': 1}),
	# ('plot_correlogram', {'time': 10e6, 'from_file': True, 'method': 'sargolini',
	# 					  'mode': 'same', 'publishable': False, 'n_cumulative': 1}),
	# ('plot_correlogram', {'time': 24e6, 'from_file': True, 'method': 'sargolini',
	# 					  'mode': 'same', 'publishable': False, 'n_cumulative': 1}),
	# ('plot_time_evolution', {'observable': 'grid_score', 'method': 'Weber',
	# 						 'data': True}),
	# ('plot_time_evolution', {'observable': 'grid_score', 'method': 'sargolini',
	# 						 'data': True}),
	# ('plot_time_evolution', {'observable': 'grid_score', 'method': 'sargolini_extended',
	# 						 'data': True}),

	# ('plot_correlogram', {'time': 3e4, 'from_file': True, 'method': 'sargolini_extended',
	# 				  'mode': 'same', 'publishable': False, 'n_cumulative': 10}),
	# ('plot_correlogram', {'time': 1.2e7, 'from_file': True, 'method': 'sargolini_extended',
	# 					  'mode': 'same', 'publishable': False}),
	# ('plot_output_rates_from_equation', {'time': -1, 'from_file': True,
	# 									 'maximal_rate': False,
	# 									 'subdimension': 'space',
	# 									 'publishable': True}),
	# ('fields_polar', {'syn_type': 'exc', 'neuron': 1234, 'publishable': True}),
	# ('plot_head_direction_polar', {'time': -1, 'from_file': True,
	# 							   'publishable': True}),
	# ('plot_correlogram', {'time': 1e7, 'from_file': True, 'method': method, 'publishable': False}),
	# ('plot_output_rates_from_equation', {'time': 0e4, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 1e6, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 10e6, 'from_file': True}),
	# ('plot_output_rates_from_equation', {'time': 24e6, 'from_file': True}),

	# ('plot_correlogram', {'time': 2e7, 'from_file': True, 'mode': 'full', 'method': method, 'publishable': False}),

	# ('input_tuning', {'neuron': 0, 'populations': ['exc']}),
	# ('input_tuning', {'neuron': 0, 'populations': ['inh']}),
	# ('input_tuning', {'neuron': 1, 'populations': ['exc']}),
	# ('input_tuning', {'neuron': 1, 'populations': ['inh']}),
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


	# # This is the good one
	# ('plot_grid_spacing_vs_parameter',
	# 		{	'from_file': True,
	# 			'varied_parameter': ('inh', 'sigma'),
	# 			'parameter_range': np.linspace(0.08, 0.36, 201),
	# 			# 'parameter_range': np.linspace(0.08, 0.36, 201),
	# 			# 'parameter_range': np.linspace(0.015, 0.055, 200),
	# 			'plot_mean_inter_peak_distance': False,
	# 			'computed_data': False}),

	# ('plot_correlogram', dict(time=4e7, mode='same', from_file=True, xlim=1.0)),
	# ('mean_correlogram', {}),


	# ('trajectory_with_firing', {'start_frame': 0.0, 'end_frame':0,
	# 							'symbol_size': 20, 'colormap': 'inferno'}),
	# ('trajectory_with_firing', {'start_frame': 0.0, 'end_frame':2100,
	# 							'symbol_size': 20, 'colormap': 'inferno'}),
	# ('trajectory_with_firing', {'start_frame': 0.0, 'end_frame':5900,
	# 							'symbol_size': 20, 'colormap': 'inferno'}),
	# ('trajectory_with_firing', {'start_frame': 0.0, 'end_frame':6100,
	# 							'symbol_size': 20, 'colormap': 'inferno'}),
	# ('trajectory_with_firing', {'start_frame': 0.0, 'end_frame':1.5e5,
	# 							'symbol_size': 20, 'colormap': 'inferno'}),
	# ('trajectory_with_firing', {'start_frame': 0.0, 'end_frame':1.5e5,
	# 							'symbol_size': 20, 'colormap': 'plasma'}),
	# ('trajectory_with_firing', {'start_frame': 0.0, 'end_frame':1.5e5,
	# 							'symbol_size': 20, 'colormap': 'magma'}),
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
]

if __name__ == '__main__':
	t1 = time.time()

	# for date_dir in ['2016-07-27-17h22m04s_1d_grf_grid_cell']:
	for date_dir in [
					# '2016-12-07-13h59m36s_different_centers_different_trajectories',
					# '2016-06-29-17h09m25s_10_conjunctive_cells',
					# '2017-04-28-12h36m43s_20_conjunctive_cells',
					# '2017-05-02-11h20m28s_20_conjunctive_cells_less_angular_noise',
			# '2017-09-01-11h38m37s_room_switch',
		# '2017-09-01-12h56m41s_room_switch',
		# '2017-09-01-14h52m33s_test',
		# '2017-09-05-14h41m29s_room_switch_1fps',
		# '2017-09-05-16h03m34s_room_switch_2fps',
		# '2017-09-05-16h04m15s_room_switch_4fps',
		'2017-10-16-13h59m03s_wernle_10_seeds'
			]:
		path, tables, psps = get_path_tables_psps(date_dir)
		save_path = False
		save_path = os.path.join(os.path.dirname(path), 'visuals')
		try:
			os.mkdir(save_path)
		except OSError:
			pass

		all_psps = psps
		# for seed in [0,1,2,3]:
		# for eta_factor in [0.2, 0.5, 1.0]:
		# 	for sigma_inh in [0.25, 0.20]:
		# for sigma in sigmaI_range:
		psps = [p for p in all_psps
				# if p[('sim', 'seed_centers')].quantity == 1
				# and  p[('sim', 'alpha_room2')].quantity == 1
				# and p[('exc', 'fields_per_synapse')].quantity == 1
				# if not (p[('sim', 'seed_centers')].quantity == 1 and p[('sim', 					'room_coherence_after_switch')].quantity == 0.25 and p[('exc', 					'fields_per_synapse')].quantity == 2)
				#
				# if p[('inh', 'weight_factor')].quantity < 1.025
				# if p[('sim', 'gaussian_process_rescale')].quantity == 'fixed_mean'
				# if general_utils.misc.approx_equal(p[('exc', 'eta')].quantity,
				# 								  eta_factor * 3e-5 / (2* 0.5 * 10. * 22),
				# 								   3e-5 / (2* 0.5 * 10. * 22) / 100.)
				]

		prefix = general_utils.plotting.get_prefix(function_kwargs)
		# prefix = 'eta_factor_{0}_sigma_inh_{1}'.format(eta_factor, sigma_inh)
		# prefix = 'seed_{0}'.format(seed)
		# prefix = 'extrema_distribution_sigma_{0}'.format(sigma)
		general_utils.snep_plotting.plot_psps(
			tables, psps, project_name='learning_grids', save_path=save_path,
			psps_in_same_figure=True, function_kwargs=function_kwargs,
			prefix=prefix, automatic_arrangement=True, file_type='png',
			dpi=300, figsize=(7, 5), transparent=True)

	# Note: interval should be <= 300, otherwise the videos are green
	# animate_psps(tables, psps, 'animate_positions', 0.0, 3e2, interval=50, save_path=save_path)
	# animate_psps(tables, psps, 'animate_output_rates', 0.0, 1e6, interval=50, save_path=save_path, take_weight_steps=True)

t2 = time.time()
# tables.close_file()
# plt.show()
print 'Plotting took % seconds' % (t2 - t1)
