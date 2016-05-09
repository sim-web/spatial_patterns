import math

import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import initialization
import general_utils
import general_utils.snep_plotting
import general_utils.arrays
import general_utils.plotting
from general_utils.plotting import color_cycle_blue3
from general_utils.plotting import color_cycle_blue4
from general_utils.plotting import color_cycle_red3
import analytics.linear_stability_analysis
import utils
import observables
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
import itertools
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from general_utils.plotting import simpleaxis
from general_utils.plotting import adjust_spines
from matplotlib import gridspec
import figures.two_dim_input_tuning

# from matplotlib._cm import cubehelix
mpl.rcParams.update({'figure.autolayout': True})
font_size=12
mpl.rc('font', size=font_size)
mpl.rc('legend', fontsize=font_size)
mpl.rcParams['font.size'] = font_size
mpl.rcParams['axes.labelsize'] = font_size
mpl.rcParams['axes.linewidth'] = font_size / 16.
mpl.rcParams['axes.titlesize'] = font_size
mpl.rcParams['axes.linewidth'] = font_size / 16.
mpl.rcParams['lines.linewidth'] = font_size / 16.
mpl.rcParams['patch.linewidth'] = font_size / 16.
mpl.rcParams['legend.fontsize'] = font_size
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size


color_cycle = general_utils.plotting.color_cycle_qualitative10
plt.rc('axes', color_cycle=color_cycle)

# print mpl.rcParams.keys()
# mpl.rcParams['animation.frame_format'] = 'jpeg'
# print mpl.rcParams['animation.frame_format']

def make_segments(x, y):
	"""
	Taken from http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

	Create list of line segments from x and y coordinates, in the correct format for LineCollection:
	an array of the form   numlines x (points per line) x 2 (x and y) array
	"""

	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)

	return segments

def colorline(x, y, z=None, cmap=plt.get_cmap('gnuplot_r'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
	'''
	Taken from http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

	Plot a colored line with coordinates x and y
	Optionally specify colors in the array z
	Optionally specify a colormap, a norm function and a line width

	Defines a function colorline that draws a (multi-)colored 2D line with coordinates x and y.
	The color is taken from optional data in z, and creates a LineCollection.

	z can be:
	- empty, in which case a default coloring will be used based on the position along the input arrays
	- a single number, for a uniform color [this can also be accomplished with the usual plt.plot]
	- an array of the length of at least the same length as x, to color according to this data
	- an array of a smaller length, in which case the colors are repeated along the curve

	The function colorline returns the LineCollection created, which can be modified afterwards.

	See also: plt.streamplot

	'''

	# Default colors equally spaced on [0,1]:
	if z is None:
		z = np.linspace(0.0, 1.0, len(x))

	# Special case if a single number:
	if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
		z = np.array([z])

	z = np.asarray(z)

	segments = make_segments(x, y)
	lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

	ax = plt.gca()
	ax.add_collection(lc)

	return lc

def plot_inputs_rates_heatmap(plot_list):
	"""
	Plots input examples, initial firing rate, heat map, final firing rate

	This function also illustrates the usage of nested gridspecs and the
	separate plotting of a colorbar in order to have aligned x axes even
	if one of the figures is a heatmap with colorbar.
	Note: tight_layout does not work in the nested gridspecs, we therefore
		use slicing to arange the sizes instead of choosing height and
		width ratios.

	It is necessary to choose the appropriate plotting functions in plot.py
	in the right order.

	For example:
	('input_tuning', {'neuron': 0, 'populations': ['exc'], 'publishable':
		True}),
	('input_tuning', {'neuron': 53, 'populations': ['inh'], 'publishable':
		True}),
	('plot_output_rates_from_equation', {'time':  0, 'from_file': True,
										 'maximal_rate': False,
										 'publishable': True}),
	('output_rate_heat_map', {'from_file': True, 'end_time': 2e5,
							  'publishable': True}),
	('plot_output_rates_from_equation', {'time':  2e5, 'from_file': True,
									 'maximal_rate': False,
									 'publishable': True}),

	Parameters
	----------
	plot_list : see somewhere else
	"""
	# The meta grid spec (the distribute the two following grid specs
	# on a vertical array of length 5)
	gs0 = gridspec.GridSpec(4, 1)
	# Along the x axis we take the same number of array points for both
	# gridspecs in order to align the axes horizontally
	nx = 50
	# Room for colorbar
	n_cb = 3
	# The number of vertical array points can be chose differently for the
	# two inner grid specs and is used to adjust the vertical distance
	# between plots withing a gridspec
	ny = 102
	n_plots = 2 # Number of plots in hte the first gridspec
	# A 'sub' gridspec place on the first fifth of the meta gridspec
	gs00 = gridspec.GridSpecFromSubplotSpec(ny, nx, subplot_spec=gs0[0])
	# Excitatory Input
	plt.subplot(gs00[0:ny/n_plots-1, :-n_cb])
	plot_list[0]()
	# Inhibitory Input
	plt.subplot(gs00[1+ny/n_plots:, :-n_cb])
	plot_list[1]()
	# Now we choose a different number of vertical array points in the
	# gridspec, to allow for independent adjustment of vertical distances
	# within the two sub-gridspecs
	ny = 40
	gs01 = gridspec.GridSpecFromSubplotSpec(ny, nx, subplot_spec=gs0[1:])
	# Initial Rate
	plt.subplot(gs01[0:ny/8, :-n_cb])
	plot_list[2]()
	# Heat Map
	vrange = [5+ny/8, 7*ny/8-3]
	plt.subplot(gs01[vrange[0]:vrange[1], :-n_cb])
	plot_list[3]()
	# Final Rate
	plt.subplot(gs01[7*ny/8:, :-n_cb])
	plot_list[4]()
	# Colorbar
	# The colorbar is plotted right next to the heat map
	plt.subplot(gs01[vrange[0]:vrange[1], nx-2:])
	output_rates = plot_list[3]()
	vmin = 0.0
	vmax = np.amax(output_rates)
	ax1 = plt.gca()
	cm = mpl.cm.gnuplot_r
	norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
	cb = mpl.colorbar.ColorbarBase(ax1, cmap=cm, norm=norm, ticks=[int(vmin), int(vmax)])
	# Negative labelpad puts the label further inwards
	# For some reason labelpad needs to be manually adjustes for different
	# simulations. No idea why!
	# Take -7.0 (for general inputs) and -1.0 for ideal inputs
	cb.set_label('Hz', rotation='horizontal', labelpad=-7.0)
	fig = plt.gcf()
	fig.set_size_inches(2.2, 2.6)
	gs0.tight_layout(fig, rect=[0, 0, 1, 1], pad=0.2)

def plot_output_rates_and_gridspacing_vs_parameter(plot_list):
	"""
	Plots gridspacing vs. parameter together with two output rate examples

	Illustrates:
	Using gridspec with a combination of slicing AND height_ratios.

	NOTE, The order of function in plot.py should be something like that:

	('plot_output_rates_from_equation', {'time':  4e7, 'from_file': True,
										 'maximal_rate': False,
										 'publishable': True}),
	('plot_output_rates_from_equation', {'time':  4e7, 'from_file': True,
										 'maximal_rate': False,
										 'publishable': True}),
	 ('plot_grid_spacing_vs_parameter',
			{	'from_file': True,
				'parameter_name': 'sigma_inh',
				'parameter_range': np.linspace(0.08, 0.38, 201),
				# 'parameter_range': np.linspace(0.08, 0.36, 201),
				# 'parameter_range': np.linspace(0.015, 0.055, 200),
				'plot_mean_inter_peak_distance': True,
				'computed_data': False})
	"""
	gs = gridspec.GridSpec(2, 2, height_ratios=[5,1])
	# Output rates small gridspacing
	plt.subplot(gs[1, 0])
	plot_list[0](xlim=np.array([-1.0, 1.0]), selected_psp=0, sigma_inh_label=False)
	# Output rates large gridspacing
	plt.subplot(gs[1, 1])
	plot_list[1](xlim=np.array([-1.0, 1.0]), selected_psp=-1, no_ylabel=True, indicate_gridspacing=False)
	# Gridspacing vs parameter
	plt.subplot(gs[0, :])
	plot_list[2](sigma_corr=True)
	fig = plt.gcf()
	# fig.set_size_inches(2.4, 2.4)
	fig.set_size_inches(2.2, 2.0)
	gs.tight_layout(fig, rect=[0, 0, 1, 1], pad=0.2)

def plot_input_initrate_finalrate_correlogram(plot_list):
	"""
	Plots one line of input examples, initial rate, final rate, correlogram

	In plot.py you should have something like:
	('fields', {'neuron': 2960, 'show_each_field': False, 'show_sum': True,
				'populations': ['exc'], 'publishable': True}),
	('fields', {'neuron': 1300, 'show_each_field': False, 'show_sum': True,
				'populations': ['exc'], 'publishable': True}),
	('fields', {'neuron': 1510, 'show_each_field': False, 'show_sum': True,
				'populations': ['inh'], 'publishable': True}),
	('fields', {'neuron': 2270, 'show_each_field': False, 'show_sum': True,
				'populations': ['inh'], 'publishable': True}),
	('plot_output_rates_from_equation', {'time': 0, 'from_file': True,
								'maximal_rate': False, 'publishable': True}),
	('plot_output_rates_from_equation', {'time': t1, 'from_file': True,
	 							'maximal_rate': False, 'publishable': True}),
	('plot_correlogram', {'time': t1, 'from_file': True,
	 				'mode': 'same', 'method': method, 'publishable': True}),
	"""
	# Grid Spec for: inputs, init rates, final rates, correlogram
	gs = gridspec.GridSpec(1, 4)
	# Input
	# Subgridspec for the inputs (2 example for the 2 synapse types)
	# Each input example is set within this sub-gridspec gs1
	gs1 = gridspec.GridSpecFromSubplotSpec(2,2,gs[0], wspace=0.0, hspace=0.1)
	# Excitation
	plt.subplot(gs1[0])
	plot_list[0]()
	plt.subplot(gs1[2])
	plot_list[1]()
	# Inhibition
	plt.subplot(gs1[1])
	plot_list[2]()
	plt.subplot(gs1[3])
	plot_list[3]()
	# Rate maps
	plt.subplot(gs[1])
	plot_list[4]()
	plt.subplot(gs[2])
	plot_list[5]()
	# Correlogram
	plt.subplot(gs[3])
	plot_list[6]()
	# It's crucial that the figure is not too high, because then the smaller
	# squares move to the top and bottom. It is a bit trick to work with
	# equal aspect ratio in a gridspec
	fig = plt.gcf()
	fig.set_size_inches(1.1*5.0, 1.1*1)
	gs.tight_layout(fig, pad=0.2, w_pad=0.0)

def plot_input_initrate_correlogram_finalrate_correlogram(plot_list):
	"""
	Plots one line of input examples, initial rate, final rate, correlogram

	In plot.py you should have something like:
	('fields', {'neuron': 2960, 'show_each_field': False, 'show_sum': True,
				'populations': ['exc'], 'publishable': True}),
	('fields', {'neuron': 1300, 'show_each_field': False, 'show_sum': True,
				'populations': ['exc'], 'publishable': True}),
	('fields', {'neuron': 1510, 'show_each_field': False, 'show_sum': True,
				'populations': ['inh'], 'publishable': True}),
	('fields', {'neuron': 2270, 'show_each_field': False, 'show_sum': True,
				'populations': ['inh'], 'publishable': True}),
	('plot_output_rates_from_equation', {'time': 0, 'from_file': True,
								'maximal_rate': False, 'publishable': True}),
	('plot_correlogram', {'time': 0, 'from_file': True,
	 				'mode': 'same', 'method': method, 'publishable': True}),
	('plot_output_rates_from_equation', {'time': t1, 'from_file': True,
	 							'maximal_rate': False, 'publishable': True}),
	('plot_correlogram', {'time': t1, 'from_file': True,
	 				'mode': 'same', 'method': method, 'publishable': True}),
	"""
	# Grid Spec for: inputs, init rates, final rates, correlogram
	gs = gridspec.GridSpec(1, 5)
	# Input
	# Subgridspec for the inputs (2 example for the 2 synapse types)
	# Each input example is set within this sub-gridspec gs1
	gs1 = gridspec.GridSpecFromSubplotSpec(2,2,gs[0], wspace=0.0, hspace=0.1)
	# Excitation
	plt.subplot(gs1[0])
	plot_list[0]()
	plt.subplot(gs1[2])
	plot_list[1]()
	# Inhibition
	plt.subplot(gs1[1])
	plot_list[2]()
	plt.subplot(gs1[3])
	plot_list[3]()
	# Rate maps
	plt.subplot(gs[1])
	plot_list[4]()
	# Correlogram
	plt.subplot(gs[2])
	plot_list[5]()
	# Rate maps
	plt.subplot(gs[3])
	plot_list[6]()
	# Correlogram
	plt.subplot(gs[4])
	plot_list[7]()
	# It's crucial that the figure is not too high, because then the smaller
	# squares move to the top and bottom. It is a bit trick to work with
	# equal aspect ratio in a gridspec
	fig = plt.gcf()
	fig.set_size_inches(1.1*6.25, 1.1*1)
	gs.tight_layout(fig, pad=0.2, w_pad=0.0)

def plot_input_rate_correlogram_hd_tuning(plot_list):
	"""
	Plots for combined spatial and head direction tuning (3 dimensions)

	Selected plotting functions in plot.py should look something like this:

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
	"""
	mpl.rc('grid', color='black', linestyle='solid', lw=0.5)
	gs = gridspec.GridSpec(1, 5)

	# Input examples: spatial tuning
	gs_spatial_input = gridspec.GridSpecFromSubplotSpec(2,2,gs[0], wspace=0.0, hspace=0.1)
	# Excitation
	plt.subplot(gs_spatial_input[0])
	plot_list[0]()
	plt.subplot(gs_spatial_input[2])
	plot_list[1]()
	# Inhibition
	plt.subplot(gs_spatial_input[1])
	plot_list[2]()
	plt.subplot(gs_spatial_input[3])
	plot_list[3]()
	# Input examples: head direction tuning
	gs_hd_iput = gridspec.GridSpecFromSubplotSpec(2,2,gs[1], wspace=0.0, hspace=0.1)
	plt.subplot(gs_hd_iput[0], polar=True)
	plot_list[4]()
	plt.subplot(gs_hd_iput[2], polar=True)
	plot_list[5]()
	plt.subplot(gs_hd_iput[1], polar=True)
	plot_list[6]()
	plt.subplot(gs_hd_iput[3], polar=True)
	plot_list[7]()
	# Rate map
	plt.subplot(gs[2])
	plot_list[8]()
	# Correlogram
	plt.subplot(gs[3])
	plot_list[9]()
	# Head direction tuning
	plt.subplot(gs[4], polar=True)
	plot_list[10]()
	fig = plt.gcf()
	fig.set_size_inches(1.1*6, 1.1*1)
	gs.tight_layout(fig, pad=0.2, w_pad=0.0)



def plot_list(fig, plot_list, automatic_arrangement=True):
	"""
	Takes a list of lambda forms of plot functions and plots them such that
	no more than four rows are used to assure readability
	"""
	n_plots = len(plot_list)
	# A title for the entire figure (super title)
	# fig.suptitle('Time evolution of firing rates', y=1.1)
	if automatic_arrangement:
		for n, p in enumerate(plot_list, start=1):
				# Check if function name contains 'polar'
				# is needed for the sublotting
				if 'polar' in str(p.func):
					polar = True
				else:
					polar = False
				if n_plots < 4:
					fig.add_subplot(n_plots, 1, n, polar=polar)
					# plt.locator_params(axis='y', nbins=4)
					# plt.ylabel('firing rate')
				else:
					fig.add_subplot(math.ceil(n_plots/2.), 2, n, polar=polar)
					# plt.locator_params(axis='y', nbins=4)
				p()

	else:
		# plot_inputs_rates_heatmap(plot_list=plot_list)
		# plot_output_rates_and_gridspacing_vs_parameter(plot_list=plot_list)
		# plot_input_initrate_finalrate_correlogram(plot_list)
		plot_input_initrate_correlogram_finalrate_correlogram(plot_list)
		# plot_input_rate_correlogram_hd_tuning(plot_list)

def set_current_output_rate(self):
	"""
	Sums exc_weights * exc_rates and substracts inh_weights * inh_rates
	"""
	rate = (
		np.dot(self.exc_syns.weights, self.exc_syns.rates) -
		np.dot(self.inh_syns.weights, self.inh_syns.rates)
	)
	self.output_rate = utils.rectify(rate)

def set_current_input_rates(self):
	"""
	Set the rates of the input neurons by using their place fields
	"""
	self.exc_syns.set_rates(self.x)
	self.inh_syns.set_rates(self.x)

class Plot(utils.Utilities,
			general_utils.snep_plotting.Snep):
	"""Class with methods related to plotting

	Parameters
	----------
	tables : snep tables object
	psps : list of paramspace points
	params, rawdata : see general_utils.snep_plotting.Snep
	"""

	def __init__(self, tables=None, psps=[None], params=None, rawdata=None,
				 latex=False):
		if latex:
			mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
			mpl.rc('text', usetex=True)
		general_utils.snep_plotting.Snep.__init__(self, params, rawdata)
		self.tables = tables
		self.psps = psps
		# self.params = params
		# self.rawdata = rawdata
		# for k, v in params['sim'].items():
		# 	setattr(self, k, v)
		# for k, v in params['out'].items():
		# 	setattr(self, k, v)
		# for k, v in rawdata.items():
		# 	setattr(self, k, v)
		self.color_cycle_blue3 = general_utils.plotting.color_cycle_blue3
		# self.box_linspace = np.linspace(-self.radius, self.radius, 200)
		# self.time = np.arange(0, self.simulation_time + self.dt, self.dt)
		self.colors = {'exc': '#D7191C', 'inh': '#2C7BB6'}
		self.population_name = {'exc': r'excitatory', 'inh': 'inhibitory'}
		self.populations = ['exc', 'inh']
		# self.fig = plt.figure()
		self.cms = {'exc': mpl.cm.Reds, 'inh': mpl.cm.Blues}
		if tables:
			self.computed_full = self.tables.get_computed(None)

	def time2frame(self, time, weight=False):
		"""Returns corresponding frame number to a given time

		Parameters
		----------
		- time: (float) time in the simulation
		- weight: (bool) decides wether every_nth_step or
					every_nth_step_weights is taken

		Returns
		(int) the frame number corresponding to the time
		-------

		"""

		if weight:
			every_nth_step = self.every_nth_step_weights
		else:
			every_nth_step = self.every_nth_step

		if time == -1:
			time = self.params['sim']['simulation_time']

		frame = time / every_nth_step / self.dt
		return int(frame)

	def frame2time(self, frame, weight=False):
		"""
		Inverse of time2frame

		Parameters
		----------
		frame : int
			Frame number
		weight : bool
			See time2frame()

		Returns
		-------
		time : float
			The time corresponding to the given frame
		"""
		if weight:
			every_nth_step = self.every_nth_step_weights
		else:
			every_nth_step = self.every_nth_step

		if frame == -1:
			time = self.params['sim']['simulation_time']

		time = frame * every_nth_step * self.dt
		return time


	def trajectory_with_firing(self, start_frame=0, end_frame=None,
				  firing_indicator='color_map', small_dt=None,
				  symbol_size=8, show_title=True, colormap='viridis',
							   max_rate_for_colormap=6.0):
		"""
		Plots trajectory with firing rates at each location

		NOTE: Currently you treat frame and time as the same, because
		in all your data you created output at each time step of 1.0.
		Be careful with other scenarios.

		Parameters
		----------
		firing_indicator : str
			'color_map': The trajectory is a scatterplot where color
						indicates the output firing rate
			'spikes': The trajectory is a black line with red dots on top
				which drawn from an inhomogeneous Poisson process whose
				rate depens on the output firing rate
		small_dt : float
			Determines the rate in the Poisson process if `firing_indicator`
			is set to 'spikes'. The larger the value, the higher the
			spike density.
		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			positions = self.rawdata['positions']
			x_positions = positions[start_frame:end_frame,0]
			y_positions = positions[start_frame:end_frame,1]
			output_rates = self.rawdata['output_rates'][start_frame:end_frame]
			if not max_rate_for_colormap:
				max_rate_for_colormap = np.amax(output_rates) * 0.5

			if firing_indicator == 'color_map':
				# cm = mpl.cm.gnuplot
				# cm = mpl.cm.viridis
				cm = getattr(mpl.cm, colormap)
				# cm = mpl.cm.jet
				color_norm = mpl.colors.Normalize(0., max_rate_for_colormap)
				# fig, ax = plt.subplots(1,1, figsize=(5,5))
				plt.scatter(
					x_positions,
					y_positions,
					c=output_rates,
					s=symbol_size, linewidths=0., edgecolors='none',
					norm=color_norm, cmap=cm, alpha=0.5)

			elif firing_indicator == 'spikes':
				plt.plot(
					positions[start_frame:end_frame,0],
					positions[start_frame:end_frame,1], color='black', linewidth=0.5)

				rates_x_y = np.nditer(
					[output_rates,
					x_positions,
					y_positions])
				for r, x, y in rates_x_y:
						# if r * small_dt > np.random.random():
						if r > 8:
							plt.plot(x, y, marker='o',
								linestyle='none', markeredgecolor='none',
								markersize=10, color='r')


			if show_title:
				# title = '%.1e to %.1e' % (start_frame, end_frame)
				time_in_minutes = int(end_frame / 3000.)
				title = 'time = {0} minutes'.format(time_in_minutes)
				plt.title(title, fontsize=12)
			ax = plt.gca()
			ax.set_xlim(-self.radius, self.radius)
			ax.set_ylim(-self.radius, self.radius)
			ax.set_aspect('equal')
			ax.set_xticks([])
			ax.set_yticks([])


	def plot_output_rates_via_walking(self, frame=0, spacing=201):
		"""
		DEPRECATED! Use get_output_rates_from_equation instead
		"""
		start_pos = -0.5
		end_pos = self.radius
		linspace = np.linspace(-self.radius, self.radius, spacing)
		# Initial equilibration
		equilibration_steps = 10000
		plt.xlim([-self.radius, self.radius])
		r = np.zeros(self.output_neurons)
		dt_tau = self.dt / self.tau
		# tau = 0.011
		# dt = 0.01
		# dt_tau = 0.1
		x = start_pos
		for s in np.arange(equilibration_steps):
			r = (
					r*(1 - dt_tau)
					+ dt_tau * ((
					np.dot(self.rawdata['exc']['weights'][frame],
						self.get_rates(x, 'exc')) -
					np.dot(self.rawdata['inh']['weights'][frame],
						self.get_rates(x, 'inh'))
					)
					- self.weight_lateral
					* (np.sum(r) - r)
					)
					)
			r[r<0] = 0
		start_r = r
		output_rate = []
		for x in linspace:
			r = (
					r*(1 - dt_tau)
					+ dt_tau * ((
					np.dot(self.rawdata['exc']['weights'][frame],
						self.get_rates(x, 'exc')) -
					np.dot(self.rawdata['inh']['weights'][frame],
						self.get_rates(x, 'inh'))
					)
					- self.weight_lateral
					* (np.sum(r) - r)
					)
					)
			r[r<0] = 0
			output_rate.append(r)
		# plt.title(start_r)
		plt.plot(linspace, output_rate)


	def rate1_vs_rate2(self, start_frame=0, three_dimensional=False, weight=0):
		target_rate = self.params['out']['target_rate']
		if three_dimensional:
			fig = plt.figure()
			ax = fig.gca(projection='3d')
			x = self.rawdata['output_rates'][start_frame:,0]
			y = self.rawdata['output_rates'][start_frame:,1]
			z = self.rawdata['inh']['weights'][start_frame:,weight,0]

			ax.plot(x, y, z)
			zlim = ax.get_zlim()
			# Plot line for target rate
			ax.plot([target_rate, target_rate],
					[target_rate, target_rate], zlim, lw=2, color='black')

			ax.set_xlabel('Rate of neuron 1')
			ax.set_ylabel('Rate of neuron 2')
			ax.set_zlabel('Weight of neuron %i' % weight)
			# ax.set_zlim(-10, 10)

			return

		else:
			plt.plot(target_rate, target_rate, marker='x', color='black', markersize=10, markeredgewidth=2)
			# plt.plot(
			# 	self.rawdata['output_rates'][start_frame:,0],
			# 	self.rawdata['output_rates'][start_frame:,1])
			x = self.rawdata['output_rates'][start_frame:,0]
			y = self.rawdata['output_rates'][start_frame:,1]
			colorline(x, y)
			# Using colorline it's necessary to set the limits again
			plt.xlim(x.min(), x.max())
			plt.ylim(y.min(), y.max())
			plt.xlabel('Output rate 1')
			plt.ylabel('Output rate 2')


		# ax = fig.gca(projection='rectilinear')


	def output_rate_vs_time(self, plot_mean=False, start_time_for_mean=0):
		"""Plot output rate of output neurons vs time

		Parameters
		----------
		- plot_mean: (boolian) If True the mean is plotted as horizontal line
		- start_time_for_mean: (float) The time from which on the mean is to
								be taken
		"""

		plt.xlabel('Time')
		plt.ylabel('Output rates')
		time = general_utils.arrays.take_every_nth(self.time, self.every_nth_step)
		plt.plot(time, self.rawdata['output_rates'])
		plt.axhline(self.target_rate, lw=4, ls='dashed', color='black',
					label='Target', zorder=3)
		if plot_mean:
			start_frame = self.time2frame(start_time_for_mean)
			mean = np.mean(self.rawdata['output_rates'][start_frame:], axis=0)
			legend = 'Mean:' + str(mean)
			plt.hlines(mean, xmin=start_time_for_mean, xmax=max(time), lw=4,
						color='red', label=legend, zorder=4)

		plt.legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize=8)

			# plt.axhline(mean[1], xmin=start_frame)

	# def output_rates_vs_position(self, start_frame=0, clipping=False):
	# 	if self.dimensions == 1:
	# 		_positions = self.positions[:,0][start_frame:,]
	# 		_output_rates = self.output_rates[start_frame:,]
	# 		plt.plot(_positions, _output_rates, linestyle='none', marker='o', alpha=0.5)
	# 	if self.dimensions == 2:
	# 		positions = self.positions[start_frame:,]
	# 		output_rates = self.output_rates[start_frame:,]
	# 		plt.xlim(-self.radius, self.radius)
	# 		plt.ylim(-self.radius, self.radius)
	# 		if clipping:
	# 			color_norm = mpl.colors.Normalize(0, np.amax(output_rates)/10000.0)
	# 		else:
	# 			color_norm = mpl.colors.Normalize(np.amin(output_rates), np.amax(output_rates))
	# 		for p, r in zip(positions, output_rates):
	# 			color = mpl.cm.YlOrRd(color_norm(r))
	# 			plt.plot(p[0], p[1], linestyle='none', marker='s', markeredgecolor='none', color=color, markersize=5, alpha=0.5)
	# 	# ax = plt.gca()
	# 	# ax.set_aspect('equal')
	# 	# ax.set_xticks([])
	# 	# ax.set_yticks([])

	def plot_sigmas_vs_centers(self):
		for t in ['exc', 'inh']:
			plt.plot(self.rawdata[t]['centers'], self.rawdata[t]['sigmas'],
				color=self.colors[t], marker='o', linestyle='none')

	def plot_sigma_distribution(self):
		if self.params['inh']['sigma_distribution'] == 'cut_off_gaussian':
			plt.xlim(0, self.params['inh']['sigma_spreading']['right'])
			for t in ['exc', 'inh']:
				plt.hist(self.rawdata[t]['sigmas'], bins=10, color=self.colors[t])
		else:
			# plt.xlim(0, )
			for t in ['exc', 'inh']:
				plt.hist(self.rawdata[t]['sigmas'], bins=10, color=self.colors[t])

	def get_rates(self, position, syn_type):
		"""
		Computes the values of all place field Gaussians at `position`

		Inherited from Synapses
		"""
		# Downward compatibility
		# DEPRECATED: Downward compatibility is now ensured in
		# snep_plotting.py
		if 'input_norm' not in self.params[syn_type]:
			self.params[syn_type]['input_norm'] = [1]

		get_rates = self.get_rates_function(position, data=self.rawdata[syn_type],
											params=self.params[syn_type])
		# return self.set_rates(position, data=self.rawdata[syn_type])
		return get_rates(position)


	def get_output_rate(self, position, frame):
		"""
		Note: if you want it for several times don't calculate set_rates every time, because it does not change!!!
		"""
		if self.lateral_inhibition:
			start_pos = -0.5
			end_pos = self.radius
			# Initial equilibration
			equilibration_steps = 10000
			plt.xlim([-self.radius, self.radius])
			r = np.zeros(self.output_neurons)
			dt_tau = self.dt / self.tau
			# tau = 0.011
			# dt = 0.01
			# dt_tau = 0.1
			x = start_pos
			for s in np.arange(equilibration_steps):
				r = (
						r*(1 - dt_tau)
						+ dt_tau * ((
						np.dot(self.rawdata['exc']['weights'][frame],
							self.get_rates(x, 'exc')) -
						np.dot(self.rawdata['inh']['weights'][frame],
							self.get_rates(x, 'inh'))
						)
						- self.weight_lateral
						* (np.sum(r) - r)
						)
						)
				r[r<0] = 0
			start_r = r
			output_rates = []
			for x in linspace:
				for s in np.arange(200):
					r = (
							r*(1 - dt_tau)
							+ dt_tau * ((
							np.dot(self.rawdata['exc']['weights'][frame],
								self.get_rates(x, 'exc')) -
							np.dot(self.rawdata['inh']['weights'][frame],
								self.get_rates(x, 'inh'))
							)
							- self.weight_lateral
							* (np.sum(r) - r)
							)
							)
					r[r<0] = 0
				output_rates.append(r)

		else:
			output_rate = (
				np.dot(self.rawdata['exc']['weights'][frame],
				 self.get_rates(position[0], 'exc'))
				- np.dot(self.rawdata['inh']['weights'][frame],
				 self.get_rates(position[0], 'inh'))
			)
		return output_rate

	def get_X_Y_positions_grid_input_rates_tuple(self, spacing):
		"""
		Returns X, Y meshgrid and position_grid and input_rates for contour plot

		RETURNS:
		- X, Y: meshgrids for contour plotting
		- positions_grid: array with all the positions in a matrix like shape:
			[
				[
					[x1, y1], [x1, y2]
				] ,
				[
					[x2, y1], [x2, y2]
				]
			]
		- input_rates: dictionary of two arrays, one exc and one inh.
				Following the matrix structure of positions_grid, each entry in this
				"matrix" (note: it is an array, not a np.matrix) is the array of
				firing rates of the neuron type at this position
		ISSUES:
		- Probably X, Y creation can be circumvented elegantly with the positions_grid
		- Since it is only used once per animation (it was created just for this purpose)
			it is low priority
		"""
		input_rates = {}
		# Set up X, Y for contour plot
		x_space = np.linspace(-self.radius, self.radius, spacing)
		y_space = np.linspace(-self.radius, self.radius, spacing)
		X, Y = np.meshgrid(x_space, y_space)
		positions_grid = np.dstack([X.T, Y.T])

		positions_grid.shape = (spacing, spacing, 1, 1, 2)
		# if self.boxtype == 'circular':
		# 	distance = np.sqrt(X*X + Y*Y)
		# 	positions_grid[distance>self.radius] = np.nan
		input_rates['exc'] = self.get_rates(positions_grid, 'exc')
		input_rates['inh'] = self.get_rates(positions_grid, 'inh')
		return X, Y, positions_grid, input_rates


	def output_rate_heat_map(self, start_time=0, end_time=-1, spacing=None,
			maximal_rate=False, number_of_different_colors=50,
			equilibration_steps=10000, from_file=False, publishable=False,
			return_output_rates=False, colormap='inferno'):
		"""Plot evolution of output rate from equation vs time

		Time is the vertical axis. Linear space is the horizontal axis.
		Output rate is given in color code.

		Parameters
		----------
		start_time, end_time : int
			Determine the time range
		spacing : int
			resolution along the horizontal axis
			Note: the resolution along the vertical axis is given by the data
		maximal_rate : float
			Above this value everything is plotted in black. This is useful
			if smaller values should appear in more detail. If left as False,
			the largest appearing value of all occurring output rates is taken.
		number_of_different_colors : int
			Number of colors used for the color coding
		return_output_rates : bool
			In 1d returns the output_rates instead of plotting anything
			(this is needed in plot_inputs_rates_heatmap)
		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			# frame = self.time2frame(time, weight=True)

			if spacing is None:
				spacing = self.spacing

			linspace = np.linspace(-self.radius , self.radius, spacing)
			# Get the output rates
			# output_rates = self.get_output_rates(frame, spacing, from_file)
			lateral_inhibition = self.params['sim']['lateral_inhibition']
			fig = plt.gcf()
			fig.set_size_inches(5.8, 3)
			# fig.set_size_inches(6, 3.5)
			first_frame = self.time2frame(start_time, weight=True)
			last_frame = self.time2frame(end_time, weight=True)
			output_rates = np.empty((last_frame-first_frame+1,
							spacing, self.params['sim']['output_neurons']))
			frames = np.arange(first_frame, last_frame+1)
			for i in frames:
				 output_rates[i-first_frame] = self.get_output_rates(
													i, spacing, from_file)
				 print 'frame: %i' % i
			time = frames * self.every_nth_step_weights
			X, Y = np.meshgrid(linspace, time)
			# color_norm = mpl.colors.Normalize(0., 50.)
			if not maximal_rate:
				# maximal_rate = int(np.ceil(np.amax(output_rates)))
				maximal_rate = np.amax(output_rates)
			V = np.linspace(0, maximal_rate, number_of_different_colors)
			plt.ylabel('Time')
			if return_output_rates:
				return output_rates[...,0]
			# plt.xlabel('Position')
			if lateral_inhibition:
				cm_list = [mpl.cm.Blues, mpl.cm.Greens, mpl.cm.Reds, mpl.cm.Greys]
				cm = mpl.cm.Blues
				for n in np.arange(int(self.params['sim']['output_neurons'])):
					cm = cm_list[n]
					my_masked_array = np.ma.masked_equal(output_rates[...,n], 0.0)
					plt.contourf(X, Y, my_masked_array, V, cmap=cm, extend='max')
			else:
				cm = getattr(mpl.cm, colormap)
				# cm = mpl.cm.binary
				plt.contourf(X, Y, output_rates[...,0], V, cmap=cm, extend='max')
			# cm.set_over('black', 1.0) # Set the color for values higher than maximum
			# cm.set_bad('white', alpha=0.0)
			ax = plt.gca()
			# plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
			# plt.locator_params(axis='y', nbins=1)
			ax.invert_yaxis()
			ticks = np.linspace(0.0, maximal_rate, 2)
			# from mpl_toolkits.axes_grid1 import make_axes_locatable
			# divider = make_axes_locatable(plt.gca())
			# cax = divider.append_axes("right", "20%", pad="3%")
			# plt.colorbar(im, cax=cax)
			# cb = plt.colorbar(format='%.0f', ticks=ticks, cax=cax)
			# cb = plt.colorbar(format='%.0f', ticks=ticks)
			# cb.set_label('Firing rate')
			# plt.sca(ax)
			plt.xticks([])
			if publishable:
				plt.locator_params(axis='y', nbins=2)
				# ax.set_ylim([0, 2e5])
				# plt.yticks([0, 5e4], [0, 5])
				# plt.ylabel('Time [a.u.]', fontsize=12)
				fig.set_size_inches(2.2, 1.4)

			return output_rates[...,0]


	def set_axis_settings_for_contour_plots(self, ax):
		if self.boxtype == 'circular':
			circle1=plt.Circle((0,0), self.radius, ec='black', fc='none', lw=2)
			ax.add_artist(circle1)
		if self.boxtype == 'linear':
			rectangle1=plt.Rectangle((-self.radius, -self.radius),
					2*self.radius, 2*self.radius, ec='black', fc='none', lw=2)
			ax.add_artist(rectangle1)
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])

	def plot_autocorrelation_vs_rotation_angle(self, time, from_file=True, spacing=51, method='Weber'):
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			frame = self.time2frame(time, weight=True)

			if spacing is None:
				spacing = self.spacing

			linspace = np.linspace(-self.radius , self.radius, spacing)
			X, Y = np.meshgrid(linspace, linspace)
			# Get the output rates
			output_rates = self.get_output_rates(frame, spacing, from_file)
			if self.dimensions == 2:
				corr_spacing, correlogram = observables.get_correlation_2d(
									output_rates, output_rates, mode='same')
				gridness = observables.Gridness(
						correlogram, self.radius, 5, 0.2, method=method)
				angles, correlations = gridness.get_correlation_vs_angle()
				title = 'Grid Score = %.2f' % gridness.get_grid_score()
				plt.title(title)
				plt.plot(angles, correlations)
				ax = plt.gca()
				y0, y1 = ax.get_ylim()
				plt.ylim((y0, y1))
				plt.vlines([30, 90, 150], y0, y1, color='red',
								linestyle='dashed', lw=2)
				plt.vlines([60, 120], y0, y1, color='green',
								linestyle='dashed', lw=2)
				plt.xlabel('Rotation angle')
				plt.ylabel('Correlation')

	# def plot_analytical_grid_spacing(self, k, ):


	def grid_spacing_vs_n_inh_fraction(self, time, spacing=None, from_file=True):
		"""
		Parameters
		----------


		Returns
		-------
		"""

		cc = general_utils.plotting.color_cycle_blue6[::-1]
		mc = general_utils.plotting.marker_cycle
		# n_excs = [280, 400, 1000, 2000]
		n_excs = [840, 1200, 3000, 6000]
		color_dict = {}
		marker_dict = {}
		for i, nexc in enumerate(n_excs):
			color_dict[nexc] = cc[i]
			marker_dict[nexc] = mc[i]
		# color_dict = {140: cc[0], 280: cc[1], 400: cc[2], 1000: cc[3], 2000: cc[4]}
		# marker_dict = {140: mc[0], 280: mc[1], 400: mc[2], 1000: mc[3], 2000: mc[4]}

		for psp in self.psps:
			fig = plt.gcf()
			fig.set_size_inches(5.5, 4.0)
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			n_exc = self.params['exc']['number_per_dimension']
			n_inh = self.params['inh']['number_per_dimension']

			if spacing is None:
				spacing = self.spacing
			frame = self.time2frame(time, weight=True)
			output_rates = self.get_output_rates(frame, spacing, from_file,
								squeeze=True)
			grid_spacing = general_utils.arrays.get_mean_inter_peak_distance(
							output_rates, 2*radius, 5, 0.1)

			plt.plot(n_exc/n_inh, grid_spacing,
							color=color_dict[n_exc[0]],
							marker=marker_dict[n_exc[0]],
							alpha=1.0,
							linestyle='none', markeredgewidth=0.0, lw=1)

		# The legend
		padding = 0.7
		handles = [plt.Line2D((0,1),(0,0), color=color_dict[n], lw=2)
					for n in n_excs]
		labels = [r'$N^E={0}$'.format(n)
					for n in n_excs]
		plt.legend(handles, labels, bbox_to_anchor=(1,0), loc='lower right', borderpad=padding,
					labelspacing=padding, handlelength=1.5)
		plt.margins(0.1)


	def plot_grid_spacing_vs_parameter(self, time=-1, spacing=None,
		from_file=False, varied_parameter=None, parameter_range=None,
		plot_mean_inter_peak_distance=False, computed_data=False,
		sigma_corr=False, plot_analytic_results=True, publishable=True,
		high_density_limit=True):
		"""Plot grid spacing vs parameter

		Plot the grid spacing vs. the parameter both from data and from
		the analytical equation.

		Publishable:
		For the publishable version, make sure that you only plot psps
		where sigma_inh < 0.38.

		Parameters
		----------
		time : float
			Time at which the grid spacing should be determined
		from_file : bool
			If True, output rates are taken from file
		spacing : int
			Needs only be specified if from_file is False
		varied_parameter : tuple
			Tuple that speciefies the parameter against which the grid spacing
			shall be plotted, e.g. ('inh', 'sigma').
		parameter_range : ndarray
			Range of this parameter. This array also determines the plotting
			range.
		plot_mean_inter
		"""
		fig = plt.gcf()
		# fig.set_size_inches(4.6, 4.1)
		fig.set_size_inches(3.2, 3.0)
		mpl.rcParams['legend.handlelength'] = 1.0
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			parameter = self.params[varied_parameter[0]][varied_parameter[1]]

			if spacing is None:
				spacing = self.spacing
			frame = self.time2frame(time, weight=True)
			output_rates = self.get_output_rates(frame, spacing, from_file,
								squeeze=True)

			if plot_mean_inter_peak_distance:
				if computed_data:
					# inter_peak_distances = self.computed['inter_peak_distances']
					# grid_spacing = np.mean(inter_peak_distances)
					grid_spacing = self.computed['mean_inter_peak_distance']
				else:
					grid_spacing = general_utils.arrays.get_mean_inter_peak_distance(
							output_rates, 2*self.radius, 5, 0.1)
				plt.plot(parameter, grid_spacing, marker='o',
							color=color_cycle_blue3[0], alpha=1.0,
							linestyle='none', markeredgewidth=0.0, lw=1)
			else:
				# Get the auto-correlogram
				correlogram = scipy.signal.correlate(
								output_rates, output_rates, mode='same')
				# Obtain grid spacing by taking the first peak of the correlogram
				gridness = observables.Gridness(correlogram, self.radius, 10, 0.1)
				gridness.set_spacing_and_quality_of_1d_grid()
				grid_spacing = gridness.grid_spacing
				plt.plot(parameter, grid_spacing, marker='o',
							color=color_cycle_blue3[0], alpha=1.0,
							linestyle='none', markeredgewidth=0.0, lw=1)
				# plt.errorbar(parameter, gridness.grid_spacing, yerr=0.0,
				# 			marker='o', color=self.color_cycle_blue3[1])

		# Plot the last points again, this time with a label, to make sure
		# that the legend only appears once
		plt.plot(parameter, grid_spacing, marker='o',
					color=color_cycle_blue3[0], alpha=1.0, label=r'Simulation',
					linestyle='none', markeredgewidth=0.0, lw=1)

		if plot_analytic_results:

			if high_density_limit:
				# The grid spacing from the high density limit approximation
				grid_spacing_theory = (
					analytics.linear_stability_analysis.grid_spacing_high_density_limit(
					params=self.params, varied_parameter=varied_parameter,
					parameter_range=parameter_range, sigma_corr=sigma_corr))

			else:
				grid_spacing_theory = (
					analytics.linear_stability_analysis.get_grid_spacing(
						self.params, varied_parameter, parameter_range))

			plt.plot(parameter_range, grid_spacing_theory, lw=2,
					 color='gray', label=r'Theory')

			plt.legend(loc='best', numpoints=1, fontsize=12, frameon=False)

		if sigma_corr:
			plt.xlabel(r'$\sigma_{\mathrm{corr}}$ (m)', labelpad=-10.0)
		else:
			plt.xlabel(r'$\sigma_{\mathrm{I}}$ (m)', labelpad=-10.0)

		ax = plt.gca()
		simpleaxis(ax)

		plt.autoscale(tight=True)
		# plt.margins(0.02)
		mpl.rcParams.update({'figure.autolayout': True})

		# Ranges for grid spacing vs. fraction of inhibitory inputs
		# plt.xlim([0, 1050])
		# plt.ylim([0.1, 0.5])

		# Ranges for grid spacing vs. sigma_inh figures
		# plt.xlim([0.05, 0.31])
		# plt.ylim([0.18, 0.72])

		# Ranges for grid spacing vs. eta_inh figures


		if publishable:
			plt.ylabel(r'$\ell$ (m)', labelpad=-10.0)
			if sigma_corr:
				plt.xlim([0.05, 0.31])
				plt.ylim([0.06, 0.54])
				# plt.xticks([0.1, 0.4])
				plt.yticks([0.1, 0.5])
				xlim = np.array([0.1, 0.4])
				plt.xticks(xlim/np.sqrt(2), xlim)
				plt.legend(loc='upper left', numpoints=1, fontsize=12, frameon=False)
			else:
				plt.xlim([0.05, 0.31])
				plt.ylim([0.15, 0.73])
				plt.xticks([0.1, 0.3])
				plt.yticks([0.2, 0.7])

	def get_correlogram(self, time, spacing=None, mode='full', from_file=False,
							subdimension=None, n_cumulative=None):
		"""Returns correlogram and corresponding linspace for plotting

		This is just a convenience function. It only creates the appropriate
		linspaces and choose the right correlogram function for different
		dimensions.

		Parameters
		----------
		time, spacing, mode, from_file as in function plot_correlogram
		"""
		dimensions = self.dimensions
		frame = self.time2frame(time, weight=True)
		if spacing is None:
			spacing = self.params['sim']['spacing']
		if mode == 'full':
			corr_radius = 2*self.radius
			corr_spacing = 2*spacing-1
		elif mode == 'same':
			corr_radius = self.radius
			corr_spacing = spacing
		# Get the output rates
		if not n_cumulative:
			output_rates = self.get_output_rates(frame, spacing, from_file)
		else:
			output_rates = self.get_cumulative_output_rates(frame, spacing,
												from_file, n=n_cumulative)

		spatial_dim_from_HD_vs_space_data = (
				self.dimensions == 2 and subdimension == 'space')
		if self.dimensions == 1 or spatial_dim_from_HD_vs_space_data:
			output_rates = (np.squeeze(output_rates) if self.dimensions == 1
								else self.get_spatial_tuning(output_rates))
			correlogram = scipy.signal.correlate(
							output_rates, output_rates, mode=mode)
		elif self.dimensions >= 2:
			a = np.squeeze(output_rates)
			if self.dimensions == 3:
				# Note that you don|t take
				a = np.mean(output_rates, axis=2)
			corr_spacing, correlogram = observables.get_correlation_2d(
								a, a, mode=mode)

		# We wan't nans to be zeros
		correlogram = np.nan_to_num(correlogram)
		corr_linspace = np.linspace(-corr_radius, corr_radius, corr_spacing)
		return corr_linspace, correlogram


	def plot_correlogram(self, time, spacing=None, mode='full', method=None,
				from_file=False, subdimension=None, publishable=False,
				show_colorbar=True, n_cumulative=None, type='hexagonal',
						 colormap='viridis', xlim=None, correlogram_title=False):
		"""Plots the autocorrelogram of the rates at given `time`

		Parameters
		----------
		time : float
			Time in the simulation
		spacing : int
			Specifies the resolution of the correlogram.
		mode : string
			See definition of observables.get_correlation_2d
		"""
		cm = getattr(mpl.cm, colormap)
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			corr_linspace, correlogram = self.get_correlogram(
										time=time, spacing=spacing, mode=mode,
										from_file=from_file,
										subdimension=subdimension,
										n_cumulative=n_cumulative)

			spatial_dim_from_HD_vs_space_data = (
				self.dimensions == 2 and subdimension == 'space')
			if self.dimensions == 1 or spatial_dim_from_HD_vs_space_data:
				plt.plot(corr_linspace, correlogram, color='black')
				gridness = observables.Gridness(correlogram, radius=self.radius,
						neighborhood_size=10, threshold_difference=0.1)
				gridness.set_spacing_and_quality_of_1d_grid()
				title = 'Spacing: %.3f, Quality: %.3f' % (
							gridness.grid_spacing, gridness.quality)
				plt.title(title)
				ax = plt.gca()
				y0, y1 = ax.get_ylim()
				plt.ylim((y0, y1))
				plt.vlines([-gridness.grid_spacing, gridness.grid_spacing], y0, y1,
								color='green', linestyle='dashed', lw=2)
				analytic_grid_spacing = analytics.linear_stability_analysis.grid_spacing_high_density_limit(
					params=self.params, varied_parameter=None,
					parameter_range=None, sigma_corr=False)
				plt.vlines([-analytic_grid_spacing, analytic_grid_spacing], y0, y1,
								color='red', linestyle='dotted', lw=2)
				mipd = self.computed['mean_inter_peak_distance']
				plt.vlines([-mipd, mipd], y0, y1,
								color='blue', linestyle='solid', lw=2, alpha=0.5)
				plt.xlim([-xlim, xlim])
			elif self.dimensions >= 2:
				X_corr, Y_corr = np.meshgrid(corr_linspace, corr_linspace)
				V = np.linspace(-1.0, 1.0, 30)
				# V = 40
				plt.contourf(X_corr, Y_corr, correlogram, V, cmap=cm)
				# plt.contourf(X_corr.T, Y_corr.T, correlogram, 30)
				# cb = plt.colorbar()
				ax = plt.gca()
				self.set_axis_settings_for_contour_plots(ax)
				title = 't=%.2e' % time
				if method != None:
					if mode == 'same':
						r = self.radius
					gridness = observables.Gridness(
						correlogram, r, method=method, type=type)
					title += ', GS = %.2f, l = %.2f' \
								% (gridness.get_grid_score(), gridness.grid_spacing)
					for r, c in [(gridness.inner_radius, 'black'),
								(gridness.outer_radius, 'black'),
								(gridness.grid_spacing, 'white'),	]:
						circle = plt.Circle((0,0), r, ec=c, fc='none', lw=2,
												linestyle='dashed')
						ax.add_artist(circle)
				ticks = np.linspace(-1, 1, 2)
				if show_colorbar:
					cb = plt.colorbar(format='%i', ticks=ticks)
					cb.ax.tick_params(width=0)
					# cb.set_label('Correlation')
				# mpl.rc('font', size=42)
				plt.title(title, fontsize=8)
				if publishable:
					mpl.rc('font', size=12)
					cb.set_label('')
					if correlogram_title:
						plt.title('Correlogram')
					else:
						plt.title('')

	def get_grid_score_suffix(self, type):
		if type == 'hexagonal':
			suffix = ''
		elif type == 'quadratic':
			suffix = '_quadratic'
		return suffix

	def get_grid_score(self, time, spacing=None, method='Weber', from_file=True,
					   data=False, n_cumulative=None, type='hexagonal'):
		"""
		Returns grid score
		Just a convenience function to avoid code duplication in add_computed()
		and plotting()

		For the moment just in 2D

		Parameters
		----------
		time : float
			Time at which the grid score is returned
		spacing : int
			Spacing of the rate maps. Only needed if from_file=False, i.e. if
			we don't use the stored rate maps but create new ones from the
			weights
		from_file : bool
			If True the stored rate map is taken as the basis for the
			correlograms
		method : str
			See Gridness class
		data : bool
			If True the grid score of the given method is taken from 'computed'
		Returns
		-------
		grid_score : float
		"""
		if not data:
			correlogram = self.get_correlogram(
								time, spacing, 'same', from_file,
								n_cumulative=n_cumulative)[1]
			gridness = observables.Gridness(
							correlogram, self.radius, method=method, type=type)
			grid_score = gridness.get_grid_score()
		else:
			suffix = self.get_grid_score_suffix(type)
			frame = self.time2frame(time, weight=True)
			grid_score = self.computed['grid_score'+suffix][method][str(n_cumulative)][frame]
		return grid_score

	def get_list_of_grid_score_arrays_over_all_psps(self,
													method, n_cumulative,
													type='hexagonal',
													from_computed_full=True):
		"""
		Returns arrays with grid scores for all psps

		Note: This only makes sense if the psps differ in seeds but
		not in parameters

		Parameters
		----------
		from_computed_full : bool
			If True, it assumes that there are grid score arrays with all
			psps in the main computed folder

		Returns
		-------
		"""
		# TODO: You should make this a general function
		# Like: get_list_of_computed_arrays_over_all_psps
		# Problem: I don't know how deep the dictionary is (here it is
		# three, but it could be anything)
		# Possible solution: specifiy the path for the tables
		# Ask Owen how to do it.
		# if grid_score:
		# 	print 'begin'
		# 	# grid_score = self.tables.get_computed(None)['grid_score'+suffix]
		# 	print 'end'
		# 	# l = grid_score[method][str(n_cumulative)]
		# 	# self.set_params_rawdata_computed(self.psps[0], set_sim_params=True)
		suffix = self.get_grid_score_suffix(type)
		if from_computed_full:
			l = self.computed_full['grid_score'+suffix][method][str(n_cumulative)]
			self.params = self.tables.as_dictionary(self.psps[0], True)
		else:
			print 'Use add_computed first to make this faster'
			l = []
			for psp in self.psps:
				suffix = self.get_grid_score_suffix(type)
				try:
					self.set_params_rawdata_computed(psp, set_sim_params=True)
					array = self.computed['grid_score'+suffix][method][str(n_cumulative)]
					l.append(array)
				except:
					pass
			l = np.asarray(l)
		return l

	def get_output_rates_over_all_psps(self):
		l = []
		for psp in self.psps:
			try:
				self.set_params_rawdata_computed(psp, set_sim_params=True)
				array = np.squeeze(self.rawdata['output_rates'])
				l.append(array)
			except:
				pass
		l = np.asarray(l)
		return l

	def mean_output_rate_time_evolution(self, end_frame=None,
										n_individual_plots=0):
		# mpl.style.use('ggplot')
		output_rates = self.get_output_rates_over_all_psps()
		mean = np.nanmean(output_rates, axis=0)[:end_frame]
		std = np.nanstd(output_rates, axis=0)[:end_frame]
		time = (np.arange(0, len(output_rates[0]))
				* self.params['sim']['every_nth_step_weights']
				* self.params['sim']['dt'])[:end_frame]
		plt.plot(time, mean)
		plt.fill_between(time,
						 mean + std,
						 mean - std, alpha=0.5)
		# Plot some invidivual traces
		for j in np.arange(n_individual_plots):
			plt.plot(time, output_rates[j][:end_frame])
		# plt.ylim([-0.5, 1.0])
		plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

	def grid_score_time_correlation(self, row_index=0):
		plt.figure(figsize=(14, 5))
		mpl.style.use('ggplot')
		gs = gridspec.GridSpec(1, 4)
		gs_dict = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
		methods = ['Weber', 'sargolini']
		for i, method in enumerate(methods):
			for k, ncum in enumerate([1, 10]):
				column_index = gs_dict[(i, k)]
				plt.subplot(gs[row_index, column_index])
				grid_scores = self.get_list_of_grid_score_arrays_over_all_psps(
								method=method, n_cumulative=ncum)
				time = (np.arange(0, len(grid_scores[0]))
						* self.params['sim']['every_nth_step_weights']
						* self.params['sim']['dt'])
				# Plot some invidivual traces
				start_frame = 100
				correlograms = []
				for j in np.arange(4):
					# corr_time = time[start_frame:]
					corr_grid_scores = grid_scores[j][start_frame:]
					correlogram = scipy.signal.correlate(corr_grid_scores,
														 corr_grid_scores,
														 mode='same')
					# correlograms.append(correlogram)
					plt.plot(correlogram)
				plt.plot(np.mean(correlograms, axis=0))
				# plt.ylim([-0.5, 1.0])
				plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
				if row_index == 0:
					plt.title('{0}, nc = {1}'.format(method, ncum), fontsize=10)


	def grid_score_histogram(self, row_index=0, type='hexagonal',
								methods=['Weber', 'sargolini'],
								n_cumulative=[1, 3],
								end_frame=-1,
							 	from_computed_full=True):
		"""
		Plots histogram of grid scores of all psps at time 0 and -1

		4 plots are returned in a row.
		2 methods ('Weber', 'sargolini') and 2 n_cumulative values (1, 3)

		Parameters
		----------
		row_index : int
			If this row is to be plotted in a grid with other plots,
			`row_index` gives the row number of this plot.
		"""
		# mpl.style.use('ggplot')
		gs = gridspec.GridSpec(2, 4)
		gs_dict = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
		for i, method in enumerate(methods):
			for k, ncum in enumerate(n_cumulative):
				column_index = gs_dict[(i, k)]
				plt.subplot(gs[row_index, column_index])
				grid_scores = self.get_list_of_grid_score_arrays_over_all_psps(
									method=method, n_cumulative=ncum, type=type,
									from_computed_full=from_computed_full)
				self.plot_grid_score_histogram(grid_scores, end_frame)
				if row_index == 0:
					plt.title('{0}, nc = {1}'.format(method, ncum))


	def plot_grid_score_histogram(self, grid_scores, end_frame=-1):
		"""
		Plots histogram of grid scores

		Note: This is not a stand alone plotting function

		Parameters
		----------
		grid_scores : ndarray
			Contains grids scores for all seeds and times
		"""
		# my_bins = np.linspace(-1.2, 1.4, 27)
		my_bins = np.linspace(-1.2, 1.4, 14)
		hist_kwargs = {'alpha': 0.5, 'bins': my_bins, 'lw': 0.}
		init_gs = grid_scores[:, 0]
		final_gs = grid_scores[:, end_frame]
		colors = {'init': color_cycle_blue4[2], 'final': color_cycle_blue4[0]}
		# colors = {'init': '0.7', 'final': '0.1'}
		n_init, bins_init, p = plt.hist(
			init_gs[~np.isnan(init_gs)], color=colors['init'], **hist_kwargs)
		n_final, bins_final, p = plt.hist(
			final_gs[~np.isnan(final_gs)], color=colors['final'], **hist_kwargs)
		gc_percentage_init = '{0} %'.format(
			int(100*np.sum(n_init[bins_init[:-1]>=0]) / np.sum(n_init)))
		gc_percentage_final = '{0} %'.format(
			int(100*np.sum(n_final[bins_final[:-1]>=0]) / np.sum(n_final)))
		ax = plt.gca()
		ax.text(0.05, 0.95, gc_percentage_init, horizontalalignment='left',
				verticalalignment='top', transform=ax.transAxes,
				color=colors['init'])
		ax.text(0.95, 0.95, gc_percentage_final, horizontalalignment='right',
				verticalalignment='top', transform=ax.transAxes,
				color=colors['final'])
		plt.xlim([-1.2, 1.4])
		plt.xlim()
		plt.locator_params(axis='y', tight=True, nbins=2)

	def plot_grid_score_evolution(self, grid_scores, end_frame=None,
								  seed_centers=None,
								  row_index=None):
		"""
		Plots time evolution of mean and stdev of grid scores

		Note: This is not a stand alone plotting function

		Parameters
		----------
		grid_scores : ndarray
			Contains grids scores for all seeds and times
		"""
		if end_frame == -1:
			end_frame = None
		grid_score_mean = np.nanmean(grid_scores, axis=0)[:end_frame]
		grid_score_std = np.nanstd(grid_scores, axis=0)[:end_frame]
		time = (np.arange(0, len(grid_scores[0]))
				* self.params['sim']['every_nth_step_weights']
				* self.params['sim']['dt'])[:end_frame]
		plt.plot(time, grid_score_mean, color='black', lw=2)
		plt.fill_between(time,
						 grid_score_mean + grid_score_std,
						 grid_score_mean - grid_score_std,
						 alpha=0.1, color='black', lw=0.)
		# Plot some invidivual traces
		for n,j in enumerate(seed_centers):
			plt.plot(time, grid_scores[j][:end_frame],
					 color=color_cycle_blue3[n])
		plt.xlim([0.0, time[-1]])
		plt.ylim([-0.45, 1.25])
		# plt.xlabel('Time [hrs]')
		plt.ylabel('Grid score')
		plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
		plt.xticks([0, 9e5, 18e5], ['0', '5h', '10h'])
		if row_index == 0:
			plt.title('{0}, nc = {1}'.format(method, ncum), fontsize=10)


	def mean_grid_score_time_evolution(self, row_index=0, end_frame=None,
									   seed_centers=[0,1,2],
									   methods=['Weber', 'sargolini'],
									   n_cumulative=[1, 3],
									   figsize=None,
									   type='hexagonal',
									   from_computed_full=True):
		"""
		Plots mean grid score of all psps with standard deviation as shade

		4 plots are returned in a row.
		2 methods ('Weber', 'sargolini') and 2 n_cumulative values (1, 3)

		Parameters
		----------
		row_index : int
			If this row is to be plotted in a grid with other plots,
			`row_index` gives the row number of this plot.
		"""
		# mpl.style.use('ggplot')
		if figsize:
			plt.figure(figsize=figsize)
		# gs = gridspec.GridSpec(2, len(methods)*len(n_cumulative))
		gs = gridspec.GridSpec(2, 4)
		gs_dict = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
		for i, method in enumerate(methods):
			for k, ncum in enumerate(n_cumulative):
				column_index = gs_dict[(i, k)]
				plt.subplot(gs[row_index, column_index])
				grid_scores = self.get_list_of_grid_score_arrays_over_all_psps(
								method=method, n_cumulative=ncum, type=type,
								from_computed_full=from_computed_full
				)
				self.plot_grid_score_evolution(grid_scores, end_frame,
											   seed_centers)

	def grid_score_evolution_and_histogram(self, type='hexagonal',
										end_frame=-1,
										methods=['Weber', 'sargolini'],
									 	n_cumulative=[1, 3],
										from_computed_full=True):
		"""
		Convenience function to arrange grid score mean and histogram
		"""
		plt.figure(figsize=(14, 6))
		kwargs = dict(type=type, end_frame=end_frame, methods=methods,
					  n_cumulative=n_cumulative,
					  from_computed_full=from_computed_full)
		self.mean_grid_score_time_evolution(row_index=0, **kwargs)
		self.grid_score_histogram(row_index=1, **kwargs)

	def grid_score_hexagonal_and_quadratic(self, n_individual_plots=1,
										   method='sargolini', n_cumulative=3,
										   end_frame=None):
		"""
		Convenience function
		"""
		plt.figure(figsize=(14, 6))
		mpl.style.use('ggplot')
		color_cycle = dict(hexagonal=color_cycle_blue3,
						   quadratic=color_cycle_red3)
		for type in ['hexagonal', 'quadratic']:
			grid_scores = self.get_list_of_grid_score_arrays_over_all_psps(
									method=method, n_cumulative=n_cumulative, type=type)
			time = (np.arange(0, len(grid_scores[0]))
							* self.params['sim']['every_nth_step_weights']
							* self.params['sim']['dt'])[:end_frame]
			for j in np.arange(n_individual_plots):
				plt.plot(time, grid_scores[j][:end_frame],
						 color=color_cycle[type][j], label=type[:4])

		plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
		plt.legend(loc='best')
		# self.mean_grid_score_time_evolution(end_frame=200, n_individual_plots=1,
		# 									methods=['sargolini'],
		# 									type='hexagonal', row_index=0,
		# 									n_cumulative=[10])
		# self.mean_grid_score_time_evolution(end_frame=200, n_individual_plots=1,
		# 									methods=['sargolini'],
		# 									type='quadratic', row_index=1,
		# 									n_cumulative=[10])

	def plot_time_evolution(self, observable, t_start=0, t_end=None,
							method='Weber', spacing=None, from_file=True,
							data=False, n_cumulative=1):
		"""Plots time evolution of given observable

		Parameters
		----------
		observable : string
			'grid_score', 'grid_spacing'
		data : bool
			If True the data is taken from 'computed'. It needs to be created
			there first using add_computed.py in a separate step.
		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			if t_end == None:
				t_end = self.simulation_time
			time_increment = self.every_nth_step_weights * self.dt
			time = np.arange(t_start, t_end+time_increment, time_increment)
			observable_list = []
			if observable == 'grid_score':
				for t in time:
					observable_list.append(self.get_grid_score(t, spacing,
													method, from_file, data))
				observable_list = self.computed['grid_score'][method][str(n_cumulative)]
				plt.ylim([-0.6, 1.4])
				plt.hlines([0.0], t_start, t_end,
								color='black',linestyle='dashed', lw=2)
				plt.ylabel('Grid score')
			plt.xlabel('Time')
			# plt.xlim([0, 1e7])
			plt.plot(time, observable_list, lw=2, marker='o', color='black')

	def get_output_rates(self, frame, spacing, from_file=False, squeeze=False):
		"""Get output rates either from file or determine them from equation

		The output rates are returned at several positions.

		Parameters
		----------
		frame : int
			The frame at which the rates should be returned
		spacing : int
			Sets the resolution of the space at which output rates are returned
			In 1D: A linear space [-radius, radius] with `spacing` points
			In 2D: A quadratic space with `spacing`**2 points

		Returns
		-------
		output_rates : ndarray
		"""

		if from_file:
			output_rates = self.rawdata['output_rate_grid'][frame].copy()

		else:
			input_rates = {}

			if self.dimensions == 1:
				limit = self.radius # +self.params['inh']['center_overlap']
				linspace = np.linspace(-limit, limit, spacing)
				positions_grid = linspace.reshape(spacing, 1, 1)
				for t in ['exc', 'inh']:
					input_rates[t] = self.get_rates(positions_grid, syn_type=t)
				output_rates = self.get_output_rates_from_equation(
					frame=frame, rawdata=self.rawdata, spacing=spacing,
					positions_grid=False, input_rates=input_rates,
					equilibration_steps=10000)
			elif self.dimensions == 2:
				X, Y, positions_grid, input_rates = self.get_X_Y_positions_grid_input_rates_tuple(spacing)
				output_rates = self.get_output_rates_from_equation(
						frame=frame, rawdata=self.rawdata, spacing=spacing,
						positions_grid=positions_grid, input_rates=input_rates)

		if squeeze:
			output_rates = np.squeeze(output_rates)
		return output_rates

	def get_cumulative_output_rates(
			self, frame, spacing, from_file=False, squeeze=False, n=1):
		"""
		Returns the cumulative sum of output rate maps

		The cumulative sum is taken over the output rate map at `frame`
		and the `n`-1 previous rate maps.
		Note : The associated time interval is determined by the
		every_nth_step_weights parameter.

		Parameters
		----------
		n : int
			Number of previous rate maps that should be summed
			n = 1 corresponds to get_output_rates function

		See get_output_rates for other parameters

		Returns
		-------
		cum_output_rates : ndarray
		"""
		cum_output_rates = np.zeros_like(self.get_output_rates(frame, spacing, from_file,
												  squeeze))
		for i in np.arange(1, n+1):
			current_frame = frame-i+1
			if (frame-i+1) >= 0:
				cum_output_rates += self.get_output_rates(
								current_frame,
								spacing, from_file, squeeze)
			else:
				break
			# except IndexError:
			# 	print 'IndexError was raised because there are not enough previous maps'
		return cum_output_rates / i

	def plot_head_direction_polar(self, time, spacing=None, from_file=False,
				show_watson_U2=False, publishable=False):
		"""Plots polar plot of head direction distribution

		NOTE: It is crucial that the name of this function contains the
				string 'polar'

		Parameters
		----------
		See parameters for plot_output_rates_from_equation
		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			frame = self.time2frame(time, weight=True)

			if spacing is None:
				spacing = self.spacing

			# Get the output rates
			output_rates = self.get_output_rates(frame, spacing, from_file)
			theta = np.linspace(-np.pi, np.pi, spacing)
			if self.dimensions == 2:
				b = output_rates[...,0].T
				r = np.mean(b, axis=1)
			elif self.dimensions == 3:
				 r = np.mean(output_rates[..., 0], axis=(1, 0)).T
			plt.polar(theta, r, color='black', lw=1)
			# fig = plt.figure(figsize=(2.5, 2.5))
			# ax = fig.add_subplot(111, polar=True)
			# mpl.rc('font', size=10)
			thetaticks = np.arange(0,360,90)
			ax = plt.gca()
			ax.set_thetagrids(thetaticks, frac=1.4)
			ax.set_aspect('equal')
			if show_watson_U2:
				hd_tuning = observables.Head_Direction_Tuning(r, spacing)
				U2, h = hd_tuning.get_watson_U2_against_uniform()
				plt.title('Watson U2: ' + str(U2))
			if publishable:
				plt.yticks([])
				ax.spines['polar'].set_visible(False)
				# ax.set_axis_bgcolor('0.9')
				# No angles written but still the 90 degree lines
				ax.set_thetagrids(thetaticks, [])
				r_max = ax.get_ylim()[-1]
				plt.ylim([0, r_max*1.1])



	def get_spatial_tuning(self, output_rates):
		"""Returns the spatial dimension of a 2D (HD vs Space) simulation

		Parameters
		----------
		output_rates : ndarray
			The 2 dimensional output firing rate array
		"""
		return np.mean(output_rates[...,0].T, axis=0)


	# def plot_grids_linear(self, time, spacing=None, from_file=False):
	# 	"""Plots linear plot of output firing rates rate vs position

	# 	Used for 2D (HD vs space) simulation

	# 	Parameters
	# 	----------
	# 	See parameters for plot_output_rates_from_equation
	# 	"""
	# 	for psp in self.psps:
	# 		self.set_params_rawdata_computed(psp, set_sim_params=True)
	# 		frame = self.time2frame(time, weight=True)

	# 		if spacing is None:
	# 			spacing = self.spacing

	# 		linspace = np.linspace(-self.radius , self.radius, spacing)
	# 		output_rates = self.get_output_rates(frame, spacing, from_file)
	# 		spatial_tuning = self.get_spatial_tuning(output_rates)
	# 		plt.plot(linspace, spatial_tuning)

	def input_tuning_extrema_distribution(self, populations=['exc', 'inh']):
		"""
		Plots histogram of maxima and minima of each input tuning function

		This is only interesting for gaussian process inputs.
		Maxima are plotted with high alpha value.
		Minima are plotted with low alpha value.
		Populations follow the usual color code.

		Parameters
		----------



		Returns
		-------
		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			# extraticks = []
			for p in populations:
				gp_min = self.rawdata[p]['gp_min']
				gp_max = self.rawdata[p]['gp_max']
				kwargs = dict(color=self.colors[p], bins=30)
				plt.hist(gp_min, alpha=0.4,
						 label='min, {0}'.format(p), **kwargs)
				plt.hist(gp_max, alpha=0.8,
						 label='max, {0}'.format(p), **kwargs)
				plt.legend()
				ax = plt.gca()
				plt.setp(ax,
						xlabel='Extreme value of input tuning',
						ylabel='Frequency')
				for gp_m in [gp_min, gp_max]:
					mean = np.mean(gp_m)
					plt.axvline(mean)
					# extraticks.append(mean)
					trans = mpl.transforms.blended_transform_factory(
							ax.transData, ax.transAxes)
					plt.text(mean, 1, '{0:.2}'.format(mean),
							 rotation='vertical', transform=trans)

	def indicate_grid_spacing(self, maxima_positions, y):
		"""
		Plots an arrow between two most central peaks and labels it

		Parameters
		----------
		maxima_positions : ndarray
			Positions of all ocurring maxima
		y : float
			Vertical position of array in data coordinates
		"""
		# Number of maxima
		n_maxima = len(maxima_positions)
		# Index of most central maximum
		central_index = int(np.floor(n_maxima/2.))
		# The two most central maxima positions
		central_maxima_pos = maxima_positions[central_index-1:central_index+1]
		# Plot the arrow
		# Note: mutation_scale can adjust the arrow head size in relation
		# to the arrow size. I don't know what a particular value means.
		# It might be percentage.
		plt.annotate(
			 '', xy=(central_maxima_pos[0], y), xycoords='data',
			xytext=(central_maxima_pos[1], y), textcoords='data',
			arrowprops={'arrowstyle': '<->', 'shrinkA': 1, 'shrinkB': 1,
						'mutation_scale': 5.})
		# Plot the text label. xytext is used as an offset here
		plt.annotate(r'$\ell$',
			xy=((central_maxima_pos[1]+central_maxima_pos[0])/2, y),
			xycoords='data',
			xytext=(0, 2), textcoords='offset points',
			ha='center')

	def plot_output_rates_from_equation(self, time, spacing=None, fill=False,
					from_file=False, number_of_different_colors=30,
					maximal_rate=False, subdimension=None, plot_maxima=False,
					publishable=False, xlim=None, selected_psp=None,
					no_ylabel=False, indicate_gridspacing=False,
					sigma_inh_label=False, show_colorbar=True,
					show_title=True, n_cumulative = None,
										colormap='viridis',
										firing_rate_title=False,
										colorbar_label=False):
		"""Plots output rates using the weights at time `time

		Publishable:
		Note that you want a higher spacing and not from file plotting
		for nice publishable figures

		Parameters
		----------
		spacing : int
			The output will contain spacing**dimensions data points
		fill : bool
			If True the contour plot will be filled,
			if False it will be just lines
		subdimension : string
			None : Every dimension is plotted
			'space' : Only the spatial dimensions are plotted
		selected_psp : int or ndarray
			Indexing for the psp list. Only the indexed items are plotted.
		no_ylabel : bool
			If True the ylabel is turned off
		indicate_gridspacing : bool
			If True a labeled arrow is plotted between two peaks to show
			what the gridspacing is.
		n_cumulative : int
			See get_cumulative_output_rates
		"""
		# Possiblity to select just one paramspace point
		if selected_psp is None:
			psps = self.psps
		else:
			psps= self.psps[selected_psp]
		for psp in np.atleast_1d(psps):
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			frame = self.time2frame(time, weight=True)
			if spacing is None:
				spacing = self.spacing

			linspace = np.linspace(-self.radius , self.radius, spacing)
			X, Y = np.meshgrid(linspace, linspace)
			distance = np.sqrt(X*X + Y*Y)
			# Get the output rates
			if not n_cumulative:
				output_rates = self.get_output_rates(frame, spacing, from_file)
			else:
				output_rates = self.get_cumulative_output_rates(frame, spacing,
													from_file, n=n_cumulative)
			# np.save('test_output_rates', output_rates)
			##############################
			##########	Plot	##########
			##############################
			spatial_dim_from_HD_vs_space_data = (
				self.dimensions == 2 and subdimension == 'space')

			if self.dimensions == 1 or spatial_dim_from_HD_vs_space_data:
				output_rates = (np.squeeze(output_rates) if self.dimensions == 1
								else self.get_spatial_tuning(output_rates))

				# color='#FDAE61'
				color = 'black'
				limit = self.radius# + self.params['inh']['center_overlap']
				linspace = np.linspace(-limit, limit, spacing)
				plt.plot(linspace, output_rates, color=color, lw=1)
				# Plot positions of centers which have been located
				# maxima_boolean = general_utils.arrays.get_local_maxima_boolean(
				# 			output_rates, 5, 0.1)
				# peak_positions = linspace[maxima_boolean]
				# plt.plot(peak_positions, np.ones_like(peak_positions),
				# 			marker='s', color='red', linestyle='none')
				maxima_positions, maxima_values, grid_score = (
						self.get_1d_grid_score(output_rates, linspace,
						neighborhood_size=7))

				if indicate_gridspacing:
					self.indicate_grid_spacing(maxima_positions, 7)

				if plot_maxima:
					plt.plot(maxima_positions, maxima_values, marker='o',
							linestyle='none', color='red')
				# title = 'GS = %.2f' % grid_score
				title = 'GS={0:.2f}, t={1:.02}'.format(grid_score, time)
				ax = plt.gca()
				# ax.set_ylim(0, ax.get_ylim()[1])
				# y0, y1 = ax.get_ylim()
				# Allows to use different transforms for x and y axis
				trans = mpl.transforms.blended_transform_factory(
							ax.transData, ax.transAxes)
				# plt.vlines([-self.radius, self.radius], 0, 1,
				# 			color='gray', lw=2, transform = trans)
				x0, x1 = ax.get_xlim()
				# plt.ylim((y0, y1))
				plt.hlines([self.params['out']['target_rate']], x0, x1,
							color='black',linestyle='dashed', lw=1)
				# plt.yticks(['rho'])
				# title = 'time = %.0e' % (frame*self.every_nth_step_weights)
				plt.title(title)
				# plt.ylim([0.4, 1.3])
				# plt.yticks([0.4, 1.0, 1.3])
				# plt.xticks([])
				# plt.locator_params(axis='y', nbins=3)
				# ax.set_yticks((0, self.params['out']['target_rate'], 5, 10))
				# ax.set_yticklabels((0, r'$\rho_0$', 5, 10), fontsize=18)
				# plt.xlabel('Position')
				# plt.ylabel('Firing rate')
				fig = plt.gcf()
				# fig.set_size_inches(5,2.1)
				fig.set_size_inches(5,3.5)
				if publishable:
					ax = plt.gca()
					# xmin = linspace.min()
					# xmax = linspace.max()
					# ymin = output_rates.min()
					ymax = output_rates.max()
					# ymax = 4.2
					# ax.set_xlim(1.01*xmin,1.01*xmax)
					# ax.set_ylim(1.01*xmin,1.01*xmax)
					ax.spines['right'].set_color('none')
					ax.spines['top'].set_color('none')
					ax.spines['bottom'].set_color('none')
					ax.xaxis.set_ticks_position('bottom')
					ax.spines['bottom'].set_position(('data', -0.1*ymax))
					ax.yaxis.set_ticks_position('left')
					# ax.spines['left'].set_position(('data', xmin))
					plt.xticks([])
					# ax.axes.get_xaxis().set_visible(False)
					plt.ylabel('')
					plt.title('')
					# plt.xlim([-0.5, 0.5])
					if xlim is None:
						plt.xlim([-limit, limit])
					else:
						plt.xlim(xlim)
					plt.margins(0.0, 0.1)
					# plt.xticks([])
					# plt.locator_params(axis='y', nbins=1)
					plt.yticks([0, int(ymax)])
					# adjust_spines(ax, ['left', 'bottom'])
					# fig.set_size_inches(1.65, 0.8)
					fig.set_size_inches(1.65, 0.3)
					if not no_ylabel:
						plt.ylabel('Hz')
					else:
						plt.ylabel('')
					# plt.yticks([0])
					# plt.xlabel('2 m')
					# plt.ylabel('')
					if sigma_inh_label:
						plt.xlabel('$\sigma_{\mathrm{I}}=' + str(self.params['inh']['sigma'][0]) + ' \mathrm{cm}$')



			elif self.dimensions >= 2:
				# title = r'$\vec \sigma_{\mathrm{inh}} = (%.2f, %.2f)$' % (self.params['inh']['sigma_x'], self.params['inh']['sigma_y'])
				# plt.title(title, y=1.04, size=36)
				title = 't=%.2e' % time
				if show_title:
					plt.title(title, fontsize=8)
				# cm = mpl.cm.jet
				cm = getattr(mpl.cm, colormap)
				cm.set_over('y', 1.0) # Set the color for values higher than maximum
				cm.set_bad('white', alpha=0.0)
				# V = np.linspace(0, 3, 20)
				if not maximal_rate:
					if self.dimensions == 3 and subdimension == 'space':
						maximal_rate = int(np.ceil(np.amax(np.mean(output_rates[..., 0], axis=2))))
					else:
						maximal_rate = int(np.ceil(np.amax(output_rates)))
				V = np.linspace(0, maximal_rate, number_of_different_colors)
				# mpl.rc('font', size=42)

				# Hack to avoid error in case of vanishing output rate at every position
				# If every entry in output_rates is 0, you define a norm and set
				# one of the elements to a small value (such that it looks like zero)
				if np.count_nonzero(output_rates) == 0:
					color_norm = mpl.colors.Normalize(0., 100.)
					output_rates[0][0] = 0.000001
					plt.contourf(X, Y, output_rates[...,0], V, norm=color_norm, cmap=cm, extend='max')
				else:
					if self.lateral_inhibition:
						# plt.contourf(X, Y, output_rates[:,:,0], V, cmap=cm, extend='max')
						cm_list = [mpl.cm.Blues, mpl.cm.Greens, mpl.cm.Reds, mpl.cm.Greys]
						# cm = mpl.cm.Blues
						for n in np.arange(int(self.params['sim']['output_neurons'])):
							cm = cm_list[n]
							my_masked_array = np.ma.masked_equal(output_rates[...,n], 0.0)
							plt.contourf(X, Y, my_masked_array.T, V, cmap=cm, extend='max')
					else:
						if self.dimensions == 3:
							if subdimension == 'space':
								# For plotting of spatial tuning
								a = np.mean(output_rates[..., 0], axis=2).T
							else:
								# For plotting of just two axes
								a = output_rates[:, :, 28, 0].T
							plt.contourf(X, Y, a, V, cmap=cm)
							# output_rates[...,0][distance>self.radius] = np.nan
						elif self.dimensions == 2:
							# plt.contourf(X, Y, output_rates[..., 0].T, V, cmap=cm, extend='max')
							plt.contourf(X, Y, output_rates[..., 0], V, cmap=cm)

				plt.margins(0.01)
				plt.axis('off')
				ticks = np.linspace(0.0, maximal_rate, 2)
				if show_colorbar and not publishable:
					plt.colorbar(format='%.2f', ticks=ticks)
				# cb = plt.colorbar(format='%i')
				# plt.colorbar()
				# cb.set_label('Firing rate')
				ax = plt.gca()
				self.set_axis_settings_for_contour_plots(ax)
				# fig = plt.gcf()
				# fig.set_size_inches(6.5,6.5)
				if publishable:
					mpl.rc('font', size=12)
					if show_colorbar:
						cb = plt.colorbar(format='%i', ticks=ticks)
						# cb.ax.set_yticklabels(['0', "{0}nn".format(maximal_rate)[:3]])
						# if no_ylabel:
						# 	cb.set_label('')
						# else:
						if ticks[-1] >= 10:
							labelpad = -9.0
						else:
							labelpad = -1.5
						if colorbar_label:
							cb.set_label('Hz', rotation='horizontal',
										 labelpad=labelpad)
						else:
							cb.set_label('')
						cb.ax.tick_params(width=0)
					if firing_rate_title:
						plt.title('Firing rate')
					else:
						plt.title('')


	def fields_times_weights(self, time=-1, syn_type='exc', normalize_sum=True):
		"""
		Plots the Gaussian Fields multiplied with the corresponding weights

		Arguments:
		- normalize_sum: If true the sum gets scaled such that it
			is comparable to the height of the weights*gaussians,
			this way it is possible to see the sum and the individual
			weights on the same plot. Otherwise the sum would be way larger.
		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			frame = self.time2frame(time, weight=True)
			# plt.title(syn_type + ' fields x weights', fontsize=8)
			limit = self.radius # + self.params[syn_type]['center_overlap']
			x = np.linspace(-limit, limit, 601)
			t = syn_type
			# colors = {'exc': 'g', 'inh': 'r'}
			summe = 0
			divisor = 1.0
			if normalize_sum:
				# divisor = 0.5 * len(rawdata[t + '_centers'])
				divisor = 0.5 * self.params[syn_type]['number_desired']
			# for c, s, w in np.nditer([
			# 				self.rawdata[t]['centers'],
			# 				self.rawdata[t]['sigmas'],
			# 				self.rawdata[t]['weights'][frame][0] ]):
			for i in np.arange(self.rawdata[syn_type]['number']):
				print i
				c = self.rawdata[t]['centers'][i]
				s = self.rawdata[t]['sigmas'][i]
				w = self.rawdata[t]['weights'][frame][0][i]
				gaussian = scipy.stats.norm(loc=c, scale=s).pdf
				# l = plt.plot(x, w * gaussian(x), color=self.colors[syn_type])
				summe += w * gaussian(x)
			plt.plot(x, summe / divisor, color=self.colors[syn_type], linewidth=4)

			summe = 0
			for i in np.arange(self.rawdata['exc']['number']):
				print i
				c = self.rawdata['exc']['centers'][i]
				s = self.rawdata['exc']['sigmas'][i]
				w = self.rawdata['exc']['weights'][frame][0][i]
				gaussian = scipy.stats.norm(loc=c, scale=s).pdf
				# l = plt.plot(x, w * gaussian(x), color=self.colors[syn_type])
				summe += w * gaussian(x)
			plt.plot(x, summe / divisor, color=self.colors['exc'], linewidth=4)
		# return l

	def input_tuning_mean_distribution(self, populations=['exc']):
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			for t in populations:
				means = np.mean(self.rawdata[t]['input_rates'], axis=1)
				plt.hist(means, color=self.colors[t])
				plt.axvline(0.5)
				plt.title('mean {0}'.format(np.mean(means)))

	def input_tuning(self, neuron=0, populations=['exc'], publishable=False,
					 plot_title=False):
		"""
		Plots input tuning from file

		Parameters
		----------
		neuron : int
			Number of input neuron whose tuning is plotted
		"""
		cmap_dict = {'exc': mpl.cm.Reds, 'inh': mpl.cm.Blues}
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			# plt.xticks([])
			# plt.yticks([])
			# plt.axis('off')
			positions = self.rawdata['positions_grid']
			if self.dimensions  == 1:
				plt.xlim([-self.radius, self.radius])
				for t in populations:
					input_rates = self.rawdata[t]['input_rates'][:, neuron]
					plt.plot(positions, input_rates, color=self.colors[t])
				if publishable:
					limit = self.radius # + self.params['inh']['center_overlap']
					linspace = np.linspace(-limit, limit, self.spacing)
					fig = plt.gcf()
					fig.set_size_inches(1.65, 1.0)
					ax = plt.gca()
					plt.setp(ax, xlim=[-self.radius, self.radius],
					xticks=[], yticks=[])
					xmin = linspace.min()
					xmax = linspace.max()
					ax.spines['right'].set_color('none')
					ax.spines['top'].set_color('none')
					ax.spines['left'].set_color('none')
					ax.spines['left'].set_position(('data', xmin))
					ax.spines['bottom'].set_position(('data', 0.0))
					# ax.yaxis.tick_right()
					ax.yaxis.set_label_position("right")
					plt.setp(ax, xlim=[-self.radius, self.radius],
					xticks=[], yticks=[0.])
					if populations[0] == 'exc':
						ax.xaxis.set_label_position("top")
						plt.ylabel('Exc', color=self.colors['exc'],
								   rotation='horizontal', labelpad=12.0)
						plt.arrow(-self.radius, 1.4, 2*self.radius, 0, lw=1,
								  length_includes_head=True, color='black',
								  head_width=0.2, head_length=0.1)
						plt.arrow(self.radius, 1.4, -2*self.radius, 0, lw=1,
								  length_includes_head=True, color='black',
								  head_width=0.2, head_length=0.1)
						plt.xlabel('2 m', fontsize=12, labelpad=0.)
					elif populations[0] == 'inh':
						plt.ylabel('Inh', color=self.colors['inh'],
									rotation='horizontal', labelpad=12.0)
					plt.ylim([0, 1.6])
					plt.margins(0.1)
			elif self.dimensions == 2:
				# plt.ylim([-self.radius, self.radius])
				linspace = np.linspace(-self.radius, self.radius, self.spacing)
				X, Y = np.meshgrid(linspace, linspace)
				for t in populations:
					input_rates = self.rawdata[t]['input_rates'][..., neuron]
					plt.contourf(X, Y, input_rates, 40, cmap=cmap_dict[populations[0]])
					# cb = plt.colorbar(format='%.2f')
					ax = plt.gca()
					ax.set_aspect('equal')
					ax.set_xticks([])
					ax.set_yticks([])
					if plot_title:
						# plt.colorbar()
						# title = '{0}'.format(int(np.amax(input_rates)))
						title = 'Exc.' if populations == ['exc'] else 'Inh.'
						plt.title(title, color=self.colors[populations[0]])
				# if publishable:
				# 	limit = self.radius # + self.params['inh']['center_overlap']
				# 	linspace = np.linspace(-limit, limit, self.spacing)
				# 	fig = plt.gcf()
				# 	fig.set_size_inches(1.65, 1.0)
				# 	ax = plt.gca()
				# 	plt.setp(ax, xlim=[-self.radius, self.radius],
				# 	xticks=[], yticks=[])
				# 	xmin = linspace.min()
				# 	xmax = linspace.max()
				# 	ax.spines['right'].set_color('none')
				# 	ax.spines['top'].set_color('none')
				# 	ax.spines['left'].set_color('none')
				# 	ax.spines['left'].set_position(('data', xmin))
				# 	ax.spines['bottom'].set_position(('data', 0.0))
				# 	# ax.yaxis.tick_right()
				# 	ax.yaxis.set_label_position("right")
				# 	plt.setp(ax, xlim=[-self.radius, self.radius],
				# 	xticks=[], yticks=[0.])



	def input_auto_correlation(self, neuron=0, populations=['exc'], publishable=False):
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			# plt.xlim([-self.radius, self.radius])
			positions = self.rawdata['positions_grid']
			if self.dimensions  == 1:
				for t in populations:
					input_rates = self.rawdata[t]['input_rates'][:, neuron]
					im = input_rates - np.mean(input_rates)
					ac = np.correlate(im, im, mode='same')
					plt.plot(positions, ac, color=self.colors[t])
					gauss = scipy.stats.norm(loc=0, scale=self.params[t]['sigma'] * np.sqrt(2)).pdf
					plt.plot(positions, gauss(positions), color=self.colors[t])

	def fields_polar(self, neuron=0, syn_type='exc', spacing=501,
					 publishable=False):
		"""
		Plots von Mises fields in a polar plot

		Note: since we use coordinates between -radius and radius also in
		the head direction, we need to scale the location and the width
		of the input field with (pi / radius) to map [-radius, radius]
		to [-pi, pi].
		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			radius = self.radius
			sigma = self.params[syn_type]['sigma'][-1]
			# The scaling
			sigma_scaled = sigma * np.pi / radius
			kappa = 1 / sigma_scaled**2
			location = self.rawdata[syn_type]['centers'][neuron,:,-1]
			location_scaled = location * np.pi / radius
			theta = np.linspace(-np.pi, np.pi, spacing)
			r = scipy.stats.vonmises(loc=location_scaled, kappa=kappa).pdf(theta)
			plt.polar(theta, r, color=self.colors[syn_type], lw=1)
			thetaticks = np.arange(0,360,90)
			ax = plt.gca()
			ax.set_aspect('equal')
			if publishable:
				plt.yticks([])
				plt.xlim([0, 20])
				ax.spines['polar'].set_visible(False)
				# ax.set_axis_bgcolor('0.9')
				# No angles written but still the 90 degree lines
				ax.set_thetagrids(thetaticks, [])
				r_max = ax.get_ylim()[-1]
				plt.ylim([0, r_max*1.1])

	# def fields(self, show_each_field=True, show_sum=False, neuron=0,
	# 		   populations=['exc'], publishable=False, alpha=1.0):
	# 	"""
	# 	Plotting of Gaussian Fields and their sum
	#
	# 	NOTE: For simulations newer than 2014/11/17, you should use
	# 			input_tuning() instead.
	#
	# 	Note: The sum gets divided by a something that depends on the
	# 			number of cells of the specific type, to make it fit into
	# 			the frame (see note in fields_times_weighs)
	# 	"""
	# 	for psp in self.psps:
	# 		self.set_params_rawdata_computed(psp, set_sim_params=True)
	# 		# Loop over different synapse types and color tuples
	# 		plt.xlim([-self.radius, self.radius])
	# 		# plt.xticks([])
	# 		# plt.yticks([])
	# 		# plt.axis('off')
	# 		if self.dimensions  == 1:
	# 			x = np.linspace(-self.radius, self.radius, 501)
	# 			# plt.xlabel('position')
	# 			# plt.ylabel('firing rate')
	# 			# plt.title('firing rate of')
	# 			for t in populations:
	# 				title = '%i fields per synapse' % len(self.rawdata[t]['centers'][neuron])
	# 				# plt.title(title)
	# 				legend = self.population_name[t]
	# 				summe = 0
	# 				for c, s in np.nditer([self.rawdata[t]['centers'][neuron], self.rawdata[t]['sigmas'][neuron]]):
	# 					gaussian = scipy.stats.norm(loc=c,
	# 														  scale=s).pdf
	# 					if show_each_field:
	# 						plt.plot(x, np.sqrt(2*np.pi*s**2)*gaussian(x),
	# 								 color=self.colors[t])
	# 					summe += np.sqrt(2*np.pi*s**2)*gaussian(x)
	# 				# for c, s in np.nditer([self.rawdata[t]['centers'][5], self.rawdata[t]['sigmas'][5]]):
	# 				# 	gaussian = scipy.stats.norm(loc=c, scale=s).pdf
	# 				# 	if show_each_field:
	# 				# 		plt.plot(x, gaussian(x), color=self.colors[t], label=legend)
	# 				# 	summe += gaussian(x)
	# 				if show_sum:
	# 					plt.plot(x, summe, color=self.colors[t], linewidth=1,
	# 							 label=legend, alpha=alpha)
	# 				# plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
	#
	# 			return x, summe
	# 			if publishable:
	# 				# fig = plt.gcf()
	# 				# fig.set_size_inches(1.65, 1.0)
	# 				# plt.margins(0.5)
	# 				# ax = plt.gca()
	# 				# plt.setp(ax, xlim=[-self.radius, self.radius],
	# 				# xticks=[], yticks=[])
	# 				limit = self.radius # + self.params['inh']['center_overlap']
	# 				linspace = np.linspace(-limit, limit, self.spacing)
	# 				fig = plt.gcf()
	# 				fig.set_size_inches(1.65, 1.0)
	# 				ax = plt.gca()
	# 				plt.setp(ax, xlim=[-self.radius, self.radius],
	# 				xticks=[], yticks=[])
	# 				xmin = linspace.min()
	# 				xmax = linspace.max()
	# 				ax.spines['right'].set_color('none')
	# 				ax.spines['top'].set_color('none')
	# 				ax.spines['left'].set_color('none')
	# 				ax.spines['left'].set_position(('data', xmin))
	# 				ax.spines['bottom'].set_position(('data', 0.0))
	# 				# ax.yaxis.tick_right()
	# 				ax.yaxis.set_label_position("right")
	# 				plt.setp(ax, xlim=[-self.radius, self.radius],
	# 				xticks=[], yticks=[0.])
	# 				if populations[0] == 'exc':
	# 					ax.xaxis.set_label_position("top")
	# 					plt.ylabel('Exc', color=self.colors['exc'],
	# 							   rotation='horizontal', labelpad=12.0)
	# 					plt.arrow(-self.radius, 1.4, 2*self.radius, 0, lw=1,
	# 							  length_includes_head=True, color='black',
	# 							  head_width=0.2, head_length=0.1)
	# 					plt.arrow(self.radius, 1.4, -2*self.radius, 0, lw=1,
	# 							  length_includes_head=True, color='black',
	# 							  head_width=0.2, head_length=0.1)
	# 					plt.xlabel('2 m', fontsize=12, labelpad=0.)
	# 				elif populations[0] == 'inh':
	# 					plt.ylabel('Inh', color=self.colors['inh'],
	# 								rotation='horizontal', labelpad=12.0)
	# 				plt.ylim([0, 1.6])
	# 				plt.margins(0.1)
	#
	# 		if self.dimensions  >= 2:
	# 			plt.ylim([-self.radius, self.radius])
	# 			ax = plt.gca()
	# 			ax.set_aspect('equal')
	# 			n_x = 100
	# 			n = np.array([n_x, n_x])
	# 			r = np.array([self.radius, self.radius])
	# 			linspace = np.linspace(-r[0], r[0], n[0])
	# 			X, Y = np.meshgrid(linspace, linspace)
	# 			positions = initialization.get_equidistant_positions(r, n)
	# 			for t in populations:
	# 				# In 2D sigma is only stored correctly in twoSigma2
	# 				# sigma = 1./np.sqrt(2.*self.rawdata[t]['twoSigma2'])[
	# 				# 	neuron,0,:]
	# 				sigma = self.params[t]['sigma'][:2]
	# 				summe = np.zeros(n)
	# 				for c in self.rawdata[t]['centers'][neuron]:
	# 					fields = figures.two_dim_input_tuning.field(
	# 								positions, c, sigma).reshape((n_x, n_x))
	# 					summe += fields
	# 					if show_each_field:
	# 						plt.contour(X, Y, fields)
	# 				if show_sum:
	# 					plt.contourf(X, Y, summe, 40, cmap=self.cms[t])
	#
	# 			plt.xticks([])
	# 			plt.yticks([])
	#
	# 	# These parameters are used in 1D to create the figures of general inputs
	# 	# to the network (on the Bernstein Poster the section 'Robustness
	# 	# to input tuning properties')
	# 	# y0, y1 = plt.ylim()
	# 	# plt.ylim([-1.5, y1+1.5])
	# 	# fig = plt.gcf()
	# 	# fig.set_size_inches(3,2)
	# 	return

	def input_current(self, time, spacing=51, populations=['exc', 'inh'], xlim=None):
		"""Plot either exc. or inh. input currents

		The input currents are given as weight vector times input rate
		vector calculated for each position x
		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			rawdata = self.rawdata
			frame = self.time2frame(time, weight=True)
			r = self.radius
			for syn_type in populations:
				n_syn = rawdata[syn_type]['number']
				if self.dimensions == 1:
					if xlim is None:
						xlim = [-r, r]
					x = np.linspace(xlim[0], xlim[1], spacing)
					x.shape = (spacing, 1, 1)
					input_rates = self.get_rates(x, syn_type)
					input_current = np.tensordot(
										rawdata[syn_type]['weights'][frame],
											input_rates, axes=([-1], [1]))
					input_current.shape = spacing
					x.shape = spacing
					plt.plot(x, input_current, lw=1, color=self.colors[syn_type])
					plt.xlim(xlim)
					ax = plt.gca()
					trans = mpl.transforms.blended_transform_factory(
							ax.transData, ax.transAxes)
					plt.vlines([-r, r], 0, 1,
							color='gray', lw=1, transform=trans, alpha=0.3)

				elif self.dimensions == 2:
					X, Y, positions_grid, input_rates = \
						self.get_X_Y_positions_grid_input_rates_tuple(spacing)
					input_current = np.tensordot(rawdata[syn_type]['weights'][
												frame],
								 input_rates[syn_type], axes=([-1],
														[self.dimensions]))
					input_current = input_current.reshape(spacing, spacing,
														  self.output_neurons)
					max = int(np.ceil(np.amax(input_current)))
					V = np.linspace(0, max, 30)
					plt.contourf(X, Y, input_current[..., 0].T, 30,
								 cmap=self.cms[syn_type], extend='max')
					cb = plt.colorbar()
					cb.set_label('Current')
					ax = plt.gca()
					ax.set_aspect('equal')
					ax.set_xticks([])
					ax.set_yticks([])



	def weight_statistics(self, time, syn_type='exc'):
		"""Plots mean, std and CV of weight vectors vs. fields per synapse

		Reasoning: We want to see if for more fields per synapse the weight
		vectors (after a grid has established) are still rather uniform.
		Rather uniform mean lower CV (coefficient of variation). This would
		mean that different states are closer in weight space and thus
		transition between states are more likely to occur. It is a possible
		explanation for the instability observed for many fields per synapse.
		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			frame = self.time2frame(time, weight=True)

			# Dictionary with key (fps, seed) tuple and value [mean, std] list
			fps_seed_mean_std = {}
			p = syn_type
			weights = self.rawdata[p]['weights'][frame]
			std = np.std(weights)
			mean = np.mean(weights)
			fps_seed_mean_std[
					(self.params['exc']['fields_per_synapse'],
						self.params['sim']['seed_centers'])] = [mean, std]

			ax = plt.gca()
			ax.set_xscale('log', basex=2)
			for k, v in fps_seed_mean_std.items():
				fps, seed = k
				# We shift the fps by +/- 5% to avoid overlapping points
				# in the plot. (Should be a nicer way to do this)
				plt.plot(0.95*fps, v[0], marker='o', color='blue')
				plt.plot(1.05*fps, v[1], marker='^', color='red')
				plt.plot(fps, v[1]/v[0], marker='s', color='green')

		# We plot the last points again separately to get the legend
		# only once.
		plt.plot(0.95*fps, v[0], marker='o', color='blue', label='mean')
		plt.plot(1.05*fps, v[1], marker='^', color='red', label='std')
		plt.plot(fps, v[1]/v[0], marker='s', color='green', label='CV')
		plt.legend(loc='best')
		plt.xlabel('Fields per synapse')
		plt.ylabel(syn_type)



	def weights_vs_centers(self, time, populations=['exc', 'inh'], marker='o'):
		"""Plots the current weight at each center"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			frame = self.time2frame(time, weight=True)
			# plt.title(syn_type + ' Weights vs Centers' + ', ' + 'Frame = ' + str(frame), fontsize=8)
			# Note: shape of centers (number_of_synapses, fields_per_synapses)
			# shape of weights (number_of_output_neurons, number_of_synapses)
			for p in populations:
				centers = self.rawdata[p]['centers']
				weights = self.rawdata[p]['weights'][frame]
				# sigma = self.params[p]['sigma']
				centers = np.mean(centers, axis=1)
				plt.plot(np.squeeze(centers), np.squeeze(weights),
					color=self.colors[p], marker=marker, markersize=1.0)
			# limit = self.radius + self.params['inh']['center_overlap']
			# plt.xlim(-limit, self.radius)
			plt.xlim([-self.radius, self.radius])

	def weight_evolution(self, syn_type='exc', time_sparsification=1,
						 weight_sparsification=1, output_neuron=0):
		"""
		Plots the time evolution of synaptic weights.

		The case of multiple output neurons needs to be treated separately
		----------
		Arguments:
		- syn_type: type of the synapse
		- time_sparsification: factor by which the time resolution is reduced
		- weight_sparsification: factor by which the number of weights
									is reduced

		----------
		Remarks:
		- If you use an already sparsified weight array as input, the center
			 color-coding won't work
		"""

		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			syn = self.rawdata[syn_type]
			plt.title(syn_type + ' weight evolution', fontsize=8)
			# Create time array, note that you need to add 1, because you also
			# have time 0.0
			time = np.linspace(
				0, self.simulation_time,
				num=(self.simulation_time / time_sparsification
					/ self.every_nth_step_weights + 1))
			# Loop over individual weights (using sparsification)
			# Note the arange takes as an (excluded) endpoint the length of the
			# first weight array
			# assuming that the number of weights is constant during the simulation
			if not self.params['sim']['lateral_inhibition']:
				for i in np.arange(0, len(syn['weights'][0]), weight_sparsification):
					# Create array of the i-th weight for all times
					weight = syn['weights'][:,i]
					center = syn['centers'][i]
					# Take only the entries corresponding to the sparsified times
					weight = general_utils.arrays.take_every_nth(
								weight, time_sparsification)
					if self.dimensions == 2:
						center = center[0]

					# if self.params['exc']['fields_per_synapse'] == 1 and self.params['inh']['fields_per_synapse'] == 1:
					# 	# Specify the range of the colormap
					# 	color_norm = mpl.colors.Normalize(-self.radius, self.radius)
					# 	# Set the color from a color map
					# 	color = mpl.cm.rainbow(color_norm(center))
					# 	plt.plot(time, weight, color=color)
					# else:
			else:
				for i in np.arange(0, self.params[syn_type]['n'],
									 weight_sparsification):
					weight = syn['weights'][:,output_neuron,i]
					center = syn['centers'][i]
					weight = general_utils.arrays.take_every_nth(weight,
								time_sparsification)

			plt.plot(weight)

	def sigma_histogram(self, populations=['exc'], bins=10):
		"""Plots histogram of sigmas for each dimension"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			fig = plt.figure()
			counter = itertools.count(1)
			for n, p in enumerate(populations):
				sigmas = self.rawdata[p]['sigmas']
				for d in np.arange(self.dimensions):
					plt.subplot(self.dimensions, len(populations), next(counter))
					plt.title('Dimension ' + str(d))
					plt.hist(sigmas[..., d], bins=bins, color=self.colors[p])

	def output_rate_distribution(self, start_time=0):
		n_bins = 100
		positions = self.positions[:,0][start_time:,]
		output_rates = self.output_rates[start_time:,]
		dx = 2*self.radius / n_bins
		bin_centers = np.linspace(dx, 2*self.radius-dx, num=n_bins)
		mean_output_rates = []
		for i in np.arange(0, n_bins):
			indexing = (positions >= i*dx) & (positions < (i+1)*dx)
			mean_output_rates.append(np.mean(output_rates[indexing]))
		plt.plot(bin_centers, mean_output_rates, marker='o')
		plt.axhline(y=self.target_rate, linewidth=3, linestyle='--', color='black')

	def position_distribution(self, start_time=0, end_time=-1,
							  bins=20):
		"""
		Plots histogram of visited locations

		Note: 1D needs to be updated
		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			if self.dimensions == 1:
				x = self.positions[:,0]
				n, bins, patches = plt.hist(x, 50, normed=True, facecolor='green',
											alpha=0.75)
			elif self.dimensions == 2:
				start_frame = self.time2frame(start_time)
				end_frame = self.time2frame(end_time)
				plt.hist2d(self.rawdata['positions'][start_frame:end_frame, 0],
						   self.rawdata['positions'][start_frame:end_frame, 1],
						   bins=bins,
						   range=np.array([[-self.radius, self.radius], [-self.radius, self.radius]])
				)
				plt.colorbar()
				ax = plt.gca()
				ax.set_aspect('equal')
				plt.title('end_time = {0:.1e}'.format(end_time))


	def get_1d_grid_score(self, output_rates, linspace, neighborhood_size=7,
							threshold_percentage=30.,
							return_maxima_arrays=True,
							at_least_n_maxima=4):
		"""Returns 1 dimensioinal grid score

		Grid score is defined as 1-CV, where CV is the coefficient of
		variation inter-maxima distances.
		The grid score is set to zero if:
		1. There are less than `at_least_n_maxima` maxima.
		2. The distance between leftmost (rightmost) maximum and left (right)
			wall is 1.5 times larger than the grid_spacing
			(This ensures that there is not just a grid on one side of the
				system)
		
		Note: Maybe add different methods
		
		Parameters
		----------
		threshold_percentage : float
			Percentage of mean with which values have to vary within
			neighborhood_size to qualify for a maximum.
		neighborhood_size : int
			see function get_local_maxima_boolean
		
		Returns
		-------
		maxima_positions : ndarray
			Positions of the maxima
		maxima_values : ndarray
			Values at the maxima positions
		grid_score : float
		"""
		threshold_difference = np.mean(output_rates) * threshold_percentage/100.
		maxima_boolean = general_utils.arrays.get_local_maxima_boolean(
					output_rates, neighborhood_size, threshold_difference)
		maxima_positions = linspace[maxima_boolean]
		maxima_values = output_rates[maxima_boolean]
		distances_between_maxima = (np.abs(maxima_positions[:-1]
										- maxima_positions[1:]))
		grid_spacing = np.mean(distances_between_maxima)
		grid_std = np.std(distances_between_maxima)

		enough_maxima = len(maxima_positions) >= at_least_n_maxima
		if enough_maxima:
			distances_to_wall = np.array([maxima_positions[0] + self.radius,
									 		self.radius - maxima_positions[-1]])
			even_maxima_distribution = np.all(distances_to_wall < 1.5*grid_spacing) 
			if even_maxima_distribution:
				grid_score = 1-grid_std/grid_spacing
			else:
				grid_score = 0.
		else:
			grid_score = 0.

		if return_maxima_arrays:
			ret = maxima_positions, maxima_values, grid_score
		else:
			ret = grid_score
		return ret

	def get_watsonU2(self, spacing, output_rates):
		"""Watson U2 from output_rates
		
		Parameters
		----------
		spacing : int
			Same spacing as used in get_output_rates function
		output_rates : ndarray
			Return from get_output_rates
		Returns
		-------
		U2, h : tuple (float, bool)
			WatsonU2 value and True if (P < 0.01)
		"""
		theta = np.linspace(0, 2*np.pi, spacing)
		b = output_rates[...,0].T
		r = np.mean(b, axis=1)
		hd_tuning = observables.Head_Direction_Tuning(r, spacing)
		U2, h = hd_tuning.get_watson_U2_against_uniform()
		return U2, h


	def watsonU2_vs_grid_score(self, time, spacing=None, from_file=True,
			precomputed=False):
		"""Plot watsunU2 vs. grid score like in Sargoling 2006 Fig. 3
		
		Parameters
		----------
		time : float
		spacing : int
		from_file : bool
		precomputed : bool
			If True grid_score and Watson U2 are read from self.computed
			Make sure to use add_computed.py to get these values first
		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			frame = self.time2frame(time, weight=True)
			if spacing is None:
				spacing = self.spacing

			if not precomputed:
				# WATSON
				output_rates = self.get_output_rates(frame, spacing, from_file)
				U2, h = self.get_watsonU2(spacing, output_rates)
				# Grid score
				spatial_tuning = self.get_spatial_tuning(output_rates)
				linspace = np.linspace(-self.radius, self.radius, spacing)
				maxima_positions, maxima_values, grid_score = self.get_1d_grid_score(
					 spatial_tuning, linspace)


			else:
				U2 = self.computed['U2'][frame]
				grid_score = self.computed['grid_score'][frame]
				
			ax = plt.gca()
			ax.set_yscale('log')
			for x, y in np.nditer([grid_score, U2]):
				# NOTE: in plt.plot below you set the alpha value of the circles
				# You need invisible circles if you want to only show the
				# numbers, because you something to annotate.
				# NOTE: Different colors correspond to different center seeds
				circle1=plt.Circle((x, y), 0.1, ec='black', fc='none', lw=2, color=color_cycle_blue3[self.params['sim']['seed_centers']])
				plt.annotate(self.params['sim']['seed_sigmas'], (x, y), va='center', ha='center', color=color_cycle_blue3[self.params['sim']['seed_centers']])
				# Use these two lines instead of the two above if you want
				# to show the time evolution of different seeds
				# circle1=plt.Circle((x, y), 0.1, ec='black', fc='none', lw=2, color=color_cycle[frame-1])
				# plt.annotate(self.params['sim']['seed_sigmas'], (x, y), va='center', ha='center', color=color_cycle[frame-1])
			
			# Use this to plot circles
			# plt.plot(grid_score, U2, marker='o', linestyle='none',
			# 	color=color_cycle[self.params['sim']['seed_centers']])
			# Use this to only plot numbers
			plt.plot(grid_score, U2, alpha=0.)
			plt.xlabel('Grid score')
			plt.ylabel("Watson's U2" )
			plt.xlim([0, 1])
			plt.ylim([min([1, plt.ylim()[0]]), max([1e3, plt.ylim()[1]])])
			plt.margins(0.05)

	def watson_u2_vs_grid_score_with_examples(self, time, spacing=None,
											 from_file=True):
		"""Like watsonU2_vs_grid_score(precomputed=True) but with 4 snapshots

		For the extrema in U2 and grid_score the correspondning firing rates
		are shown and connected to the point in the Sargolini figue with an
		arrow
		
		Note: Requires precomputed WatsonU2 and grid_score
		
		Parameters
		----------
		See function watsonU2_vs_grid_score
		"""
		##############################################################
		##########	Find extrem grid score and U2 values	##########
		##############################################################
		GS_U2_seedSigma = []
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			frame = self.time2frame(time, weight=True)
			GS_U2_seedSigma.append(
				(	self.computed['grid_score'][frame],
					self.computed['U2'][frame],
					self.params['sim']['seed_sigmas']
					)
				)

		max_GS_seed = max(GS_U2_seedSigma, key=lambda tup: tup[0])[2]
		min_GS_seed = min(GS_U2_seedSigma, key=lambda tup: tup[0])[2]
		max_U2_seed = max(GS_U2_seedSigma, key=lambda tup: tup[1])[2]
		min_U2_seed = min(GS_U2_seedSigma, key=lambda tup: tup[1])[2]

		seed_sigmas_list = list(set([max_GS_seed, min_GS_seed, max_U2_seed, min_U2_seed]))
		counter = itertools.count(0)
		while len(seed_sigmas_list) < 4:
			c = next(counter)
			seed_sigmas_list = list(set(seed_sigmas_list + [c]))

		# Set list manually
		# seed_sigmas_list = [34, 29, 0, 6]

		gs = GridSpec(4, 3)
		ax1 = plt.subplot(gs[:, 1])

		ax2 = plt.subplot(gs[0, 2])
		ax3 = plt.subplot(gs[1, 2], polar=True)
		ax4 = plt.subplot(gs[2, 2])
		ax5 = plt.subplot(gs[3, 2], polar=True)
		ax6 = plt.subplot(gs[0, 0])
		ax7 = plt.subplot(gs[1, 0], polar=True)
		ax8 = plt.subplot(gs[2, 0])
		ax9 = plt.subplot(gs[3, 0], polar=True)

		ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
		counter = itertools.count(1)
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			seed_sigmas = self.params['sim']['seed_sigmas']
			seed_centers = self.params['sim']['seed_centers']
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			frame = self.time2frame(time, weight=True)
			if spacing is None:
				spacing = self.spacing

			U2 = self.computed['U2'][frame]
			grid_score = self.computed['grid_score'][frame]
			
			plt.sca(ax_list[0])
			ax = plt.gca()
			ax.set_yscale('log')
			for x, y in np.nditer([grid_score, U2]):
				circle1=plt.Circle((x, y), 0.1, ec='black', fc='none', lw=2, color=color_cycle_blue3[self.params['sim']['seed_centers']])
				ax.annotate(seed_sigmas, (x, y), va='center', ha='center', color=color_cycle_blue3[self.params['sim']['seed_centers']])
			# plt.plot(grid_score, U2, marker='o', linestyle='none',
				# color=color_cycle[self.params['sim']['seed_centers']])
			plt.plot(grid_score, U2, alpha=0.)
			plt.xlabel('Grid score')
			plt.ylabel("Watson's U2" )
			plt.xlim([0, 1])
			plt.ylim([min([1, ax.get_ylim()[0]]), max([1e3, ax.get_ylim()[1]])])
			plt.margins(0.05)


			# seed_sigmas_list = [5, 7, 15, 18]
			# seed_centers_list = [1, 0, 0, 1]
			# NOTE: Finally you want a figure with just one center seed,
			# because we are interested in the influence of the sigma
			# distribution.
			if seed_sigmas in seed_sigmas_list and seed_centers == 0:
				##################################
				##########	Grid Plots	##########
				##################################
				c = next(counter)
				print c
				plt.sca(ax_list[c])
				ax = plt.gca()
				output_rates = self.get_output_rates(frame, spacing, from_file)
				spatial_tuning = self.get_spatial_tuning(output_rates)
				color = 'black'
				limit = self.radius
				linspace = np.linspace(-limit, limit, spacing)
				plt.plot(linspace, spatial_tuning, color=color, lw=2)
				maxima_positions, maxima_values, grid_score = (
					self.get_1d_grid_score(spatial_tuning, linspace,
						neighborhood_size=7))
				plt.plot(maxima_positions, maxima_values, marker='o',
							linestyle='none', color='red')
				title = 'GS = %.2f' % grid_score
				plt.ylim(0, ax.get_ylim()[1])
				# y0, y1 = ax.get_ylim()
				# Allows to use different transforms for x and y axis
				trans = mpl.transforms.blended_transform_factory(
							ax.transData, ax.transAxes)
				plt.vlines([-self.radius, self.radius], 0, 1,
							color='gray', lw=2, transform = trans)
				x0, x1 = ax.get_xlim()
				plt.hlines([self.params['out']['target_rate']], x0, x1,
							color='black',linestyle='dashed', lw=2)
				plt.title(title, size=10)
				plt.xticks([])
				plt.locator_params(axis='y', nbins=3)
				plt.xlabel('Position')
				plt.ylabel('Firing rate')
				##################################
				##########	HD Plots	##########
				##################################
				plt.sca(ax_list[next(counter)])
				ax = plt.gca()
				theta = np.linspace(0, 2*np.pi, spacing)
				# if self.dimensions == 2:
				b = output_rates[...,0].T
				r = np.mean(b, axis=1)
				plt.polar(theta, r)
				# mpl.rc('font', size=10)
				thetaticks = np.arange(0,360,180)
				ax.set_thetagrids(thetaticks, frac=1.4)
				ax.set_aspect('equal')
				# if show_watson_U2:
				# hd_tuning = observables.Head_Direction_Tuning(r, spacing)
				# U2, h = hd_tuning.get_watson_U2_against_uniform()
				plt.title('Watson U2: %.2f' % U2, size=10)
				######################################
				##########	Connection Patch	##########
				######################################

				# We want different start of the path on the right and on
				# on the left side
				xyA = (0.0, 1.0) if c<5 else (1.0, 1.0)
				con = ConnectionPatch(
					xyA=xyA, xyB=(grid_score, U2),
					coordsA='axes fraction', coordsB='data',
					axesA=ax, axesB=ax_list[0],
					arrowstyle='->',
					shrinkA=0,
					shrinkB=0
					)
				ax.add_artist(con)

	def print_init_weight_ratio(self):
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			weight_ratio = self.rawdata['exc']['weights'][0] / self.rawdata['inh']['weights'][0]
			print '******* WEIGHT RATIO (exc/inh) *******'
			print 'Fields per synapse'
			print self.params['exc']['fields_per_synapse']
			print 'All'
			print weight_ratio
			print 'Mean'
			print np.mean(weight_ratio)
			print '****************************'


	def input_norm(self, populations=['exc', 'inh'], ylim=None):
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			rawdata = self.rawdata
			for p in populations:
				if self.dimensions == 1:
					plt.plot(rawdata[p]['input_norm'], color=self.colors[p])
					plt.ylim(ylim)

				elif self.dimensions == 2:
					centers_x = rawdata[p]['centers'][:,0,0]
					centers_y = rawdata[p]['centers'][:,0,1]
					colors =  rawdata[p]['input_norm']
					# color_norm = mpl.colors.Normalize(np.amin(colors), np.amax(colors))
					# plt.scatter(centers_x, centers_y, c=colors, alpha=0.5, cmap=mpl.cm.jet, norm=color_norm)
					plt.scatter(centers_x, centers_y, c='blue', alpha=0.1)
					# plt.colorbar()


					lim = [-self.radius-0.05, self.radius+0.05]
					plt.xlim(lim)
					plt.ylim(lim)
					ax = plt.gca()
					ax.set_aspect('equal')
					plt.hlines([-self.radius, self.radius], -self.radius, self.radius)
					plt.vlines([-self.radius, self.radius], -self.radius, self.radius)

					# plt.contourf(rawdata['exc']['centers'][:,0,0].reshape((10, 10)))

	def fields(self, show_each_field=True, show_sum=False, neuron=0,
				   populations=['exc'], publishable=False, alpha=1.0):
			"""
			Plotting of Gaussian Fields and their sum

			NOTE: For simulations newer than 2014/11/17, you should use
					input_tuning() instead.

			Note: The sum gets divided by a something that depends on the
					number of cells of the specific type, to make it fit into
					the frame (see note in fields_times_weighs)
			"""
			for psp in self.psps:
				self.set_params_rawdata_computed(psp, set_sim_params=True)
				# Loop over different synapse types and color tuples
				plt.xlim([-self.radius, self.radius])
				# plt.xticks([])
				# plt.yticks([])
				# plt.axis('off')
				if self.dimensions  == 1:
					x = np.linspace(-self.radius, self.radius, 501)
					# plt.xlabel('position')
					# plt.ylabel('firing rate')
					# plt.title('firing rate of')
					for t in populations:
						title = '%i fields per synapse' % len(self.rawdata[t]['centers'][neuron])
						# plt.title(title)
						legend = self.population_name[t]
						summe = 0
						for c, s in np.nditer([self.rawdata[t]['centers'][neuron], self.rawdata[t]['sigmas'][neuron]]):
							gaussian = scipy.stats.norm(loc=c,
																  scale=s).pdf
							if show_each_field:
								plt.plot(x, np.sqrt(2*np.pi*s**2)*gaussian(x),
										 color=self.colors[t])
							summe += np.sqrt(2*np.pi*s**2)*gaussian(x)
						# for c, s in np.nditer([self.rawdata[t]['centers'][5], self.rawdata[t]['sigmas'][5]]):
						# 	gaussian = scipy.stats.norm(loc=c, scale=s).pdf
						# 	if show_each_field:
						# 		plt.plot(x, gaussian(x), color=self.colors[t], label=legend)
						# 	summe += gaussian(x)
						if show_sum:
							plt.plot(x, summe, color=self.colors[t], linewidth=1,
									 label=legend, alpha=alpha)
						# plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

					if publishable:
						# fig = plt.gcf()
						# fig.set_size_inches(1.65, 1.0)
						# plt.margins(0.5)
						# ax = plt.gca()
						# plt.setp(ax, xlim=[-self.radius, self.radius],
						# xticks=[], yticks=[])
						limit = self.radius # + self.params['inh']['center_overlap']
						linspace = np.linspace(-limit, limit, self.spacing)
						fig = plt.gcf()
						fig.set_size_inches(1.65, 1.0)
						ax = plt.gca()
						plt.setp(ax, xlim=[-self.radius, self.radius],
						xticks=[], yticks=[])
						xmin = linspace.min()
						xmax = linspace.max()
						ax.spines['right'].set_color('none')
						ax.spines['top'].set_color('none')
						ax.spines['left'].set_color('none')
						ax.spines['left'].set_position(('data', xmin))
						ax.spines['bottom'].set_position(('data', 0.0))
						# ax.yaxis.tick_right()
						ax.yaxis.set_label_position("right")
						plt.setp(ax, xlim=[-self.radius, self.radius],
						xticks=[], yticks=[0.])
						if populations[0] == 'exc':
							ax.xaxis.set_label_position("top")
							plt.ylabel('Exc', color=self.colors['exc'],
									   rotation='horizontal', labelpad=12.0)
							# arrow_kwargs = dict(head_width=0.2,
							# 					head_length=0.05,
							# 					color='black',
							# 					length_includes_head=True,
							# 					lw=1)
							# plt.arrow(0, 1.4, 0.97*self.radius, 0,
							# 		  **arrow_kwargs)
							# plt.arrow(0, 1.4, -0.97*self.radius, 0,
							# 		  **arrow_kwargs)
							# plt.xlabel('2 m', fontsize=12, labelpad=0.)
						elif populations[0] == 'inh':
							plt.ylabel('Inh', color=self.colors['inh'],
										rotation='horizontal', labelpad=12.0)
						plt.ylim([0, 1.6])
						plt.margins(0.1)

				if self.dimensions  >= 2:
					plt.ylim([-self.radius, self.radius])
					ax = plt.gca()
					ax.set_aspect('equal')
					n_x = 100
					n = np.array([n_x, n_x])
					r = np.array([self.radius, self.radius])
					linspace = np.linspace(-r[0], r[0], n[0])
					X, Y = np.meshgrid(linspace, linspace)
					positions = initialization.get_equidistant_positions(r, n)
					for t in populations:
						# In 2D sigma is only stored correctly in twoSigma2
						# sigma = 1./np.sqrt(2.*self.rawdata[t]['twoSigma2'])[
						# 	neuron,0,:]
						sigma = self.params[t]['sigma'][:2]
						summe = np.zeros(n)
						print summe.shape
						for c in self.rawdata[t]['centers'][neuron]:
							fields = figures.two_dim_input_tuning.field(
										positions, c, sigma).reshape((n_x, n_x))
							summe += fields
							if show_each_field:
								print fields.shape
								plt.contour(X, Y, fields)
						if show_sum:
							plt.contourf(X, Y, summe, 40, cmap=self.cms[t])

					plt.xticks([])
					plt.yticks([])