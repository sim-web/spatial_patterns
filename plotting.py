import matplotlib as mpl
import math
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy import signal
import initialization
import general_utils.snep_plotting
import general_utils.arrays
import general_utils.plotting
from general_utils.plotting import color_cycle_blue3
import analytics.linear_stability_analysis
import utils
import observables
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D

# from matplotlib._cm import cubehelix
mpl.rcParams.update({'figure.autolayout': True})
# print mpl.rcParams.keys()
# mpl.rcParams['animation.frame_format'] = 'jpeg'
# print mpl.rcParams['animation.frame_format']

def make_segments(x, y):
	'''
	Taken from http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

	Create list of line segments from x and y coordinates, in the correct format for LineCollection:
	an array of the form   numlines x (points per line) x 2 (x and y) array
	'''

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


def plot_list(fig, plot_list):
	"""
	Takes a list of lambda forms of plot functions and plots them such that
	no more than four rows are used to assure readability
	"""
	n_plots = len(plot_list)
	# A title for the entire figure (super title)
	# fig.suptitle('Time evolution of firing rates', y=1.1)
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
		# ax = plt.gca()
		# if n == 1 or n == 2:
		# 	# title = r'$\sigma_{\mathrm{inh}} = %.1f $' % 0.05
		# 	# plt.title(title, y=1.02, size=26)
		# 	ax.get_xaxis().set_ticklabels([])
		# if n == 1 or n == 3:
		# 	# plt.title('Initially')
		# 	plt.ylabel('firing rate')
		# if n == 3 or n == n_plots:
		# 	# plt.title('Finally')
		# 	plt.xlabel('position')
		p()

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

class Plot(initialization.Synapses, initialization.Rat,
			general_utils.snep_plotting.Plot):
	"""Class with methods related to plotting

	Parameters
	----------
	tables : snep tables object
	psps : list of paramspace points
	params, rawdata : see general_utils.snep_plotting.Plot
	"""

	def __init__(self, tables=None, psps=[None], params=None, rawdata=None):
		general_utils.snep_plotting.Plot.__init__(self, params, rawdata)
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


	def spike_map(self, small_dt, start_frame=0, end_frame=-1):
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			positions = self.rawdata['positions']
			output_rates = self.rawdata['output_rates']
			print self.every_nth_step_weights
			# print positions
			plt.xlim(-self.radius, self.radius)
			plt.ylim(-self.radius, self.radius)

			plt.plot(
				positions[start_frame:end_frame,0],
				positions[start_frame:end_frame,1], color='black', linewidth=0.5)

			rates_x_y = np.nditer(
				[output_rates[start_frame:end_frame],
				positions[start_frame:end_frame, 0],
				positions[start_frame:end_frame, 1]])
			for r, x, y in rates_x_y:
					if r * small_dt > np.random.random():
						plt.plot(x, y, marker='o',
							linestyle='none', markeredgecolor='none', markersize=3, color='r')
			title = '%.1e to %.1e' % (start_frame, end_frame)
			plt.title(title, fontsize=8)

			ax = plt.gca()
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
		print r
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
			# print start_frame
			mean = np.mean(self.rawdata['output_rates'][start_frame:], axis=0)
			legend = 'Mean:' + str(mean)
			plt.hlines(mean, xmin=start_time_for_mean, xmax=max(time), lw=4,
						color='red', label=legend, zorder=4)

		plt.legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize=8)

			# plt.axhline(mean[1], xmin=start_frame)
			# print mean

	def output_rates_vs_position(self, start_frame=0, clipping=False):
		if self.dimensions == 1:
			_positions = self.positions[:,0][start_frame:,]
			_output_rates = self.output_rates[start_frame:,]
			plt.plot(_positions, _output_rates, linestyle='none', marker='o', alpha=0.5)
		if self.dimensions == 2:
			positions = self.positions[start_frame:,]
			output_rates = self.output_rates[start_frame:,]
			plt.xlim(-self.radius, self.radius)
			plt.ylim(-self.radius, self.radius)
			if clipping:
				color_norm = mpl.colors.Normalize(0, np.amax(output_rates)/10000.0)
			else:
				color_norm = mpl.colors.Normalize(np.amin(output_rates), np.amax(output_rates))
			for p, r in zip(positions, output_rates):
				color = mpl.cm.YlOrRd(color_norm(r))
				plt.plot(p[0], p[1], linestyle='none', marker='s', markeredgecolor='none', color=color, markersize=5, alpha=0.5)
		# ax = plt.gca()
		# ax.set_aspect('equal')
		# ax.set_xticks([])
		# ax.set_yticks([])

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
		Computes the values of all place field Gaussians at `position

		Inherited from Synapses
		"""
		get_rates = self.get_rates_function(position, data=self.rawdata[syn_type])
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
			print r
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

	def get_X_Y_positions_grid_rates_grid_tuple(self, spacing):
		"""
		Returns X, Y meshgrid and position_grid and rates_grid for contour plot

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
		- rates_grid: dictionary of two arrays, one exc and one inh.
				Following the matrix structure of positions_grid, each entry in this
				"matrix" (note: it is an array, not a np.matrix) is the array of
				firing rates of the neuron type at this position
		ISSUES:
		- Probably X, Y creation can be circumvented elegantly with the positions_grid
		- Since it is only used once per animation (it was created just for this purpose)
			it is low priority
		"""
		rates_grid = {}
		# Set up X, Y for contour plot
		x_space = np.linspace(-self.radius, self.radius, spacing)
		y_space = np.linspace(-self.radius, self.radius, spacing)
		X, Y = np.meshgrid(x_space, y_space)
		positions_grid = np.dstack([X.T, Y.T])

		positions_grid.shape = (spacing, spacing, 1, 1, 2)
		# if self.boxtype == 'circular':
		# 	distance = np.sqrt(X*X + Y*Y)
		# 	positions_grid[distance>self.radius] = np.nan
		rates_grid['exc'] = self.get_rates(positions_grid, 'exc')
		rates_grid['inh'] = self.get_rates(positions_grid, 'inh')
		return X, Y, positions_grid, rates_grid


	def output_rate_heat_map(self, start_time=0, end_time=-1, spacing=None,
			maximal_rate=False, number_of_different_colors=50,
			equilibration_steps=10000, from_file=False):
		"""Plot evolution of output rate from equation vs time

		Time is the vertical axis. Linear space is the horizontal axis.
		Output rate is given in color code.

		Parameters
		----------
		- start_time, end_time: (int) determine the time range
		- spacing: (int) resolution along the horizontal axis
						(note: the resolution along the vertical axis is given
							by the data)
		- maximal_rate: (float) Above this value everything is plotted in black.
 						This is useful if smaller values should appear in
 						more detail. If left as False, the largest appearing
 						value of all occurring output rates is taken.
 		- number_of_different_colors: (int) Number of colors used for the
 											color coding
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
			fig = plt.figure()
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
				maximal_rate = int(np.ceil(np.amax(output_rates)))
			V = np.linspace(0, maximal_rate, number_of_different_colors)
			plt.ylabel('Time')
			# plt.xlabel('Position')
			if lateral_inhibition:
				cm_list = [mpl.cm.Blues, mpl.cm.Greens, mpl.cm.Reds, mpl.cm.Greys]
				cm = mpl.cm.Blues
				for n in np.arange(int(self.params['sim']['output_neurons'])):
					cm = cm_list[n]
					my_masked_array = np.ma.masked_equal(output_rates[...,n], 0.0)
					plt.contourf(X, Y, my_masked_array, V, cmap=cm, extend='max')
			else:
				cm = mpl.cm.gnuplot_r
				# cm = mpl.cm.binary
				plt.contourf(X, Y, output_rates[...,0], V, cmap=cm, extend='max')
			cm.set_over('black', 1.0) # Set the color for values higher than maximum
			cm.set_bad('white', alpha=0.0)
			ax = plt.gca()
			plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
			plt.locator_params(axis='y', nbins=3)
			ax.invert_yaxis()
			ticks = np.linspace(0.0, maximal_rate, 3)
			cb = plt.colorbar(format='%.0f', ticks=ticks)
			cb.set_label('Firing rate')
			plt.xticks([])

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


	def plot_grid_spacing_vs_parameter(self, time=-1, spacing=None,
		from_file=False, parameter_name=None, parameter_range=None,
		plot_mean_inter_peak_distance=False):
		"""Plot grid spacing vs parameter

		Plot the grid spacing vs. the parameter both from data and from
		the analytical equation.

		Parameters
		----------
		time : float
			Time at which the grid spacing should be determined
		from_file : bool
			If True, output rates are taken from file
		spacing : int
			Needs only be specified if from_file is False
		parameter_name : str
			Name of parameter against which the grid spacing shall be plotted
		parameter_range : ndarray
			Range of this parameter. This array also determines the plotting
			range.
		"""
		# param_vs_auto_corr = []
		# param_vs_interpeak_distance = []
		fig = plt.gcf()
		fig.set_size_inches(4.6, 4.1)
		mpl.rcParams['legend.handlelength'] = 1.0
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)

			# Map string of parameter name to parameters in the file
			# TO BE CHANGED: cumbersome!
			if parameter_name == 'sigma_inh':
				parameter = self.params['inh']['sigma']
			elif parameter_name == 'sigma_exc':
				parameter = self.params['exc']['sigma']

			if spacing is None:
				spacing = self.spacing
			frame = self.time2frame(time, weight=True)
			output_rates = self.get_output_rates(frame, spacing, from_file,
								squeeze=True)
			# Get the auto-correlogram
			correlogram = scipy.signal.correlate(
							output_rates, output_rates, mode='same')
			# Obtain grid spacing by taking the first peak of the correlogram
			gridness = observables.Gridness(correlogram, self.radius, 10, 0.1)
			gridness.set_spacing_and_quality_of_1d_grid()
			# plt.errorbar(parameter, gridness.grid_spacing, yerr=0.0,
			# 			marker='o', color=self.color_cycle_blue3[1])
			# param_vs_auto_corr.append(np.array([parameter, gridness.grid_spacing]))
			# Plot grid spacing from inter peak distance of firing rates
			if plot_mean_inter_peak_distance:
				maxima_boolean = general_utils.arrays.get_local_maxima_boolean(
								output_rates, 5, 0.1)
				x_space = np.linspace(-self.radius, self.radius, spacing)
				peak_positions = x_space[maxima_boolean]
				distances_between_peaks = (np.abs(peak_positions[:-1]
												- peak_positions[1:]))
				grid_spacing = np.mean(distances_between_peaks)
				plt.plot(parameter, grid_spacing, marker='o',
							color=color_cycle_blue3[0], alpha=1.0,
							linestyle='none', markeredgewidth=0.0, lw=1)
				# param_vs_interpeak_distance.append(np.array([parameter, grid_spacing]))
		plt.plot(parameter, grid_spacing, marker='o',
							color=color_cycle_blue3[0], alpha=1.0, label=r'Simulation',
							linestyle='none', markeredgewidth=0.0, lw=1)
		plt.autoscale(tight=True)
		plt.margins(0.02)
		mpl.rcParams.update({'figure.autolayout': True})
		mpl.rc('font', size=18)
		# plt.legend(loc='best', numpoints=1)

		# np.save('temp_data/sigma_inh_vs_auto_corr_R7',
		# 			np.array(param_vs_auto_corr))
		# np.save('temp_data/sigma_inh_vs_interpeak_distance_R7',
		# 			np.array(param_vs_interpeak_distance))


		# If a parameter name and parameter are given, the grid spacing
		# is plotted from the analytical results
		if parameter_name and parameter_range is not None:
			# Set the all the values
			self.target_rate = self.params['out']['target_rate']
			self.w0E = self.params['exc']['init_weight']
			self.eta_inh = self.params['inh']['eta']
			self.sigma_inh = self.params['inh']['sigma']
			self.n_inh = self.params['inh']['number_desired']
			self.eta_exc = self.params['exc']['eta']
			self.sigma_exc = self.params['exc']['sigma']
			self.n_exc = self.params['exc']['number_desired']
			self.boxlength = 2*self.radius

			# Set the varied parameter values again
			setattr(self, parameter_name, parameter_range)

			analytics.linear_stability_analysis.plot_grid_spacing_vs_parameter(
				self.target_rate, self.w0E, self.eta_inh, self.sigma_inh,
				self.n_inh, self.eta_exc, self.sigma_exc, self.n_exc,
				self.boxlength, parameter_name)
			# Set xlabel manually
			# plt.xlabel(r'Excitatory width $\sigma_{\mathrm{E}}$')
			plt.xlabel(r'Inhibitory width $\sigma_{\mathrm{I}}$')

		plt.locator_params(axis='x', nbins=5)
		plt.locator_params(axis='y', nbins=5)

		# ax = plt.gca()
		# ax.set_xticks(np.linspace(0.015, 0.045, 3))
		# plt.ylim(0.188, 0.24)
		# plt.ylim(0.18, 0.84)

	def plot_correlogram(self, time, spacing=None, mode='full', method=False,
				from_file=False):
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

		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			radius = self.radius
			dimensions = self.dimensions
			frame = self.time2frame(time, weight=True)
			if spacing is None:
				spacing = self.params['sim']['spacing']
			if mode == 'full':
				corr_radius = 2*radius
				corr_spacing = 2*spacing-1
			elif mode == 'same':
				corr_radius = radius
				corr_spacing = spacing

			corr_linspace = np.linspace(-corr_radius, corr_radius, corr_spacing)
			# Get the output rates
			output_rates = self.get_output_rates(frame, spacing, from_file,
								squeeze=True)
			if dimensions == 1:
				correlogram = scipy.signal.correlate(
								output_rates, output_rates, mode=mode)
				plt.plot(corr_linspace, correlogram)
				gridness = observables.Gridness(correlogram, radius, 10, 0.1)
				gridness.set_spacing_and_quality_of_1d_grid()
				title = 'Spacing: %.3f, Quality: %.3f' % (
							gridness.grid_spacing, gridness.quality)
				plt.title(title)
				ax = plt.gca()
				y0, y1 = ax.get_ylim()
				plt.ylim((y0, y1))
				plt.vlines([-gridness.grid_spacing, gridness.grid_spacing], y0, y1,
								color='green', linestyle='dashed', lw=2)
			if dimensions == 2:
				corr_spacing, correlogram = observables.get_correlation_2d(
									output_rates, output_rates, mode=mode)
				corr_linspace = np.linspace(-corr_radius, corr_radius, corr_spacing)
				X_corr, Y_corr = np.meshgrid(corr_linspace, corr_linspace)
				# V = np.linspace(-0.21, 1.0, 40)
				V = 40
				plt.contourf(X_corr.T, Y_corr.T, correlogram, V)
				# plt.contourf(X_corr.T, Y_corr.T, correlogram, 30)
				# cb = plt.colorbar()
				ax = plt.gca()
				self.set_axis_settings_for_contour_plots(ax)
				title = 't=%.2e' % time
				if method:
					gridness = observables.Gridness(
						correlogram, radius, method=method)
					title += ', grid score = %.2f, spacing = %.2f' \
								% (gridness.get_grid_score(), gridness.grid_spacing)
					for r, c in [(gridness.inner_radius, 'black'),
								(gridness.outer_radius, 'black'),
								(gridness.grid_spacing, 'white'),	]:
						circle = plt.Circle((0,0), r, ec=c, fc='none', lw=2,
												linestyle='dashed')
						ax.add_artist(circle)
				ticks = np.linspace(-0.05, 1.0, 2)
				cb = plt.colorbar(format='%.1f', ticks=ticks)
				cb.set_label('Correlation')
				mpl.rc('font', size=42)

				# plt.title(title, fontsize=8) 

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
			output_rates = self.rawdata['output_rate_grid'][frame]

		else:
			rates_grid = {}

			if self.dimensions == 1:
				limit = self.radius # +self.params['inh']['center_overlap']
				linspace = np.linspace(-limit, limit, spacing)
				positions_grid = linspace.reshape(spacing, 1, 1)
				for t in ['exc', 'inh']:
					rates_grid[t] = self.get_rates(positions_grid, syn_type=t)
				output_rates = self.get_output_rates_from_equation(
					frame=frame, rawdata=self.rawdata, spacing=spacing,
					positions_grid=False, rates_grid=rates_grid,
					equilibration_steps=10000)
			if self.dimensions == 2:
				X, Y, positions_grid, rates_grid = self.get_X_Y_positions_grid_rates_grid_tuple(spacing)
				output_rates = self.get_output_rates_from_equation(
						frame=frame, rawdata=self.rawdata, spacing=spacing,
						positions_grid=positions_grid, rates_grid=rates_grid)

		if squeeze:
			output_rates = np.squeeze(output_rates)
		return output_rates

	def plot_head_direction_polar(self, time, spacing=None, from_file=False):
		"""Plots polar plot of head direction distribution

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
			theta = np.linspace(0, 2*np.pi, spacing)
			if self.dimensions == 2:
				b = output_rates[...,0].T
				r = np.mean(b, axis=1)
			elif self.dimensions == 3:
				 r = np.mean(output_rates[..., 0], axis=(1, 0)).T
			plt.polar(theta, r)

	def plot_grids_linear(self, time, spacing=None, from_file=False):
		"""Plots linear plot of grid firing rate vs position

		Parameters
		----------
		See parameters for plot_output_rates_from_equation
		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			frame = self.time2frame(time, weight=True)

			if spacing is None:
				spacing = self.spacing

			linspace = np.linspace(-self.radius , self.radius, spacing)
			X, Y = np.meshgrid(linspace, linspace)
			# Get the output rates
			output_rates = self.get_output_rates(frame, spacing, from_file)
			b = output_rates[...,0].T
			plt.plot(linspace, np.mean(b, axis=0))


	def plot_output_rates_from_equation(self, time, spacing=None, fill=False,
					from_file=False, number_of_different_colors=30,
					maximal_rate=False, plot_spatial_tuning=True):
		"""Plots output rates using the weights at time `time

		Parameters
		----------
		- frame: (int) Frame of the simulation that shall be plotted
		- spacing: (int) The output will contain spacing**dimensions data points
		- fill: (boolean) If True the contour plot will be filled, if False
							it will be just lines
		Returns
		-------

		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			frame = self.time2frame(time, weight=True)
			if spacing is None:
				spacing = self.spacing

			linspace = np.linspace(-self.radius , self.radius, spacing)
			X, Y = np.meshgrid(linspace, linspace)
			distance = np.sqrt(X*X + Y*Y)
			# Get the output rates
			output_rates = self.get_output_rates(frame, spacing, from_file)

			# np.save('test_output_rates', output_rates)
			##############################
			##########	Plot	##########
			##############################
			if self.dimensions == 1:
				output_rates = np.squeeze(output_rates)
				# plt.ylim(0.0, 2.0)
				# plt.xlim(-5, 5)
				# color='#FDAE61'
				color = 'black'
				limit = self.radius # + self.params['inh']['center_overlap']
				linspace = np.linspace(-limit, limit, spacing)
				plt.plot(linspace, output_rates, color=color, lw=2)
				# Plot positions of centers which have been located
				# maxima_boolean = general_utils.arrays.get_local_maxima_boolean(
				# 			output_rates, 5, 0.1)
				# peak_positions = linspace[maxima_boolean]
				# plt.plot(peak_positions, np.ones_like(peak_positions),
				# 			marker='s', color='red', linestyle='none')
				ax = plt.gca()
				y0, y1 = ax.get_ylim()
				# plt.ylim((y0, y1))
				plt.vlines([-self.radius, self.radius], y0, y1,
							color='gray', lw=2)
				x0, x1 = ax.get_xlim()
				# plt.ylim((y0, y1))
				plt.hlines([self.params['out']['target_rate']], x0, x1,
							color='black',linestyle='dashed', lw=2)
				# plt.yticks(['rho'])
				# title = 'time = %.0e' % (frame*self.every_nth_step_weights)
				# plt.title(title, size=16)
				# plt.ylim([0, 10.0])
				plt.xticks([])
				plt.locator_params(axis='y', nbins=3)
				# ax.set_yticks((0, self.params['out']['target_rate'], 5, 10))
				# ax.set_yticklabels((0, r'$\rho_0$', 5, 10), fontsize=18)
				plt.xlabel('Position')
				plt.ylabel('Firing rate')
				fig = plt.gcf()
				# fig.set_size_inches(5,2.1)
				fig.set_size_inches(5,3.5)

			if self.dimensions >= 2:
				# title = r'$\vec \sigma_{\mathrm{inh}} = (%.2f, %.2f)$' % (self.params['inh']['sigma_x'], self.params['inh']['sigma_y'])
				# plt.title(title, y=1.04, size=36)
				title = 't=%.2e' % time
				# plt.title(title, fontsize=8)
				cm = mpl.cm.jet
				cm.set_over('y', 1.0) # Set the color for values higher than maximum
				cm.set_bad('white', alpha=0.0)
				# V = np.linspace(0, 3, 20)
				if not maximal_rate:
					maximal_rate = int(np.ceil(np.amax(output_rates)))
				V = np.linspace(0, maximal_rate, number_of_different_colors)
				# mpl.rc('font', size=42)
				
				# Hack to avoid error in case of vanishing output rate at every position
				# If every entry in output_rates is 0, you define a norm and set
				# one of the elements to a small value (such that it looks like zero)
				if np.count_nonzero(output_rates) == 0:
					color_norm = mpl.colors.Normalize(0., 100.)
					output_rates[0][0] = 0.000001
					plt.contourf(X, Y, output_rates[...,0].T, V, norm=color_norm, cmap=cm, extend='max')
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
							# print self.rawdata['exc']['centers']
							if plot_spatial_tuning:
								# For plotting of spatial tuning
								a = np.mean(output_rates[..., 0], axis=2).T
							else:
								# For plotting of just two axes
								a = output_rates[11, :, :, 0].T
							plt.contourf(X, Y, a, V, cmap=cm, extend='max')
							# output_rates[...,0][distance>self.radius] = np.nan
						elif self.dimensions == 2:
							plt.contourf(X, Y, output_rates[..., 0].T, V, cmap=cm, extend='max')

				plt.margins(0.01)
				# plt.axis('off')
				# ticks = np.linspace(0.0, maximal_rate, 2)
				# cb = plt.colorbar(format='%i', ticks=ticks)
				# cb = plt.colorbar(format='%i')
				plt.colorbar()
				# cb.set_label('Firing rate')
				ax = plt.gca()
				self.set_axis_settings_for_contour_plots(ax)
				# fig = plt.gcf()
				# fig.set_size_inches(6.5,6.5)
				# else:

				# 	if np.count_nonzero(output_rates) == 0:
				# 		color_norm = mpl.colors.Normalize(0., 100.)
				# 		output_rates[0][0] = 0.000001
				# 		if self.boxtype == 'circular':
				# 			distance = np.sqrt(X*X + Y*Y)
				# 			output_rates[distance>self.radius] = np.nan
				# 		plt.contour(X, Y, output_rates.T, V, norm=color_norm, cmap=cm, extend='max')
				# 	else:
				# 		if self.boxtype == 'circular':
				# 			distance = np.sqrt(X*X + Y*Y)
				# 			output_rates[distance>self.radius] = np.nan
				# 		if self.lateral_inhibition:
				# 			plt.contour(X, Y, output_rates[:,:,0].T, V, cmap=cm, extend='max')
				# 		else:
				# 			plt.contour(X, Y, output_rates.T, V, cmap=cm, extend='max')

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

	def fields(self, show_each_field=True, show_sum=False, neuron=0):
		"""
		Plotting of Gaussian Fields and their sum

		Note: The sum gets divided by a something that depends on the
				number of cells of the specific type, to make it fit into
				the frame (see note in fields_times_weighs)
		"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			# Loop over different synapse types and color tuples
			plt.xlim([-self.radius, self.radius])
			x = np.linspace(-self.radius, self.radius, 501)
			plt.xticks([])
			plt.yticks([])
			plt.axis('off')
			# plt.xlabel('position')
			# plt.ylabel('firing rate')
			# plt.title('firing rate of')
			self.populations = ['inh']
			for t in self.populations:
				title = '%i fields per synapse' % len(self.rawdata[t]['centers'][neuron])
				# plt.title(title)
				legend = self.population_name[t]
				summe = 0
				for c, s in np.nditer([self.rawdata[t]['centers'][neuron], self.rawdata[t]['sigmas'][neuron]]):
					gaussian = scipy.stats.norm(loc=c, scale=s).pdf
					if show_each_field:
						plt.plot(x, gaussian(x), color=self.colors[t])
					summe += gaussian(x)
				# for c, s in np.nditer([self.rawdata[t]['centers'][5], self.rawdata[t]['sigmas'][5]]):
				# 	gaussian = scipy.stats.norm(loc=c, scale=s).pdf
				# 	if show_each_field:
				# 		plt.plot(x, gaussian(x), color=self.colors[t], label=legend)
				# 	summe += gaussian(x)
				if show_sum:
					plt.plot(x, summe, color=self.colors[t], linewidth=2, label=legend)
				# plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
		y0, y1 = plt.ylim()
		plt.ylim([-1, y1+1])
		fig = plt.gcf()
		fig.set_size_inches(3,2)
		return

	def weights_vs_centers(self, time, syn_type='exc'):
		"""Plots the current weight at each center"""
		for psp in self.psps:
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			frame = self.time2frame(time, weight=True)
			# plt.title(syn_type + ' Weights vs Centers' + ', ' + 'Frame = ' + str(frame), fontsize=8)
			# limit = self.radius + self.params['inh']['center_overlap']
			# plt.xlim(-limit, self.radius)
			# Note: shape of centers (number_of_synapses, fields_per_synapses)
			# shape of weights (number_of_output_neurons, number_of_synapses)
			populations = [syn_type]
			if syn_type == 'both':
				populations = ['exc', 'inh']
			for p in populations:
				centers = self.rawdata[p]['centers']
				weights = self.rawdata[p]['weights'][frame]
				# sigma = self.params[p]['sigma']
				plt.plot(np.squeeze(centers), np.squeeze(weights), color=self.colors[p], marker='o')

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
					# 	print center
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

	def position_distribution(self):
		x = self.positions[:,0]
		n, bins, patches = plt.hist(x, 50, normed=True, facecolor='green', alpha=0.75)