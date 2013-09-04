import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.stats
mpl.rcParams.update({'figure.autolayout': True})
# print mpl.rcParams.keys()
# mpl.rcParams['animation.frame_format'] = 'jpeg'
# print mpl.rcParams['animation.frame_format']

class Animation:
	"""Class for different animations"""
	def __init__(self, rat, rawdata, start_time):
		self.rat = rat
		self.params = rat.params
		for k, v in rat.params.items():
			setattr(self, k, v)
		for k, v in rawdata.items():
			setattr(self, k, v)

		self.start_time = start_time
		start_value = int(start_time / self.dt)
		self.positions = self.positions[start_value:,]
		self.exc_weights = self.exc_weights[start_value:,]
		self.inh_weights = self.inh_weights[start_value:,]
		self.output_rates = self.output_rates[start_value:,]
		self.exc_centers = rat.exc_syns.centers
		self.exc_sigmas = rat.exc_syns.sigmas
		self.inh_centers = rat.inh_syns.centers
		self.inh_sigmas = rat.inh_syns.sigmas

		self.fig = plt.figure()
		self.fig.patch.set_alpha(0.5)

	def animate_all_synapses(self, save_path=False, interval=50):
		"""
		Returns (or saves) an animation with all synapses and the moving rat.
		"""
		# We use gridspec to arrange the plots
		# See the matplotlib tutorial
		# plt.title('test asdf')
		# Create grid with 1 rows and 15 columns
		gs = mpl.gridspec.GridSpec(10, 15)
		artist_frame_tuples = (
			self.get_artist_frame_tuples_position(plot_location=gs[-1, :-1])
			+ self.get_artist_frame_tuples_time(plot_location=gs[0:1, 0:3])
			+ self.get_artist_frame_tuples_weights(plot_location=gs[1:-1, :-1])
			+ self.get_artist_frame_tuples_output_rate(plot_location=gs[1:-1, -1])
		)
		artists = []
		for i in xrange(0, len(self.positions)):
			artists.append([a[0] for a in artist_frame_tuples if a[1] == i])
		ani = animation.ArtistAnimation(
			self.fig, artists, interval=interval, repeat_delay=3000, blit=True)
		if save_path:
			Writer = animation.writers['ffmpeg']
			writer = Writer(fps=(1000/interval), metadata=self.params, bitrate=1)
			ani.save(
				save_path,
				writer=writer)
			return
		else:
			plt.draw()
			return

	def animate_positions(self, save_path=False, interval=50):
		gs = mpl.gridspec.GridSpec(1, 1)
		artist_frame_tuples = (
			self.get_artist_frame_tuples_position(plot_location=gs[0:1])
			+ self.get_artist_frame_tuples_time(plot_location=gs[0:1])
			# + self.get_artist_frame_tuples_weights(plot_location=gs[1:-1, :-1])
			# + self.get_artist_frame_tuples_output_rate(plot_location=gs[1:-1, -1])
		)
		artists = []
		for i in xrange(0, len(self.positions)):
			artists.append([a[0] for a in artist_frame_tuples if a[1] == i])
		ani = animation.ArtistAnimation(
			self.fig, artists, interval=interval, repeat_delay=3000, blit=True)
		if save_path:
			Writer = animation.writers['ffmpeg']
			writer = Writer(fps=(1000/interval), metadata=self.params, bitrate=1)
			ani.save(
				save_path,
				writer=writer)
			return
		else:
			plt.draw()
			return

	def get_artist_frame_tuples_output_rates_from_equation(self, plot_location=111):
		ax = self.fig.add_subplot(plot_location)
		ax.set_xticks([])
		ax.set_yticks([])
		a_f_tuples = []

		for n, p in enumerate(self.positions):
			# This conditional is just to make the initial
			# position dot disappear
			if n == 0:
				color = 'white'
			else:
				color = 'b'
			l, = ax.plot(p[0], p[1], marker='o', color=color, markersize=18)
			a_f_tuples.append((l, n))
		return a_f_tuples

	def get_artist_frame_tuples_weights(self, plot_location=111):
		"""
		Returns tuples of artists and frame numbers for the synapse Gaussians
		"""
		ax = self.fig.add_subplot(plot_location)
		### Setting the axes limits ###
		# x axis is simple
		ax.set_xlim(0, self.boxlength)
		ax.set_ylabel('Synaptic Strength')
		# plt.title('Norm = ' + self.params['normalization']
		# 			 + 'n_exc = ' + str(self.params['n_exc'])
		# 			 + 'n_inh = ' + str(self.params['n_inh'])
		# 			 + 'sigma_exc = ' + str(self.params['sigma_exc']))
		# For the y axis we need to get the maximum of all ocurring values
		# and maybe the minimum
		maxlist = []
		# minlist = []
		# Append two values to the list maxlist (one exc, one inh)
		# Late take the maximum of both
		# Note: Dividing by sqrt(2 pi sigma^2) gives the
		# maximum of the Gaussian (that is what we want)
		maxlist.append((
			np.amax(self.exc_weights)
			/ np.sqrt(2 * np.pi * self.sigma_exc**2)))
		maxlist.append((
			np.amax(self.inh_weights)
			/ np.sqrt(2 * np.pi * self.sigma_inh**2)))
		ylimit = max(maxlist)  # Take the larger of the two values
		ax.set_ylim(0, ylimit)
		time_steps = len(self.exc_weights)
		time_steps_array = np.arange(0, time_steps)
		x = np.linspace(0, self.boxlength, 200)
		a_f_tuples = []
		for n in time_steps_array:
			# exc synapses
			for c, s, w in np.nditer(
					[self.exc_centers, self.exc_sigmas, self.exc_weights[n]]):
				gaussian = scipy.stats.norm(loc=c, scale=s).pdf
				l, = ax.plot(x, w * gaussian(x), color='g')
				a_f_tuples.append((l, n))
			# inh synapses
			for c, s, w in np.nditer(
					[self.inh_centers, self.inh_sigmas, self.inh_weights[n]]):
				gaussian = scipy.stats.norm(loc=c, scale=s).pdf
				l, = ax.plot(x, w * gaussian(x), color='r')
				a_f_tuples.append((l, n))
		return a_f_tuples

	def get_artist_frame_tuples_time(self, plot_location=111):
		"""
		Returns tuples of artists and frame numbers for the time box
		"""
		dt = self.params['dt']
		ax = self.fig.add_subplot(plot_location)
		ax.set_xticks([])
		ax.set_yticks([])
		a_f_tuples = []
		for n, p in enumerate(self.positions):
			txt = ax.text(
				# Specified in axis coords, (see transform below)
				0.02, 0.95, 'Time = ' + str(n * dt + self.start_time),
				horizontalalignment='left',
				verticalalignment='top',
				bbox=dict(facecolor='gray', alpha=0.2),  # Draw box around text
				transform=ax.transAxes)  # Text positions in axis coords,not in data coords
			a_f_tuples.append((txt, n))
		return a_f_tuples

	def get_artist_frame_tuples_position(self, plot_location=111):
		"""
		Returns tuples of artists and frame numbers for a single moving dot
		"""
		ax = self.fig.add_subplot(plot_location)
		ax.set_xlim(0, self.boxlength)
		ax.set_ylim(0, self.boxlength)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlabel('Rat Position')
		a_f_tuples = []

		for n, p in enumerate(self.positions):
			# This conditional is just to make the initial
			# position dot disappear
			if n == 0:
				color = 'white'
			else:
				color = 'b'
			l, = ax.plot(p[0], p[1], marker='o', color=color, markersize=18)
			a_f_tuples.append((l, n))
		return a_f_tuples

	def get_artist_frame_tuples_output_rate(self, plot_location=111):
		"""
		Returns tuples of artists and frame numbers of the output rate as a bar
		"""
		ax = self.fig.add_subplot(plot_location)
		a_f_tuples = []
		ax.set_xlim(0, 1)
		ax.set_ylim(0, max(self.output_rates))
		ax.set_xticks([])
		ax.yaxis.set_ticks_position('right')
		ax.axhline(y=self.target_rate, linewidth=5, linestyle='-', color='black')
		ax.text(
			1.1, self.target_rate, 'Target Rate',
			fontsize=14, verticalalignment='center')
		for n, r in enumerate(self.output_rates):
			if r >= self.target_rate:
				l, = ax.bar(0.25, r, 0.5, color='g')
			else:
				l, = ax.bar(0.25, r, 0.5, color='r')
			a_f_tuples.append((l, n))
		return a_f_tuples


# def positions_animation(params, positions, save_path=False):
# 	"""
# 	Creates an animation of the rat moving through space.

# 	- positions: Array: [[x1, y2], [x2, y2], ...]
# 	- save: Save a video .mp4
# 	"""
# 	boxlength = params['boxlength']
# 	dt = params['dt']
# 	fig = plt.figure()
# 	plt.xlim(0, 1)
# 	ax = fig.add_subplot(111)
# 	ax.set_xlim(0, boxlength)
# 	ax.set_ylim(0, boxlength)
# 	ax.set_xlabel('Position')
# 	ax.set_title('Moving Rat')

# 	l, = ax.plot([], [], 'o-')
# 	txt = ax.text(
# 			0.02, 0.95, '',  # Specified in axis coords, (see transform below)
# 			horizontalalignment='left',
# 			verticalalignment='top',
# 			bbox=dict(facecolor='gray', alpha=0.2),  # Draw box around text
# 			transform=ax.transAxes)  # Text positions in axis coords, not in data coords

# 	def _update_lines_and_text(num, positions, line, text):
# 		line.set_data(positions[num])
# 		text.set_text('Time = ' + str(num*dt))
# 		return line, text

# 	def _init():
# 		l.set_data([], [])
# 		txt.set_text('')
# 		return l, txt

# 	ani = animation.FuncAnimation(
# 			fig, _update_lines_and_text, np.arange(0, len(positions)),
# 			fargs=(positions, l, txt),
# 			interval=50, blit=True, repeat_delay=2000, repeat=True, init_func=_init)

# 	if save_path:
# 		ani.save(
# 			save_path,
# 			writer=animation.FFMpegFileWriter(),
# 			metadata={'artist': 'Simon Weber'})
# 	else:
# 		return ani


# def positions_and_weigths_animation(
# 	params, positions,
# 	exc_centers, inh_centers,
# 	exc_sigmas, inh_sigmas,
# 	exc_weights_for_all_times, inh_weights_for_all_times, save_path=False):
# 	"""
# 	Animation with moving rat and weighted place fields
# 	"""
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)
# 	n_exc = params['n_exc']

# 	### Setting the axes limits ###
# 	# x axis is simple
# 	ax.set_xlim(0, params['boxlength'])
# 	# For the y axis we need to get the maximum of all ocurring values
# 	# and maybe the minimu
# 	maxlist = []
# 	minlist = []
# 	for a in exc_weights_for_all_times:
# 		maxlist.append(np.amax(a))
# 		minlist.append(np.amin(a))
# 	# Now we have the maximum of all weights
# 	# Dividing by sqrt(2 pi sigma^2) gives the maximum of the Gaussian
# 	ax.set_ylim(0, max(maxlist) / np.sqrt(2 * np.pi * 0.05**2))
# 	time_steps = len(exc_weights_for_all_times)
# 	time_steps_array = np.arange(0, time_steps)
# 	x = np.linspace(0, params['boxlength'], 200)

# 	lines = []
# 	# It has to be done like this
# 	# (see my_examples/matplotlib/animation_with_arrays.py)
# 	for i in time_steps_array:
# 		l, = ax.plot([], [])
# 		lines.append(l)

# 	def _update_lines_and_text(exc_weights, *lines):
# 		"""
# 		Update function that is passed to FuncAnimation

# 		Arguments:
# 			- exc_weights: Array of exc_weights
# 			- *lines: List of Line2D instances
# 		"""
# 		# Zip numpy arrays to make them simultaneously iterable
# 		it_exc = np.nditer(
# 			[np.arange(0, n_exc), exc_centers, exc_sigmas, exc_weights])
# 		for n, c, s, w in it_exc:
# 			l = lines[n]
# 			gaussian = norm(loc=c, scale=s).pdf
# 			l.set_data(x, w * gaussian(x))
# 		return lines

# 	def _init():
# 		for l in lines:
# 			l.set_data([], [])
# 		return lines

# 	# The Animation
# 	# Note that exc_weights_for_all_times is an array of exc weight arrays
# 	# which come from each time step
# 	ani = animation.FuncAnimation(
# 		fig, _update_lines_and_text, exc_weights_for_all_times, fargs=(lines),
# 		interval=100, blit=True, repeat_delay=3000, init_func=_init)
# 	writer = animation.FFMpegFileWriter()
# 	if save_path:
# 		ani.save(
# 			save_path,
# 			writer=writer,
# 			metadata={'artist': 'Simon Weber'})
# 		return
# 	else:
# 		return ani
