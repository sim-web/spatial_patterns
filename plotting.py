import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.stats import norm


class Animation:
	"""Class for different animations"""
	def __init__(self, rat):
		self.rat = rat
		self.params = rat.params
		self.sigma_exc = self.params['sigma_exc']
		self.sigma_inh = self.params['sigma_inh']
		self.boxlength = self.params['boxlength']
		self.positions = rat.positions
		self.exc_centers = rat.exc_syns.centers
		self.exc_sigmas = rat.exc_syns.sigmas
		self.exc_weights_for_all_times = rat.exc_weights
		self.inh_centers = rat.inh_syns.centers
		self.inh_sigmas = rat.inh_syns.sigmas
		self.inh_weights_for_all_times = rat.inh_weights
		self.output_rates = rat.output_rates
		self.target_rate = self.params['target_rate']
		self.fig = plt.figure()

	def animate_all_synapses(self, save_path=False, interval=50):
		"""
		Returns (or saves) an animation with all synapses and the moving rat.
		"""
		# We use gridspec to arrange the plots
		# See the matplotlib tutorial

		# Create grid with 1 rows and 15 columns
		gs = mpl.gridspec.GridSpec(1, 15)
		artist_frame_tuples = (
			self.get_artist_frame_tuples_position(plot_location=gs[0, :-1])
			+ self.get_artist_frame_tuples_time(plot_location=gs[0, :-1])
			+ self.get_artist_frame_tuples_weights(plot_location=gs[0, :-1])
			+ self.get_artist_frame_tuples_output_rate(plot_location=gs[0, -1])
		)
		artists = []
		for i in xrange(0, len(self.positions)):
			artists.append([a[0] for a in artist_frame_tuples if a[1] == i])
		ani = animation.ArtistAnimation(
			self.fig, artists, interval=interval, repeat_delay=3000, blit=True)
		if save_path:
			ani.save(
				save_path,
				writer=animation.FFMpegFileWriter(),
				metadata={'artist': 'Simon Weber'})
			return
		else:
			plt.draw()
			return

	def get_artist_frame_tuples_weights(self, plot_location=111):
		"""
		Returns tuples of artists and frame numbers for the synapse Gaussians
		"""
		ax = self.fig.add_subplot(plot_location)
		### Setting the axes limits ###
		# x axis is simple
		ax.set_xlim(0, self.boxlength)
		ax.set_ylabel('Synaptic Strength')
		# For the y axis we need to get the maximum of all ocurring values
		# and maybe the minimum
		maxlist = []
		# minlist = []
		# Append two values to the list maxlist (one exc, one inh)
		# Late take the maximum of both
		# Note: Dividing by sqrt(2 pi sigma^2) gives the
		# maximum of the Gaussian (that is what we want)
		maxlist.append((
			np.amax(self.exc_weights_for_all_times)
			/ np.sqrt(2 * np.pi * self.sigma_exc**2)))
		maxlist.append((
			np.amax(self.inh_weights_for_all_times)
			/ np.sqrt(2 * np.pi * self.sigma_inh**2)))
		ylimit = max(maxlist)  # Take the larger of the two values
		ax.set_ylim(0, ylimit)
		time_steps = len(self.exc_weights_for_all_times)
		time_steps_array = np.arange(0, time_steps)
		x = np.linspace(0, self.boxlength, 200)
		a_f_tuples = []
		for n in time_steps_array:
			# exc synapses
			for c, s, w in np.nditer(
					[self.exc_centers, self.exc_sigmas, self.exc_weights_for_all_times[n]]):
				gaussian = norm(loc=c, scale=s).pdf
				l, = ax.plot(x, w * gaussian(x), color='g')
				a_f_tuples.append((l, n))
			# inh synapses
			for c, s, w in np.nditer(
					[self.inh_centers, self.inh_sigmas, self.inh_weights_for_all_times[n]]):
				gaussian = norm(loc=c, scale=s).pdf
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
				0.02, 0.95, 'Time = ' + str(n * dt),
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
		a_f_tuples = []

		for n, p in enumerate(self.positions):
			l, = ax.plot(p[0], p[1], marker='o', color='b', markersize=18)
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





def positions_ArtistAnimation(params, positions):
	fig = plt.figure()
	artist_frame_tuples = (
		get_artist_frame_tuples_position(params, fig, positions)
		+ get_artist_frame_tuples_time(params, fig, positions)
	)
	print artist_frame_tuples
	artists = []
	for i in xrange(0, len(positions)):
		artists.append([a[0] for a in artist_frame_tuples if a[1] == i])
	print artists
	ani = animation.ArtistAnimation(
		fig, artists, interval=500, repeat_delay=3000, blit=True)
	plt.draw()
	return

def weights_ArtistAnimaton(
	params, positions,
	exc_centers, exc_sigmas, exc_weights_for_all_times):
	fig = plt.figure()
	artist_frame_tuples = get_artist_frame_tuples_weights(
								params, fig, positions, exc_centers, exc_sigmas, exc_weights_for_all_times)
	print artist_frame_tuples
	artists = []
	for i in xrange(0, len(positions)):
		artists.append([a[0] for a in artist_frame_tuples if a[1] == i])
	print artists
	ani = animation.ArtistAnimation(
		fig, artists, interval=100, repeat_delay=3000, blit=True)
	plt.draw()
	return


def get_artist_frame_tuples_weights(
	params, fig, positions,
	exc_centers, exc_sigmas, exc_weights_for_all_times,
	plot_location=111):
	ax = fig.add_subplot(plot_location)
	n_exc = params['n_exc']

	### Setting the axes limits ###
	# x axis is simple
	ax.set_xlim(0, params['boxlength'])
	ax.set_ylabel('Synaptic Strength')
	# For the y axis we need to get the maximum of all ocurring values
	# and maybe the minimu
	maxlist = []
	minlist = []
	for a in exc_weights_for_all_times:
		maxlist.append(np.amax(a))
		minlist.append(np.amin(a))
	# Now we have the maximum of all weights
	# Dividing by sqrt(2 pi sigma^2) gives the maximum of the Gaussian
	ax.set_ylim(0, max(maxlist) / np.sqrt(2 * np.pi * 0.05**2))
	time_steps = len(exc_weights_for_all_times)
	time_steps_array = np.arange(0, time_steps)
	x = np.linspace(0, params['boxlength'], 200)

	a_f_tuples = []
	for n in time_steps_array:
		lines = []
		for c, s, w in np.nditer([exc_centers, exc_sigmas, exc_weights_for_all_times[n]]):
			gaussian = norm(loc=c, scale=s).pdf
			l, = ax.plot(x, w * gaussian(x))
			a_f_tuples.append((l, n))
	return a_f_tuples




def get_artist_frame_tuples_time(params, fig, positions, plot_location=111):
	"""
	Returns Text artist and frame number tuple to show the time
	"""
	dt = params['dt']
	ax = fig.add_subplot(plot_location)
	a_f_tuples = []
	for n, p in enumerate(positions):
		txt = ax.text(
			# Specified in axis coords, (see transform below)
			0.02, 0.95, 'Time = ' + str(n * dt),
			horizontalalignment='left',
			verticalalignment='top',
			bbox=dict(facecolor='gray', alpha=0.2),  # Draw box around text
			transform=ax.transAxes)  # Text positions in axis coords, not in data coords
		a_f_tuples.append((txt, n))
	return a_f_tuples


def get_artist_frame_tuples_position(
	params, fig, positions, plot_location=111):
	"""
	Returns artist frame_number tuple of a single moving dot
	"""
	boxlength = params['boxlength']
	ax = fig.add_subplot(plot_location)
	ax.set_xlim(0, boxlength)
	ax.set_ylim(0, boxlength)
	a_f_tuples = []

	for n, p in enumerate(positions):
		l, = ax.plot(p[0], p[1], 'o-', color='b')
		a_f_tuples.append((l, n))
	return a_f_tuples


def positions_and_weigths_animation(
	params, positions,
	exc_centers, inh_centers,
	exc_sigmas, inh_sigmas,
	exc_weights_for_all_times, inh_weights_for_all_times, save_path=False):
	"""
	Animation with moving rat and weighted place fields
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111)
	n_exc = params['n_exc']

	### Setting the axes limits ###
	# x axis is simple
	ax.set_xlim(0, params['boxlength'])
	# For the y axis we need to get the maximum of all ocurring values
	# and maybe the minimu
	maxlist = []
	minlist = []
	for a in exc_weights_for_all_times:
		maxlist.append(np.amax(a))
		minlist.append(np.amin(a))
	# Now we have the maximum of all weights
	# Dividing by sqrt(2 pi sigma^2) gives the maximum of the Gaussian
	ax.set_ylim(0, max(maxlist) / np.sqrt(2 * np.pi * 0.05**2))
	time_steps = len(exc_weights_for_all_times)
	time_steps_array = np.arange(0, time_steps)
	x = np.linspace(0, params['boxlength'], 200)

	lines = []
	# It has to be done like this
	# (see my_examples/matplotlib/animation_with_arrays.py)
	for i in time_steps_array:
		l, = ax.plot([], [])
		lines.append(l)

	def _update_lines_and_text(exc_weights, *lines):
		"""
		Update function that is passed to FuncAnimation

		Arguments:
			- exc_weights: Array of exc_weights
			- *lines: List of Line2D instances
		"""
		# Zip numpy arrays to make them simultaneously iterable
		it_exc = np.nditer(
			[np.arange(0, n_exc), exc_centers, exc_sigmas, exc_weights])
		for n, c, s, w in it_exc:
			l = lines[n]
			gaussian = norm(loc=c, scale=s).pdf
			l.set_data(x, w * gaussian(x))
		return lines

	def _init():
		for l in lines:
			l.set_data([], [])
		return lines

	# The Animation
	# Note that exc_weights_for_all_times is an array of exc weight arrays
	# which come from each time step
	ani = animation.FuncAnimation(
		fig, _update_lines_and_text, exc_weights_for_all_times, fargs=(lines),
		interval=100, blit=True, repeat_delay=3000, init_func=_init)
	if save_path:
		ani.save(
			save_path,
			writer=animation.FFMpegFileWriter(),
			metadata={'artist': 'Simon Weber'})
		return
	else:
		return ani


def fields(params, centers, sigmas):
	"""
	Plotting of Gaussian Fields

	Arguments:
	- centers: numpy array of centers for the gaussians
	- sigmas: numpy array of standard deviations for the gaussians
	"""

	x = np.linspace(0, params['boxlength'], 200)
	it = np.nditer([centers, sigmas])
	summe = 0
	for c, s in it:
		gaussian = norm(loc=c, scale=s).pdf
		plt.plot(x, gaussian(x))
		summe += gaussian(x)
	plt.plot(x, 2*summe/(len(centers)))
	#plt.show()
	return


def sum_of_symmetric_gaussians():
	np.exp(np.power((x - m / N), 2))


def positions_animation(params, positions, save_path=False):
	"""
	Creates an animation of the rat moving through space.

	- positions: Array: [[x1, y2], [x2, y2], ...]
	- save: Save a video .mp4
	"""
	boxlength = params['boxlength']
	dt = params['dt']
	fig = plt.figure()
	plt.xlim(0, 1)
	ax = fig.add_subplot(111)
	ax.set_xlim(0, boxlength)
	ax.set_ylim(0, boxlength)
	ax.set_xlabel('Position')
	ax.set_title('Moving Rat')

	l, = ax.plot([], [], 'o-')
	txt = ax.text(
			0.02, 0.95, '',  # Specified in axis coords, (see transform below)
			horizontalalignment='left',
			verticalalignment='top',
			bbox=dict(facecolor='gray', alpha=0.2),  # Draw box around text
			transform=ax.transAxes)  # Text positions in axis coords, not in data coords

	def _update_lines_and_text(num, positions, line, text):
		line.set_data(positions[num])
		text.set_text('Time = ' + str(num*dt))
		return line, text

	def _init():
		l.set_data([], [])
		txt.set_text('')
		return l, txt

	ani = animation.FuncAnimation(
			fig, _update_lines_and_text, np.arange(0, len(positions)),
			fargs=(positions, l, txt),
			interval=50, blit=True, repeat_delay=2000, repeat=True, init_func=_init)

	if save_path:
		ani.save(
			save_path,
			writer=animation.FFMpegFileWriter(),
			metadata={'artist': 'Simon Weber'})
	else:
		return ani


def positions_movie_old(params, positions, save_path=False):
	"""
	BETA
	Creates a movie of how the rat moves through space.

	- positions: Array: [[x1, y2], [x2, y2], ...]
	"""
	boxlength = params['boxlength']
	fig = plt.figure(0)
	ax = fig.add_subplot(111)
	ax.set_xlim(0, boxlength)
	ax.set_ylim(0, boxlength)
	ims = []
	#ax.text(0.8, 1.02, 'Init Text', fontsize=14)

	#txt=plt.text(0.8, 1.02, 'TEST', fontsize=14,  bbox=dict(alpha=0.1))
	for p in positions:
		#ax.draw_artist(ax.text(0.8, p[0], 'bla', fontsize=14))
		line = ax.plot([p[0]], [p[1]], 'o', color='b', markersize=20)
		#print ax.get_lines()
		# line = ax.get_lines(1)
		#print dir(ax)
		#print line
		ims.append(line)
		#ims.append(text)
		#ims.append(ax.text(0.8, 1.02, str(p), fontsize=14))
	im_ani = animation.ArtistAnimation(
		fig, ims, interval=50, repeat_delay=1000, blit=True)

	if save_path:
		im_ani.save(
			save_path,
			writer=animation.FFMpegFileWriter(),
			metadata={'artist': 'Simon Weber'})
	else:
		plt.show()

# def centers(params, centers_array):
# 	x = np.linspace(0, params['boxlength'], 200)
# 	s = 0
# 	for c in centers_array:
# 		y = norm(loc=c, scale=params['sigma_exc']).pdf
# 		s += y(x)
# 		plt.plot(x, y(x))
# 	plt.plot(x, 2*s/(len(centers_array)))
# 	return
