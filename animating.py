import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotting
import numpy as np
import scipy.stats
import types
mpl.rcParams.update({'figure.autolayout': True})
# print mpl.rcParams.keys()
# mpl.rcParams['animation.frame_format'] = 'jpeg'
# print mpl.rcParams['animation.frame_format']

class Animation(plotting.Plot):
	"""
	Class for Animations

	This class is used to create different animations.
	It inherits from the Plot Class in plotting.py.
	----------
	Arguments:
	- start_time, end_time: simulation time values for the beginning
		and the end of the animation
	- step_factor: If the interval between two simulation points (determined
		by self.every_nth_step) is too small, it can be enhance by multiplication
		with step_factor
	- take_weight_steps: if True, every_nth_step is set to every_nth_step_weights,
		to ensure that the frames are chosen only where weight data exists
	"""
	def __init__(self, params, rawdata, start_time, end_time, step_factor=1, take_weight_steps=False):
		plotting.Plot.__init__(self, params, rawdata)

		try:
			self.every_nth_step_weights
		except AttributeError:
			pass
		else:
			if self.every_nth_step != self.every_nth_step_weights:
				print "WARNING: every_nth_step and every_nth_step_weights differ!"

		if take_weight_steps:
			self.every_nth_multiplicator = self.every_nth_step_weights / self.every_nth_step
			self.every_nth_step = self.every_nth_step_weights


		self.start_time = start_time
		start_frame = int(start_time / (self.dt * self.every_nth_step))
		end_frame = int(end_time / (self.dt * self.every_nth_step)) + 1

		self.frames = np.arange(start_frame, end_frame, step_factor, dtype=int)

		self.fig = plt.figure()
		self.fig.patch.set_alpha(0.5)

	def animate_all_synapses(self, save_path=False, interval=50):
		"""
		Animation with all synapses and the moving rat.
		"""
		# We use gridspec to arrange the plots
		# See the matplotlib tutorial
		# plt.title('test asdf')
		# Create grid with 10 rows and 15 columns
		gs = mpl.gridspec.GridSpec(10, 15)
		artist_frame_tuples = (
			self.get_artist_frame_tuples_position(plot_location=gs[-1, :-1])
			+ self.get_artist_frame_tuples_time(plot_location=gs[0:1, 0:3])
			+ self.get_artist_frame_tuples_weights(plot_location=gs[1:-1, :-1])
			+ self.get_artist_frame_tuples_output_rate(plot_location=gs[1:-1, -1])
		)
		self.create_animation(artist_frame_tuples, save_path, interval)

	def animate_output_rates(self, save_path=False, interval=50):
		"""
		Animation of output rates vs position determined from the current weights
		"""
		gs = mpl.gridspec.GridSpec(10, 10)
		artist_frame_tuples = (
			# self.get_artist_frame_tuples_position(plot_location=gs[0:1])
			self.get_artist_frame_tuples_time(plot_location=gs[0:1, 0:3])
			+ self.get_artist_frame_tuples_position(plot_location=gs[1:-1, :-1], every_nth_multiplicator=self.every_nth_multiplicator)
			# + self.get_artist_frame_tuples_trace(plot_location=gs[1:-1, :-1])
			+ self.get_artist_frame_tuples_output_rates_from_equation(plot_location=gs[1:-1, :-1])

		)		
		self.create_animation(artist_frame_tuples, save_path, interval)		

	def animate_positions(self, save_path=False, interval=50):
		gs = mpl.gridspec.GridSpec(1, 1)
		artist_frame_tuples = (
			self.get_artist_frame_tuples_time(plot_location=gs[0:1])
			+ self.get_artist_frame_tuples_position(plot_location=gs[0:1])
			+ self.get_artist_frame_tuples_trace(plot_location=gs[0:1])
		)		
		self.create_animation(artist_frame_tuples, save_path, interval)

	def create_animation(self, artist_frame_tuples, save_path=False, interval=50):
		"""
		Creates (and saves) animation from tuples of artists and frames

		How it works:
		artist_frame_tuples is a list of tuples of artists and a frame number.
		This function takes all the artists which belong to the same frame and
		adds them to a list of artists <artists>. This is used for ArtistAnimation
		"""
		artists = []
		for i in self.frames:
			# Append all the artists which correspond to the same frame
			artists.append([a[0] for a in artist_frame_tuples if a[1] == i and a[0] != []])
		ani = animation.ArtistAnimation(
			self.fig, artists, interval=interval, blit=False, repeat_delay=1000)
		if save_path:
			Writer = animation.writers['ffmpeg']
			writer = Writer(fps=1000/interval, metadata=self.params, bitrate=1)
			ani.save(
				save_path,
				writer=writer)
			return
		else:
			plt.draw()
			return	

	def get_artist_frame_tuples_output_rates_from_equation(self, plot_location=111):
		"""
		Notes:
		- In 2D we use ax.contour, which does not return an artist. We therefore
			have to duck punch it to look like an artist. This is done using
			the type module 
		"""
		ax = self.fig.add_subplot(plot_location)


		a_f_tuples = []
		if self.dimensions == 2:
			X, Y, positions_grid, rates_grid = self.get_X_Y_positions_grid_rates_grid_tuple(spacing=51)
		for f in self.frames:
			if self.dimensions == 1:
				linspace, output_rates = self.get_output_rates_from_equation(f, spacing=201)
				l, = ax.plot(linspace, output_rates, color='b')
			if self.dimensions == 2:
				ax.set_aspect('equal')
				ax.set_xticks([])
				ax.set_yticks([])
				ax.axis('off')
				if self.boxtype == 'circular':
					circle1=plt.Circle((0,0),.497, ec='black', fc='none', lw=4)
					ax.add_artist(circle1)
				if self.boxtype == 'linear':
					rectangle1=plt.Rectangle((-self.radius, -self.radius),
						2*self.radius, 2*self.radius, ec='black', fc='none', lw=4)
					ax.add_artist(rectangle1)				
				# Make the background transparent, so that the time is still visible
				ax.patch.set_facecolor('none')
				output_rates = self.get_output_rates_from_equation(f, 51, positions_grid, rates_grid)
				
				# Hack to avoid error in case of vanishing output rate at every position
				# If every entry in output_rates is 0, you define a norm and set
				# one of the elements to a small value (such that it looks like zero)
				if np.count_nonzero(output_rates) == 0:
					color_norm = mpl.colors.Normalize(0., 100.)
					output_rates[0][0] = 0.000001
					im = ax.contour(X, Y, output_rates, norm=color_norm)
				else:
					im = ax.contour(X, Y, output_rates)			
				# The DUCK PUNCH
				def setvisible(self,vis):
	   				for c in self.collections: c.set_visible(vis)
				im.set_visible = types.MethodType(setvisible,im,None)
				im.axes = ax
				im.figure = self.fig
				im.draw = ax.draw
				# END OF DUCK PUNCH
				l, = [im]
			a_f_tuples.append((l, f))
		return a_f_tuples


	def get_artist_frame_tuples_weights(self, plot_location=111):
		"""
		Returns tuples of artists and frame numbers for the synapse Gaussians
		"""
		ax = self.fig.add_subplot(plot_location)
		### Setting the axes limits ###
		# x axis is simple
		ax.set_xlim(-self.radius, self.radius)
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
			np.amax(self.rawdata['exc']['weights'])
			/ np.sqrt(2 * np.pi * self.params['exc']['sigma']**2)))
		maxlist.append((
			np.amax(self.rawdata['inh']['weights'])
			/ np.sqrt(2 * np.pi * self.params['inh']['sigma']**2)))
		ylimit = max(maxlist)  # Take the larger of the two values
		ax.set_ylim(0, ylimit)
		time_steps = len(self.rawdata['exc']['weights'])
		time_steps_array = np.arange(0, time_steps)
		x = np.linspace(-self.radius, self.radius, 200)
		a_f_tuples = []
		for n in time_steps_array:
			# exc synapses
			for c, s, w in np.nditer(
					[self.rawdata['exc']['centers'], self.rawdata['exc']['sigmas'], self.rawdata['exc']['weights'][n]]):
				gaussian = scipy.stats.norm(loc=c, scale=s).pdf
				l, = ax.plot(x, w * gaussian(x), color='g')
				a_f_tuples.append((l, n))
			# inh synapses
			for c, s, w in np.nditer(
					[self.rawdata['inh']['centers'], self.rawdata['inh']['sigmas'], self.rawdata['inh']['weights'][n]]):
				gaussian = scipy.stats.norm(loc=c, scale=s).pdf
				l, = ax.plot(x, w * gaussian(x), color='r')
				a_f_tuples.append((l, n))
		return a_f_tuples

	def get_artist_frame_tuples_time(self, plot_location=111):
		"""
		Returns tuples of artists and frame numbers for the time box
		"""
		dt = self.dt
		ax = self.fig.add_subplot(plot_location)
		ax.set_xticks([])
		ax.set_yticks([])
		a_f_tuples = []
		for f in self.frames:
			txt = ax.text(
				# Specified in axis coords, (see transform below)
				0.02, 0.95, 'Time = ' + str(f * dt * self.every_nth_step),
				horizontalalignment='left',
				verticalalignment='top',
				bbox=dict(facecolor='gray', alpha=0.2),  # Draw box around text
				transform=ax.transAxes)  # Text positions in axis coords,not in data coords
			a_f_tuples.append((txt, f))
		return a_f_tuples

	def get_artist_frame_tuples_position(self, plot_location=111, every_nth_multiplicator=1):
		"""Returns tuples of artists and frame numbers for a single moving dot
		
		Parameters
		----------
		every_nth_multiplicator: (int). If the output density differs between different values
			this can be used to keep them in phase.
			Example: position output on each time step (i.e. every_nth_step=1) but weight
				output on every 2000th timestep (i.e. every_nth_step_weights=2000). Then
				using every_nth_multiplicator = 2000 takes the position at the correct
				positions according to the time snap shots of the weights.
		
		Returns
		-------
		
		"""
		ax = self.fig.add_subplot(plot_location)
		ax.set_xlim(-self.radius, self.radius)
		ax.set_ylim(-self.radius, self.radius)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')
		# ax.set_xlabel('Rat Position')
		a_f_tuples = []

		# frames = self.frames * every_nth_multiplicator
		for f in self.frames:
			# This conditional is just to make the initial
			# position dot disappear
			# if f == 0:
			# 	color = 'white'
			# else:
			# print f
			f_new = f * every_nth_multiplicator
			color = 'b'
			l, = ax.plot(
					self.positions[f_new][0], self.positions[f_new][1],
					 marker='o', color=color, markersize=18)
			a_f_tuples.append((l, f))
		return a_f_tuples

	def get_artist_frame_tuples_trace(self, plot_location=111):
		"""
		Returns tuples of artists and frame numbers for the trace of the rat
		"""
		ax = self.fig.add_subplot(plot_location)
		ax.set_xlim(-self.radius, self.radius)
		ax.set_ylim(-self.radius, self.radius)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')
		a_f_tuples = []

		for f in self.frames:
			if f > 0:
				l, = ax.plot(
					self.positions[0:f+1, 0], self.positions[0:f+1, 1], color='black')
				a_f_tuples.append((l, f))
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
		for f in self.frames:
			r = self.output_rates[f]
			if r >= self.target_rate:
				l, = ax.bar(0.25, r, 0.5, color='g')
			else:
				l, = ax.bar(0.25, r, 0.5, color='r')
			a_f_tuples.append((l, f))
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
