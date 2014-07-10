# import pdb
import numpy as np
import utils
import scipy.special as sps
# from memory_profiler import profile
# import plotting
# import output
# from scipy.stats import norm

def get_equidistant_positions(r, n_x, n_y, boxtype='linear', distortion=0.):
	"""Returns equidistant, symmetrically distributed 2D coordinates
	
	Note: The number of returned positions will be <= n, because not
		because only numbers where n = k**2 where k in Integers, can
		exactly tile the space and only for 'linear' arrangements anyway.
		In the case of circular boxtype many positions need to be thrown
		away.

	Parameters
	----------
	r : ndarray
		Dimensions of the box
	n_x, n_y: int, int
		Number of centers along x and y axis
	boxtype : string
		'linear': A quadratic arrangement of positions is returned
		'circular': A ciruclar arrangement instead 
	distortion : float or ndarray
		Maximal length by which each lattice coordinate (x and y separately)
		is shifted randomly (uniformly)
	
	Returns
	-------
	(ndarray) of shape (m, 2), where m < n but close to n for linear boxtype
				and signficantly smaller than n for circular boxtype
	"""
	dx = 2*r[0]/(2*n_x)
	dy = 2*r[1]/(2*n_y)
	x_space = np.linspace(-r[0]+dx, r[0]-dx, n_x)
	y_space = np.linspace(-r[1]+dy, r[1]-dy, n_y)
	positions_grid = np.empty((n_x, n_y, 2))
	X, Y = np.meshgrid(x_space, y_space)
	# Put all the positions in positions_grid
	for n_y, y in enumerate(y_space):
		for n_x, x in enumerate(x_space):
			positions_grid[n_x][n_y] =  [x, y]
	# Flatten for easier reshape
	positions = positions_grid.flatten()
	# Reshape to the desired shape
	positions = positions.reshape(positions.size/2, 2)
	if boxtype == 'circular':
		r = np.amax(r)
		distance = np.sqrt(X*X + Y*Y)
		# Set all position values outside the circle to NaN
		positions_grid[distance.T>r] = np.nan
		positions = positions_grid.flatten()
		isnan = np.isnan(positions)
		# Delete all positions outside the circle
		positions = np.delete(positions, np.nonzero(isnan))
		# Bring into desired shape
		positions = positions.reshape(positions.size/2, 2)
	distortion_array = 2*distortion * np.random.random_sample(positions.shape) - distortion
	return positions + distortion_array

def get_random_positions_within_circle(n, r, multiplicator=10):
	"""Returns n random 2 D positions within radius (rejection sampling)

	Parameters		
	----------
	n: number of positions
	r: radius
	multiplicator: to ensure that the rejection doesn't reduce the
					positions below n
	Returns
	-------
	ndarray of shape (n, 2)
	"""
	# random coords shape (n, 2)
	# random_nrs = 2*r * np.random.random_sample((multiplicator * n, 2)) - r
	random_nrs = np.random.uniform(-r, r, (multiplicator * n, 2))
	# difference squared
	ds = np.sum(np.power(random_nrs, 2), axis=1)
	# boolean arra
	b = ds < r**2
	# survivors: points within the circle
	survivors = random_nrs[b] 
	# slice the survivors to keep only n
	return survivors[:n]

def get_random_numbers(n, mean, spreading, distribution):
	"""Returns random numbers with specified distribution
	
	Parameters
	----------
	n: (int) number of random numbers to be returned
	mean: (float) mean value for the distributions
	spreading: (float or array) specifies the spreading of the random nubmers
	distribution: (string) a certain distribution
		- uniform: uniform distribution with mean mean and percentual spreading spreading
		- cut_off_gaussian: normal distribution limited to range
			(spreading[1] to spreading[2]) with stdev spreading[0]
			Values outside the range are thrown away
	
	Returns
	-------
	Array of n random numbers
	"""
		
	if distribution == 'uniform':
		rns = np.random.uniform(mean * (1. - spreading), mean * (1. + spreading), n)

	if distribution == 'cut_off_gaussian':
		# Draw 100 times more numbers, because those outside the range are thrown away
		rns = np.random.normal(mean, spreading['stdev'], 100*n)
		rns = rns[rns>spreading['left']]
		rns = rns[rns<spreading['right']]
		rns = rns[:n]

	if distribution == 'cut_off_gaussian_with_standard_limits':
		rns = np.random.normal(mean, spreading, 100*n)
		left = 0.001
		right = 2 * mean - left
		rns = rns[rns>left]
		rns = rns[rns<right]
		rns = rns[:n]

	return rns


class Synapses:
	"""
	The class of excitatory and inhibitory synapses

	Notes:
		- Given the synapse_type, it automatically gets the appropriate
			parameters from params
	"""
	def __init__(self, sim_params, type_params, seed_centers, seed_init_weights):
		# self.params = params
		for k, v in sim_params.items():
			setattr(self, k, v)

		for k, v in type_params.items():
			setattr(self, k, v)

		##############################
		##########	centers	##########
		##############################
		np.random.seed(int(seed_centers))
		# You might want to change it such that center_overlap is an array
		# already in the parameters, however, to this end you need to find out
		# how you handle arrays of arrays with snep
		# self.center_overlap = np.array([self.center_overlap_x, self.center_overlap_y])
		if self.dimensions == 1:
			# Take the first entry of weight overlap in case you forgotten
			# to make it one dimensional
			limit = self.radius + self.center_overlap[0]
			if self.symmetric_centers:
				self.centers = np.linspace(-limit, limit, self.number_desired)
				self.centers = self.centers.reshape(
					(self.number_desired, self.fields_per_synapse))
			else:
				self.centers = np.random.uniform(
					-limit, limit,
					(self.number_desired, self.fields_per_synapse))
			# self.centers.sort(axis=0)
		if self.dimensions == 2:
			if self.boxtype == 'linear':
				# self.centers = np.random.uniform(-limit, limit,
				# 			(self.number_desired, self.fields_per_synapse, 2))
				limit = self.radius + self.center_overlap
				centers_x = np.random.uniform(-limit[0], limit[0],
							(self.number_desired, self.fields_per_synapse))
				centers_y = np.random.uniform(-limit[1], limit[1],
							(self.number_desired, self.fields_per_synapse))
				self.centers = np.dstack((centers_x, centers_y))
			if self.boxtype == 'circular':
				limit = self.radius + np.amax(self.center_overlap)
				random_positions_within_circle = get_random_positions_within_circle(
						self.number_desired*self.fields_per_synapse, limit)
				self.centers = random_positions_within_circle.reshape(
							(self.number_desired, self.fields_per_synapse, 2))
			if self.symmetric_centers:
				limit = self.radius + self.center_overlap
				self.centers = get_equidistant_positions(limit,
								self.n_x, self.n_y, self.boxtype,
									self.distortion)
				self.centers = self.centers.reshape(self.centers.shape[0], 1, 2)

		self.number = self.centers.shape[0]
		##############################
		##########	sigmas	##########
		##############################

		# The following four lines should just be temporary. We just use it
		# to test if we can change sigma to an array (in the two dimensional
		# case). Once this is done, it can also be done nicer below.
		# if self.dimensions == 2:
		self.sigma_x = self.sigma[0]
		self.sigma_y = self.sigma[1]
		self.sigma = self.sigma[0]


		self.sigmas = get_random_numbers(
			self.number*self.fields_per_synapse, self.sigma, self.sigma_spreading,
			self.sigma_distribution).reshape(self.number, self.fields_per_synapse)

		self.norm = 1. / (self.sigmas * np.sqrt(2 * np.pi))
		self.norm2 = 1. / (np.power(self.sigmas, 2) * 2 * np.pi)
		self.twoSigma2 = 1. / (2. * np.power(self.sigmas, 2))
		# This looks a bit dodgy, but it if done otherwise, the arrays
		# everything gets slower
		# if self.dimensions == 1:
		# for a in ['norm', 'norm2', 'twoSigma2']:
		# 	my_a = getattr(self, a)
		# 	setattr(self, a, my_a.reshape(self.n, self.fields_per_synapse))
		if self.sigma_x != self.sigma_y:
			# Needs to be an array to be saved by snep
			self.norm2 = np.array([1. / (self.sigma_x * self.sigma_y * 2 * np.pi)])
		self.twoSigma2_x = 1. / (2. * self.sigma_x**2)
		self.twoSigma2_y = 1. / (2. * self.sigma_y**2)
		# self.Sigma2_x = 1. / self.sigma_x**2
		# self.Sigma2_y = 1. / self.sigma_y**2
		##############################################
		##########	von Mises distribution	##########
		##############################################
		# The von Mises distribution is periodic in the 2pi interval. We want it to be
		# periodic in the -radius, radius interval.We therefore do the following mapping:
		# Normal von Mises: e^(kappa*cos(x-x_0)) / 2 pi I_0(kappa)
		# We take: pi/r * e^((kappa*r^2/pi^2)*cos((x-x_0)pi/r) / 2 pi I_0(kappa*r^2/pi^2)
		# This function is periodic on -radius, radius and it's norm is one.
		# Also the width is just specified as in the spatial dimension.
		# If height 1.0 is desired we take:
		# e^((kappa*r^2/pi^2)*cos((x-x_0) pi/r) / e^(kappa*r^2/pi^2)
		# We achieve this by defining the norm accordingly
		self.norm_x = np.array([1. / (self.sigma_x * np.sqrt(2 * np.pi))])		
		self.scaled_kappa = np.array([(limit[1] / (np.pi*self.sigma_y))**2])
		self.pi_over_r = np.array([np.pi / limit[1]])
		self.norm_von_mises = np.array(
				[np.pi / (limit[1]*2*np.pi*sps.iv(0, self.scaled_kappa))])

		if self.gaussians_with_height_one:
			self.norm = np.ones_like(self.norm)
			self.norm2 = np.ones_like(self.norm2)
			self.norm_x = np.ones_like(self.norm_x)
			self.norm_von_mises = np.ones_like(self.norm_von_mises) /  np.exp(self.scaled_kappa)

		# Create weights array adding some noise to the init weights
		np.random.seed(int(seed_init_weights))
		self.weights = get_random_numbers((self.output_neurons, self.number),
			self.init_weight, self.init_weight_spreading,
			self.init_weight_distribution)
		# Later in the normalization we keep the initial weight sum
		# constant for each output neuron indivdually
		self.initial_weight_sum = np.sum(self.weights, axis=1)
		self.initial_squared_weight_sum = np.sum(np.square(self.weights),
													axis=1)
		
		self.eta_dt = self.eta * self.dt

	def get_rates_function(self, position, data=False):
		"""Returns function which computes values of place field Gaussians at <position>.
		
		Depending on the parameters and the desired simulation, the rates need to
		be set by a different function. To prevent the conditionals from ocurring
		in each time step, this function returns a function which is ready for later
		usage.

		Note that the outer most sum in all these functions is over the
		fields per synapse.

		If position is an array of positions (so of length > 2) and not a
		single position, an array of rates at all the given positions is
		returned. This is useful for plotting.

		Parameters
		----------
		position: (ndarray) [x, y] 
		data: e.g. rawdata['exc']

		Returns
		-------
		Function get_rates, which is used in each time step.
		"""
		# If data is given (used for plotting), then the values from the data are chosen
		# instead of the ones inside this class
		if data:
			for k, v in data.iteritems():
				setattr(self, k, v)

		# Set booleans to choose the desired functions for the rates
		symmetric_fields = (self.twoSigma2_x == self.twoSigma2_y)
		von_mises = (self.motion == 'persistent_semiperiodic')


		if self.dimensions == 1:
			if len(np.atleast_1d(position)) > 2:
				axis = 2
			else:
				axis = 1		
			# The outer most sum is over the fields per synapse
			def get_rates(position):
				rates = (
					np.sum(
						self.norm
						* np.exp(
							-np.power(
								position-self.centers, 2)
							*self.twoSigma2), 
					axis=axis))
				return rates

		if self.dimensions == 2:
			# For contour plots we pass grids with many positions
			# where len(position) > 2. For these grids we need to some along axis 4.
			if len(position) > 2:
				axis = 4
			else:
				axis = 2

			if symmetric_fields and not von_mises:
				def get_rates(position):
					# The outer most sum is over the fields per synapse
					rates = (
						np.sum(
							self.norm2
							*np.exp(
								-np.sum(
									np.power(position - self.centers, 2),
								axis=axis)
							*self.twoSigma2),
							axis=axis-1)
					)
					return rates
			# For band cell simulations
			elif not symmetric_fields and not von_mises:
				def get_rates(position):
					rates = (
						np.sum(
							self.norm2
							* np.exp(
								-np.power(
									position[...,0] - self.centers[...,0], 2)
								*self.twoSigma2_x
								-np.power(
									position[...,1] - self.centers[...,1], 2)
								*self.twoSigma2_y),
						axis=axis-1)
					)
					return rates
			elif von_mises:
				def get_rates(position):
					rates = (
						np.sum(
							self.norm_x
							* np.exp(
								-np.power(
									position[...,0] - self.centers[...,0], 2)
								*self.twoSigma2_x)
							* self.norm_von_mises
							* np.exp(
								self.scaled_kappa
								* np.cos(
									self.pi_over_r*(position[...,1] - self.centers[...,1]))
								),
							axis=axis-1)
					)
					return rates							
		return get_rates


class Rat:
	"""
	The class of the rat
	"""
	def __init__(self, params):
		self.params = params
		for k, v in params['sim'].items():
			setattr(self, k, v)
		for k, v in params['out'].items():
			setattr(self, k, v)
		self.x = self.initial_x
		self.y = self.initial_y
		np.random.seed(int(self.params['sim']['seed_trajectory']))
		self.phi = np.random.random_sample() * 2. * np.pi
		self.move_right = True
		self.turning_probability = self.dt * self.velocity / self.persistence_length
		self.angular_sigma = np.sqrt(2.*self.velocity*self.dt/self.persistence_length)
		self.velocity_dt = self.velocity * self.dt
		self.dspace = np.sqrt(2.0*self.diff_const*self.dt)
		self.steps = np.arange(1, self.simulation_time / self.dt + 1)
		self.populations = ['exc', 'inh']
		self.radius_sq = self.radius**2
		self.synapses = {}
		self.get_rates = {}
		self.dt_tau = self.dt / self.tau

		self.get_rates_grid = {}
		self.rates_grid = {}
		if self.dimensions == 1:
			self.positions_grid = np.empty(self.spacing)
		if self.dimensions == 2:
			self.positions_grid = np.empty((self.spacing, self.spacing, 2))
		# Set up X, Y for contour plot
		x_space = np.linspace(-self.radius, self.radius, self.spacing)
		y_space = np.linspace(-self.radius, self.radius, self.spacing)
		self.X, self.Y = np.meshgrid(x_space, y_space)

		if self.dimensions == 1:
			for n_x, x in enumerate(x_space):
					self.positions_grid[n_x] = x
			self.positions_grid.shape = (self.spacing, 1, 1)
		if self.dimensions == 2:
			for n_y, y in enumerate(y_space):
				for n_x, x in enumerate(x_space):
					self.positions_grid[n_x][n_y] =  [x, y]
			self.positions_grid.shape = (self.spacing, self.spacing, 1, 1, 2)		
		for n, p in enumerate(self.populations):
			# We want different seeds for the centers of the two populations
			# We therfore add a number to the seed depending. This number
			# is different for each population. We add 1000, because then
			# we could in principle take seed values up to 1000 until the
			# first population would have the same seed as the second 
			# population already had before. Note: it doesn't really matter.
			seed_centers = self.seed_centers + (n+1) * 1000
			seed_init_weights = self.seed_init_weights + (n+1) * 1000
			self.synapses[p] = Synapses(params['sim'], params[p],
			 	seed_centers=seed_centers, seed_init_weights=seed_init_weights)
			
			self.get_rates_grid[p] = self.synapses[p].get_rates_function(
									position=self.positions_grid, data=False)
			# Here we set the rate grid
			self.rates_grid[p] = self.get_rates_grid[p](self.positions_grid)

			if self.dimensions == 1:
				self.get_rates[p] = self.synapses[p].get_rates_function(position=self.x, data=False)
			else:
				self.get_rates[p] = self.synapses[p].get_rates_function(position=np.array([self.x, self.y]), data=False)
	
		if self.params['sim']['first_center_at_zero']:
			if self.dimensions == 1:
				self.synapses['exc'].centers[0] = np.zeros(
						self.params['exc']['fields_per_synapse'])
			if self.dimensions == 2:
				self.synapses['exc'].centers[0] = np.zeros(
						(self.params['exc']['fields_per_synapse'], 2))

		if self.params['sim']['same_centers']:
			self.synapses['inh'].centers = self.synapses['exc'].centers
		self.rates = {}

		# Store input rates
		self.input_rates = {}
		# Take the limit such that the rat will never be at a position
		# oustide of the limit
		self.limit = self.radius + 2*self.velocity_dt
		possible_x_positions = np.arange(-self.limit+self.input_space_resolution, self.limit, self.input_space_resolution)
		if self.input_space_resolution != -1 and self.dimensions == 1:
			possible_positions = possible_x_positions
			self.input_rates['exc'] = np.empty((possible_positions.shape[0], self.synapses['exc'].number))
			self.input_rates['inh'] = np.empty((possible_positions.shape[0], self.synapses['inh'].number))
			for n, p in enumerate(possible_positions):
				self.input_rates['exc'][n] = self.get_rates['exc'](p)
				self.input_rates['inh'][n] = self.get_rates['inh'](p)

		if self.input_space_resolution != -1 and self.dimensions == 2:
			# possible_positions = np.empty((possible_x_positions.shape[0], possible_x_positions.shape[0], 2))
			self.input_rates['exc'] =  np.empty((possible_x_positions.shape[0], possible_x_positions.shape[0], self.synapses['exc'].number))
			self.input_rates['inh'] =  np.empty((possible_x_positions.shape[0], possible_x_positions.shape[0], self.synapses['inh'].number))
			for ny, y in enumerate(possible_x_positions):
				for nx, x in enumerate(possible_x_positions):
					self.input_rates['exc'][nx][ny] = self.get_rates['exc'](np.array([x, y]))
					self.input_rates['inh'][nx][ny] = self.get_rates['inh'](np.array([x, y]))

	def move_diffusively(self):
		"""
		Update position of rat by number drawn from gauss with stdev = dspace
		"""
		if self.dimensions == 1:
			self.x += self.dspace*np.random.randn()
		if self.dimensions == 2:
			self.x += self.dspace*np.random.randn()
			self.y += self.dspace*np.random.randn()

	def dont_move(self):
		pass

	def move_persistently_semi_periodic(self):
		# Boundary conditions and movement are interleaved here
		pos = np.array([self.x, self.y])
		is_bound_trespassed = np.logical_or(pos < -self.radius, pos > self.radius)
		# Reflection at the corners
		if np.all(is_bound_trespassed):
			self.phi = np.pi - self.phi
			self.x += self.velocity_dt * np.cos(self.phi)
			if self.y > self.radius:
				self.y -= 2 * self.radius
			else:
				self.y += 2 * self.radius
		# Reflection at left and right
		elif is_bound_trespassed[0]:
			self.phi = np.pi - self.phi
			self.x += self.velocity_dt * np.cos(self.phi)
			self.y += self.velocity_dt * np.sin(self.phi)
		# Reflection at top and bottom
		elif self.y > self.radius:
			self.y -= 2 * self.radius
			self.x += self.velocity_dt * np.cos(self.phi)
			self.y += self.velocity_dt * np.sin(self.phi)
		elif self.y < -self.radius:
			self.y += 2 * self.radius
			self.x += self.velocity_dt * np.cos(self.phi)
			self.y += self.velocity_dt * np.sin(self.phi)
		# Normal move without reflection	
		else:
			self.phi += self.angular_sigma * np.random.randn()
			self.x += self.velocity_dt * np.cos(self.phi)
			self.y += self.velocity_dt * np.sin(self.phi)		

	def move_persistently(self):
		"""
		Move rat along direction phi and update phi according to persistence length
		"""
		if self.dimensions == 1:
			if self.x > self.radius:
				self.move_right = False
			elif self.x < -self.radius:
				self.move_right = True
			elif np.random.random() < self.turning_probability:
				self.move_right = not self.move_right

			if self.move_right:
				self.x += self.velocity_dt
			else:
				self.x -= self.velocity_dt


		if self.dimensions == 2:
			# Boundary conditions and movement are interleaved here
			x, y, r = self.x, self.y, self.radius
			out_of_bounds_vertical = (y < -r or y > r)
			out_of_bounds_horizontal = (x < -r or x > r)
			# Reflection at the corners
			if (out_of_bounds_vertical and out_of_bounds_horizontal):
				self.phi += np.pi
			# Reflection at left and right
			elif out_of_bounds_horizontal:
				self.phi = np.pi - self.phi
			# Reflection at top and bottom
			elif out_of_bounds_vertical:
				self.phi = -self.phi
			# Normal move without reflection	
			else:
				self.phi += self.angular_sigma * np.random.randn()
			self.x += self.velocity_dt * np.cos(self.phi)
			self.y += self.velocity_dt * np.sin(self.phi)

	def move_persistently_circular(self):
		# Check if rat is outside and reflect it
		if self.x**2 + self.y**2 > self.radius_sq:			
			# Reflection algorithm
			# Get theta (polar coordinate angle)
			theta = np.arctan2(self.y, self.x)
			# Get unit vector along tangent (counterclockwise)
			u_tangent = [-np.sin(theta), np.cos(theta)]
			# Get unit vector along velocity that jumped outside
			u = [np.cos(self.phi), np.sin(self.phi)]
			# Get angle between these two unit vectors
			alpha = np.arccos(np.dot(u_tangent, u))
			# Update phi by adding 2 alpha (makes sense, make sketch)
			self.phi += 2 * alpha
			# Update position
			self.x += self.velocity_dt * np.cos(self.phi)
			self.y += self.velocity_dt * np.sin(self.phi)
			# # Straight away algorithms 
			# theta = np.arctan2(self.y, self.x)
			# self.phi = theta + np.pi
			# self.x += self.velocity_dt * np.cos(self.phi)
			# self.y += self.velocity_dt * np.sin(self.phi)
		# Normal move without reflection
		else:
			self.phi += self.angular_sigma * np.random.randn()
			self.x += self.velocity_dt * np.cos(self.phi)
			self.y += self.velocity_dt * np.sin(self.phi)			


	def reflective_BCs(self):
		"""
		Reflective Boundary Conditions

		If the rat moves beyond the boundary, it gets reflected inside the boundary
		by the amount it was beyond the boundary
		"""
		if self.dimensions == 1:
			dimension_list = ['x']
		if self.dimensions == 2:
			dimension_list = ['x', 'y']
		for d in dimension_list:
			v = getattr(self, d)
			if v < -self.radius:
				setattr(self, d, - v - 2. * self.radius)
			if v > self.radius:
				setattr(self, d, - v + 2. * self.radius)

	def periodic_BCs(self):
		"""
		Periodic Boundary Conditions
		"""
		if self.dimensions == 1:
			dimension_list = ['x']
		if self.dimensions == 2:
			dimension_list = ['x', 'y']
		for d in dimension_list:
			v = getattr(self, d)
			if v < -self.radius:
				setattr(self, d, v + 2*self.radius)
			if v > self.radius:
				setattr(self, d, v - 2*self.radius)

	def billiard_BCs(self):
		"""
		Billiard Boundary Conditions

		Incidence Angle = Emergent Angle
		"""
		pass
		# # Right and Left wall
		# if self.x > self.boxlength or self.x < 0:
		# 	self.phi = np.pi - self.phi
		# # Top and Bottom wall
		# if self.y > self.boxlength or self.y < 0:
		# 	self.phi = 2. * np.pi - self.phi

	def set_current_output_rate(self):
		"""
		Sums exc_weights * exc_rates and substracts inh_weights * inh_rates
		"""
		rate = (
			np.dot(self.synapses['exc'].weights, self.rates['exc']) -
			np.dot(self.synapses['inh'].weights, self.rates['inh'])
		)

		rate[rate<0] = 0
		self.output_rate = rate

	def set_current_output_rate_lateral_inhibition(self):
		
		rate = (
				self.output_rate*(1 - self.dt_tau)
				+ self.dt_tau * ((
				np.dot(self.synapses['exc'].weights, self.rates['exc']) -
				np.dot(self.synapses['inh'].weights, self.rates['inh'])
				)
				- self.weight_lateral
				* (np.sum(self.output_rate) - self.output_rate)
				)
				)

		rate[rate<0] = 0
		self.output_rate = rate	

	def set_current_output_rate_lateral_inhibition_fixed_point(self):
		A_inv_neg = np.array([[1, -self.weight_lateral], [-self.weight_lateral, 1]])/(1-self.weight_lateral**2)
		rate = np.dot(A_inv_neg, np.dot(self.synapses['exc'].weights, self.rates['exc']) - np.dot(self.synapses['inh'].weights, self.rates['inh']))
		rate[rate<0] = 0
		self.output_rate = rate

	def set_current_input_rates(self):
		"""
		Set the rates of the input neurons by using their place fields
		"""
		if self.dimensions == 1:
			if self.input_space_resolution != -1:
				index =  (self.x + self.limit)/self.input_space_resolution - 1
				self.rates['exc'] = self.input_rates['exc'][index]
				self.rates['inh'] = self.input_rates['inh'][index]
			else:
				self.rates['exc'] = self.get_rates['exc'](self.x)
				self.rates['inh'] = self.get_rates['inh'](self.x)
		if self.dimensions == 2:
			if self.input_space_resolution != -1:
				index_x =  (self.x + self.limit)/self.input_space_resolution - 1
				index_y =  (self.y + self.limit)/self.input_space_resolution - 1
				self.rates['exc'] = self.input_rates['exc'][index_x][index_y]
				self.rates['inh'] = self.input_rates['inh'][index_x][index_y]
			else:
				self.rates['exc'] = self.get_rates['exc'](np.array([self.x, self.y]))
				self.rates['inh'] = self.get_rates['inh'](np.array([self.x, self.y]))

		
	def update_exc_weights(self):
		self.synapses['exc'].weights += (
			(self.rates['exc'] * self.synapses['exc'].eta_dt) * self.output_rate[:, np.newaxis] 
		)

	def update_inh_weights(self):
		self.synapses['inh'].weights += (
			self.rates['inh'] *
				((self.output_rate[:, np.newaxis] - self.target_rate)
				* self.synapses['inh'].eta_dt)
		)
		# self.synapses['inh'].weights += (
		# 	np.outer((self.output_rate - self.target_rate), self.rates['inh']) * self.synapses['inh'].eta_dt
		# )
	def update_weights(self):
		"""
		Update both weights (convenience function)
		"""
		self.update_exc_weights()
		self.update_inh_weights()

	def normalize_exc_weights_linear_substractive(self):
		"""Normalize substractively, keeping the linear sum constant"""
		# Get a vector with entries of ones and zeroes
		# For each synapse with positive values you get a one
		# For each synapase with negative values you get a zero
		# See Dayan, Abbott p. 290 for schema
		substraction_value = (
			self.synapses['exc'].eta_dt * self.output_rate
			* np.sum(self.rates['exc']) / self.synapses['exc'].number)
		n_vector = (self.synapses['exc'].weights > substraction_value).astype(int)

		substractive_norm = (
			self.synapses['exc'].eta_dt * self.output_rate
			* np.dot(self.rates['exc'], n_vector) * n_vector
			/ np.sum(n_vector)
		)
		self.synapses['exc'].weights -= substractive_norm

	def normalize_exc_weights_linear_multiplicative(self):
		"""Normalize multiplicatively, keeping the linear sum constant"""
		self.synapses['exc'].weights = (
			(self.synapses['exc'].initial_weight_sum
				/ np.sum(self.synapses['exc'].weights))
			* self.synapses['exc'].weights
		)

	# def normalize_exc_weights_quadratic_multiplicative(self):
	# 	"""Normalize  multiplicatively, keeping the quadratic sum constant"""
	# 	self.synapses['exc'].weights = (
	# 		np.sqrt((self.synapses['exc'].initial_squared_weight_sum /
	# 									np.sum(np.square(self.synapses['exc'].weights))))
	# 			*self.synapses['exc'].weights
	# 	)

	def normalize_exc_weights_quadratic_multiplicative(self):
		"""Normalize  multiplicatively, keeping the quadratic sum constant"""
		factor = np.sqrt(
					(self.synapses['exc'].initial_squared_weight_sum /
					np.einsum('...j,...j->...', self.synapses['exc'].weights,
						self.synapses['exc'].weights)))
		self.synapses['exc'].weights *= factor[:, np.newaxis]

	def get_output_rates_from_equation(self, frame, rawdata, spacing,
		positions_grid=False, rates_grid=False, equilibration_steps=10000):
		"""	Return output rates at many positions

		***
		Note:
		This function used to be in plotting.py, but now it is used here
		to output arrays containing the output rates. This makes 
		quick plotting and in particular time traces of Grid Scores feasible.
		***

		For normal plotting in 1D and for contour plotting in 2D.
		It is differentiated between cases with and without lateral inhibition.

		With lateral inhibition the output rate has to be determined via
		integration (but fixed weights).
		In 1 dimensions we start at one end of the box, integrate for a
		time specified by equilibration steps and than walk to the 
		other end of the box.
		
		Parameters
		----------
		frame : int
			Frame at which the output rates are plotted
		rawdata : dict
			Contains the synaptic weights
		spacing : int
			The spacing, describing the detail richness of the plor or contour plot (spacing**2)
		positions_grid, rates_grid : ndarray
			Arrays as described in get_X_Y_positions_grid_rates_grid_tuple
		equilibration_steps : int
			Number of steps of integration to reach the correct
			value of the output rates for the case of lateral inhibition
	
		Returns
		-------
		output_rates : ndarray
			Array with output rates at several positions tiling the space
			For 1 dimension with shape (spacing)
			Fro 2 dimensions with shape (spacing, spacing)
		"""
			
		# plt.title('output_rates, t = %.1e' % (frame * self.every_nth_step_weights), fontsize=8)
		if self.dimensions == 1:
			linspace = np.linspace(-self.radius, self.radius, spacing)

			if self.lateral_inhibition:
				output_rates = np.empty((spacing, self.output_neurons))
			
				start_pos = -self.radius
				end_pos = self.radius
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
							np.dot(rawdata['exc']['weights'][frame],
								rates_grid['exc'][0]) -
							np.dot(rawdata['inh']['weights'][frame], 
								rates_grid['inh'][0])
							)
							- self.weight_lateral
							* (np.sum(r) - r)
							)
							)
					r[r<0] = 0
				start_r = r
				# output_rates = []
				for n, x in enumerate(linspace):
					for s in np.arange(200):
						r = (
								r*(1 - dt_tau)
								+ dt_tau * ((
								np.dot(rawdata['exc']['weights'][frame],
									rates_grid['exc'][n]) -
								np.dot(rawdata['inh']['weights'][frame], 
									rates_grid['inh'][n])
								)
								- self.weight_lateral
								* (np.sum(r) - r)
								)
								)
						r[r<0] = 0
					output_rates[n] = r

			else:
				output_rates = (
					np.tensordot(rawdata['exc']['weights'][frame],
										rates_grid['exc'], axes=([-1], [1]))
					- np.tensordot(rawdata['inh']['weights'][frame],
						 				rates_grid['inh'], axes=([-1], [1]))
				)
				output_rates = output_rates.T
			output_rates[output_rates<0] = 0
			return output_rates

		if self.dimensions == 2:
			if self.lateral_inhibition:
				output_rates = np.empty((spacing, spacing, self.output_neurons))
				start_pos = positions_grid[0, 0, 0, 0]
				r = np.zeros(self.output_neurons)
				dt_tau = self.dt / self.tau

				pos = start_pos
				for s in np.arange(equilibration_steps):
					r = (
							r*(1 - dt_tau)
							+ dt_tau * ((
							np.dot(rawdata['exc']['weights'][frame],
								rates_grid['exc'][0][0]) -
							np.dot(rawdata['inh']['weights'][frame], 
								rates_grid['inh'][0][0])
							)
							- self.weight_lateral
							* (np.sum(r) - r)
							)
							)
					r[r<0] = 0
				# start_r = r
				# print r
				# output_rates = []

				for ny in np.arange(positions_grid.shape[1]):
					for nx in np.arange(positions_grid.shape[0]):
						pos = positions_grid[nx][ny]
						for s in np.arange(200):
							r = (
									r*(1 - dt_tau)
									+ dt_tau * ((
									np.dot(rawdata['exc']['weights'][frame],
										rates_grid['exc'][nx][ny]) -
									np.dot(rawdata['inh']['weights'][frame], 
										rates_grid['inh'][nx][ny])
									)
									- self.weight_lateral
									* (np.sum(r) - r)
									)
									)
							r[r<0] = 0

						output_rates[nx][ny] = r

				# for i in np.arange(self.output_neurons):
				# 	output_rates[:,:,i] = np.transpose(output_rates[:,:,i])

			else:
				output_rates = (
					np.tensordot(rawdata['exc']['weights'][frame],
										rates_grid['exc'], axes=([-1], [2]))
					- np.tensordot(rawdata['inh']['weights'][frame],
						 				rates_grid['inh'], axes=([-1], [2]))
				)
				# Transposing is now done in the contourplot
				# output_rates = np.transpose(output_rates)
				# Rectification
				output_rates[output_rates < 0] = 0.
				output_rates = output_rates.reshape(
									spacing, spacing, self.output_neurons)
			return output_rates

	def run(self, rawdata_table=False, configuration_table=False):
		"""
		Let the rat move and learn

		Arguments:
		- 	position_output: if True, self.positions gets all the rat positions
			appended
		"""
		
		np.random.seed(int(self.params['sim']['seed_trajectory']))
		print 'Type of Normalization: ' + self.normalization
		print 'Type of Motion: ' + self.motion
		# print 'Boundary Conditions: ' + self.boundary_conditions
		##########################################################
		##########	Choose Motion and Boundary Conds 	##########
		##########################################################
		if self.motion == 'diffusive':
			self.move = self.move_diffusively
			if self.boundary_conditions == 'reflective':
				self.apply_boundary_conditions = self.reflective_BCs
			elif self.boundary_conditions == 'periodic':
				self.apply_boundary_conditions = self.periodic_BCs
			else:
				self.apply_boundary_conditions = self.reflective_BCs

		if self.motion == 'persistent' and self.boxtype == 'linear':
			self.move = self.move_persistently
		if self.motion == 'persistent_semiperiodic' and self.boxtype == 'linear':
			self.move = self.move_persistently_semi_periodic
		if self.motion == 'persistent' and self.boxtype == 'circular':
			self.move = self.move_persistently_circular
		if self.params['sim']['stationary_rat']:
			self.move = self.dont_move
			
		# if self.boundary_conditions == 'periodic':
		# 	self.apply_boundary_conditions = self.periodic_BCs
		# self.apply_boundary_conditions = getattr(self,self.boundary_conditions+'_BCs')

		# Choose the normalization scheme
		normalize_exc_weights = getattr(self,'normalize_exc_weights_'+self.normalization)

		# Choose the update functions and the output_rate functions
		if self.lateral_inhibition:
			self.my_set_output_rate = self.set_current_output_rate_lateral_inhibition
		else:
			self.my_set_output_rate = self.set_current_output_rate



		rawdata = {'exc': {}, 'inh': {}}

		n_time_steps = 1 + self.simulation_time / self.dt
		for p in ['exc', 'inh']:
			rawdata[p]['norm'] = self.synapses[p].norm
			rawdata[p]['norm2'] = self.synapses[p].norm2
			rawdata[p]['norm_x'] = self.synapses[p].norm_x
			rawdata[p]['norm_von_mises'] = self.synapses[p].norm_von_mises
			rawdata[p]['pi_over_r'] = self.synapses[p].pi_over_r
			rawdata[p]['scaled_kappa'] = self.synapses[p].scaled_kappa
			rawdata[p]['number'] = np.array([self.synapses[p].number])
			rawdata[p]['twoSigma2'] = self.synapses[p].twoSigma2
			rawdata[p]['twoSigma2_x'] = np.array([self.synapses[p].twoSigma2_x])
			rawdata[p]['twoSigma2_y'] = np.array([self.synapses[p].twoSigma2_y])
			# rawdata[p]['sigma_x'] = self.synapses[p].twoSigma2_y
			# rawdata[p]['sigma_y'] = self.synapses[p].twoSigma2_y
			rawdata[p]['centers'] = self.synapses[p].centers
			rawdata[p]['sigmas'] = self.synapses[p].sigmas
			weights_shape = (np.ceil(
								n_time_steps / self.every_nth_step_weights),
									self.output_neurons, self.synapses[p].number)

			rawdata[p]['weights'] = np.empty(weights_shape)
			rawdata[p]['weights'][0] = self.synapses[p].weights.copy()

		rawdata['positions'] = np.empty((np.ceil(
								n_time_steps / self.every_nth_step), 2))
		rawdata['phi'] = np.empty(np.ceil(
								n_time_steps / self.every_nth_step))
		
		if self.dimensions == 1:
			rawdata['output_rate_grid'] = np.empty((np.ceil(
										n_time_steps / self.every_nth_step_weights),
											self.spacing, self.output_neurons))
			rawdata['output_rate_grid'][0] = self.get_output_rates_from_equation(
							frame=0, rawdata=rawdata, spacing=self.spacing,
							positions_grid=self.positions_grid,
							rates_grid=self.rates_grid,
							equilibration_steps=self.equilibration_steps)				

		if self.dimensions == 2:
			rawdata['output_rate_grid'] = np.empty((np.ceil(
										n_time_steps / self.every_nth_step_weights),
											self.spacing, self.spacing,
											self.output_neurons))
			rawdata['output_rate_grid'][0] = self.get_output_rates_from_equation(
							frame=0, rawdata=rawdata, spacing=self.spacing,
							positions_grid=self.positions_grid,
							rates_grid=self.rates_grid,
							equilibration_steps=self.equilibration_steps)

		rawdata['output_rates'] = np.empty((np.ceil(
									n_time_steps / self.every_nth_step),
									self.output_neurons))	
			
		rawdata['phi'][0] = self.phi
		rawdata['positions'][0] = np.array([self.x, self.y])
		rawdata['output_rates'][0] = 0.0

		if self.lateral_inhibition:
			self.output_rate = 0.
		rawdata['time_steps'] = self.steps
		for step in self.steps:
			self.move()
			# if self.apply_boundary_conditions:
			try:
				self.apply_boundary_conditions()
			except AttributeError:
				pass
			self.set_current_input_rates()
			self.my_set_output_rate()
			self.update_weights()
			self.synapses['exc'].weights[self.synapses['exc'].weights<0] = 0.
			self.synapses['inh'].weights[self.synapses['inh'].weights<0] = 0.
			normalize_exc_weights()
			
			if step % self.every_nth_step == 0:
				index = step / self.every_nth_step
				# print 'step = %f' % step
				# Store Positions
				rawdata['positions'][index] = np.array([self.x, self.y])
				rawdata['phi'][index] = np.array(self.phi)
				rawdata['output_rates'][index] = self.output_rate
				# print 'current step: %i' % step

			if step % self.every_nth_step_weights == 0:
				print 'Current step: %i' % step
				index = step / self.every_nth_step_weights
				rawdata['exc']['weights'][index] = self.synapses['exc'].weights.copy()
				rawdata['inh']['weights'][index] = self.synapses['inh'].weights.copy()
				rawdata['output_rate_grid'][index] = self.get_output_rates_from_equation(
						frame=index, rawdata=rawdata, spacing=self.spacing,
						positions_grid=self.positions_grid,
						rates_grid=self.rates_grid,
						equilibration_steps=self.equilibration_steps)

		# Convert the output into arrays
		# for k in rawdata:
		# 	rawdata[k] = np.array(rawdata[k])
		# rawdata['output_rates'] = np.array(rawdata['output_rates'])
		print 'Simulation finished'
		return rawdata

