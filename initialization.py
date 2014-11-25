# import pdb
import numpy as np
import utils
import scipy.special as sps
# from memory_profiler import profile
# import plotting
# import output
# from scipy.stats import norm
from scipy import stats


def get_gaussian_process(radius, sigma, linspace, white_noise, factor=1.1):
	"""
	Returns function within radius with autocorrelation length sqrt(2)*sigma

	Note: By stretching the border with a factor, we enable the
	desired linspace to be larger then the box. This is
	necessary, because the rat can slightly move beyond the boundary.
	The returned scaled function has its minumum and maximum in the
	range [-factor*radius, factor*radius], so if you want the function to
	typically have its extrema within the box, you should make the factor
	not much larger than 1. Since you only need this for the space
	discretization in the lookup of input rates factor=1.1 is sufficient,
	because the rat does not move further out of the box than velocity*dt=0.01
	and 1.1*radius = 1.1*0.5 = 0.55, so the rat could move 0.05 out of the
	box an the function would still be defined.

	Parameters
	----------
	radius : float
	sigma : float
		The autocorrelation lenght of the resulting function will be the
		same as of a Gaussian with std of `sigma`
	white_noise : ndarray
		Large array of white noise. Good length for our purposes: 6e4
	linspace : ndarray
		Linear space on which the returned function should lie.
		Typically (-limit, limit, spacing), where `limit` either equals
		`radius` or is slightly larger (see note in description).
	factor : float
		Factor for which the limits can be larger than `radius`

	Return
	------
	output : ndarray
		An interpolation of a random function with the same autocorrelation
		length as a Gaussian of std = sigma, interpolated to the
		discretization defined given in `linspace`
	"""
	half_len_wn = int(len(white_noise) / 2.)
	# Put Gauss array on half the length of the white noise to use convolve
	# in mode 'valid'
	gauss_space = np.linspace(-factor * radius, factor * radius, half_len_wn)
	# Linspace of the convolution
	conv_space = np.linspace(-factor * radius, factor * radius,
								half_len_wn + 1)
	# Centered Gaussian
	gaussian = np.sqrt(2 * np.pi * sigma ** 2) * stats.norm(loc=0.0,
						scale=sigma).pdf(gauss_space)
	# Convolve the Gaussian with the white_noise
	convolution = np.convolve(gaussian, white_noise, mode='valid')
	# Rescale the result such that its maximum is 1.0 and its minimum 0.0
	gp = (convolution - np.amin(convolution)) / (np.amax(convolution) - np.amin(
		convolution))
	# Interpolate the outcome to the desired output discretization
	return np.interp(linspace, conv_space, gp)
	# return conv_space, gp


def get_fixed_point_initial_weights(dimensions, radius, center_overlap_exc,
		center_overlap_inh,
		target_rate, init_weight_exc, n_exc, n_inh,
		sigma_exc=None, sigma_inh=None, von_mises=False,
		fields_per_synapse_exc=1,
		fields_per_synapse_inh=1):
	"""Initial inhibitory weights chosen s.t. firing rate = target rate

	From the analytics we know which combination of initial excitatory
	and inhibitory weights leads to an overall output rate of the
	target rate.

	NOTE: This function used to be in experiment_using_snep.py and it was
		made to work with higher dimensional arrays. However, here it is
		better located, because then you don't have to worry about complicated
		parameter linking.

	Parameters
	----------
	dimensions : int
	sigma_exc : float or ndarray
	sigma_inh : float or ndarray
		`sigma_exc` and `sigma_inh` must be of same shape
	von_mises : bool
		If True it is assumed that the bell curves are periodic in y direction

	Returns
	-------
	output : float
		Value for the initial inhibitory weight
	"""

	limit_exc = radius + center_overlap_exc
	limit_inh = radius + center_overlap_inh

	# Change n such that it accounts for multiple fields per synapse
	n_exc *= fields_per_synapse_exc
	n_inh *= fields_per_synapse_inh
	if dimensions == 1:
		init_weight_inh = ( (n_exc * init_weight_exc
								* sigma_exc[0]/ limit_exc[0]
						- target_rate*np.sqrt(2/np.pi))
						/ ( n_inh * sigma_inh[0]
							/ limit_inh[0]) )

	elif dimensions == 2:
		if not von_mises:
			init_weight_inh = (
						(n_exc * init_weight_exc * sigma_exc[0] * sigma_exc[1]
							/ (limit_exc[0]*limit_exc[1])
							- 2 * target_rate / np.pi)
							/ (n_inh * sigma_inh[0] * sigma_inh[1]
							/ (limit_inh[0]*limit_inh[1]))
							)
		else:
			scaled_kappa_exc = (limit_exc[1] / (np.pi*sigma_exc[1]))**2
			scaled_kappa_inh = (limit_inh[1] / (np.pi*sigma_inh[1]))**2
			init_weight_inh = (
					(n_exc * init_weight_exc * sigma_exc[0] * sps.iv(0, scaled_kappa_exc)
						/ (limit_exc[0] * np.exp(scaled_kappa_exc))
						- np.sqrt(2/np.pi) * target_rate)
						/ (n_inh * sigma_inh[0] * sps.iv(0, scaled_kappa_inh)
							/ (limit_inh[0] * np.exp(scaled_kappa_inh)))
							)
	elif dimensions == 3:
		scaled_kappa_exc = (limit_exc[2] / (np.pi*sigma_exc[2]))**2
		scaled_kappa_inh = (limit_inh[2] / (np.pi*sigma_inh[2]))**2
		init_weight_inh = (
			(n_exc * init_weight_exc * sigma_exc[0] * sigma_exc[1] * sps.iv(0, scaled_kappa_exc)
				/ (limit_exc[0] * limit_exc[1] * np.exp(scaled_kappa_exc))
				- 2 * target_rate / np.pi)
			/ (n_inh * sigma_inh[0] * sigma_inh[1] * sps.iv(0, scaled_kappa_inh)
				/ (limit_inh[0] * limit_inh[1] * np.exp(scaled_kappa_inh)))
			)
	return init_weight_inh

def get_equidistant_positions(r, n, boxtype='linear', distortion=0., on_boundary=False):
	"""Returns equidistant, symmetrically distributed coordinates

	Works in dimensions higher than One.
	The coordinates are taken such that they don't lie on the boundaries
	of the environment but instead half a lattice constant away on each
	side.
	Note: In the case of circular boxtype positions outside the cirlce
		are thrown away.

	Parameters
	----------
	r : array_like
		Dimensions of the box [Rx, Ry, Rz, ...]
		If `boxtype` is 'circular', then r can just be an integer, if it
		is an array the first entry is taken as the radius
	n : array_like
		Array of same shape as r, number of positions along each direction
	boxtype : string
		'linear': A quadratic arrangement of positions is returned
		'circular': A ciruclar arrangement instead
	distortion : float or array_like
		Maximal length by which each lattice coordinate (x and y separately)
		is shifted randomly (uniformly)
	on_boundary : bool
		If True, positions can also lie on the system boundaries
	Returns
	-------
	(ndarray) of shape (m, len(n)), where m < np.prod(n) for boxtype
	'circular', because points at the edges are thrown away.
	"""
	r, n, distortion = np.asarray(r), np.asarray(n), np.asarray(distortion)
	if not on_boundary:
		# Get the distance from the boundaries
		d = 2*r/(2*n)
	else:
		# Set the distance from the boundaries to zero
		d = np.zeros_like(r)
	# Get linspace for each dimension
	spaces = [np.linspace(-ra+da, ra-da, na) for (ra, na, da) in zip(r, n, d)]
	# Get multidimensional meshgrid
	Xs = np.meshgrid(*spaces)
	if boxtype == 'circular':
		distance = np.sqrt(np.sum([x**2 for x in Xs], axis=0))
		# Set grid values outside the circle to NaN. Note: This sets the x
		# and the y component (or higher dimensions) to NaN
		for x in Xs:
			x[distance>r[0]] = np.nan
	# Obtain positions file (shape: (n1*n2*..., dimensions)) from meshgrids
	positions = np.array(zip(*[x.flat for x in Xs]))
	# Remove any subarray which contains at least one NaN
	# You do this by keeping only those that do not contain NaN (negation: ~)
	positions = positions[~np.isnan(positions).any(axis=1)]
	dist = 2*distortion * np.random.random_sample(positions.shape) - distortion
	return positions + dist

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

	elif distribution == 'cut_off_gaussian':
		# Draw 100 times more numbers, because those outside the range are thrown away
		rns = np.random.normal(mean, spreading['stdev'], 100*n)
		rns = rns[rns>spreading['left']]
		rns = rns[rns<spreading['right']]
		rns = rns[:n]

	elif distribution == 'cut_off_gaussian_with_standard_limits':
		rns = np.random.normal(mean, spreading, 100*n)
		left = 0.001
		right = 2 * mean - left
		rns = rns[rns>left]
		rns = rns[rns<right]
		rns = rns[:n]

	elif distribution == 'gamma':
		k = (mean/spreading)**2
		theta = spreading**2 / mean
		rns = np.random.gamma(k, theta, n)

	elif distribution == 'gamma_with_cut_off':
		k = (mean/spreading)**2
		theta = spreading**2 / mean
		rns = np.random.gamma(k, theta, n)
		rns[rns<0.01] = 0.01
	return rns



class Synapses:
	"""
	The class of excitatory and inhibitory synapses

	Notes:
		- Given the synapse_type, it automatically gets the appropriate
			parameters from params
	"""
	def __init__(self, sim_params, type_params, seed_centers, seed_init_weights,
					seed_sigmas, positions=None):
		# self.params = params
		for k, v in sim_params.items():
			setattr(self, k, v)

		for k, v in type_params.items():
			setattr(self, k, v)

		##############################
		##########	centers	##########
		##############################
		np.random.seed(int(seed_centers))
		limit = self.radius + self.center_overlap
		self.set_centers(limit)
		self.number = self.centers.shape[0]

		##########################################################
		#################### Gasssian Process ####################
		##########################################################
		if self.gaussian_process:
			self.set_gaussian_process_rates(positions)

		##############################
		##########	sigmas	##########
		##############################
		np.random.seed(int(seed_sigmas))
		# The following four lines should just be temporary. We just use it
		# to test if we can change sigma to an array (in the two dimensional
		# case). Once this is done, it can also be done nicer below.
		# if self.dimensions == 2:
		# self.twoSigma2 = 1. / (2. * self.sigma**2)

		# This doesn't allow for spreading of sigma for asymmetric gaussians
		# Should be generalized

		# self.sigmas = get_random_numbers(
		# 	self.number*self.fields_per_synapse, self.sigma[0], self.sigma_spreading,
		# 	self.sigma_distribution).reshape(self.number, self.fields_per_synapse)

		# This is necessary so that the loop below works for 1 dimension
		self.sigma, self.sigma_spreading, self.sigma_distribution = np.atleast_1d(
			self.sigma, self.sigma_spreading, self.sigma_distribution)

		self.sigmas = np.empty(
					(self.number, self.fields_per_synapse, self.dimensions))
		for i in np.arange(self.dimensions):
			self.sigmas[..., i] = get_random_numbers(
					self.number*self.fields_per_synapse,
					self.sigma[i], self.sigma_spreading[i],
					self.sigma_distribution[i]).reshape(self.number,
						self.fields_per_synapse)
		if self.dimensions == 1:
			self.sigmas.shape = (self.number, self.fields_per_synapse)

		self.twoSigma2 = 1. / (2. * np.power(self.sigmas, 2))
		# if self.dimensions == 2 and self.sigma[0] != self.sigma[1]:
			# Needs to be an array to be saved by snep
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
		# self.norm_x = np.array([1. / (self.sigma[0] * np.sqrt(2 * np.pi))])
		# In either two or three dimensions we take the last dimension te be
		# the one with priodic boundaries
		self.scaled_kappa = np.array([(limit[-1] / (np.pi*self.sigmas[..., -1]))**2])
		self.pi_over_r = np.array([np.pi / limit[-1]])
		self.norm_von_mises = np.array(
				[np.pi / (limit[-1]*2*np.pi*sps.iv(0, self.scaled_kappa))])

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

	def set_gaussian_process_rates(self, positions):
		"""
		Sets self.gaussian_process_rates

		Parameters
		----------
		positions : ndarray
			Positions on which inputs should be defined
		"""
		if self.dimensions == 1:
			self.gaussian_process_rates = np.empty((len(
				positions), self.number_desired))
			for i in np.arange(self.number_desired):
				print i
				white_noise = np.random.random(6e4)
				self.gaussian_process_rates[:,i] = get_gaussian_process(
					self.radius, self.sigma, positions,
					white_noise)


	def set_centers(self, limit):
		"""
		Sets self.centers

		Parameters
		----------
		limit : ndarray
			Specifies the coordinate limits of the place field centers.
			Values can be outside the enclosure to decrease boundary effects.
		"""
		if self.dimensions == 1:
			if self.symmetric_centers:
				self.centers = np.linspace(-limit[0], limit[0], self.number_desired)
				self.centers = self.centers.reshape(
					(self.number_desired, self.fields_per_synapse))
			else:
				self.centers = np.random.uniform(
					-limit, limit,
					(self.number_desired, self.fields_per_synapse)).reshape(
						self.number_desired, self.fields_per_synapse)
			# self.centers.sort(axis=0)

		if self.dimensions >= 2:
			if self.boxtype == 'linear':
				# self.centers = np.random.uniform(-limit, limit,
				# 			(self.number_desired, self.fields_per_synapse, 2))
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
				self.centers = get_equidistant_positions(limit,
								self.number_per_dimension, self.boxtype,
									self.distortion)
				N = self.centers.shape[0]
				fps = self.fields_per_synapse
				# In the case of several fields per synapse (fps) and rather
				# symmetric  distribution of centers, we create fps many
				# distorted lattices and cocaneta the all
				# Afterwards we randomly permute this array so that the inputs
				# to one synapse are drawn randomyl from all these centers
				# Then we reshape it
				if fps > 1:
					for i in np.arange(fps-1):
						b = get_equidistant_positions(limit,
								self.number_per_dimension, self.boxtype,
									self.distortion)
						self.centers = np.concatenate((self.centers, b), axis=0)
					self.centers = np.random.permutation(
										self.centers)
				self.centers = self.centers.reshape(N, fps, self.dimensions)


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
		if self.dimensions == 2:
			symmetric_fields = np.all(self.twoSigma2[..., 0] == self.twoSigma2[..., 1])
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
						np.exp(
							-np.power(
								position-self.centers, 2)
							*self.twoSigma2),
					axis=axis))
				return rates

		if self.dimensions == 2:
			# For contour plots we pass grids with many positions
			# where len(position) > 2. For these grids we need to some along axis 4.
			if len(position) > 2:
				axis = 3
			else:
				axis = 1

			if symmetric_fields and not von_mises:
				def get_rates(position):
					# The outer most sum is over the fields per synapse
					rates = (
						np.sum(
							np.exp(
								-np.sum(
									np.power(position - self.centers, 2),
								axis=axis+1)
							*self.twoSigma2[..., 0]),
							axis=axis)
					)
					return rates
			# For band cell simulations
			elif not symmetric_fields and not von_mises:
				def get_rates(position):
					rates = (
						np.sum(
							np.exp(
								-np.power(
									position[...,0] - self.centers[...,0], 2)
								*self.twoSigma2[..., 0]
								-np.power(
									position[...,1] - self.centers[...,1], 2)
								*self.twoSigma2[..., 1]),
						axis=axis)
					)
					return rates
			elif von_mises:
				def get_rates(position):
					rates = (
						np.sum(
							np.exp(
								-np.power(
									position[...,0] - self.centers[...,0], 2)
								*self.twoSigma2[..., 0]
								)
							* self.norm_von_mises
							* np.exp(
								self.scaled_kappa
								* np.cos(
									self.pi_over_r*(position[...,1]
									- self.centers[...,1]))
								),
							axis=axis)
					)
					return rates
		if self.dimensions == 3:
			if len(position) > 3:
				axis = 4
			else:
				axis = 1

			def get_rates(position):
				rates = (
					np.sum(
						np.exp(
							-np.power(
								position[...,0] - self.centers[...,0], 2)
							*self.twoSigma2[..., 0]
							-np.power(
								position[...,1] - self.centers[...,1], 2)
							*self.twoSigma2[..., 1])
						* self.norm_von_mises
						* np.exp(
							self.scaled_kappa
							* np.cos(
								self.pi_over_r*(position[...,2]
								- self.centers[...,2]))
							),
					axis=axis)
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
		self.z = self.initial_y
		self.position = np.array([self.x, self.y, self.z][:self.dimensions])
		np.random.seed(int(self.params['sim']['seed_trajectory']))
		self.phi = np.random.random_sample() * 2. * np.pi
		self.theta = 2 * np.pi * np.random.random_sample() - np.pi
		self.move_right = True
		self.turning_probability = self.dt * self.velocity / self.persistence_length
		self.angular_sigma = np.sqrt(2.*self.velocity*self.dt/self.persistence_length)
		self.velocity_dt = self.velocity * self.dt
		self.dspace = np.sqrt(2.0*self.diff_const*self.dt)
		self.steps = np.arange(1, self.simulation_time / self.dt + 1)
		self.populations = ['exc', 'inh']
		self.radius_sq = self.radius**2
		self.synapses = {}
		self.get_rates_at_single_position = {}
		self.dt_tau = self.dt / self.tau
		#asdf
		self.input_rates_low_resolution = {}
		self.rates = {}

		if self.params['sim']['first_center_at_zero']:
			if self.dimensions == 1:
				self.synapses['exc'].centers[0] = np.zeros(
						self.params['exc']['fields_per_synapse'])
			if self.dimensions == 2:
				self.synapses['exc'].centers[0] = np.zeros(
						(self.params['exc']['fields_per_synapse'], 2))
		if self.params['sim']['same_centers']:
			self.synapses['inh'].centers = self.synapses['exc'].centers


		self.positions_grid = self.get_positions(
								self.radius,  self.dimensions, self.spacing)
		if self.dimensions >= 2:
			# TODO: You do this transposing here, but not in the high resolution
			# input rates grid. Why? What if you don't do it at all and change
			# the transposing stuff in the plotting functions? You have to do
			# this careful. Bear in mind: Also the projection and polar plots
			# need to be done with care.
			self.positions_grid = np.transpose(self.positions_grid,
										(1, 0, 2, 3, 4, 5)[:self.dimensions+3])


		######################################################
		##########	Discretize space for efficiency	##########
		######################################################
		# Store input rates
		self.input_rates = {}
		# Take the limit such that the rat will never be at a position
		# oustide of the limit
		self.limit = self.radius + 2*self.velocity_dt
		if self.discretize_space:
			positions, self.n_discretize = self.get_positions(
									self.limit, self.dimensions,
									resolution=self.input_space_resolution,
									return_discretization=True)

		#######################################################################
		###################### Instatiating the Synapses ######################
		#######################################################################
		if self.take_fixed_point_weights:
			self.set_fixed_point_initial_weights()
		for n, p in enumerate(self.populations):
			# We want different seeds for the centers of the two populations
			# We therfore add a number to the seed depending. This number
			# is different for each population. We add 1000, because then
			# we could in principle take seed values up to 1000 until the
			# first population would have the same seed as the second
			# population already had before. Note: it doesn't really matter.
			seed_centers = self.seed_centers + (n+1) * 1000
			seed_init_weights = self.seed_init_weights + (n+1) * 1000
			seed_sigmas = self.seed_sigmas + (n+1) * 1000

			self.synapses[p] = Synapses(params['sim'], params[p],
			 	seed_centers=seed_centers, seed_init_weights=seed_init_weights,
			 	seed_sigmas=seed_sigmas, positions=np.squeeze(positions))

			if self.gaussian_process:
				nu = self.params[p]['number_desired']
				self.input_rates_low_resolution[p] = np.empty(
									(self.spacing, nu))
				for i in np.arange(nu):
					self.input_rates_low_resolution[p][:,i] = np.interp(
						np.linspace(-self.radius, self.radius, self.spacing),
						np.squeeze(positions),
						self.synapses[p].gaussian_process_rates[:,i])
				self.input_rates[p] = self.synapses[
							p].gaussian_process_rates
			else:
				# Here we set the low resolution input rates grid
				self.input_rates_low_resolution[p] = \
					self.get_input_rates_grid(self.positions_grid,  self.synapses[p])
				# Here we create a function that returns the firing rate of each
				# input neuron at a single position
				self.get_rates_at_single_position[p] = self.synapses[p].get_rates_function(
							position=self.position, data=False)
				if self.discretize_space:
					print 'Creating the large input rates grid'
					self.input_rates[p] = self.get_input_rates_grid(
						positions, self.synapses[p])

	def get_input_rates_grid(self, positions, synapses):
		"""
		Returns input_rates of synapses at positions

		Parameters
		----------
		positions : ndarray
			Positions grid as defined in get_positions
		synapses : class
			See Synapses class
		Returns
		-------
		ret : ndarray
			See above
		"""
		rates_function = synapses.get_rates_function(positions, data=False)
		return rates_function(positions)

	def get_positions(self, limit, dimensions, spacing=None, resolution=None,
					  return_discretization=False):
		"""
		Returns positions in 1,2 or 3 dimensional space in correct shape
		
		Parameters
		----------
		limit : float
			The limit of the coordinate space. This is take for all sides
			in higher dimensions.
		spacing : int
			If specified the positons are taken on a linear space with that
			spacing
		resolution : float
			If specified the positions tile the space evenly with a binsize
			of `resolution`
		return_discretization : bool
			If True, the number of bins along each dimension is also returned
		Returns
		-------
		ret : ndarray
			Positions in desired shape
			If `return_discretization` is True, it returns a tuple of the
			positions and another array specifying the bin number along
			each dimension
		"""
		

		if dimensions == 1:
			if spacing != None:
				positions = np.linspace(-limit, limit, spacing)
				positions.shape = (spacing, 1, 1)
			elif resolution != None:
				n = None
				# Note that resolution is always an array, even in 1 dimension
				positions = np.arange(-limit+resolution[0], limit,
									  resolution[0])
				positions.shape = (positions.shape[0], 1, 1)

		elif dimensions >= 2:
			r = np.array([limit, limit, limit])[:dimensions]
			if spacing != None:
				n = np.array([spacing, spacing, spacing])[:dimensions]
				positions = get_equidistant_positions(r, n, on_boundary=True)
			elif resolution != None:
				n = np.ceil(2 * limit / resolution)
				# Note that we always take a linear boxtype for the
				# lookup table of all the input rates.
				positions = get_equidistant_positions(
					r, n, boxtype='linear', on_boundary=False)
			# The n[dimenions-1] is just to avoid an error for dimensions=2
			shape = (n[1], n[0], n[dimensions-1])[:dimensions] + (1, 1,
																dimensions)
			positions.shape = shape

		if return_discretization:
			return positions, n
		else:
			return positions


	def set_fixed_point_initial_weights(self):
		"""
		Sets the initial inhibitory weights s.t. rate is close to target rate

		Explanation:
		For Gaussian inputs:
			See mathematical analysis (linear_stablitiy_analysis)
		For Gaussian process inputs:
			The mean of such an input is typically close to 0.5.
			The mean of all the means should be 0.5.
			Therefore we assume the mean input rate to be 0.5 and choose
			the weights accordingly.
		"""
		params = self.params
		if self.gaussian_process:
			wexc = params['exc']['init_weight']
			nexc = np.prod(params['exc']['number_per_dimension'])
			ninh = np.prod(params['inh']['number_per_dimension'])
			rho = self.target_rate
			self.params['inh']['init_weight'] = (
				params['inh']['weight_factor'] * (nexc * wexc - 2 * rho) / ninh
			)
			print "params['inh']['init_weight']"
			print params['inh']['init_weight']
		else:
			self.params['inh']['init_weight'] = get_fixed_point_initial_weights(
				dimensions=self.dimensions, radius=self.radius,
				center_overlap_exc=params['exc']['center_overlap'],
				center_overlap_inh=params['inh']['center_overlap'],
				sigma_exc=params['exc']['sigma'],
				sigma_inh=params['inh']['sigma'],
				target_rate=self.target_rate,
				init_weight_exc=params['exc']['init_weight'],
				n_exc=np.prod(params['exc']['number_per_dimension']),
				n_inh=np.prod(params['inh']['number_per_dimension']),
				von_mises=self.von_mises,
				fields_per_synapse_exc=params['exc']['fields_per_synapse'],
				fields_per_synapse_inh=params['inh']['fields_per_synapse'])

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
		"""Motion in box which is periodic along the last axis

		Note: in 3D it might not be comletely istropic.
		"""
		if self.dimensions == 2:
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

		if self.dimensions == 3:
			x, y, z, r, phi, theta, angular_sigma, velocity_dt = (
				self.x, self.y, self.z, self.radius, self.phi, self.theta,
				self.angular_sigma, self.velocity_dt)
			out_of_bounds_y = (y < -r or y > r)
			out_of_bounds_x = (x < -r or x > r)
			out_of_bounds_z = (z < -r or z > r)
			if (out_of_bounds_x and out_of_bounds_y and out_of_bounds_z):
				phi += np.pi
				theta += np.pi
				if z > r:
					z -= 2 * r
				else:
					z += 2 * r
			# Reflection at the edges
			elif (out_of_bounds_x and out_of_bounds_y):
				phi += np.pi
				z += 0.5*velocity_dt * np.cos(theta)
			elif (out_of_bounds_x and out_of_bounds_z):
				theta += np.pi
				if z > r:
					z -= 2 * r
				else:
					z += 2 * r
			elif (out_of_bounds_y and out_of_bounds_z):
				theta += np.pi
				if z > r:
					z -= 2 * r
				else:
					z += 2 * r
			# Reflection at x
			elif out_of_bounds_x:
				phi = np.pi - phi
				z += 0.5*velocity_dt * np.cos(theta)
			# Reflection at y
			elif out_of_bounds_y:
				phi = -phi
				z += 0.5*velocity_dt * np.cos(theta)
			# Reflection at z
			elif z > r:
				z -= 2 * r
			elif z < -r:
				z += 2 * r
			# Normal move without reflection
			else:
				phi += angular_sigma * np.random.randn()
				theta += angular_sigma * np.random.randn()
				z += 0.5*velocity_dt * np.cos(theta)
			x += velocity_dt * np.cos(phi) * np.sin(theta)
			y += velocity_dt * np.sin(phi) * np.sin(theta)
			self.x, self.y, self.z, self.phi, self.theta = (
														x, y, z, phi, theta)

	def move_persistently(self):
		"""
		Move rat along direction phi and update phi according to persistence length

		Note: The 3D case hasn't yet been tested in a simulation. The 0.5
			in the z coordinate is not understood. Plotting
			indcates that it work though. However, I cannot guarantee that
			the random walk is completely isotropic.
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

		if self.dimensions == 3:
			x, y, z, r = self.x, self.y, self.z, self.radius
			out_of_bounds_y = (y < -r or y > r)
			out_of_bounds_x = (x < -r or x > r)
			out_of_bounds_z = (z < -r or z > r)
			if (out_of_bounds_x and out_of_bounds_y and out_of_bounds_z):
				self.phi += np.pi
				self.theta += np.pi
			# Reflection at the edges
			elif (out_of_bounds_x and out_of_bounds_y):
				self.phi += np.pi
			elif (out_of_bounds_x and out_of_bounds_z):
				self.theta += np.pi
			elif (out_of_bounds_y and out_of_bounds_z):
				self.theta += np.pi
			# Reflection at x
			elif out_of_bounds_x:
				self.phi = np.pi - self.phi
			# Reflection at y
			elif out_of_bounds_y:
				self.phi = -self.phi
			# Reflection at z
			elif out_of_bounds_z:
				self.theta = np.pi - self.theta
			# Normal move without reflection
			else:
				self.phi += self.angular_sigma * np.random.randn()
				self.theta += self.angular_sigma * np.random.randn()
			self.x += self.velocity_dt * np.cos(self.phi) * np.sin(self.theta)
			self.y += self.velocity_dt * np.sin(self.phi) * np.sin(self.theta)
			# This 0.5 is not understood
			self.z += 0.5*self.velocity_dt * np.cos(self.theta)

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
			if self.discretize_space:
				index =  (self.x + self.limit)/self.input_space_resolution - 1
				self.rates = {p: self.input_rates[p][tuple(index)]
										for p in self.populations}
			else:
				self.rates = {p: self.get_rates_at_single_position[p](self.x)
										for p in self.populations}
		if self.dimensions >= 2:
			position = np.array([self.x, self.y, self.z][:self.dimensions])
			if self.discretize_space:
				# index = (position + self.limit)/self.input_space_resolution - 1
				r = self.limit
				n = self.n_discretize
				index = np.ceil((position + r)*n/(2*r)) - 1
				if self.dimensions == 2:
					self.rates = {p: self.input_rates[p][tuple([index[1], index[0]])]
										for p in self.populations}
				elif self.dimensions == 3:
					self.rates = {p: self.input_rates[p][tuple([index[1], index[0], index[2]])]
										for p in self.populations}
				# self.rates = {p: self.input_rates[p][tuple(index)]
				# 						for p in self.populations}
			else:
				self.rates = {p: self.get_rates_at_single_position[p](position)
										for p in self.populations}


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
		positions_grid=False, input_rates=False, equilibration_steps=10000):
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
		positions_grid, input_rates : ndarray
			Arrays as described in get_X_Y_positions_grid_input_rates_tuple
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
								input_rates['exc'][0]) -
							np.dot(rawdata['inh']['weights'][frame],
								input_rates['inh'][0])
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
									input_rates['exc'][n]) -
								np.dot(rawdata['inh']['weights'][frame],
									input_rates['inh'][n])
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
										input_rates['exc'], axes=([-1], [1]))
					- np.tensordot(rawdata['inh']['weights'][frame],
						 				input_rates['inh'], axes=([-1], [1]))
				)
				output_rates = output_rates
			output_rates[output_rates<0] = 0
			output_rates = output_rates.reshape(spacing, self.output_neurons)
			return output_rates

		if self.dimensions >= 2:
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
								input_rates['exc'][0][0]) -
							np.dot(rawdata['inh']['weights'][frame],
								input_rates['inh'][0][0])
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
										input_rates['exc'][nx][ny]) -
									np.dot(rawdata['inh']['weights'][frame],
										input_rates['inh'][nx][ny])
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
										input_rates['exc'], axes=([-1], [self.dimensions]))
					- np.tensordot(rawdata['inh']['weights'][frame],
						 				input_rates['inh'], axes=([-1], [self.dimensions]))
				)
				# Rectification
				output_rates[output_rates < 0] = 0.
				if self.dimensions == 2:
					output_rates = output_rates.reshape(
									spacing, spacing, self.output_neurons)
				elif self.dimensions == 3:
					output_rates = output_rates.reshape(
									spacing, spacing, spacing, self.output_neurons)
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

		time_shape = int(np.ceil(n_time_steps / self.every_nth_step))
		time_shape_weights =  int(np.ceil(n_time_steps
										/ self.every_nth_step_weights))

		for p in ['exc', 'inh']:
			rawdata[p]['norm_von_mises'] = self.synapses[p].norm_von_mises
			rawdata[p]['pi_over_r'] = self.synapses[p].pi_over_r
			rawdata[p]['scaled_kappa'] = self.synapses[p].scaled_kappa
			rawdata[p]['number'] = np.array([self.synapses[p].number])
			rawdata[p]['twoSigma2'] = self.synapses[p].twoSigma2
			rawdata[p]['centers'] = self.synapses[p].centers
			rawdata[p]['sigmas'] = self.synapses[p].sigmas
			weights_shape = (time_shape_weights, self.output_neurons,
												self.synapses[p].number)
			rawdata[p]['weights'] = np.empty(weights_shape)
			rawdata[p]['weights'][0] = self.synapses[p].weights.copy()
			if self.save_input_rates:
				rawdata[p]['input_rates'] = self.input_rates_low_resolution[p]


		rawdata['positions'] = np.empty((time_shape, 3))
		rawdata['phi'] = np.empty(time_shape)

		rawdata['positions_grid'] = np.squeeze(self.positions_grid)

		output_rate_grid_shape = (time_shape_weights, )
		output_rate_grid_shape += tuple([self.spacing for i in
											np.arange(self.dimensions)])
		output_rate_grid_shape += (self.output_neurons, )

		rawdata['output_rate_grid'] = np.empty(output_rate_grid_shape)
		rawdata['output_rate_grid'][0] = self.get_output_rates_from_equation(
						frame=0, rawdata=rawdata, spacing=self.spacing,
						positions_grid=self.positions_grid,
						input_rates=self.input_rates_low_resolution,
							equilibration_steps=self.equilibration_steps)

		rawdata['output_rates'] = np.empty((time_shape, self.output_neurons))

		rawdata['phi'][0] = self.phi
		rawdata['positions'][0] = np.array([self.x, self.y, self.z])
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
				rawdata['positions'][index] = np.array([self.x, self.y, self.z])
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
						input_rates=self.input_rates_low_resolution,
						equilibration_steps=self.equilibration_steps)

		# Convert the output into arrays
		# for k in rawdata:
		# 	rawdata[k] = np.array(rawdata[k])
		# rawdata['output_rates'] = np.array(rawdata['output_rates'])
		print 'Simulation finished'
		return rawdata

