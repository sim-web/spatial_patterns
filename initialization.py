# import pdb
import numpy as np
import scipy.special as sps
# from memory_profiler import profile
# import plotting
# import output
# from scipy.stats import norm
from scipy import stats
from scipy import signal
import scipy
from scipy.integrate import dblquad
import utils

def get_gaussian_process(radius, sigma, linspace, dimensions=1, rescale='stretch',
						 stretch_factor=1.0, extremum='none', untuned=False):
	"""
	Returns function with autocorrelation length sqrt(2)*sigma

	So the returned function has the same autocorrelation length like a
	gaussian of standard deviation sigma.

	Parameters
	----------
	radius : float
	sigma : float or ndarray
		The autocorrelation length of the resulting function will be the
		same as of a Gaussian with standard deviation of `sigma`
	linspace : ndarray
		Linear space on which the returned function should lie.
		Typically (-limit, limit, spacing), where `limit` either equals
		`radius` or is slightly larger if there's a chance that the
		rat moves outside the box. Typically limit = 1.1 * radius.
		Note: The limit must be <= 2*radius. If you ever want to change this,
		make the value of agp larger than 2.
	dimensions : int
		Number of dimensions of the gaussian process function
	rescale : str
		If 'stretch' the final function is scaled between 0 and 1
		If 'fixed_mean', the mean of the final function is set to
		a desired value (currently 0.5)
	untuned : bool
		If True, than independent of the sigma value the input will have
		no spatial tuning and will fire at the desired mean value everywhere.
	Return
	------
	output : ndarray
		An interpolation of a random function with the same autocorrelation
		length as a Gaussian of std = sigma, interpolated to the
		discretization defined given in `linspace`.
	"""
	if dimensions == 1:
		# The bin width
		dx = sigma / 20.
		if sigma < radius/8.:
			# For small enough sigma it's enough to define the gaussian
			# in the range [-r, r]
			agauss = 1.0
		else:
			# For larger sigma we need the gaussian on a larger array,
			# to avoid the curve from being cut off
			agauss = np.ceil(8*sigma)
			# We need to take a smaller discretization than given by the
			# large sigma in order to have enough points within the radius
			dx /= agauss
		# agp determines how much larger the
		# Note: Maybe put agp as function argument with default 2
		agp = 2
		###
		# The number of bins for each ocurring array
		# We specifiy them all on a radius, so only on haft the track.
		# This is why later we multiply each of them by 2.
		###
		# The discretization of space
		bins_per_radius = np.ceil(radius / dx)
		# The array size of the Gaussian
		bins_gauss = agauss * bins_per_radius
		# We take the white noise on a larger array
		bins_wn = (agauss + agp) * bins_per_radius
		bins_gp = agp * bins_per_radius
		# White noise between -0.5 and 0.5 (zero mean)
		# Note: The range doesn't matter.
		white_noise = np.random.random(2*bins_wn) - 0.5
		gauss_limit = agauss*radius
		gauss_space = np.linspace(-gauss_limit, gauss_limit, 2*bins_gauss)
		conv_limit = agp*radius
		# Note: you need to add +1 to the number of bins
		conv_space = np.linspace(-conv_limit, conv_limit, 2*bins_gp + 1)
		# Centered Gaussian on gauss_space
		# gaussian = np.sqrt(2 * np.pi * sigma**2) * stats.norm(loc=0.0,
		# 					scale=sigma).pdf(gauss_space)
		gaussian = stats.norm(loc=0.0, scale=sigma).pdf(gauss_space)
		# Convolve the Gaussian with the white_noise
		# Note: in fft convolve the larger array must be the first argument
		convolution = signal.fftconvolve(white_noise, gaussian, mode='valid')
		# The convolution doesn't consider dx
		# It treats each bin as a distance of 1
		# This leads to a larger value of the convolutions.
		# This can be reverted by the following:
		gp = convolution * dx	# / (sigma * np.sqrt(2 * np.pi))
		gp = np.interp(linspace, conv_space, gp)
		if extremum=='none':
			gp_min, gp_max = np.amin(gp), np.amax(gp)
		else:
			gp_min, gp_max = extremum[0], extremum[1]
		if rescale == 'stretch':
			# Rescale the result such that its maximum is 1 and its minimum 0
			gp = stretch_factor * (gp - gp_min) / (gp_max - gp_min)
			# Rectfication is necessary if the extrema are global values
			# because then gp_min could exceed the actual minimum
			gp[gp<0.] = 0.
		elif rescale == 'fixed_mean':
			# mean_of_single_field = np.sqrt(2*np.pi*sigma**2)/(2*radius)
			desired_mean = 0.5
			gp_min = np.amin(gp)
			divisor = np.mean(gp - gp_min)
			gp_max = divisor
			gp = desired_mean * (gp - gp_min) / divisor
			if untuned:
				gp = desired_mean
		# Interpolate the outcome to the desired output discretization given
		# in `linspace`
		return gp, gp_min, gp_max

	elif dimensions == 2:
		# Works like in 1D but we take a larger dx, for faster initialization
		dx = sigma / 10.
		# We choose 1.0 as the standard
		agauss = np.array([1.0, 1.0])
		# Only in the dimensions where sigma>radius/8 we change it
		agauss[sigma>radius/8.] = np.ceil(8*sigma)[sigma>radius/8.]
		# Change dx as in 1D case (unchanged values will just be divided by 1)
		dx /= agauss
		agp = np.array([2, 2])
		# The number of bins for each ocurring array
		bins_per_radius = np.ceil(radius / dx)
		bins_wn = (agauss + agp) * bins_per_radius
		bins_gauss = agauss * bins_per_radius
		bins_gp = agp * bins_per_radius
		white_noise = np.random.random(2*bins_wn)
		# Now we need to differentiate between x and y
		gauss_limit = agauss*radius
		gauss_space_x = np.linspace(-gauss_limit[0], gauss_limit[0], 2*bins_gauss[0])
		gauss_space_y = np.linspace(-gauss_limit[1], gauss_limit[1], 2*bins_gauss[1])
		conv_limit = agp*radius
		conv_space_x = np.linspace(-conv_limit[0], conv_limit[0], 2*bins_gp[0] + 1)
		conv_space_y = np.linspace(-conv_limit[1], conv_limit[1], 2*bins_gp[1] + 1)
		# Note: meshgrid leads to shape (len(gauss_space_y), len(gauss_space_x))
		X_gauss, Y_gauss = np.meshgrid(gauss_space_x, gauss_space_y)
		pos = np.empty(X_gauss.shape + (2,))
		pos[:, :, 0] = X_gauss
		pos[:, :, 1] = Y_gauss
		gaussian = ((2*np.pi*sigma[0]**1) *
					stats.multivariate_normal(
						None,
						[[sigma[0]**2, 0.0], [0.0, sigma[1]**2]]).pdf(pos))
		# Since gaussian now has switched x and y we transpose it to make
		# it fit the shape of the white noise.
		# Note: now plotting the result with plt.contour shows switched x and y
		convolution = signal.fftconvolve(white_noise, gaussian.T, mode='valid')
		# Interpolate, i.e. only look at the gp in the region of interest
		gp = scipy.interpolate.RectBivariateSpline(
					conv_space_x, conv_space_y, convolution)(linspace, linspace)
		if rescale == 'stretch':
			gp = (stretch_factor * (gp - np.amin(gp))
				  / (np.amax(gp) - np.amin(gp)))
		elif rescale == 'fixed_mean':
			# mean_of_single_field = np.sqrt(2*np.pi*sigma**2)/(2*radius)
			desired_mean = 0.5
			gp_min = np.amin(gp)
			gp = (desired_mean
				  * (gp - gp_min) / np.mean(gp - gp_min))
			if untuned:
				gp = desired_mean
		else:
			print "The proper scaling is not yet implemented in 2D"
			sys.exit()
		return gp

def get_input_tuning_mass(sigma, tuning_function, limit,
						  integrate_within_limits=False, dimensions=1,
						  loc=None, gaussian_height=1):
	"""
	Returns the normalization factor M (see Notability)

	It is the normalization factor from the probablity distribution
	function, which we dropped because we use functions with peaks
	of height 1.
	So this is just the integral of the tuning function from
	[-infinity, infinity] or [-limit, limit].
	In one dimension there are analytical expressions.
	In two dimensions integrals from [-limit, limit] are done with dblquad.

	Note: Integration within limits does not yet work with von_mises,
		because we probably won't ever need it. However, it still
		requires a limit (see analytics).

	Parameters
	----------
	tuning_function : string
		'gaussian' : Gaussian of height 1.0
			exp( -x^2 / (2*sigma^2) )
		'lorentzian' : Lorentzian of height 1.0
			1 / ( 1 + ((x-x0)/sigma)^2 )
			See also Notability.
	limit : ndarray of shape (dimensions)
		The integration of the input function will be carried out in
		the interval [-limit[j], limit[j]] along dimension j
	inside_only : bool
		If True, the area of an input tuning curve located at `loc` INSIDE
		the box will be returned.
		Note: Use this to get the normalization factor
	integrate_within_limits : bool
		If True, the area of an input tuning curved located at 'loc' within
		`limit` will be returned.
		Note: that's teh expression used in the analytics
		(See stability_analysis.pdf).
	loc : Location of the center of the tuning curve
		If None, it is assumed to be in the center of the box

	Returns
	-------
	m : float or ndarray depending on the arguments
	"""
	if loc is None:
		loc = np.zeros(dimensions)
	if dimensions == 1:
		if tuning_function == 'gaussian':
			if integrate_within_limits:
				m = -gaussian_height * sigma * np.sqrt(np.pi/2) * (sps.erf((-limit-loc)/(sigma*np.sqrt(2))) + sps.erf((-limit+loc)/(sigma*np.sqrt(2))))
			else:
				m = gaussian_height * np.sqrt(2. * np.pi * sigma**2)
		elif tuning_function == 'lorentzian':
			if integrate_within_limits:
				m = sigma * (np.arctan((limit-loc)/sigma) - np.arctan((-limit-loc)/sigma))
			else:
				m = np.pi * sigma
		elif tuning_function == 'gaussian_process':
			m = 0.5
	elif dimensions == 2:
		if tuning_function == 'gaussian':
			if integrate_within_limits:
				m = dblquad(lambda x, y: np.exp(-(((x-loc[0])/(2*sigma[0]))**2 + ((y-loc[1])/(2*sigma[1]))**2)),
					   -limit[0], limit[0], lambda y: -limit[1], lambda y: limit[1])[0]
			else:
				m = 2 * np.pi * sigma[0] * sigma[1]
		elif tuning_function == 'lorentzian':
			if integrate_within_limits:
				m = dblquad(lambda x, y: 1. / (np.power(1 + ((x-loc[0])/sigma[0])**2 + ((y-loc[1])/sigma[1])**2, 1.5)),
					   -limit[0], limit[0], lambda y: -limit[1], lambda y: limit[1])[0]
			else:
				m = 2 * np.pi * sigma[0] * sigma[1]
		elif tuning_function == 'von_mises':
			scaled_kappa = (limit[1] / (np.pi*sigma[1]))**2
			m = (
					(sigma[0] * limit[1]
					 * sps.iv(0, scaled_kappa) * np.sqrt(8*np.pi))
						/ np.exp(scaled_kappa)
				)
		elif tuning_function == 'periodic':
			scaled_kappas = (limit / (np.pi*sigma))**2
			m = (
					4 * limit[0] * limit[1]
					* sps.iv(0, scaled_kappas[0]) * sps.iv(0, scaled_kappas[1])
					/ (np.exp(scaled_kappas[0]) * np.exp(scaled_kappas[1]))
				)
		elif tuning_function == 'gaussian_process':
			m = 0.5
	elif dimensions == 3:
		if tuning_function == 'von_mises':
			scaled_kappa = (limit[2] / (np.pi*sigma[2]))**2
			m = (
				4 * np.pi * sigma[0] * sigma[1] * limit[2]
				* sps.iv(0, scaled_kappa) / np.exp(scaled_kappa)
			)
		elif tuning_function == 'gaussian_process':
			m = 0.5
	return m

def get_fixed_point_initial_weights(dimensions, radius, center_overlap_exc,
		center_overlap_inh,
		target_rate, init_weight_exc, n_exc, n_inh,
		sigma_exc=None, sigma_inh=None,
		fields_per_synapse_exc=1,
		fields_per_synapse_inh=1,
		tuning_function='gaussian',
		gaussian_height_exc=1,
		gaussian_height_inh=1):
	"""Initial inhibitory weights chosen s.t. firing rate = target rate

	From the analytics we know which combination of initial excitatory
	and inhibitory weights leads to an overall output rate of the
	target rate. See stability_analysis.pdf

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

	m_exc = get_input_tuning_mass(sigma_exc, tuning_function, limit_exc,
								  dimensions=dimensions,
								  integrate_within_limits=True,
								  gaussian_height=gaussian_height_exc)
	m_inh = get_input_tuning_mass(sigma_inh, tuning_function, limit_inh,
								  dimensions=dimensions,
								  integrate_within_limits=True,
								  gaussian_height=gaussian_height_inh)

	if dimensions == 1:
		init_weight_inh = ( (n_exc * init_weight_exc * m_exc / limit_exc[0]
							- 2 * target_rate) / (n_inh * m_inh / limit_inh[0]))

	elif dimensions == 2:
		init_weight_inh = (
						(init_weight_exc * n_exc * m_exc
						/ (limit_exc[0] * limit_exc[1])
						- 4 * target_rate)
						/ (n_inh * m_inh  / (limit_inh[0] * limit_inh[1]))
		)

	elif dimensions == 3:
		init_weight_inh = (
			(init_weight_exc * n_exc * m_exc
			 / (limit_exc[0] * limit_exc[1] * limit_exc[2])
			 - 8 * target_rate)
			/ (n_inh * m_inh  / (limit_inh[0] * limit_inh[1] * limit_inh[2]))
		)

	return init_weight_inh

def get_equidistant_positions(r, n, boxtype='linear', distortion=0., on_boundary=False):
	"""Returns equidistant, symmetrically distributed coordinates

	Works in dimensions higher than One.
	The coordinates are taken such that they don't lie on the boundaries
	of the environment but instead half a lattice constant away on each
	side (now optional).
	Note: In the case of circular boxtype, positions outside the cirlce
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
	distortion : float or array_like or string
		If float or array: Maximal length by which each lattice coordinate (x and y separately)
		is shifted randomly (uniformly)
		If string:
			'half_spacing': The distortion is taken as half the distance between
			two points along a perfectly symmetric lattice (along each dimension)
	on_boundary : bool
		If True, positions can also lie on the system boundaries
	Returns
	-------
	(ndarray) of shape (m, len(n)), where m < np.prod(n) for boxtype
	'circular', because points at the edges are thrown away. Note
	len(n) is the dimensionality.
	"""
	if distortion == 'half_spacing':
		distortion = r / (n-1)
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
	n: (tuple) shape of random numbers array to be returned
	mean: (float) mean value for the distributions
	spreading: (float or array) specifies the spreading of the random nubmers
	distribution: (string) a certain distribution
		- uniform: uniform distribution with mean mean and percentual spreading spreading
		- cut_off_gaussian: normal distribution limited to range
			(spreading[1] to spreading[2]) with stdev spreading[0]
			Values outside the range are thrown away
		- gaussian_peak: Does not create random numbers, but a gaussian shaped
			weight profile for symmetric centers. Works only with one
			output neuron.

	Returns
	-------
	Array of n random numbers
	"""

	if distribution == 'uniform':
		rns = np.random.uniform(mean * (1. - spreading), mean * (1. + spreading), n)

	# elif distribution == 'factor':
	# 	rns = np.random.uniform(mean / spreading, mean * spreading, n)

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

	elif distribution == 'gaussian_peak':
		linspace = np.linspace(-1, 1, n[1])
		rns = stats.norm(loc=mean, scale=spreading).pdf(linspace).reshape(1, n[1]) / 8

	return rns



class Synapses(utils.Utilities):
	"""
	The class of excitatory and inhibitory synapses

	Notes:
		- Given the synapse_type, it automatically gets the appropriate
			parameters from params

	Parameters
	----------
	positions : ndarray
		Positions on which the firing rate of each input neuron should be
		defined in the case of gaussian process inputs.
	"""
	def __init__(self, sim_params, type_params, seed_centers, seed_init_weights,
					seed_sigmas, positions=None):
		# self.input_tuning = utils.InputTuning()
		for k, v in sim_params.items():
			setattr(self, k, v)

		for k, v in type_params.items():
			setattr(self, k, v)

		self.n_total = np.prod(self.number_per_dimension)

		##############################
		##########	centers	##########
		##############################
		np.random.seed(int(seed_centers))
		limit = self.radius + self.center_overlap
		self.set_centers(limit)
		self.number = self.centers.shape[0]

		#######################################################################
		############################ Normalization ############################
		#######################################################################
		# if self.normalization == 'mass_fraction_inside':
		# self.set_input_norm()


		##########################################################
		#################### Gasssian Process ####################
		##########################################################
		if self.gaussian_process:
			self.set_gaussian_process_rates(positions)

		##############################
		##########	sigmas	##########
		##############################
		np.random.seed(int(seed_sigmas))
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
		# self.scaled_kappa = np.array([(limit[-1] / (np.pi*self.sigmas[..., -1]))**2])
		self.scaled_kappas = ((limit[-1] / (np.pi*self.sigmas))**2)

		self.pi_over_r = np.array([np.pi / limit[-1]])
		self.norm_von_mises = 1 / np.exp(self.scaled_kappas)

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

		For each input neuron a firing rate is defined at each
		position in `positions`. So this defines all of the input
		tuning.

		Parameters
		----------
		positions : ndarray
			Positions on which inputs should be defined (high resolution)
		"""
		n = np.prod(self.number_per_dimension[:self.dimensions])
		self.gp_min, self.gp_max = np.empty(n), np.empty(n)
		if self.dimensions == 1:
			shape = (len(positions), n)
			self.gaussian_process_rates = np.empty(shape)
			for i in np.arange(n):
				if i % 100 == 0:
					print i
				# white_noise = np.random.random(6e4)
				self.gaussian_process_rates[:,i], self.gp_min[i], self.gp_max[i]\
					= get_gaussian_process(
					self.radius, self.sigma, positions,
					rescale=self.gaussian_process_rescale,
					stretch_factor=self.gp_stretch_factor,
					extremum=self.gp_extremum,
					untuned=self.untuned)
		elif self.dimensions == 2:
			linspace = positions[0,:,0]
			shape = (linspace.shape[0], linspace.shape[0], n)
			self.gaussian_process_rates = np.empty(shape)
			for i in np.arange(n):
				print i
				# white_noise = np.random.random((1e3, 1e3))
				self.gaussian_process_rates[..., i] = get_gaussian_process(
					self.radius, self.sigma, linspace,
					dimensions=self.dimensions,
					rescale=self.gaussian_process_rescale,
					stretch_factor=self.gp_stretch_factor,
					extremum=self.gp_extremum,
					untuned=self.untuned)

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
				self.centers = np.linspace(-limit[0], limit[0], self.number_per_dimension[0])
				self.centers = self.centers.reshape(
					(self.number_per_dimension[0], self.fields_per_synapse))
			else:
				self.centers = np.random.uniform(
					-limit, limit,
					(self.number_per_dimension[0], self.fields_per_synapse)).reshape(
						self.number_per_dimension[0], self.fields_per_synapse)
			# self.centers.sort(axis=0)

		if self.dimensions >= 2:
			if self.boxtype == 'linear' and not self.symmetric_centers:
				centers_x = np.random.uniform(-limit[0], limit[0],
							(self.n_total, self.fields_per_synapse))
				centers_y = np.random.uniform(-limit[1], limit[1],
							(self.n_total, self.fields_per_synapse))
				self.centers = np.dstack((centers_x, centers_y))
			elif self.boxtype == 'circular' and not self.symmetric_centers:
				limit = self.radius + self.center_overlap
				random_positions_within_circle = get_random_positions_within_circle(
						self.n_total*self.fields_per_synapse, limit[0])
				self.centers = random_positions_within_circle.reshape(
							(self.n_total, self.fields_per_synapse, 2))
			elif self.symmetric_centers:
				self.centers = get_equidistant_positions(limit,
								self.number_per_dimension, self.boxtype,
									self.distortion)
				N = self.centers.shape[0]
				fps = self.fields_per_synapse
				# In the case of several fields per synapse (fps) and rather
				# symmetric  distribution of centers, we create fps many
				# distorted lattices and concatenate them all
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

class Rat(utils.Utilities):
	"""
	The class of the rat
	"""
	def __init__(self, params):
		self.params = params
		for k, v in params['sim'].items():
			setattr(self, k, v)
		for k, v in params['out'].items():
			setattr(self, k, v)
		self.set_initial_position()
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
		# self.params['exc']['center_overlap'] = np.atleast_1d(self.params['exc']['center_overlap'])
		# self.params['inh']['center_overlap'] = np.atleast_1d(self.params['inh']['center_overlap'])
		self.set_center_overlap()
		if self.tuning_function == 'von_mises':
			condition = (self.motion == 'persistent_semiperiodic'
						 or self.motion == 'sargolini_data')
			if not condition:
				raw_input('The motion is not semiperiodic but the input function are!')
			if self.boxtype != 'linear':
				raw_input('The boxtype is not linear even though the input function is Von Mises!')

		self.positions_grid = self.get_positions(
								self.radius, self.dimensions, self.spacing)

		# TODO: You stopped the transposing here and in plotting.py
		# You will have to change something in the 3D case also, otherwise
		# it's proabably not going to work anymore.
		# if self.dimensions >= 2:
		#
		# 	# input rates grid. Why? What if you don't do it at all and change
		# 	# the transposing stuff in the plotting functions? You have to do
		# 	# this careful. Bear in mind: Also the projection and polar plots
		# 	# need to be done with care.
		# 	self.positions_grid = np.transpose(self.positions_grid,
		# 								(1, 0, 2, 3, 4, 5)[:self.dimensions+3])


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
		###################### Instantiating the Synapses ######################
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
				# Here we set the high resolution input rates grid
				# Note: it already has the correct precision, because
				# `positions` is the desired discretization
				self.input_rates[p] = self.synapses[
							p].gaussian_process_rates
				# Set the min and max of unscaled gp inputs to find
				# their distribution
				# self.gp_min[p] = self.synapses[p].gp_min
				# self.gp_max[p] = self.synapses[p].gp_max
				# Here we set the low resolution input rates grid
				self.set_input_rates_low_resolution(p, positions)
				self.synapses[p].input_norm = np.array([1])

			else:
				self.set_input_norm(positions=self.positions_grid, syn_type=p)

				# Here we set the low resolution input rates grid
				self.input_rates_low_resolution[p] = \
					self.get_input_rates_grid(self.positions_grid,
											  self.synapses[p])

				if self.discretize_space:
					print 'Creating the large input rates grid'
					self.input_rates[p] = self.get_input_rates_grid(
						positions, self.synapses[p])
				else:
					# Here we create a function that returns the firing rate
					# of each input neuron at a single position
					self.get_rates_at_single_position[p] = \
						self.synapses[p].get_rates_function(
								position=self.position, data=False)


		if self.params['sim']['first_center_at_zero']:
			if self.dimensions == 1:
				self.synapses['exc'].centers[0] = np.zeros(
						self.params['exc']['fields_per_synapse'])
			if self.dimensions == 2:
				self.synapses['exc'].centers[0] = np.zeros(
						(self.params['exc']['fields_per_synapse'], 2))
		if self.params['sim']['same_centers']:
			self.synapses['inh'].centers = self.synapses['exc'].centers

	def set_initial_position(self):
		"""
		Sets the inital x, y and z values

		Note: self.position is only used once in the code and should
				be deprecated
		"""
		if self.motion == 'sargolini_data':
			# self.sargolini_norm = 51.6182218615
			self.sargolini_data = np.load('data/sargolini_trajectories_610min.npy')
			self.x, self.y = self.sargolini_data[0]
			self.z = None
			# self.x *= self.radius / self.sargolini_norm
			# self.y *= self.radius / self.sargolini_norm
		else:
			self.x = self.initial_x
			self.y = self.initial_y
			self.z = self.initial_z
		self.position = np.array([self.x, self.y, self.z][:self.dimensions])


	def set_center_overlap(self):
		"""
		Sets the center overlap of the inputs depending on the periodicity

		First we set the center_overlap to some multiple of the field width.
		Along perodic dimensions we then set the overlap to zero,
		because along the periodic dimensions there is no such thing as
		'outside'.
		"""
		is_tuning_function_without_overlap = (
			self.tuning_function == 'gaussian_process'
			or self.tuning_function == 'periodic'
		)
		for p in ['exc', 'inh']:
			self.params[p]['center_overlap'] = (
				np.atleast_1d(self.params[p]['sigma'])
				* self.params[p]['center_overlap_factor']
			)
			if self.tuning_function == 'von_mises':
				self.params[p]['center_overlap'][-1] = 0.
			elif is_tuning_function_without_overlap:
				self.params[p]['center_overlap'] = np.zeros_like(
											self.params[p]['center_overlap'])

	def set_input_norm(self, positions, syn_type='exc'):
		"""
		Sets the normalization array `input_norm` of the input rates

		For the time being this is the ratio:
		integration of input function over infinity / integration of
		input function over the box.
		So receptive fields with area outside the box in which the rat
		moves get an increased field to account for that loss.

		We always have an analytical expression for the integral
		over infinity.
		self.input_normalization specifies how the integral over the box
		is determined.

		'analytics':
			Uses an analytical expression for the integral.
			Works only in 1 dimension.
		'rates_sum':
			Uses an un-normalized input rates array to get the approximate
			integral by summing all the values inside the box and
			multiplying with the discretization
			dx = 2 * radius / (# intervals - 1)
			The precision depends on the discretization of `positions`.

		Parameters
		----------
		positions : ndarray
			An array of positions that discretizes the box.
			See get_positions function.
		syn_type : string
			'exc' or 'inh'
		"""
		syn = self.synapses[syn_type]

		syn.input_norm = [1]

		# Get the un-normalized input array
		input_rates = self.get_input_rates_grid(positions, syn)
		m_total = get_input_tuning_mass(syn.sigma, self.tuning_function,
									np.array([self.radius, self.radius, self.radius]),
									dimensions=self.dimensions,
									gaussian_height=syn.gaussian_height)

		if self.input_normalization == 'rates_trapz':
			m_inside = np.trapz(input_rates, positions[:,0,0], axis=0)

		elif self.input_normalization == 'rates_sum':
			if self.dimensions == 1:
				m_inside = (2*self.radius*np.sum(input_rates, axis=0)
										/ (positions.shape[0] - 1))
			elif self.dimensions == 2:
				m_inside = 	(4 * self.radius**2 * np.sum(
										np.sum(input_rates, axis=0), axis=0)
								/ ((positions.shape[0] - 1)*(positions.shape[1] - 1))
				)

		elif self.input_normalization == 'analytics':
			m_inside = get_input_tuning_mass(syn.sigma,
							self.tuning_function, self.radius,
							integrate_within_limits=True,
							dimensions=self.dimensions, loc=syn.centers[:,0],
							gaussian_height=self.gaussian_height)

		# elif self.input_normalization == 'numerics':
		# 	syn.input_norm = np.empty(syn.number)
		# 	for i in np.arange(syn.number):
		# 		syn.input_norm[i] =  (m / quad(
		# 			lambda x: np.exp(-(x-syn.centers[i,0])**2/(2*syn.sigma**2)),
		# 			-self.radius, self.radius)[0]
		# 		)
		else:
			m_inside = m_total
		syn.input_norm = np.atleast_1d(m_total / m_inside)

	def set_input_rates_low_resolution(self, syn_type, positions):
		"""
		Interpolate input_rates towards smaller resolution for each input cell

		Interpolation needs to be done on each input individually so we
		have to loop over the total number of input neurons.

		Note: Only needed for Gaussian process input.
		For normal input we make use of get_rates_function

		Parameters
		----------
		syn_type : string
			Type of the synapse
		positions : ndarray
			Positions on which the high resultion input rates are defined
		"""
		if self.dimensions  == 1:
			self.input_rates_low_resolution[syn_type] = np.empty(
								(self.spacing, self.synapses[syn_type].n_total))
			for i in np.arange(self.synapses[syn_type].n_total):
				self.input_rates_low_resolution[syn_type][:,i] = np.interp(
					np.squeeze(self.positions_grid),
					np.squeeze(positions),
					self.input_rates[syn_type][:,i])

		elif self.dimensions == 2:
			linspace_low_resolution = np.linspace(-self.radius, self.radius, self.spacing)
			linspace = np.squeeze(positions)[0,:,0]
			self.input_rates_low_resolution[syn_type] = np.empty(
								(self.spacing, self.spacing,
								 self.synapses[syn_type].n_total))
			for i in np.arange(self.synapses[syn_type].n_total):
				self.input_rates_low_resolution[syn_type][...,i] = scipy.interpolate.interp2d(
					linspace,
					linspace,
					self.input_rates[syn_type][...,i])(linspace_low_resolution,
													 linspace_low_resolution)


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
			In 1 dimension: (spacing, 1, 1)
			In 2 dimensions: (spacing_y, spacing_x, 1, 2)
			In 3 dimensions: (spacing_y, spacing_x, spacing_z, 1, 3)
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
			shape = (n[1], n[0], n[dimensions-1])[:dimensions] + (1, dimensions)
			positions.shape = shape

		if return_discretization:
			return positions, n
		else:
			return positions


	def set_fixed_point_initial_weights(self):
		"""
		Sets the initial inhibitory weights s.t. rate is close to target rate

		See mathematical analysis (linear_stablitiy_analysis)
		"""
		params = self.params
		self.params['inh']['init_weight'] = params['inh']['weight_factor'] * get_fixed_point_initial_weights(
			dimensions=self.dimensions, radius=self.radius,
			center_overlap_exc=params['exc']['center_overlap'],
			center_overlap_inh=params['inh']['center_overlap'],
			sigma_exc=params['exc']['sigma'],
			sigma_inh=params['inh']['sigma'],
			target_rate=self.target_rate,
			init_weight_exc=params['exc']['init_weight'],
			n_exc=np.prod(params['exc']['number_per_dimension']),
			n_inh=np.prod(params['inh']['number_per_dimension']),
			fields_per_synapse_exc=params['exc']['fields_per_synapse'],
			fields_per_synapse_inh=params['inh']['fields_per_synapse'],
			tuning_function=self.tuning_function,
			gaussian_height_exc=params['exc']['gaussian_height'],
			gaussian_height_inh=params['inh']['gaussian_height'])
		print self.tuning_function

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

	def move_persistently_periodic(self):
		"""
		Motion in 2D box which is periodic along both sides
		"""
		if self.dimensions == 2:
			# Boundary conditions and movement are interleaved here
			pos = np.array([self.x, self.y])
			# Define array 'too_what?' of kind [v_x, v_y], where v_i is +1/0/-1
			# if coordinate i is too_positive/inside_box/too_negative
			too_what = (pos > self.radius) * 1 - (pos < -self.radius) * 1
			self.x -= 2* self.radius * too_what[0]
			self.y -= 2* self.radius * too_what[1]
			self.phi += self.angular_sigma * np.random.randn()
			self.x += self.velocity_dt * np.cos(self.phi)
			self.y += self.velocity_dt * np.sin(self.phi)

		else:
			print 'The motion function is only defined for two dimensions!'
			sys.exit()

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
	# def iterate_sargolini_data(self):
	# 	for pos in sargolini_data:
	# 		yield pos

	def move_sargolini_data(self):
		# Ensure that you dont run out of data an loop backt to the beginning
		# 1829127 is the length of the 610 minutes data array
		step = self.step % 1829126
		self.x, self.y = self.sargolini_data[step]
		if self.dimensions == 3:
			x_future, y_future = self.sargolini_data[step + 1]
			# Get phi as direction of motion and add noise
			phi = np.arctan2(y_future - self.y,
							 x_future - self.x) + np.random.randn() * self.head_direction_sigma
			# Ensure that phi is in [-pi, pi]
			phi = (phi + np.pi) % (2 * np.pi) - np.pi
			self.z = phi * self.radius / np.pi

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
		n_vector = (self.synapses['exc'].weights[0,:] > substraction_value).astype(int)

		substractive_norm = (
			self.synapses['exc'].eta_dt * self.output_rate
			* np.dot(self.rates['exc'], n_vector) * n_vector
			/ np.sum(n_vector)
		)
		self.synapses['exc'].weights -= substractive_norm

	def normalize_exc_weights_linear_multiplicative(self):
		"""Normalize multiplicatively, keeping the linear sum constant"""
		self.synapses['exc'].weights *= (
			self.synapses['exc'].initial_weight_sum
				/ np.sum(self.synapses['exc'].weights))


	def normalize_exc_weights_inactive(self):
		"""
		No normalization
		"""
		pass

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

	# def get_output_rates_from_equation(self, frame, rawdata, spacing,
	# 	positions_grid=False, input_rates=False, equilibration_steps=10000):
	# 	"""	Return output rates at many positions
	#
	# 	***
	# 	Note:
	# 	This function used to be in plotting.py, but now it is used here
	# 	to output arrays containing the output rates. This makes
	# 	quick plotting and in particular time traces of Grid Scores feasible.
	# 	***
	#
	# 	For normal plotting in 1D and for contour plotting in 2D.
	# 	It is differentiated between cases with and without lateral inhibition.
	#
	# 	With lateral inhibition the output rate has to be determined via
	# 	integration (but fixed weights).
	# 	In 1 dimensions we start at one end of the box, integrate for a
	# 	time specified by equilibration steps and than walk to the
	# 	other end of the box.
	#
	# 	Parameters
	# 	----------
	# 	frame : int
	# 		Frame at which the output rates are plotted
	# 	rawdata : dict
	# 		Contains the synaptic weights
	# 	spacing : int
	# 		The spacing, describing the detail richness of the plor or contour plot (spacing**2)
	# 	positions_grid, input_rates : ndarray
	# 		Arrays as described in get_X_Y_positions_grid_input_rates_tuple
	# 	equilibration_steps : int
	# 		Number of steps of integration to reach the correct
	# 		value of the output rates for the case of lateral inhibition
	#
	# 	Returns
	# 	-------
	# 	output_rates : ndarray
	# 		Array with output rates at several positions tiling the space
	# 		For 1 dimension with shape (spacing)
	# 		Fro 2 dimensions with shape (spacing, spacing)
	# 	"""
	#
	# 	# plt.title('output_rates, t = %.1e' % (frame * self.every_nth_step_weights), fontsize=8)
	# 	if self.dimensions == 1:
	# 		linspace = np.linspace(-self.radius, self.radius, spacing)
	#
	# 		if self.lateral_inhibition:
	# 			output_rates = np.empty((spacing, self.output_neurons))
	#
	# 			start_pos = -self.radius
	# 			end_pos = self.radius
	# 			r = np.zeros(self.output_neurons)
	# 			dt_tau = self.dt / self.tau
	# 			# tau = 0.011
	# 			# dt = 0.01
	# 			# dt_tau = 0.1
	# 			x = start_pos
	# 			for s in np.arange(equilibration_steps):
	# 				r = (
	# 						r*(1 - dt_tau)
	# 						+ dt_tau * ((
	# 						np.dot(rawdata['exc']['weights'][frame],
	# 							input_rates['exc'][0]) -
	# 						np.dot(rawdata['inh']['weights'][frame],
	# 							input_rates['inh'][0])
	# 						)
	# 						- self.weight_lateral
	# 						* (np.sum(r) - r)
	# 						)
	# 						)
	# 				r[r<0] = 0
	# 			start_r = r
	# 			# output_rates = []
	# 			for n, x in enumerate(linspace):
	# 				for s in np.arange(200):
	# 					r = (
	# 							r*(1 - dt_tau)
	# 							+ dt_tau * ((
	# 							np.dot(rawdata['exc']['weights'][frame],
	# 								input_rates['exc'][n]) -
	# 							np.dot(rawdata['inh']['weights'][frame],
	# 								input_rates['inh'][n])
	# 							)
	# 							- self.weight_lateral
	# 							* (np.sum(r) - r)
	# 							)
	# 							)
	# 					r[r<0] = 0
	# 				output_rates[n] = r
	#
	# 		else:
	# 			output_rates = (
	# 				np.tensordot(rawdata['exc']['weights'][frame],
	# 									input_rates['exc'], axes=([-1], [1]))
	# 				- np.tensordot(rawdata['inh']['weights'][frame],
	# 					 				input_rates['inh'], axes=([-1], [1]))
	# 			)
	# 			output_rates = output_rates
	# 		output_rates[output_rates<0] = 0
	# 		output_rates = output_rates.reshape(spacing, self.output_neurons)
	# 		return output_rates
	#
	# 	if self.dimensions >= 2:
	# 		if self.lateral_inhibition:
	# 			output_rates = np.empty((spacing, spacing, self.output_neurons))
	# 			start_pos = positions_grid[0, 0, 0, 0]
	# 			r = np.zeros(self.output_neurons)
	# 			dt_tau = self.dt / self.tau
	#
	# 			pos = start_pos
	# 			for s in np.arange(equilibration_steps):
	# 				r = (
	# 						r*(1 - dt_tau)
	# 						+ dt_tau * ((
	# 						np.dot(rawdata['exc']['weights'][frame],
	# 							input_rates['exc'][0][0]) -
	# 						np.dot(rawdata['inh']['weights'][frame],
	# 							input_rates['inh'][0][0])
	# 						)
	# 						- self.weight_lateral
	# 						* (np.sum(r) - r)
	# 						)
	# 						)
	# 				r[r<0] = 0
	# 			# start_r = r
	# 			# print r
	# 			# output_rates = []
	#
	# 			for ny in np.arange(positions_grid.shape[1]):
	# 				for nx in np.arange(positions_grid.shape[0]):
	# 					pos = positions_grid[nx][ny]
	# 					for s in np.arange(200):
	# 						r = (
	# 								r*(1 - dt_tau)
	# 								+ dt_tau * ((
	# 								np.dot(rawdata['exc']['weights'][frame],
	# 									input_rates['exc'][nx][ny]) -
	# 								np.dot(rawdata['inh']['weights'][frame],
	# 									input_rates['inh'][nx][ny])
	# 								)
	# 								- self.weight_lateral
	# 								* (np.sum(r) - r)
	# 								)
	# 								)
	# 						r[r<0] = 0
	#
	# 					output_rates[nx][ny] = r
	#
	# 			# for i in np.arange(self.output_neurons):
	# 			# 	output_rates[:,:,i] = np.transpose(output_rates[:,:,i])
	#
	# 		else:
	# 			output_rates = (
	# 				np.tensordot(rawdata['exc']['weights'][frame],
	# 									input_rates['exc'], axes=([-1], [self.dimensions]))
	# 				- np.tensordot(rawdata['inh']['weights'][frame],
	# 					 				input_rates['inh'], axes=([-1], [self.dimensions]))
	# 			)
	# 			# Rectification
	# 			output_rates[output_rates < 0] = 0.
	# 			if self.dimensions == 2:
	# 				output_rates = output_rates.reshape(
	# 								spacing, spacing, self.output_neurons)
	# 			elif self.dimensions == 3:
	# 				output_rates = output_rates.reshape(
	# 								spacing, spacing, spacing, self.output_neurons)
	# 		return output_rates

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
		if self.motion == 'persistent_periodic':
			self.move = self.move_persistently_periodic
		if self.motion == 'sargolini_data':
			self.move = self.move_sargolini_data

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

		# Add parameters to rawdata that are generated in the initialization
		for p in ['exc', 'inh']:
			name_list = [
				'norm_von_mises',
				'input_norm',
				'pi_over_r',
				'scaled_kappas',
				# 'twoSigma2',
				'fields_per_synapse',
				'centers',
				# 'sigmas',
				'gp_min',
				'gp_max',
			]
			if self.params['sim']['store_twoSigma2']:
				name_list.append('sigmas')
				name_list.append('twoSigma2')

			# Check if sigmas are drawn from a distribution, because
			# only then do we save them all
			if np.count_nonzero(self.params[p]['sigma_spreading']) > 0:
				name_list.append('sigmas')
				print 'sigma array is stored'
			for name in name_list:
				try:
					rawdata[p][name] = getattr(self.synapses[p], name)
				except AttributeError as e:
					print e
					rawdata[p][name] = None

			rawdata[p]['number'] = np.array([self.synapses[p].number])
			weights_shape = (time_shape_weights, self.output_neurons,
												self.synapses[p].number)
			rawdata[p]['weights'] = np.empty(weights_shape)
			rawdata[p]['weights'][0] = self.synapses[p].weights.copy()
			rawdata[p]['input_rates'] = self.input_rates_low_resolution[p][
										..., :self.save_n_input_rates]

		rawdata['positions'] = np.empty((time_shape, 3))
		if 'persistent' in self.params['sim']['motion']:
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

		if 'persistent' in self.params['sim']['motion']:
			rawdata['phi'][0] = self.phi
		rawdata['positions'][0] = np.array([self.x, self.y, self.z])
		rawdata['output_rates'][0] = 0.0

		if self.lateral_inhibition:
			self.output_rate = 0.
		for self.step in self.steps:
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

			if self.step % self.every_nth_step == 0:
				index = self.step / self.every_nth_step
				# Store Positions
				rawdata['positions'][index] = np.array([self.x, self.y, self.z])
				if 'persistent' in self.params['sim']['motion']:
					rawdata['phi'][index] = np.array(self.phi)
				rawdata['output_rates'][index] = self.output_rate

			if self.step % self.every_nth_step_weights == 0:
				print 'Current step: %i' % self.step
				index = self.step / self.every_nth_step_weights
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

		########################################################################
		################# Test to add computed in run function #################
		########################################################################

		# all_data = {}
		# add_comp = add_computed.Add_computed(
		# 				params=self.params, rawdata=rawdata)
		# for f in self.params['compute']:
		# 	all_data.update(getattr(add_comp, f)())
		#
		# print all_data
		# rawdata.update({'computed': all_data})


		print 'Simulation finished'
		return rawdata

