import numpy as np
from scipy.stats import norm


class Synapses:
	"""
	The class of excitatory and inhibitory synapses

	Notes:
		- Given the synapse_type, it automatically gets the appropriate
			parameters from params
	"""
	def __init__(self, params, synapse_type):
		self.params = params
		self.boxlength = params['boxlength']
		self.dimensions = params['dimensions']
		self.normalization = params['normalization']
		self.type = synapse_type
		self.n = params['n_' + self.type]
		# So far we take the sigma fixed
		# Maybe change to sigma array one day
		self.sigma = params['sigma_' + self.type]
		self.sigmas = np.ones(self.n) * self.sigma
		self.twoSigma2 = 1. / (2 * self.sigma**2)
		self.norm = 1. / (self.sigma * np.sqrt(2 * np.pi))
		self.init_weight_noise = params['init_weight_noise_' + self.type]
		# Create weights array adding some noise to the init weights
		self.weights = (
			(1 + self.init_weight_noise *
			(2 * np.random.random_sample(self.n) - 1)) *
			params['init_weight_' + self.type]
		)
		#self.weights = np.ones(self.n) * params['init_weight_' + self.type]
		self.initial_weight_sum = np.sum(self.weights)
		self.initial_squared_weight_sum = np.sum(np.square(self.weights))
		self.dt = params['dt']
		self.eta_dt = params['eta_' + self.type] * self.dt
		if self.dimensions == 1:
			self.centers = np.random.random_sample(self.n) * self.boxlength

	def set_rates(self, position):
		"""
		Computes the values of all place field Gaussians at <position>

		Future Tasks:
			- 	Maybe do not normalize because the normalization can be put into the
				weights anyway
			- 	Make it work for arbitrary dimensions
		"""
		if self.dimensions == 1:
			rates = self.norm*np.exp(-np.power(position - self.centers, 2)*self.twoSigma2)
			self.rates = rates


class Rat:
	"""
	The class of the rat
	"""
	def __init__(self, params):
		self.params = params
		self.boxlength = params['boxlength']
		self.x = params['initial_x']
		self.y = params['initial_y']
		self.diff_const = params['diff_const']
		self.dt = params['dt']
		self.dspace = np.sqrt(2.0*self.diff_const*self.dt)
		self.exc_syns = Synapses(params, 'exc')
		self.inh_syns = Synapses(params, 'inh')
		self.dimensions = params['dimensions']
		# self.eta_exc = params['eta_exc']
		# self.eta_inh = params['eta_inh']
		self.dt = params['dt']
		# self.eta_exc_dt = self.eta_exc * self.dt
		# self.eta_inh_dt = self.eta_inh * self.dt
		self.target_rate = params['target_rate']
		self.normalization = params['normalization']
		self.steps = np.arange(0, params['simulation_time']/params['dt'])
		# self.n_exc = params['n_exc']
		# Set the gaussion of diffusive steps
		# Note: You can either take a normal distribution and in each time step multiply
		# it with sqrt(2*D*dt) or directly define a particular Gaussian
		# self.diffusion_gauss = norm(scale=np.sqrt(2.0*self.diff_const*self.dt)).pdf

	def move_diffusively(self):
		"""
		Update position of rat by number drawn from gauss with stdev = dspace
		"""
		self.x += self.dspace*np.random.randn()

	def reflective_BCs(self):
		"""
		Reflective Boundary Conditions

		If the rat moves beyond the boundary, it gets reflected inside the boundary
		by the amount it was beyond the boundary
		"""
		if self.x < 0:
			self.x -= 2.0*self.x
		if self.x > self.boxlength:
			self.x -= 2.0*(self.x - self.boxlength)

	def rectification(self, value):
		"""
		Rectification of Firing Rates
		"""
		if value < 0:
			value = 0
		return value

	def set_current_output_rate(self):
		"""
		Sums exc_weights * exc_rates and substracts inh_weights * inh_rates
		"""
		rate = (
			np.dot(self.exc_syns.weights, self.exc_syns.rates) -
			np.dot(self.inh_syns.weights, self.inh_syns.rates)
		)
		self.output_rate = self.rectification(rate)

	def set_current_input_rates(self):
		"""
		Set the rates of the input neurons by using their place fields
		"""
		self.exc_syns.set_rates(self.x)
		self.inh_syns.set_rates(self.x)

	def update_exc_weights(self):
		"""
		Update exc weights according to Hebbian learning
		"""
		self.exc_syns.weights += (
			self.exc_syns.rates * self.output_rate * self.exc_syns.eta_dt
		)

	def update_inh_weights(self):
		"""
		Update inh weights according to Hebbian learning with target rate
		"""
		self.inh_syns.weights += (
			self.inh_syns.rates *
				(self.output_rate - self.target_rate) * self.inh_syns.eta_dt
		)

	def update_weights(self):
		"""
		Update both weights (convenience function)
		"""
		self.update_exc_weights()
		self.update_inh_weights()

	def normalize_exc_weights(self):
		"""
		Synaptic normalization for the excitatory weights

		Different types of normalization:
			- linear: L_1 norm of weights is kept constant
			- quadratic: L_2 norm of weights is kept contant
			- substractive: substraction is used to keep weights constant
			- multiplicative: multiplication is used to keep weights constant
		"""
		if self.normalization == 'linear_substractive':
			substractive_norm = (
				self.exc_syns.eta_dt * self.output_rate * np.sum(self.exc_syns.rates)
				/ self.exc_syns.n
			)
			self.exc_syns.weights -= substractive_norm
		if self.normalization == 'linear_multiplicative':
			self.exc_syns.weights = (
				(self.exc_syns.initial_weight_sum / np.sum(self.exc_syns.weights)) *
				self.exc_syns.weights
			)
		if self.normalization == 'quadratic_multiplicative':
			self.exc_syns.weights = (
				np.sqrt((self.exc_syns.initial_squared_weight_sum /
												np.sum(np.square(self.exc_syns.weights)))) *
					self.exc_syns.weights
			)


	# asdf  asdf asdf asdf adsf asdf asdf a asdf adsf asdf asdf asdf adsf asdf asdf 
	def run(self, position_output=False):
		"""
		Let the rat move and learn

		Arguments:
		- 	position_output: if True, self.positions gets all the rat positions
			appended
		"""
		print 'Type of Normalization: ' + self.normalization
		self.positions = [[self.x, self.y]]
		for step in self.steps:
			print self.inh_syns.weights
			#print self.exc_syns.weights	
			self.set_current_input_rates()
			self.set_current_output_rate()
			self.update_weights()
			self.normalize_exc_weights()
			self.move_diffusively()
			self.reflective_BCs()
			# Keep positions
			if position_output is True:
				self.positions.append([self.x, self.y])


# def get_weight_arrays(exc_synapses, inh_synapses):


# def get_synapse_lists(params):
# 	"""
# 	Returns the exc and inh synpase classes in two separate lists.
# 	"""
# 	exc_synapses = []
# 	inh_synapses = []
# 	for i in xrange(0, params['n_exc']):
# 		exc_synapses.append(Synapse(params, 'exc'))
# 	for j in xrange(0, params['n_inh']):
# 		inh_synapses.append(Synapse(params, 'inh'))
# 	return exc_synapses, inh_synapses

# class PlaceCells:
# 	"""
# 	bla
# 	"""
# 	def __init__(self, params):
# 		self.params = params
# 		self.boxtype = params['boxtype']
# 		self.boxlength = params['boxlength']
# 		self.dimensions = params['dimensions']
# 		self.n_exc = params['n_exc']
# 		self.n_inh = params['n_inh']

# 		if self.dimensions == 1:
# 			# Create sorted arrays with random numbers between 0 and boxlength
# 			self.exc_centers = np.sort(np.random.random_sample(self.n_exc)*self.boxlength)
# 			self.inh_centers = np.sort(np.random.random_sample(self.n_inh)*self.boxlength)

# 	def get_centers(self):
# 		if self.dimensions == 1:
# 			# Create sorted arrays with random numbers between 0 and boxlength
# 			exc_centers = np.sort(np.random.random_sample(self.n_exc)*self.boxlength)
# 			inh_centers = np.sort(np.random.random_sample(self.n_inh)*self.boxlength)
# 			return exc_centers, inh_centers

# 	# def get_functions(self):
# 	# 	exc_centers, inh_centers = self.get_centers()
# 	# 	p

# 	def print_params(self):
# 		print self.params


# class Synapse:
# 	"""
# 	bla
# 	"""
# 	def __init__(self, params, type):
# 		self.params = params
# 		self.boxlength = params['boxlength']
# 		self.dimensions = params['dimensions']
# 		self.type = type
# 		# Set the center of this synapse randomly
# 		self.center = np.random.random_sample()*self.boxlength
# 		### Exitatory Synapses ###
# 		if self.type == 'exc':
# 			self.weight = params['init_weight_exc']
# 			if self.dimensions == 1:
# 				self.field = norm(loc=self.center, scale=params['sigma_exc']).pdf
# 		### Inhibitory Synapses ###
# 		if self.type == 'inh':
# 			self.weight = params['init_weight_inh']
# 			if self.dimensions == 1:
# 				self.field = norm(loc=self.center, scale=params['sigma_inh']).pdf

# 	def get_rate(self, position):
# 		"""
# 		Get the firing rate of the synapse at the current position.

# 		- position: [x, y]
# 		"""
# 		if self.dimensions == 1:
# 			return self.field(position[0])
