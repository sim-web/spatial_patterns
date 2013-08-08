# import pdb
import numpy as np
import utils
# from scipy.stats import norm


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
		self.norm2 = 1. / (self.sigma**2 * 2 * np.pi)
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
			# self.centers = np.linspace(0.0, 1.0, self.n)
			self.centers = np.random.random_sample(self.n) * self.boxlength
			# sort the centers
			self.centers.sort(axis=0)
		if self.dimensions == 2:
			self.centers = np.random.random_sample((self.n, 2)) * self.boxlength
	def set_rates(self, position):
		"""
		Computes the values of all place field Gaussians at <position>

		Future Tasks:
			- 	Maybe do not normalize because the normalization can be put into the
				weights anyway
			- 	Make it work for arbitrary dimensions
		"""
		if self.dimensions == 1:
			self.rates = self.norm*np.exp(-np.power(position - self.centers, 2)*self.twoSigma2)
		if self.dimensions == 2:
			# print np.power(position - self.centers), 2), axis=1)
			self.rates =  (
				self.norm2
				* np.exp(-np.sum(np.power(position - self.centers, 2), axis=1) * self.twoSigma2)
			)

class Rat:
	"""
	The class of the rat
	"""
	def __init__(self, params):
		self.params = params
		for k, v in params.items():
			setattr(self, k, v)
		self.x = params['initial_x']
		self.y = params['initial_y']
		self.dspace = np.sqrt(2.0*self.diff_const*self.dt)
		self.exc_syns = Synapses(params, 'exc')
		self.inh_syns = Synapses(params, 'inh')
		self.steps = np.arange(0, params['simulation_time']/params['dt'])

	def move_diffusively(self):
		"""
		Update position of rat by number drawn from gauss with stdev = dspace
		"""
		if self.dimensions == 1:
			self.x += self.dspace*np.random.randn()
		if self.dimensions == 2:
			self.x += self.dspace*np.random.randn()
			self.y += self.dspace*np.random.randn()

	def reflective_BCs(self):
		"""
		Reflective Boundary Conditions

		If the rat moves beyond the boundary, it gets reflected inside the boundary
		by the amount it was beyond the boundary
		"""
		if self.dimensions == 1:
			if self.x < 0:
				self.x -= 2.0*self.x
			if self.x > self.boxlength:
				self.x -= 2.0*(self.x - self.boxlength)

		if self.dimensions == 2:
			if self.x < 0:
				self.x -= 2.0*self.x
			if self.x > self.boxlength:
				self.x -= 2.0*(self.x - self.boxlength)
			if self.y < 0:
				self.y -= 2.0*self.y
			if self.y > self.boxlength:
				self.y -= 2.0*(self.y - self.boxlength)	


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
		if self.dimensions == 1:
			self.exc_syns.set_rates(self.x)
			self.inh_syns.set_rates(self.x)
		if self.dimensions == 2:
			self.exc_syns.set_rates([self.x, self.y])
			self.inh_syns.set_rates([self.x, self.y])	

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
			# Get a vector with entries of ones and zeroes
			# For each synapse with positive values you get a one
			# For each synapase with negative values you get a zero
			# See Dayan, Abbott p. 290 for schema
			substraction_value = (
				self.exc_syns.eta_dt * self.output_rate
				* np.sum(self.exc_syns.rates) / self.exc_syns.n)
			n_vector = (self.exc_syns.weights > substraction_value).astype(int)
			# if np.sum(n_vector) != self.exc_syns.n:
			# 	print np.sum(n_vector)
			substractive_norm = (
				self.exc_syns.eta_dt * self.output_rate
				* np.dot(self.exc_syns.rates, n_vector) * n_vector
				/ np.sum(n_vector)
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

	def run(self, output=False):
		"""
		Let the rat move and learn

		Arguments:
		- 	position_output: if True, self.positions gets all the rat positions
			appended
		"""
		print 'Type of Normalization: ' + self.normalization
		rawdata = {}
		rawdata['exc_centers'] = self.exc_syns.centers
		rawdata['inh_centers'] = self.inh_syns.centers
		rawdata['exc_sigmas'] = self.exc_syns.sigmas
		rawdata['inh_sigmas'] = self.inh_syns.sigmas
		rawdata['positions'] = [[self.x, self.y]]
		rawdata['exc_weights'] = [self.exc_syns.weights.copy()]
		rawdata['inh_weights'] = [self.inh_syns.weights.copy()]
		rawdata['output_rates'] = [0.0]
		self.positions = [[self.x, self.y]]
		self.exc_weights = [self.exc_syns.weights.copy()]
		self.inh_weights = [self.inh_syns.weights.copy()]
		self.output_rates = [0.0]
		rawdata['time_steps'] = self.steps
		for step in self.steps:
			self.set_current_input_rates()
			self.set_current_output_rate()
			self.update_weights()
			# self.rectify_array(self.exc_syns.weights)
			utils.rectify_array(self.inh_syns.weights)
			self.normalize_exc_weights()
			if step % self.params['every_nth_step'] == 0 and output:
				# Store Positions
				# print 'step %f position %f outputrate %f' % (step, self.x, self.output_rate)
				rawdata['positions'].append([self.x, self.y])
				# Store weights
				rawdata['exc_weights'].append(self.exc_syns.weights.copy())
				rawdata['inh_weights'].append(self.inh_syns.weights.copy())
				# Store Rates
				rawdata['output_rates'].append(self.output_rate)			
			self.move_diffusively()
			self.reflective_BCs()


		# Convert the output into arrays
		for k in rawdata:
			rawdata[k] = np.array(rawdata[k])
		print 'Simulation finished'
		return rawdata

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
