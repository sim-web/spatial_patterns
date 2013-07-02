import numpy as np
from scipy.stats import norm

class PlaceCells:
	"""
	bla
	"""
	def __init__(self, params):
		self.params = params
		self.boxtype = params['boxtype']
		self.boxlength = params['boxlength']
		self.dimensions = params['dimensions']
		self.n_exc = params['n_exc']
		self.n_inh = params['n_inh']

		if self.dimensions == 1:
			# Create sorted arrays with random numbers between 0 and boxlength
			self.exc_centers = np.sort(np.random.random_sample(self.n_exc)*self.boxlength)
			self.inh_centers = np.sort(np.random.random_sample(self.n_inh)*self.boxlength)

	def get_centers(self):
		if self.dimensions == 1:
			# Create sorted arrays with random numbers between 0 and boxlength
			exc_centers = np.sort(np.random.random_sample(self.n_exc)*self.boxlength)
			inh_centers = np.sort(np.random.random_sample(self.n_inh)*self.boxlength)
			return exc_centers, inh_centers

	# def get_functions(self):
	# 	exc_centers, inh_centers = self.get_centers()
	# 	p

	def print_params(self):
		print self.params


class Synapse:
	"""
	bla
	"""
	def __init__(self, params, type):
		self.params = params
		self.boxlength = params['boxlength']		
		self.dimensions = params['dimensions']
		self.type = type
		self.center = np.random.random_sample()*self.boxlength
		### Exitatory Synapses ###
		if self.type == 'exc':
			self.weight = params['init_weight_exc']
			if self.dimensions == 1:
				self.field = norm(loc=self.center, scale=params['sigma_exc']).pdf	
		### Inhibitory Synapses ###
		if self.type == 'inh':
			self.weight = params['init_weight_inh']
			if self.dimensions == 1:
				self.field = norm(loc=self.center, scale=params['sigma_inh']).pdf

	def get_rate(self, position):
		"""
		Get the firing rate of the synapse at the current position.

		- position: [x, y]
		"""
		if self.dimensions == 1:
			return self.field(position[0])


class Rat:
	"""
	bla
	"""
	def __init__(self, params):
		self.params = params
		self.boxlength = params['boxlength']
		self.x = params['initial_x']
		self.y = params['initial_y']
		self.diff_const = params['diff_const']
		self.dt = params['dt']
		self.dspace = np.sqrt(2.0*self.diff_const*self.dt)

		# Set the gaussion of diffusive steps
		# Note: You can either take a normal distribution and in each time step multiply
		# it with sqrt(2*D*dt) or directly define a particular Gaussian
		# self.diffusion_gauss = norm(scale=np.sqrt(2.0*self.diff_const*self.dt)).pdf


	def move_diffusively(self):
		self.x += self.dspace*np.random.randn()

	def reflective_BCs(self):
		if self.x < 0:
			self.x -= 2.0*self.x
		if self.x > self.boxlength:
			self.x -= 2.0*(self.x - self.boxlength)
