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

		self.type = type
		self.center = np.random.random_sample()*self.boxlength
		if self.type == 'exc':
			self.weight = params['init_weight_exc']
		if self.type == 'inh':
			self.weight = params['init_weight_inh']
		self.field = norm(loc=self.center, scale=params['sigma_exc']).pdf
