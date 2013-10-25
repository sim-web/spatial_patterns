import numpy as np

def psp2params(psp):
	params = {}
	for k, v in psp.iteritems():
		params[k[1]] = v
	return params

def rectify(value):
	"""
	Rectification of Firing Rates
	"""
	if value < 0:
		value = 0.
	return value

def rectify_array(array):
	"""
	Rectification of array entries using fancy indexing
	"""
	array[array < 0] = 0.
	return array

# def get_noisy_array(value, noise, size):
# 	"""Returns array with uniformly distributed values in range [value-noise, value+noise]
	
# 	Parameters
# 	----------
	
# 	Returns
# 	-------
	
# 	"""
# 	retunr np.uniform(value-noise, value+noise, )
# 	return ((1 + noise * (2 * np.random.random_sample(size) - 1)) * value)
