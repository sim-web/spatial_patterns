import numpy as np

##############################################
##########	Measures for Learning	##########
##############################################
def sum_difference_squared(old_weights, new_weights):
	"""
	Sum of squared differences between old and new weights

	- old_weights, new_weights are numpy arrays
	"""
	return np.sum(np.square(new_weights - old_weights))
