import numpy as np

def normalize_weights(
	n_synapses, weights, rates, eta_dt, output_rate, initial_weight_sum,
	initial_squared_weight_sum, normalization):
	"""
	Synaptic Normalization for the Excitatory Weights

	Returned are the normalized weights!
	"""
	if normalization == 'linear_substractive':
		_norm = eta_exc_dt * output_rate * np.sum(exc_rates) / n_exc
		return exc_weights - _norm
	if normalization == 'linear_multiplicative':
		weight_sum = np.sum(weights)
		return (initial_weight_sum / weight_sum) * weights
	if normalization == 'quadratic_multiplicative':
		weight_sum = np.sum(np.square(weights))
		return np.sqrt((initial_squared_weight_sum / weight_sum)) * weights


def get_exc_weights_update(eta_exc, exc_rates, output_rate, dt):
	return eta_exc*exc_rates*output_rate*dt

def rectification(value):
	if value < 0:
		value = 0
	return value

def get_output_rate(exc_weights, inh_weights, exc_rates, inh_rates):
	rate = np.dot(exc_weights, exc_rates) - np.dot(inh_weights, inh_rates)
	return rectification(rate)
