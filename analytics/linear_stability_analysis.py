import numpy as np
import matplotlib.pyplot as plt
##################################
##########	Notation	##########
##################################
# e: eta
# N: number of synapses
# L: boxlength
# s: sigma
# k: wavenumber

##################################
##########	The Kernels	##########
##################################
def K(e,N,L,s,k):
	f = 2*np.pi*e*(N/L**2)*np.power(s,2)*np.exp(-np.power(k*s, 2))
	return f

######################################
##########	The eigenvalues	##########
######################################
def eigenvalue(sign_exp, k, eI, sigma_inh, NI, eE, sE, NE, L, beta):
	"""The eigenvalues of the dynamical system
	
	Parameters
	----------
	sign_exp : int
		1 for lambdaMinus
		2 for lambdaPlus
	
	Returns
	-------
	function
	"""
		
	f = (
		0.5
		* (
			(K(eE, NE, L, sE, k) - K(eI,NI,L,sigma_inh,k) - beta)
			+ (-1)**sign_exp
			* np.sqrt(
				(K(eI,NI,L,sigma_inh,k) - K(eE, NE, L, sE, k + beta))**2
				- 4*beta*K(eI,NI,L,sigma_inh,k)
				)
			)
		)
	return f


def get_max_k(sign_exp, k, target_rate, w0E, eta_inh, sigma_inh, n_inh,
					eta_exc, sigma_exc, n_exc, boxlength):
	"""The k value which maximizes the eigenvalue

	The eigenvalue (lambdaPlus, i.e sign_exp=2) has a maximum which is
	obtained by this function. One (and only one) of the input parameters
	must be an ndarray (can be of shape (1), though).
	An array of k values is returned. One value for each input parameter set.
	
	Parameters
	----------
	sign_exp : int
		If 1: Lambda Minus
		If 2: Lambda Plus
	k : ndarray
		Array of k values for which the eigenvalue is taken. It is crucial
		that this array contains the maximum. This array should also have a
		good resolution, because this determines the precision of the maximum.
	target_rate, w0E, eta_inh, sigma_inh, n_inh, eta_exc,
		sigma_exc, n_exc, boxlength : int or float or ndarray
		One and only one of these values MUST BE an array.
	Returns
	-------
	maxk : ndarray
		`maxk` is an array of the same length as the one input parameter which
		is an array (there has to be one and only one),
		where the k value which maximizes lambda is given as a function of the
		varied input parameter.
	"""
	# Mean excitatory input rate as determined in the analysis
	uEbar = np.sqrt(2*np.pi*sigma_exc**2) / boxlength
	# Beta is defined as the factor obtained from the normalization
	beta = eta_exc*target_rate*uEbar/w0E
	# The number of k values (this is crucial for the precision of the maximum)
	n_k = len(k)
	# If all the inputs are single values:
	# n_values = 1
	# # Get the maximal value of lambda. We take nanmax, because there
	# # might be nan values because of imaginary values.
	# maxlambda = np.nanmax(eigenvalue(sign_exp, k, eta_inh, sigma_inh, n_inh,
	# 						eta_exc, sigma_exc, n_exc, boxlength, beta))
	# Check if one of the inputs is an array
	d = dict(locals())
	for key, v in d.items():
		if key != 'k' and isinstance(v, np.ndarray):
			n_values = len(v)
			# Add an axis to make it broadcastable
			d[key] = v[..., np.newaxis]
			# Get the maximal values of lambda
			maxlambda = np.nanmax(eigenvalue(d['sign_exp'], d['k'],
					d['eta_inh'], d['sigma_inh'],
			 		d['n_inh'], d['eta_exc'], d['sigma_exc'],
			 		d['n_exc'], d['boxlength'], beta), axis=1)
			# Tile it such that you can set it equal to the eigenvalue array
			maxlambda = np.repeat(maxlambda, n_k, axis=0).reshape(n_values, n_k)
	k = np.tile(k, n_values).reshape(n_values, n_k)
	# Get corresponding k value(s)
	maxk = k[eigenvalue(d['sign_exp'], d['k'],
					d['eta_inh'], d['sigma_inh'],
			 		d['n_inh'], d['eta_exc'], d['sigma_exc'],
			 		d['n_exc'], d['boxlength'], beta) == maxlambda]
	return maxk

##################################
##########	Plotting	##########
##################################
def plot_grid_spacing_vs_parameter(target_rate, w0E, eta_inh, sigma_inh, n_inh,
					eta_exc, sigma_exc, n_exc, boxlength):
	for k, v in locals().items():
		if isinstance(v, np.ndarray):
			x = v
			xlabel = k	
	k = np.linspace(0, 100, 10000)
	sign_exp = 2
	maxk = get_max_k(sign_exp, k, target_rate, w0E, eta_inh, sigma_inh, n_inh,
		eta_exc, sigma_exc, n_exc, boxlength)
	grid_spacing = 2 * np.pi / maxk
	plt.plot(x, grid_spacing)
	plt.xlabel(xlabel)


k = np.arange(0, 100, 0.01)
print get_max_k(2, k, 
			1.0, 2.0, 1e-3, np.array([0.1]), 400,
			1e-4, 0.03, 400, 4.0)

sigma_inh = np.linspace(0.05, 0.5, 200)
plot_grid_spacing_vs_parameter(1.0, 2.0, 1e-3, sigma_inh, 400,
			1e-4, 0.03, 400, 4.0)
plt.show()

