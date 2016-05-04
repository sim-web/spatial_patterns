import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from general_utils.plotting import color_cycle_blue3
from copy import deepcopy

mpl.rc('font', size=12)
mpl.rc('legend', fontsize=12)

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
def kernel(k, params, type):
	"""The Kernel (either EE or II)"""
	number = np.prod(params[type]['number_per_dimension'])
	return (
		2 * np.pi * params[type]['eta']
		* (number / (2*params['sim']['radius'])**2)
		*np.power(params[type]['sigma'],2)
		*np.exp(-np.power(k*params[type]['sigma'], 2))
	)

# def squareroot(k, eI, sI, NI, eE, sE, NE, L, beta):
# 	"""The square root in the eigenvalue"""
# 	f = ((K(eE, NE, L, sE, k) - K(eI, NI, L, sI, k) - beta)**2
# 				- 4*beta*K(eI, NI, L, sI, k))
# 	return f

######################################
##########	The eigenvalues	##########
######################################
def eigenvalue(sign_exp, k, params):
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
	term = (params['exc']['eta'] * params['out']['target_rate']
			* np.sqrt(2 * np.pi * params['exc']['sigma']**2)
			/ (params['exc']['init_weight'] * 2 * params['sim']['radius'])
	)
	f = (
		0.5
		* (
			(kernel(k, params, 'exc') - kernel(k, params, 'inh') - term)
			+ (-1)**sign_exp
			* np.sqrt(
				(kernel(k, params, 'exc') - kernel(k, params, 'inh') - term)**2
				- 4 * term * kernel(k, params, 'inh')
				)
			)
		)
	return f


def lambda_p_high_density_limit(k, params):
	"""
	The eigenvalue with a '+' before the squareroot

	Note: In the high density limit the eigenvalue with a '-' before
			the squareroot is always zero.

	Parameters
	----------
	k : ndarray
		Wavevector

	Returns
	-------
	ret : ndarray
		The eigenvalue as a function of the wavevector, i.e. the spectrum.
	"""
	ret = (
		(2. * np.pi / (2*params['sim']['radius'])**2)
		* (
			params['exc']['eta'] * params['exc']['sigma']**2
				* np.prod(params['exc']['number_per_dimension'])
				* np.exp(-k**2 * params['exc']['sigma']**2)
			-
			params['inh']['eta'] * params['inh']['sigma']**2
				* np.prod(params['inh']['number_per_dimension'])
				* np.exp(-k**2 * params['inh']['sigma']**2)
		  )
	)
	return ret

def get_gamma(prms):
	if prms['sim']['gaussian_process']:
		### Check if global rescaling factor is defined ###
		gamma = {}
		for p in ['exc', 'inh']:
			try:
				gamma[p] = (prms[p]['gp_stretch_factor']
							/ (prms[p]['gp_extremum'][1] - prms[p]['gp_extremum'][0]))
			except:
				print 'COULD NOT FIND the gamma'
				gamma[p] = 1.0
	else:
		for p in ['exc', 'inh']:
			gamma[p] = 1.0

	return gamma

def grid_spacing_high_density_limit(params, varied_parameter=None,
									parameter_range=None,
									sigma_corr=False):

	prms = deepcopy(params)
	if varied_parameter is not None:
		prms[varied_parameter[0]][varied_parameter[1]] = parameter_range
		if varied_parameter[1] == 'number_per_dimension':
			prms[varied_parameter[0]][varied_parameter[1]] = parameter_range.reshape(parameter_range.shape[0], 1)

	# if sigma_corr:
	# 	factor=np.sqrt(2)
	# else:

	gamma = get_gamma(prms)

	factor=1.0
	ret = (
		2. * np.pi * np.sqrt(
			((prms['inh']['sigma']/factor)**2 - prms['exc']['sigma']**2)
			/
			np.log(
				(gamma['inh']**2 * prms['inh']['eta'] * prms['inh']['sigma']**4
					* np.atleast_2d(prms['inh']['number_per_dimension'])[:, 0]
					* prms['inh']['gaussian_height']**2
				 	# * prms['inh']['sigma']**2
				)
				/
				(gamma['exc']**2 * prms['exc']['eta'] * prms['exc']['sigma']**4
					* np.atleast_2d(prms['exc']['number_per_dimension'])[:, 0]
					* prms['exc']['gaussian_height']**2
				 	# * prms['exc']['sigma']**2
				)
			)
		)
	)
	return ret

def get_max_k(sign_exp, k, params, varied_parameter=None, parameter_range=None):
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
	varied_parameter : tuple
		Specifies which parameter is the one that is varied,
		 e.g. ('inh', 'sigma)
	parameter_range : ndarray
		Array that specifies the range of the varied parameter
	Returns
	-------
	maxk : ndarray
		`maxk` is an array of the same length as `parameter_range`,
		and contains the wavevector that maximizes the eigenvalue for each
		parameter set.
	"""
	if varied_parameter is not None:
		params[varied_parameter[0]][varied_parameter[1]] = parameter_range
	# The number of k values (this is crucial for the precision of the maximum)
	n_k = len(k)
	n_values = len(parameter_range)
	params[varied_parameter[0]][varied_parameter[1]] = (
		params[varied_parameter[0]][varied_parameter[1]][..., np.newaxis])
	# Get the maximal values of lambda
	maxlambda = np.nanmax(eigenvalue(sign_exp, k, params), axis=1)
	# Tile it such that you can set it equal to the eigenvalue array
	maxlambda = np.repeat(maxlambda, n_k, axis=0).reshape(n_values, n_k)
	k = np.tile(k, n_values).reshape(n_values, n_k)
	# Get corresponding k value(s)
	maxk = k[eigenvalue(sign_exp, k, params) == maxlambda]
	return maxk


def get_grid_spacing(params, varied_parameter, parameter_range):
	params[varied_parameter[0]][varied_parameter[1]] = parameter_range
	k = np.linspace(0, 100, 10000)
	sign_exp = 2
	maxk = get_max_k(sign_exp, k, params, varied_parameter, parameter_range)
	grid_spacing = 2 * np.pi / maxk
	return grid_spacing

# def plot_eigenvalues(params):
# 	fig = plt.figure()
# 	fig.set_size_inches(4, 2.5)
# 	k = np.linspace(0, 100, 1000)
# 	plt.plot(k, np.zeros_like(k),
# 			 color='#d8b365', label=r'$\lambda_-$')
# 	plt.plot(k, lambda_p_high_density_limit(k, params),
# 			 color='#5ab4ac', label=r'$\lambda_-$')
# 	plt.ylim(-1e-6, 2.5e-6)
# 	plt.legend()


radius = 7.0
params = {
		'sim': {
			'radius': radius,
		},
		'out': {
			'target_rate': 1.0,
		},
		'exc': {
			'eta': 1e-3 / (2*radius),
			'sigma': 0.03,
			'number_per_dimension': np.array([2000]),
			'init_weight': 1.0,
		},
		'inh': {
			'eta': 1e-2 / (2*radius),
			'sigma': 0.1,
			'number_per_dimension': np.array([500]),
			# 'number_per_dimension': np.array([2000]),
		},
	}



if __name__ == '__main__':
	# Use TeX fonts
	# mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
	# mpl.rc('text', usetex=True)
	sigma_inh = np.linspace(0.08, 0.36, 201)
	sigma_exc = np.linspace(0.01, 0.05, 200)
	eta_inh = np.linspace(1e-1, 1e-5, 200)
	n_inh = np.linspace(100, 1000, 200)
	n_exc = np.linspace(100, 1000, 200)
	k = np.linspace(0, 100, 1000)

	fig = plt.figure()
	fig.set_size_inches(4, 2.5)

	# gs = 2 * np.pi / get_max_k(2, k, params, varied_parameter=('inh', 'sigma'),
	# 				parameter_range=sigma_inh)
	# plt.plot(sigma_inh, gs, color='red')

	# plt.plot(k, eigenvalue(2, k, params), lw=2,
	# 			label=r'$\lambda_+$', color=color_cycle_blue3[0])
	# plt.plot(k, eigenvalue(1, k, params), lw=2,
	# 			label=r'$\lambda_-$', color=color_cycle_blue3[2])

	# grid_spacing = grid_spacing_high_density_limit(params,
	# 									  varied_parameter=('inh', 'sigma'),
	# 									  parameter_range=sigma_inh)
	#
	# plt.plot(sigma_inh, grid_spacing, color='red', label='4/1')
	#
	# params['inh']['number_per_dimension'] = np.array([2000])
	# grid_spacing = grid_spacing_high_density_limit(params,
	# 									  varied_parameter=('inh', 'sigma'),
	# 									  parameter_range=sigma_inh)
	#
	# plt.plot(sigma_inh, grid_spacing, color='blue', label='1/1')


	# plt.legend()
	# print sigma_inh
	# print grid_spacing
	# plt.plot(sigma_inh, grid_spacing)
	# ax = plt.gca()
	# plt.margins(0.001)
	# y0, y1 = ax.get_ylim()
	# plt.ylim(-0.0002, y1)

	# maxk = get_max_k(2, k, target_rate, w0E, eI, np.array([0.1]), NI,
	# 				eE, sE, NE, L)
	# ax.set_xticks(maxk)
	# ax.set_xticklabels([r'$k_{\mathrm{max}}$'])
	# ax.set_yticks([0])
	# plt.xlabel(r'Wavevector $k$', fontsize=18)
	# plt.ylabel(r'Eigenvalue', fontsize=18)
	# y0, y1 = ax.get_ylim()
	# plt.ylim((y0, y1))
	# plt.axvline(maxk, color='black',
	# 			linestyle='dotted', lw=1)
	# plt.axhline(0, color='black')

	# plot_eigenvalues(params)
	print (
			params['inh']['eta'] * params['inh']['sigma']**4
			* params['inh']['number_per_dimension']
			/
			(params['exc']['eta'] * params['exc']['sigma']**4
			* params['exc']['number_per_dimension'])
	)
	gs = grid_spacing_high_density_limit(params,
									varied_parameter=('exc', 'number_per_dimension'),
									parameter_range=np.array([5000]))

	plt.plot([5000], gs, marker='o')
	plt.show()


