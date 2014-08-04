import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from general_utils.plotting import color_cycle_blue3

mpl.rc('font', size=18)
mpl.rc('legend', fontsize=18)

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

def squareroot(k, eI, sigma_inh, NI, eE, sE, NE, L, beta):
	"""The square root in the eigenvalue

	Parameters
	----------

	Returns
	-------

	"""
	f = ((K(eI,NI,L,sigma_inh,k) - K(eE, NE, L, sE, k) + beta)**2
				- 4*beta*K(eI,NI,L,sigma_inh,k)
				)
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
				(K(eI,NI,L,sigma_inh,k) - K(eE, NE, L, sE, k) + beta)**2
				- 4*beta*K(eI,NI,L,sigma_inh,k)
				)
			)
		)
	return f


def get_max_k(sign_exp, k, target_rate, w0E, eta_inh, sigma_inh, n_inh,
					eta_exc, sigma_exc, n_exc, boxlength, parameter_name):
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
	# #
	# for key, v in d.items():
	# 	if key != 'k' and isinstance(v, np.ndarray):
	# 		n_values = len(v)
	# 		# Add an axis to make it broadcastable
	# 		d[key] = v[..., np.newaxis]
	n_values = len(d[parameter_name])
	d[parameter_name] = d[parameter_name][..., np.newaxis]
	# Get the maximal values of lambda
	maxlambda = np.nanmax(eigenvalue(d['sign_exp'], d['k'],
			d['eta_inh'], d['sigma_inh'],
	 		d['n_inh'], d['eta_exc'], d['sigma_exc'],
	 		d['n_exc'], d['boxlength'], d['beta']), axis=1)
	# Tile it such that you can set it equal to the eigenvalue array
	n_values = 3
	maxlambda = np.repeat(maxlambda, n_k, axis=0).reshape(n_values, n_k)
	k = np.tile(k, n_values).reshape(n_values, n_k)
	# Get corresponding k value(s)
	maxk = k[eigenvalue(d['sign_exp'], d['k'],
					d['eta_inh'], d['sigma_inh'],
			 		d['n_inh'], d['eta_exc'], d['sigma_exc'],
			 		d['n_exc'], d['boxlength'], d['beta']) == maxlambda]
	return maxk

##################################
##########	Plotting	##########
##################################
def plot_grid_spacing_vs_parameter(target_rate, w0E, eta_inh, sigma_inh, n_inh,
					eta_exc, sigma_exc, n_exc, boxlength, parameter_name):
	# for k, v in locals().items():
	# 	if isinstance(v, np.ndarray):
	# 		x = v
	# 		xlabel = k
	d = dict(locals())
	x = d[parameter_name]
	k = np.linspace(0, 100, 10000)
	sign_exp = 2
	maxk = get_max_k(sign_exp, k, target_rate, w0E, eta_inh, sigma_inh, n_inh,
		eta_exc, sigma_exc, n_exc, boxlength, parameter_name)
	grid_spacing = 2 * np.pi / maxk
	plt.plot(x, grid_spacing, lw=2, color='gray', label=r'Theory')
	plt.legend(bbox_to_anchor=(1, 0), loc='lower right')
	plt.xlabel(parameter_name)
	plt.ylabel(r'Grid spacing $a$')



if __name__ == '__main__':
	# Use TeX fonts
	# mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
	# mpl.rc('text', usetex=True)
	sigma_inh = np.linspace(0.05, 0.5, 200)
	sigma_exc = np.linspace(0.01, 0.05, 200)
	target_rate = np.linspace(0.5, 4., 500)
	# w0E = np.linspace(0.5, 200.0, 200)
	eta_inh = np.linspace(1e-1, 1e-5, 200)
	n_inh = np.linspace(100, 1000, 200)
	boxlength = np.linspace(1.0, 10.0, 200)
	# plot_grid_spacing_vs_parameter(1.0, 2.0, 1e-3, 0.1, 400,
	# 			1e-4, sigma_exc, 400, 4.0)
	sign_exp = 2
	w0E=2
	k = np.linspace(0, 100, 1000)
	eI = 1e-2
	eE = 1e-3
	# eE = np.linspace(1e-3, 4e-3, 200)
	sE = 0.03
	sI = 0.1
	N = 1e15
	L = np.power(N / 100., 0.5)
	NI = N
	NE = N
	target_rate=1.0
	uEbar = np.sqrt(2*np.pi*sE**2) / L
	# uEbar = 0.0
	beta = eE*target_rate*uEbar/w0E
	fig = plt.figure()
	# fig.add_subplot(211)
	fig.set_size_inches(4, 2.5)



	plt.ylim(-0.0002, 0.0004)
	plt.plot(k, eigenvalue(2, k, eI, sI, NI, eE, sE, NE, L, beta), lw=2,
				label=r'$\lambda_+$', color=color_cycle_blue3[0])
	plt.plot(k, eigenvalue(1, k, eI, sI, NI, eE, sE, NE, L, beta), lw=2,
				label=r'$\lambda_-$', color=color_cycle_blue3[2])
	plt.legend()
	ax = plt.gca()
	maxk = get_max_k(2, k, target_rate, w0E, eI, np.array([0.1]), NI,
					eE, sE, NE, L)

	# print maxk
	# ax.set_xticks([])
	ax.set_xticks(maxk)
	ax.set_xticklabels([r'$k_{\mathrm{max}}$'])
	ax.set_yticks([0])
	# plt.xlabel(r'Learning rate $\eta^\mathrm{E}$', fontsize=16)
	plt.xlabel(r'Wavevector $k$', fontsize=18)
	plt.ylabel(r'Eigenvalue', fontsize=18)
	y0, y1 = ax.get_ylim()
	# plt.ylim((y0, y1))
	plt.axvline(maxk, color='black',
				linestyle='dotted', lw=1)
	plt.axhline(0, color='black')
	# plt.title(r'$\sigma_{\mathrm{E}} \approx \sigma_{\mathrm{I}}, k=2$')
	# plt.title(r'$\sigma_{\mathrm{E}} < \sigma_{\mathrm{I}}$')

	# fig.add_subplot(212)
	# print np.amin(squareroot(k, eI, 0.1, NI, eE, sE, NE, L, beta))
	# plt.plot(k, squareroot(k, eI, 0.1, NI, eE, sE, NE, L, beta))
	# plt.ylim(-0.0000002, 0.0000002)
	# plt.show()
	plt.savefig('eigenvalues_new.pdf', bbox_inches='tight', pad_inches=0.01)
	plt.show()


