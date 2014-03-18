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


def eigenvalue(sign_exp, k, eI, sI, NI, eE, sE, NE, L, beta):
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
			(K(eE, NE, L, sE, k) - K(eI,NI,L,sI,k) - beta)
			+ (-1)**sign_exp
			* np.sqrt(
				(K(eI,NI,L,sI,k) - K(eE, NE, L, sE, k + beta))**2
				- 4*beta*K(eI,NI,L,sI,k)
				)
			)
		)
	return f

def beta(eE, rtarget, w0E, uEbar):
	return eE*rtarget*uEbar/w0E

eI = 5e-3
eE = 5e-4
sI = 0.1
sE = 0.03
L = 2.0
rtarget = 1.0
NI = 200
NE = 200
w0E = 2.0
uEbar = np.sqrt(2*np.pi*sE**2) / L
k = np.arange(0, 100, 0.01)
beta = beta(eE, rtarget, w0E, uEbar)
# te
fig = plt.figure()
############################################################
##########	Plotting the Kernels and the EVs	##########
############################################################
# fig.add_subplot(211)
# plt.plot(k, K(eE, NE, L, sE, k), lw=2, color='green')
# plt.plot(k, K(eI, NI, L, sI, k), lw=2, color='blue')

# fig.add_subplot(212)
# plt.plot(k, eigenvalue(1, k, eI, sI, NI, eE, sE, NE, L, beta), lw=2)
# plt.plot(k, eigenvalue(2, k, eI, sI, NI, eE, sE, NE, L, beta), lw=2, color='red')
# plt.ylim(-0.0002, 0.0004)

############################################################
##########	Plot wavelength 2 pi / k as function of Parameters	##########
############################################################
wavelength = []
sigma_inh = np.linspace(0.05, 0.1, 200)
for s in sigma_inh:
	# Get largest value of eigenvalue
	maxlambda = np.nanmax(eigenvalue(2, k, eI, s, NI, eE, sE, NE, L, beta))
	# Get corresponding k
	maxk = k[eigenvalue(2, k, eI, s, NI, eE, sE, NE, L, beta) == maxlambda]
	wavelength.append(2*np.pi/maxk)
fig.add_subplot(211)
plt.plot(sigma_inh, wavelength)
plt.xlabel('Inhibitory width')
plt.ylabel('Wavelength')

wavelength = []
sigma_exc = np.linspace(0.001, 0.05, 200)
for s in sigma_exc:
	# Get largest value of eigenvalue
	maxlambda = np.nanmax(eigenvalue(2, k, eI, sI, NI, eE, s, NE, L, beta))
	# Get corresponding k
	maxk = k[eigenvalue(2, k, eI, sI, NI, eE, s, NE, L, beta) == maxlambda]
	wavelength.append(2*np.pi/maxk)
fig.add_subplot(212)
plt.plot(sigma_exc, wavelength)
plt.xlabel('Excitatory width')
plt.ylabel('Wavelength')

plt.show()