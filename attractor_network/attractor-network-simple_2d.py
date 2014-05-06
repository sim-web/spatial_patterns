from pylab import *
import numpy as np

# Simple simulation from Henning
# Several models can be tested:
	# Couey Model: only inhibitory connections
		# Take negative W_EE (let them be inhibitory).
		# Comment out any other weights
		# Don't use normalization
		# Effect of reducing the weights: grid spacing goes down
		# Use input I = 0.1
	# Classical Head Direction cell in attractor model (see McNaughton 2006):
		# Take positive W_EE, Comment out any other weights
		# Take substractive normalization
		# Use input I = 0.0
	# Mexican Hat:
		# Don't use normalization
		# Use all the inibitory weights and change the equation accordingly
		# Use input I = 0.1

N = 128**2
length = np.sqrt(N)

positions_grid = np.empty((length, length, 2))
x_space = np.arange(length)
y_space = np.arange(length)
X, Y = np.meshgrid(x_space, y_space)
for y in y_space:
	for x in x_space:
		positions_grid[x][y] =  [x, y]

position_of_neuron = positions_grid.reshape(N, 2)
def d(i, j):
	d = position_of_neuron[i] - position_of_neuron[j]
	# if np.any(d>length/2.):
	d[d>length/2.] = length - d[d>length/2.]
	# if np.any(d<-length/2.):
	d[d<-length/2.] = d[d<-length/2.] + length
	distance = np.sqrt(np.sum(np.square(d)))
	return distance

# d = np.array([3,4])
# d[d>3] = N - d[d>3]
# print d

# dist = []
# for i in np.arange(N):
# 	dist.append(d(10, i))
# plot(dist)
# dist = []
# for i in np.arange(N):
# 	dist.append(d(4, i))
# plot(dist)
# show()

seed(3)

T = 1000.0
dt = 0.5
NT = int(T/dt)
tau = 10.0
Ntau = int(tau/dt)

I = 1.0

factor_GABA = 1.0

W_EE = zeros((N,N))
W_EI = zeros((N,N))
W_IE = zeros((N,N))
W_II = zeros((N,N))

# sigma_exc = 7.
# sigma_inh = 1.5 * sigma_exc
# sigma_EE = sigma_exc
# sigma_II = sigma_inh
# sigma_EI = sigma_inh
# sigma_IE = sigma_exc
beta = 3./(13**2)
gamma = 1.05 * beta

for i in range(N):
	print i
	for j in range(N):
		W_EE[i,j] = 1. * exp(-gamma * d(i,j)**2)
		W_EI[i,j] = 1. * exp(- beta * d(i,j)**2)
		W_IE[i,j] = 1. * exp(-gamma * d(i,j)**2)
		W_II[i,j] = 1. * exp(- beta * d(i,j)**2)
	W_EE[i,i] = 0
	W_EI[i,i] = 0
	W_IE[i,i] = 0
	W_II[i,i] = 0
	
r_E = random(N)/100.0
# r_E[N/2] = 1.0
r_I = random(N)/100.0

rates = []
activity = sum(r_E)
print activity

for t in range(0,NT):
	r_E += dt/tau * (- r_E + dot(W_EE,r_E) - factor_GABA*dot(W_EI, r_I) + I)
	
	# NORMALIZATION
	# current_activity = sum(r_E)
	# Multiplicative normalization
	# r_E *= activity/current_activity
	# Substractive Multiplication
	# r_E -= (current_activity-activity)/N

	r_E[r_E<0] = 0
	# r_E[r_E>1] = 1
	rates.append(r_E.copy())

	r_I += dt/tau * (- r_I + dot(W_IE,r_E) - factor_GABA*dot(W_II, r_I) + I)
	r_I[r_I<0] = 0
	#r_I[r_I>1] = 1

figure()
subplot(211)
np.save('r_E', r_E)
np.save('rates', rates)
# contourf(r_E.reshape(length, length))
# ax = gca()
# ax.set_aspect('equal')

# subplot(212)
# plot(array(rates)[0:NT,:])

# show()
		

