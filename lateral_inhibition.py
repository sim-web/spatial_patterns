from pylab import *
# import numpy as np
# import matplotlib.pyplot as plt

output_neurons = 2
input_neurons = 1
# initial firing rates
r = rand(output_neurons)

# input currents
Iexc, Iinh = 10., 10.
target_rate = 1.

# inhibitory weights
winh = zeros((output_neurons, input_neurons))

# excitatory weights
# wexc = zeros(2)

dt, tau, T, eta = 0.1, 10., 1000, 1e-2

# The recurrent weight matrix
rec_strength = 0.01
wrec = rec_strength * ones((2,2))
wrec[0,0] = 0
wrec[1,1] = 0

dt_tau = dt/tau

def update_rates(r, I):
	r = r * (1-dt_tau) + dt_tau * (I - dot(wrec,r)) 
	r[r<0] = 0
	return r
	
	
def update_weights(winh, r):
	winh += eta * Iinh * (r - target_rate)
	winh[winh<0] = 0
	return winh

log = []
for t in range(int(T/dt)):
	r = update_rates(r, Iexc - winh * Iinh)
	winh = update_weights(winh, r)
	# Plasticity on lateral inhibition
	# wrec += eta * outer(r - target_rate, r)
	wrec[wrec<0] = 0
	wrec[0,0] = 0
	wrec[1,1] = 0
	log.append(r)
	
	
print winh
print wrec

log = array(log)
plot(log)
show()





