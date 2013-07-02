import initialization
import learning
import plotting
import observables
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


##################################
##########	Parameters	##########
##################################
params = {
	'dimensions': 1,
	'boxtype': 'line',
	'boxlength': 1.0,
	'n_exc': 100,
	'n_inh': 25,
	'sigma_exc': 0.05,
	'sigma_inh': 0.1,
	'init_weight_exc': 0.1,
	'init_weight_inh': 0.1,
	'simulation_time': 2.0,
	'target_rate': 1.0,
	'diff_const': 1.0,
	'dt': 0.01,
	'eta_exc': 1.0,
	'eta_inh': 2.0,
	}
params['initial_x'] = params['boxlength']/2.0
params['initial_y'] = params['boxlength']/2.0
# Number of simulation steps
steps = np.arange(0, params['simulation_time']/params['dt'])
eta_exc = params['eta_exc']
eta_inh = params['eta_inh']
dt = params['dt']
# Define the product because you need it frequently
eta_exc_dt = eta_exc * dt
eta_inh_dt = eta_inh * dt
target_rate = params['target_rate']
n_exc = params['n_exc']


##########################################
##########	Initialize Synapses	##########
##########################################
exc_synapses=[]
inh_synapses=[]
for i in xrange(0, params['n_exc']):
	exc_synapses.append(initialization.Synapse(params, 'exc'))
for j in xrange(0, params['n_inh']):
	inh_synapses.append(initialization.Synapse(params, 'inh'))

#exc_synapses_array = np.array(exc_synapses)
#print exc_synapses_array.weight

# Get the attribute (or call the method) of all elements in a list
	# This is done with a lambda form
	# It returns a list, we convert this list into a numpy array
exc_weights = np.array(map(lambda x: x.weight, exc_synapses))
inh_weights = np.array(map(lambda x: x.weight, inh_synapses))

initial_weight_sum_exc = np.sum(exc_weights)
initial_squared_weight_sum = np.sum(np.square(exc_weights))
# Get the initial excitatory and inhibitory rates
exc_rates = np.array(map(lambda x: x.get_rate([params['initial_x'], params['initial_y']]), exc_synapses))
inh_rates = np.array(map(lambda x: x.get_rate([params['initial_x'], params['initial_y']]), inh_synapses))

# Get the initial output rate
output_rate = learning.get_output_rate(
				exc_weights, inh_weights, exc_rates, inh_rates)

######################################
##########	Initialize Rat	##########
######################################
rat = initialization.Rat(params)

##################################
##########	Move Rat	##########
##################################
# Set initial position in a proper array
positions = [[params['initial_x'], params['initial_y']]]
# for step in steps:
# 	old_weights = exc_weights
# 	exc_weights += exc_rates * output_rate * eta_exc_dt
# 	inh_weights += inh_rates * (output_rate - target_rate) * eta_inh_dt
# 	# Normalization of excitatory weights
# 	exc_weights = learning.normalize_weights(
# 		n_synapses=n_exc, weights=exc_weights, rates=exc_rates,
# 		eta_dt=eta_exc_dt, output_rate=output_rate,
# 		initial_weight_sum=initial_weight_sum_exc,
# 		initial_squared_weight_sum=initial_squared_weight_sum,
# 		normalization='linear_multiplicative')
# 	print "sum_diff_squared: "
# 	print observables.sum_difference_squared(old_weights, exc_weights)
# 	rat.move_diffusively()
# 	rat.reflective_BCs()
# 	position = [rat.x, 0.5]
# 	positions.append([rat.x, 0.5])
# 	exc_rates = np.array(map(lambda x: x.get_rate(position), exc_synapses))
# #	print exc_rates
# 	inh_rates = np.array(map(lambda x: x.get_rate(position), inh_synapses))
# 	output_rate = learning.get_output_rate(
# 					exc_weights, inh_weights, exc_rates, inh_rates)
	
# 	print "output_rate: "
# 	print output_rate

##################################
##########	Plotting	##########
##################################
# plotting.positions_animation(params, positions)
#plotting.fields(params, exc_synapses)
plotting.fields(params, inh_synapses)