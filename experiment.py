import matplotlib as mpl
mpl.use('TkAgg')
import initialization
import plotting
import animating
import observables
import numpy as np
import matplotlib.pyplot as plt
import math
# import functools

##################################
##########	Parameters	##########
##################################
params = {
	'dimensions': 1,
	'boxtype': 'line',
	'boxlength': 1.0,
	'n_exc': 30,
	'n_inh': 30,
	'sigma_exc': 0.03,
	'sigma_inh': 0.1,
	'init_weight_noise_exc': 0.05,  # Noise on weights (as relative value)
	'init_weight_noise_inh': 0.05,  # 0.05 corresponds to +- 5 percent
	'simulation_time': 100.0,
	'target_rate': 5.0,
	'diff_const': 1.0,
	'dt': 0.01,
	'eta_exc': 0.0001,
	'eta_inh': 0.2,
	'normalization': 'quadratic_multiplicative',
	# 'normalization': 'linear_multiplicative',
	# 'normalization': 'quadratic_multiplicative',
	'seed': 1
}

params['init_weight_exc'] = 20. * params['target_rate'] / params['n_exc']
params['init_weight_inh'] = 5.0 * params['target_rate'] / params['n_inh']

params['initial_x'] = params['boxlength']/2.0
params['initial_y'] = params['boxlength']/2.0

np.random.seed(params['seed'])
######################################
##########	Initialize Rat	##########
######################################
rat = initialization.Rat(params)

##################################
#########	Move Rat	##########
##################################
rawdata = rat.run(output=True)
# print rawdata['output_rates']
# print rawdata
##################################
##########	Plotting	##########
##################################
plot = plotting.Plot(params, rawdata)
fig = plt.figure()
plot_list = [
	lambda: [plot.fields_times_weights(syn_type='exc'), 
				plot.weights_vs_centers(syn_type='exc')],
	lambda: [plot.fields_times_weights(syn_type='inh'), 
				plot.weights_vs_centers(syn_type='inh')],
	lambda: plot.weights_vs_centers(syn_type='exc'),	
	lambda: plot.weights_vs_centers(syn_type='inh'),
	lambda: plot.weight_evolution(syn_type='exc'),
	lambda: plot.weight_evolution(syn_type='inh'),
	lambda: plot.output_rate_distribution(n_last_steps=10000),	
]
plotting.plot_list(fig, plot_list)

##################################
##########	Animating	##########
##################################
# save_path = '/Users/simonweber/Desktop/ani.mp4'
# animation = animating.Animation(rat, rawdata, start_time=0)
# animation.animate_all_synapses(interval=200)

print 'Plotting finished'
plt.show()
