import initialization
import plotting
import observables
import numpy as np
import matplotlib.pyplot as plt

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
	'init_weight_noise_exc': 0.05,  # Noise on weights (as relative value)
	'init_weight_noise_inh': 0.05,  # 0.05 corresponds to +- 5 percent
	'simulation_time': 1.0,
	'target_rate': 5.0,
	'diff_const': 1.0,
	'dt': 0.01,
	'eta_exc': 1.0,
	'eta_inh': 2.0,
	'normalization': 'quadratic_multiplicative'
}

params['init_weight_exc'] = 2. * params['target_rate'] / params['n_exc']
params['init_weight_inh'] = 0.5 * params['target_rate'] / params['n_inh']

params['initial_x'] = params['boxlength']/2.0
params['initial_y'] = params['boxlength']/2.0

######################################
##########	Initialize Rat	##########
######################################
rat = initialization.Rat(params)


##################################
#########	Move Rat	##########
##################################
rat.run(position_output=True)


##################################
##########	Plotting	##########
##################################
#plotting.positions_animation(params, rat.positions)
symmetric_centers = np.linspace(0, params['boxlength'], params['n_exc'])

plotting.fields(params, symmetric_centers, rat.exc_syns.sigmas)
#plotting.fields(params, rat.exc_syns.centers, rat.exc_syns.sigmas)
plt.show()
#plotting.fields(params, inh_synapses)
