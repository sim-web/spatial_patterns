import initialization
import plotting
import observables
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


##################################
##########	Parameters	##########
##################################
params = {
	'dimensions': 1,
	'boxtype': 'line',
	'boxlength': 1.0,
	'n_exc': 1,
	'n_inh': 1,
	'sigma_exc': 0.2,
	'sigma_inh': 0.4,
	'init_weight_noise_exc': 0.05,  # Noise on weights (as relative value)
	'init_weight_noise_inh': 0.05,  # 0.05 corresponds to +- 5 percent
	'simulation_time': 5.0,
	'target_rate': 5.0,
	'diff_const': 0.1,
	'dt': 0.01,
	'eta_exc': 0.1,
	'eta_inh': 0.2,
	'normalization': 'quadratic_multiplicative'
}

params['init_weight_exc'] = 2. * params['target_rate'] / params['n_exc']
params['init_weight_inh'] = 0.5 * params['target_rate'] / params['n_inh']

params['initial_x'] = params['boxlength']/2.0
params['initial_y'] = 0.0

######################################
##########	Initialize Rat	##########
######################################
rat = initialization.Rat(params)

##################################
#########	Move Rat	##########
##################################
rat.run(output=True)

##################################
##########	Plotting	##########
##################################
# save_path = '/Users/simonweber/Desktop/ani.mp4'
animation = plotting.Animation(rat)

animation.animate_all_synapses(interval=50)
# plotting.fields(params, rat.exc_syns.centers, rat.exc_syns.sigmas)
plt.show()
