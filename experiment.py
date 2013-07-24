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
	'n_exc': 4,
	'n_inh': 1,
	'sigma_exc': 0.05,
	'sigma_inh': 0.1,
	'init_weight_noise_exc': 0.05,  # Noise on weights (as relative value)
	'init_weight_noise_inh': 0.05,  # 0.05 corresponds to +- 5 percent
	'simulation_time': 1.0,
	'target_rate': 5.0,
	'diff_const': 1.0,
	'dt': 0.01,
	'eta_exc': 0.1,
	'eta_inh': 0.2,
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
rat.run(position_output=True, weight_output=True)


##################################
##########	Plotting	##########
##################################
# exc_centers = np.array([[0.6, 0.7], [0.8, 0.9]])
# exc_sigmas = np.array([[0.1, 0.2], [0.3, 0.5]])
# exc_weights = np.array([[1, 2], [3, 4], [5, 6], [0.1, 0.2], [0.03, 0.04], [0.05, 0.06]])
# # inh_centers = np.array([[0.1, 0.2], [0.3, 0.5]])
# # inh_sigmas = np.array([[0.6, 0.7], [0.8, 0.9]])
# inh_weights = np.array([[1, 2], [3, 4], [5, 6]])

# save_path = '/Users/simonweber/Desktop/ani.mp4'
# print len(rat.exc_weights)
weights_ani = plotting.positions_and_weigths_animation(
	params, rat.positions,
	rat.exc_syns.centers, rat.inh_syns.centers,
	rat.exc_syns.sigmas, rat.inh_syns.sigmas,
	rat.exc_weights, rat.inh_syns.weights)

# positions_ani = plotting.positions_animation(params, rat.positions)

# weights_ani.save(
# 	save_path,
# 	writer=animation.FFMpegFileWriter(),
# 	extra_anim=positions_ani)

print rat.positions
# plotting.positions_ArtistAnimation(params, rat.positions)
plotting.weights_ArtistAnimaton(params, rat.positions, rat.exc_syns.centers, rat.exc_syns.sigmas, rat.exc_weights)
#symmetric_centers = np.linspace(0, params['boxlength'], params['n_exc'])
#plotting.fields(params, symmetric_centers, rat.exc_syns.sigmas)
#plotting.fields(params, rat.exc_syns.centers, rat.exc_syns.sigmas)
plt.show()
#plotting.fields(params, inh_synapses)
