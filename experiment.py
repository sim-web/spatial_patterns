import initialization
import plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

params = {
	'dimensions': 1,
	'boxtype': 'line',
	'boxlength': 1.0,
	'n_exc': 10,
	'n_inh': 2,
	'sigma_exc': 0.1,
	'sigma_inh': 0.1,
	'init_weight_exc': 1.0,
	'init_weight_inh': 1.0,
	# 'initial_x': boxlength/2.0,
	# 'initial_y': boxlength/2.0,
	'diff_const': 1.0,
	'dt': 0.01,
	}

params['initial_x'] = params['boxlength']/2.0
params['initial_y'] = params['boxlength']/2.0

# place_cells = initialization.PlaceCells(params)


# plotting.centers(params, place_cells.exc_centers)

exc_synapses=[]
inh_synapses=[]
for i in xrange(0, params['n_exc']):
	exc_synapses.append(initialization.Synapse(params, 'exc'))
for j in xrange(0, params['n_inh']):
	inh_synapses.append(initialization.Synapse(params, 'inh'))

rat = initialization.Rat(params)
print rat.x
rat.move_diffusively()
print rat.x

x = np.linspace(-1, params['boxlength'], 200)
plt.plot(x, rat.diffusion_gauss(x))
plt.plot(x, rat.dspace*rat.diffusion_gauss_test(x))

# plotting.fields(params, exc_synapses)
plt.show()
