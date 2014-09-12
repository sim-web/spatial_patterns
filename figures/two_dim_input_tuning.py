import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os
from learning_grids import initialization
import itertools

os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

##############################################
##########	General Plotting Stuff	##########
##############################################
mpl.rc('font', size=18)
mpl.rc('legend', fontsize=18)
mpl.rcParams.update({'figure.autolayout': True})

# If you comment this out, then everything works, but in matplotlib fonts
# mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
# mpl.rc('text', usetex=True)

colors = {'exc': '#D7191C', 'inh': '#2C7BB6', 'diff': '0.4'}
legend = {'exc': 'Excitation', 'inh': 'Inhibition', 'diff': 'Difference'}
legend_short = {'exc': 'Exc.', 'inh': 'Inh.', 'diff': 'Difference'}
signs = {'exc': 1, 'inh': -1}
sigma = {'exc': 0.025, 'inh': 0.075}
marker = {'exc': '^', 'inh': 'o'}
populations = ['exc', 'inh']
scaling_factor = {'exc': 1.0, 'inh': 0.5}


fig = plt.figure(figsize=(5.0, 5.0))
def field(positions, location, sigma):
	"""Two dimensional asyemmtric Gaussian
	
	Parameters
	----------
	positions : ndarray of shape (n_x, n_y)
		Determines the resolution of the color map
	location : ndarray of shape (2)
		Location of the center of the asymmetric Gaussian
	sigma : ndarray of shape (2)
		Standard deviation in x and y direction

	Returns
	-------
	output : ndarray of shape (n_x, n_y)
	"""
	sigma = np.asarray(sigma)
	location = np.asarray(location)
	twoSigma2 = 1. / (2. * sigma**2)
	return np.exp(
			-np.power(
				location[0] - positions[...,0], 2)
			*twoSigma2[0]
			-np.power(
				location[1] - positions[...,1], 2)
			*twoSigma2[1])


if __name__ == '__main__':
	n_x = 2000
	# r = 5 * np.array([0.5, 0.5])
	r = np.array([0.5, 0.5])
	n = np.array([n_x, n_x])

	linspace = np.linspace(-r[0], r[0], n[0])
	X, Y = np.meshgrid(linspace, linspace)
	positions = initialization.get_equidistant_positions(r, n)
	location = np.array([0., 0.])
	distance = np.sqrt(X*X + Y*Y)
	##################################
	##########	Input type	##########
	##################################
	# Choose one
	# syn_type = 'exc'
	# syn_type = 'inh'
	syn_type = 'both' # Creates center surround field

	##################################
	##########	Cell Types	##########
	##################################
	# Choose a cell type by commenting out all others

	# Grid cell
	cell_type = 'grid_cell'
	sigma_inh = 1*np.array([0.12, 0.12])
	sigma_exc = 1*np.array([0.06, 0.06])

	# Band cell
	# cell_type = 'band_cell'
	# sigma_inh = 1*np.array([0.2, 0.04])
	# sigma_exc = 1*np.array([0.05, 0.07])

	# Place cell
	# Increase radius by factor (e.g. 5) to make it fit the area
	# cell_type = 'place_cell'
	# sigma_inh = 1*np.array([1.5, 1.5])
	# sigma_exc = 1*np.array([0.07, 0.07])

	# vanishing_value decides which values should be masked
	vanishing_value = 1e-1
	if syn_type == 'exc':
		fields = field(positions, location, sigma_exc).reshape((n_x, n_x))
		cm = mpl.cm.Reds
		my_masked_array = np.ma.masked_less(fields, vanishing_value)
	elif syn_type == 'inh':
		fields = field(positions, location, sigma_inh).reshape((n_x, n_x))
		cm = mpl.cm.Blues
		my_masked_array = np.ma.masked_less(fields, vanishing_value)
	elif syn_type == 'both':
		# We arbitrarily multiply the inhibitory field with 0.7 to get a mexican hat
		fields = (
				field(positions, location, sigma_exc).reshape((n_x, n_x))
				- 0.7*field(positions, location, sigma_inh).reshape((n_x, n_x))
				)
		cm = mpl.cm.RdBu_r
		if cell_type == 'place_cell':
			fields[distance>2.4] = 0.
		my_masked_array = np.ma.masked_inside(fields,
			 							-vanishing_value, vanishing_value)

	# color_norm = mpl.colors.Normalize(-1.0, 1.0)
	plt.contourf(X, Y, my_masked_array, 40, cmap=cm)
	ax = plt.gca()
	ax.set_aspect('equal')
	plt.xticks([])
	plt.yticks([])
	plt.axis('off')

	plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/2dim_input_tuning/' 
					+ cell_type + '_' + syn_type + '_conjunctive.pdf',
					bbox_inches='tight', pad_inches=0.001, transparent=True)
	plt.show()