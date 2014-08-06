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
def field(centers, position, sigma, radius):
	"""Two dimensional asyemmtric Gaussian
	
	Parameters
	----------
	centers :
	positions : 
	sigma : ndarray of shape (2)
		Standard deviation in x and y direction

	Returns
	-------
	output : ndarray of shape (n_x, n_y)
	"""
	twoSigma2 = 1. / (2. * sigma**2)
	# scaled_kappa = radius[-1] / (np.pi*sigma[-1])**2
	# pi_over_r = np.pi / radius[-1]
	return np.sum(np.exp(
			-np.power(
				position[...,0] - centers[...,0], 2)
			*twoSigma2[0]
			-np.power(
				position[...,1] - centers[...,1], 2)
			*twoSigma2[1]
			-np.power(
				position[...,2] - centers[...,2], 2)
			*twoSigma2[2]), axis=4
			)
			# *np.exp(
			# 		scaled_kappa
			# 		* np.cos(
			# 		pi_over_r*(position[...,2]
			# 		- centers[...,2])))
			# )



# Setting up the positions grid
spacing = 10
radius = 0.5
dimensions = 3
r = np.array([radius, radius, radius])

# Check
n_x, n_y, n_z = 6, 5, 4
n = np.array([n_x, n_y, n_z])

linspace = np.linspace(-r[0], r[0], spacing)
X, Y = np.meshgrid(linspace, linspace)

linspaces = [np.linspace(-radius, radius, spacing)
			for i in np.arange(dimensions)]
Xs = np.meshgrid(*linspaces, indexing='xy')
# positions_grid = np.dstack([x for x in Xs])
# positions_grid.shape = Xs[0].shape + (1, dimensions)
# positions_grid = positions_grid.reshape(spacing**3, 3)
positions_grid = initialization.get_equidistant_positions(r, np.array([spacing, spacing, spacing]))
print 'positions_grid.shape'
print positions_grid.shape

# Getting the center positions
centers = initialization.get_equidistant_positions(r, n)
# centers = centers.reshape(3)
centers = centers.reshape(centers.shape[0], 1, dimensions)
print 'centers.shape'
print centers.shape
position = np.array([0., 0., 0.])

# Setting the input widths
sigma_inh = np.array([0.4, 0.4, 0.4])

# Getting the fields
fields = field(centers, positions_grid.reshape(spacing, spacing, spacing, 1, 1, 3), sigma_inh, r)
print 'fields.shape'
print fields.shape

cm = mpl.cm.gnuplot_r

plt.contourf(X, Y, fields[:, :, 0, 78], 40, cmap=cm)
ax = plt.gca()
ax.set_aspect('equal')
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')

# # plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/3dim_input_tuning/' 
# # 				+ cell_type + '_' + syn_type + '.pdf',
# # 				bbox_inches='tight', pad_inches=0.001, transparent=True)
plt.show()