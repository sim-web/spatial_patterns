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


fig = plt.figure(figsize=(2.5, 2.5))

def field(positions, location, sigma):
	twoSigma2 = 1. / (2. * sigma**2)
	return np.exp(
			-np.power(
				location[0] - positions[...,0], 2)
			*twoSigma2[0]
			-np.power(
				location[1] - positions[...,1], 2)
			*twoSigma2[1])


r = np.array([0.5, 0.5])
n = np.array([70, 70])

linspace = np.linspace(-r[0], r[0], n[0])
X, Y = np.meshgrid(linspace, linspace)
positions = initialization.get_equidistant_positions(r, n)

# Place cell
# sigma = np.array([0.1, 0.1])
# location = np.array([-0.15, -0.25])
# field = field(positions, location, sigma).reshape((70, 70))

# Invariant cell
sigma = np.array([1e5, 1e5])
location = np.array([-0.15, -0.25])
field = field(positions, location, sigma).reshape((70, 70))

# Band cell
# sigma = np.array([0.05, 100.])
# location1 = np.array([-0.35, 0])
# location2 = np.array([0.0, 0])
# location3 = np.array([+0.35, 0])
# field = (field(positions, location1, sigma).reshape((70, 70))
# 			+ field(positions, location2, sigma).reshape((70, 70))
# 			+ field(positions, location3, sigma).reshape((70, 70)))


# Grid cell
# # For some reason you need to compile often to get it

# # Hexagonal coordinates
# hexcoords = np.array([[0,0,0],
# 			[1,-1,0], [0,-1,1], [-1,0,1], [-1,1,0], [0,1,-1], [1,0,-1],
# 			[2, -2, 0], [1,-2,1], [0,-2,2], [-1, -1, 2], [-2, 0, 2], [-2,1,1], [-2,2,0], [-1,2,-1], [0,2,-2], [1,1,-2], [2,0,-2], [2,-1,-1]])
# xs = []
# ys = []

# # Spacing (more or less)
# s = 0.2
# # From hexagonal coords to cartesian coords
# for h in hexcoords:
# 	r = h[0]
# 	g = h[1]
# 	b = h[2]
# 	x = np.sqrt(3.) * s * ( b/2. + r)
# 	y = 3./2 * s * b
# 	xs.append(x)
# 	ys.append(y)	

# sigma = np.array([0.05, 0.05])
# fields = np.empty((70, 70))
# for x, y in zip(xs, ys):
# 	print x
# 	print y
# 	fields += field(positions, np.array([x, y]), sigma).reshape((70, 70))
# field = fields

# End grid cell


# plt.contourf(X, Y, field, 30)
# print np.amax(field)
V = np.linspace(0, 2, 40)
plt.contourf(X, Y, field, V)
# cbar = plt.colorbar()
# cbar.set_ticks([])
ax = plt.gca()
ax.set_aspect('equal')
plt.xticks([])
plt.yticks([])
plt.axis('off')

# plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/2dim_cell_types/test'
# 			'.pdf',
# 	bbox_inches='tight', pad_inches=0.001)
plt.show()