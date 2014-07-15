import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_equidistant_positions(r, n, boxtype='linear', distortion=0.):
	"""Returns equidistant, symmetrically distributed coordinates
	
	The coordinates are taken such that they don't lie on the boundaries
	of the environment but instead half a lattice constant away on each
	side.
	Note: In the case of circular boxtype positions outside the cirlce
		are thrown away.

	Parameters
	----------
	r : ndarray
		Dimensions of the box
	n : ndarray
		Array of same shape as r, number of positions along each direction
	boxtype : string
		'linear': A quadratic arrangement of positions is returned
		'circular': A ciruclar arrangement instead 
	distortion : float or ndarray
		Maximal length by which each lattice coordinate (x and y separately)
		is shifted randomly (uniformly)
	Returns
	-------
	(ndarray) of shape (m, len(n)), where m < np.prod(n) for boxtype
	'circular', because points at the edges are thrown away.
	"""
	dimensions = len(n)
	d = 2*r/(2*n)
	spaces = [np.linspace(-ra+da, ra-da, na) for (ra, na, da) in zip(r, n, d)]
	Xs = np.meshgrid(*spaces)
	if boxtype == 'circular':
		distance = np.sqrt(np.sum([x**2 for x in Xs], axis=0))
		for x in Xs:
			x[distance>r[0]] = np.nan
	positions = np.array(zip(*[x.flat for x in Xs]))
	positions = positions.flatten()
	isnan = np.isnan(positions)
	np.delete(positions, np.nonzero(isnan))
	positions = positions.reshape(positions.size/dimensions, dimensions)
	distortion_array = 2*distortion * np.random.random_sample(positions.shape) - distortion
	return positions + distortion_array


n_x = 20
n_y = 10
n_z = 6

n = np.array([n_x, n_y, n_z])
r = np.array([0.5, 0.5, 0.5])
# r = np.array([0.5, 0.5])
boxtype = 'linear' 
distortion = r/n
# distortion = 0.0
centers = get_equidistant_positions(r=r, n=n, boxtype=boxtype, distortion=distortion)
fig = plt.figure()

if len(n) == 2:
	ax = fig.add_subplot(111)
	ax.scatter(centers[...,0], centers[...,1], marker='x')
elif len(n) == 3:
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(centers[...,0], centers[...,1], centers[...,2], marker='x')

ax.add_artist(plt.Circle((0,0), r[0], fc='none'))
ax.add_artist(plt.Rectangle((-r[0],-r[0]), 2*r[0], 2*r[0], fc='none'))
ax.set_aspect('equal')
plt.show()

# def plot_polar(radius, spacing, a):
# 	def __str__():
# 		return 'bla'
# 	linspace = np.linspace(-radius , radius, spacing)
# 	X, Y = np.meshgrid(linspace, linspace)
# 	b = a[...,0].T
# 	ax = plt.gca()
# 	plt.xlim([-radius, radius])
# 	# ax.set_aspect('equal')
# 	theta = np.linspace(0, 2*np.pi, spacing)
# 	r = np.mean(b, axis=1)
# 	plt.polar(theta, r)
# 	# plt.plot(linspace, np.mean(b, axis=0))
# 	# plt.show()

# def plot_linear(radius, spacing, a):
# 	linspace = np.linspace(-radius , radius, spacing)
# 	X, Y = np.meshgrid(linspace, linspace)
# 	b = a[...,0].T
# 	ax = plt.gca()
# 	plt.xlim([-radius, radius])
# 	plt.plot(linspace, np.mean(b, axis=0))
# 	# plt.show()


# radius = 0.5
# spacing = 51
# a = np.load('/Users/simonweber/programming/workspace/learning_grids/test_output_rates.npy')

# # plt.contourf(X, Y, a[...,0].T)

# # plt.plot(linspace, np.mean(b, axis=1))

# # # Now axis 1 of b corresponds to angle (y direction)
# plot_linear(radius, spacing, a)
# plt.show()