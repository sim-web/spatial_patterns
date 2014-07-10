import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def get_equidistant_positions(r, n_x, n_y, boxtype='linear', distortion=0.):
	"""Returns equidistant, symmetrically distributed 2D coordinates
	
	Note: The number of returned positions will be <= n, because not
		because only numbers where n = k**2 where k in Integers, can
		exactly tile the space and only for 'linear' arrangements anyway.
		In the case of circular boxtype many positions need to be thrown
		away.

	Parameters
	----------
	r : ndarray
		Dimensions of the box
	n_x, n_y: int, int
		Number of centers along x and y axis
	boxtype : string
		'linear': A quadratic arrangement of positions is returned
		'circular': A ciruclar arrangement instead 
	distortion : float or ndarray
		Maximal length by which each lattice coordinate (x and y separately)
		is shifted randomly (uniformly)
	
	Returns
	-------
	(ndarray) of shape (m, 2), where m < n but close to n for linear boxtype
				and signficantly smaller than n for circular boxtype
	"""
	dx = 2*r[0]/(2*n_x)
	dy = 2*r[1]/(2*n_y)
	x_space = np.linspace(-r[0]+dx, r[0]-dx, n_x)
	y_space = np.linspace(-r[1]+dy, r[1]-dy, n_y)
	positions_grid = np.empty((n_x, n_y, 2))
	X, Y = np.meshgrid(x_space, y_space)
	# Put all the positions in positions_grid
	for n_y, y in enumerate(y_space):
		for n_x, x in enumerate(x_space):
			positions_grid[n_x][n_y] =  [x, y]
	# Flatten for easier reshape
	positions = positions_grid.flatten()
	# Reshape to the desired shape
	positions = positions.reshape(positions.size/2, 2)
	if boxtype == 'circular':
		r = np.amax(r)
		distance = np.sqrt(X*X + Y*Y)
		# Set all position values outside the circle to NaN
		positions_grid[distance.T>r] = np.nan
		positions = positions_grid.flatten()
		isnan = np.isnan(positions)
		# Delete all positions outside the circle
		positions = np.delete(positions, np.nonzero(isnan))
		# Bring into desired shape
		positions = positions.reshape(positions.size/2, 2)
	distortion_array = 2*distortion * np.random.random_sample(positions.shape) - distortion
	return positions + distortion_array


n_x = 70
n_y = 70
r = np.array([0.5, 0.5])
boxtype = 'linear'
distortion = r/np.array([n_x, n_y])
centers = get_equidistant_positions(r=r, n_x=n_x, n_y=n_y, boxtype=boxtype, distortion=distortion)
ax = plt.gca()
plt.scatter(centers[...,0], centers[...,1], marker='x')
ax.add_artist(plt.Circle((0,0), r[0], fc='none'))
ax.add_artist(plt.Rectangle((-r[0],-r[0]), 2*r[0], 2*r[0], fc='none'))
ax.set_aspect('equal')
print 'bla'
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