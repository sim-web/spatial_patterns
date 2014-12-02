import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy import signal
import scipy
import cProfile
import pstats

###########################################################################
########## Playing with gaussian process and its autocorrelation ##########
###########################################################################

def get_gaussian_process(radius, sigma, linspace, white_noise, factor=1.1):
	"""
	Returns function within radius with autocorrelation length sqrt(2)*sigma

	Note: By stretching the border with a factor, we enable the
	desired linspace to be larger then the box. This is
	necessary, because the rat can slightly move beyond the boundary.
	The returned scaled function has its minumum and maximum in the
	range [-factor*radius, factor*radius], so if you want the function to
	typically have its extrema within the box, you should make the factor
	not much larger than 1. Since you only need this for the space
	discretization in the lookup of input rates factor=1.1 is sufficient,
	because the rat does not move further out of the box than velocity*dt=0.01
	and 1.1*radius = 1.1*0.5 = 0.55, so the rat could move 0.05 out of the
	box and the function would still be defined.

	Parameters
	----------
	radius : float
	sigma : float
		The autocorrelation length of the resulting function will be the
		same as of a Gaussian with std of `sigma`
	white_noise : ndarray
		Large array of white noise. Good length for our purposes: 6e4
	linspace : ndarray
		Linear space on which the returned function should lie.
		Typically (-limit, limit, spacing), where `limit` either equals
		`radius` or is slightly larger (see note in description).
	factor : float
		Factor for which the limits can be larger than `radius`

	Return
	------
	output : ndarray
		An interpolation of a random function with the same autocorrelation
		length as a Gaussian of std = sigma, interpolated to the
		discretization defined given in `linspace`
	"""
	half_len_wn = int(len(white_noise) / 2.)
	# Put Gauss array on half the length of the white noise to use convolve
	# in mode 'valid'
	gauss_space = np.linspace(-factor * radius, factor * radius, half_len_wn)
	# Linspace of the convolution
	conv_space = np.linspace(-factor * radius, factor * radius,
								half_len_wn + 1)
	# Centered Gaussian
	gaussian = np.sqrt(2 * np.pi * sigma ** 2) * stats.norm(loc=0.0,
						scale=sigma).pdf(gauss_space)
	# Convolve the Gaussian with the white_noise
	# convolution = np.convolve(gaussian, white_noise, mode='valid')
	convolution = signal.fftconvolve(white_noise, gaussian, mode='valid')

	# Rescale the result such that its maximum is 1.0 and its minimum 0.0
	gp = (convolution - np.amin(convolution)) / (np.amax(convolution) - np.amin(
		convolution))
	# Interpolate the outcome to the desired output discretization
	return np.interp(linspace, conv_space, gp)

np.random.seed(1)
radius = 5.0
sigma = 0.03
linspace = np.linspace(-radius, radius, 1001)
white_noise = np.random.random(6e4)
gp = get_gaussian_process(radius, sigma, linspace, white_noise, factor=1.0)

def create_some_gps(n=10):
	for i in np.arange(n):
		print i
		white_noise = np.random.random(6e4)
		get_gaussian_process(radius, sigma, linspace, white_noise, factor=1.1)

if __name__ == '__main__':
	cProfile.run('create_some_gps()', 'profile_gps')
	pstats.Stats('profile_gps').sort_stats('cumulative').print_stats(20)



# plt.subplots(2,1)
# plt.subplot(2,1,1)
# plt.plot(linspace, gp)
# plt.subplot(2,1,2)
# gp_zero_mean = gp - np.mean(gp)
# ac = np.correlate(gp_zero_mean, gp_zero_mean, mode='same')
# plt.plot(linspace, ac)
# gauss = scipy.stats.norm(loc=0, scale=sigma * np.sqrt(2)).pdf
# gauss_scaling = np.amax(ac) * np.sqrt(2*np.pi*(sigma*np.sqrt(2))**2)
# plt.plot(linspace, gauss_scaling * gauss(linspace), color='red')
# plt.show()



# def get_random_numbers(n, mean, spreading, distribution):
# 	"""Returns random numbers with specified distribution
#
# 	Parameters
# 	----------
# 	n: (int) number of random numbers to be returned
# 	mean: (float) mean value for the distributions
# 	spreading: (float or array) specifies the spreading of the random nubmers
# 	distribution: (string) a certain distribution
# 		- uniform: uniform distribution with mean mean and percentual spreading spreading
# 		- cut_off_gaussian: normal distribution limited to range
# 			(spreading[1] to spreading[2]) with stdev spreading[0]
# 			Values outside the range are thrown away
#
# 	Returns
# 	-------
# 	Array of n random numbers
# 	"""
#
# 	if distribution == 'uniform':
# 		rns = np.random.uniform(mean * (1. - spreading), mean * (1. + spreading), n)
#
# 	elif distribution == 'cut_off_gaussian':
# 		# Draw 100 times more numbers, because those outside the range are thrown away
# 		rns = np.random.normal(mean, spreading['stdev'], 100 * n)
# 		rns = rns[rns > spreading['left']]
# 		rns = rns[rns < spreading['right']]
# 		rns = rns[:n]
#
# 	elif distribution == 'cut_off_gaussian_with_standard_limits':
# 		rns = np.random.normal(mean, spreading, 100 * n)
# 		left = 0.001
# 		right = 2 * mean - left
# 		rns = rns[rns > left]
# 		rns = rns[rns < right]
# 		rns = rns[:n]
#
# 	elif distribution == 'gamma':
# 		k = (mean / spreading) ** 2
# 		theta = spreading ** 2 / mean
# 		rns = np.random.gamma(k, theta, n)
# 	return rns
#
#
# n = 1e6
# # mean = 0.11
# # spreading = 0.03
# mean = 0.10
# spreading = 0.03
# print spreading
# # plt.xlim([0, 2.0])
# # mean = 6
# # spreading = 2*np.sqrt(3)
# rns = get_random_numbers(n, mean, spreading, 'gamma')
#
# # plt.hist(rns, bins=50, range=(0, 2.0))
# # Plot histogram for exc / inh ratio
# # rns2 = get_random_numbers(n, 0.2, 0.13, 'gamma')
# plt.hist(rns, bins=50, range=(0, 3*mean), alpha=0.5)
# # plt.hist(rns2 / 0.7, bins=50, range=(0, 2.0), color='green', alpha=0.5)
# plt.show()

# def get_equidistant_positions(r, n, boxtype='linear', distortion=0., on_boundary=False):
# """Returns equidistant, symmetrically distributed coordinates

# Works in dimensions higher than One.
# 	The coordinates are taken such that they don't lie on the boundaries
# 	of the environment but instead half a lattice constant away on each
# 	side.
# 	Note: In the case of circular boxtype positions outside the cirlce
# 		are thrown away.

# 	Parameters
# 	----------
# 	r : array_like
# 		Dimensions of the box [Rx, Ry, Rz, ...]
# 		If `boxtype` is 'circular', then r can just be an integer, if it
# 		is an array the first entry is taken as the radius
# 	n : array_like
# 		Array of same shape as r, number of positions along each direction
# 	boxtype : string
# 		'linear': A quadratic arrangement of positions is returned
# 		'circular': A ciruclar arrangement instead
# 	distortion : float or array_like
# 		Maximal length by which each lattice coordinate (x and y separately)
# 		is shifted randomly (uniformly)
# 	on_boundary : bool
# 		If True, positions can also lie on the system boundaries
# 	Returns
# 	-------
# 	(ndarray) of shape (m, len(n)), where m < np.prod(n) for boxtype
# 	'circular', because points at the edges are thrown away.
# 	"""
# 	r, n, distortion = np.asarray(r), np.asarray(n), np.asarray(distortion)
# 	if not on_boundary:
# 		# Get the distance from the boundaries
# 		d = 2*r/(2*n)
# 	else:
# 		# Set the distance from the boundaries to zero
# 		d = np.zeros_like(r)
# 	# Get linspace for each dimension
# 	spaces = [np.linspace(-ra+da, ra-da, na) for (ra, na, da) in zip(r, n, d)]
# 	# Get multidimensional meshgrid
# 	Xs = np.meshgrid(*spaces)
# 	if boxtype == 'circular':
# 		distance = np.sqrt(np.sum([x**2 for x in Xs], axis=0))
# 		# Set grid values outside the circle to NaN. Note: This sets the x
# 		# and the y component (or higher dimensions) to NaN
# 		for x in Xs:
# 			x[distance>r[0]] = np.nan
# 	# Obtain positions file (shape: (n1*n2*..., dimensions)) from meshgrids
# 	positions = np.array(zip(*[x.flat for x in Xs]))
# 	# Remove any subarray which contains at least one NaN
# 	# You do this by keeping only those that do not contain NaN (negation: ~)
# 	positions = positions[~np.isnan(positions).any(axis=1)]
# 	dist = 2*distortion * np.random.random_sample(positions.shape) - distortion
# 	return positions + dist


# n_x = 100
# n_y = 77
# n_z = 5

# dimensions = 2
# # n = np.array([n_x, n_y])
# n = np.array([n_x, n_y, n_z])[:dimensions]
# r = np.array([0.5, 0.5, 0.5])[:dimensions]
# # r = np.array([0.5, 0.1])
# boxtype = 'linear' 
# # distortion = r/n
# distortion = 0.0
# centers = get_equidistant_positions(r=r, n=n, boxtype=boxtype, distortion=distortion, on_boundary=False)
# centers = centers.reshape(n_y, n_x, dimensions)
# # centers = centers.reshape(n_y, n_x, n_z, dimensions)
# print 'centers.shape'
# print centers.shape
# # print centers

# # x = centers[:, 0].copy()
# # centers[:, 0] = centers[:, 1]
# # centers[:, 1] = x

# print centers[0, 3, :]
# # centers[y, x, :]

# x = -0.03
# y = 0.4
# z = 0.43
# position = np.array([x, y, z])[:dimensions]
# index = np.ceil((position + r)*n/(2*r)) - 1
# print 'index'
# print index
# print centers[tuple([index[1], index[0]])]
# # print tuple(index)
# # print centers[-1, -1, :]


# fig = plt.figure()

# if len(n) == 2:
# 	ax = fig.add_subplot(111)
# 	ax.scatter(centers[...,0], centers[...,1], marker='x')
# elif len(n) == 3:
# 	ax = fig.add_subplot(111, projection='3d')
# 	ax.scatter(centers[...,0], centers[...,1], centers[...,2], marker='x')

# ax.add_artist(plt.Circle((0,0), r[0], fc='none'))
# ax.add_artist(plt.Rectangle((-r[0],-r[0]), 2*r[0], 2*r[0], fc='none'))
# ax.set_aspect('equal')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.show()



# def move_persistently_semi_periodic(x, y, phi):
# 	# Boundary conditions and movement are interleaved here
# 	pos = np.array([x, y])
# 	is_bound_trespassed = np.logical_or(pos < -radius, pos > radius)
# 	# Reflection at the corners
# 	if np.all(is_bound_trespassed):
# 		phi = np.pi - phi
# 		x += velocity_dt * np.cos(phi)
# 		if y > radius:
# 			y -= 2 * radius
# 		else:
# 			y += 2 * radius
# 	# Reflection at left and right
# 	elif is_bound_trespassed[0]:
# 		phi = np.pi - phi
# 		x += velocity_dt * np.cos(phi)
# 		y += velocity_dt * np.sin(phi)
# 	# Reflection at top and bottom
# 	elif y > radius:
# 		y -= 2 * radius
# 		x += velocity_dt * np.cos(phi)
# 		y += velocity_dt * np.sin(phi)
# 	elif y < -radius:
# 		y += 2 * radius
# 		x += velocity_dt * np.cos(phi)
# 		y += velocity_dt * np.sin(phi)
# 	# Normal move without reflection
# 	else:
# 		phi += angular_sigma * np.random.randn()
# 		x += velocity_dt * np.cos(phi)
# 		y += velocity_dt * np.sin(phi)

# 	# drop that later
# 	return x, y, phi

# def move_persistently_3d(x, y, z, phi, theta):
# 	# Boundary conditions and movement are interleaved here
# 	x, y, z, r = x, y, z, radius
# 	out_of_bounds_y = (y < -r or y > r)
# 	out_of_bounds_x = (x < -r or x > r)
# 	out_of_bounds_z = (z < -r or z > r)
# 	if (out_of_bounds_x and out_of_bounds_y and out_of_bounds_z):
# 		phi += np.pi
# 		theta += np.pi
# 	# Reflection at the edges
# 	elif (out_of_bounds_x and out_of_bounds_y):
# 		phi += np.pi
# 	elif (out_of_bounds_x and out_of_bounds_z):
# 		theta += np.pi
# 	elif (out_of_bounds_y and out_of_bounds_z):
# 		theta += np.pi
# 	# Reflection at x
# 	elif out_of_bounds_x:
# 		phi = np.pi - phi
# 	# Reflection at y
# 	elif out_of_bounds_y:
# 		phi = -phi
# 	# Reflection at z
# 	elif out_of_bounds_z:
# 		theta = np.pi - theta
# 	# Normal move without reflection
# 	else:
# 		phi += angular_sigma * np.random.randn()
# 		theta += angular_sigma * np.random.randn()
# 	x += velocity_dt * np.cos(phi) * np.sin(theta)
# 	y += velocity_dt * np.sin(phi) * np.sin(theta)
# 	z += 0.5*velocity_dt * np.cos(theta)
# 	# drop that later
# 	return x, y, z, phi, theta

# def move_persistently_semi_periodic_3d(x, y, z, phi, theta):
# 	# Boundary conditions and movement are interleaved here
# 	x, y, z, r = x, y, z, radius
# 	out_of_bounds_y = (y < -r or y > r)
# 	out_of_bounds_x = (x < -r or x > r)
# 	out_of_bounds_z = (z < -r or z > r)
# 	if (out_of_bounds_x and out_of_bounds_y and out_of_bounds_z):
# 		phi += np.pi
# 		theta += np.pi
# 		if z > r:
# 			z -= 2 * r
# 		else:
# 			z += 2 * r
# 	# Reflection at the edges
# 	elif (out_of_bounds_x and out_of_bounds_y):
# 		phi += np.pi
# 		z += 0.5*velocity_dt * np.cos(theta)
# 	elif (out_of_bounds_x and out_of_bounds_z):
# 		theta += np.pi
# 		if z > r:
# 			z -= 2 * r
# 		else:
# 			z += 2 * r
# 	elif (out_of_bounds_y and out_of_bounds_z):
# 		theta += np.pi
# 		if z > r:
# 			z -= 2 * r
# 		else:
# 			z += 2 * r
# 	# Reflection at x
# 	elif out_of_bounds_x:
# 		phi = np.pi - phi
# 		z += 0.5*velocity_dt * np.cos(theta)
# 	# Reflection at y
# 	elif out_of_bounds_y:
# 		phi = -phi
# 		z += 0.5*velocity_dt * np.cos(theta)
# 	# Reflection at z
# 	elif z > r:
# 		z -= 2 * r
# 	elif z < -r:
# 		z += 2 * r
# 	# Normal move without reflection
# 	else:
# 		phi += angular_sigma * np.random.randn()
# 		theta += angular_sigma * np.random.randn()
# 		z += 0.5*velocity_dt * np.cos(theta)
# 	x += velocity_dt * np.cos(phi) * np.sin(theta)
# 	y += velocity_dt * np.sin(phi) * np.sin(theta)
# 	# drop that later
# 	return x, y, z, phi, theta

# # np.random.seed(2)
# radius = 0.5
# x = 0.4
# y = 0.4
# z = 0.4
# theta = 0.01
# phi = 0.2
# velocity = 0.01
# dt = 1.0
# velocity_dt = velocity * dt
# persistence_length = radius
# angular_sigma = np.sqrt(2.*velocity*dt/persistence_length)
# steps = np.arange(30000)
# # for s in steps:
# # 	x, y, phi = move_persistently_semi_periodic(x, y, phi)
# # 	print x, y, phi


# fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # plt.xlim([-radius, radius])
# # plt.ylim([-radius, radius])
# # ax = plt.gca()
# # ax.set_aspect('equal')
# x_y_phi = []
# x_y_z_phi_theta = []
# # for s in steps:
# # 	x, y, phi = move_persistently_semi_periodic(x, y, phi)
# # 	x_y_phi.append((x, y, phi))
# # 	print x
# # x_y_phi = np.array(x_y_phi)
# # ax.set_xlabel('X Label')
# # ax.set_ylabel('Y Label')
# # ax.set_zlabel('Z Label')

# for s in steps:
# 	x, y, z, phi, theta = move_persistently_semi_periodic_3d(x, y, z, phi, theta)
# 	x_y_z_phi_theta.append((x, y, z, phi, theta))
# 	# print x
# x_y_z_phi_theta = np.array(x_y_z_phi_theta)


# # plt.scatter(x_y_phi[:,0], x_y_phi[:,1], s=0.2)
# # ax.scatter(x_y_z_phi_theta[:,0], x_y_z_phi_theta[:,1], x_y_z_phi_theta[:,2], s=0.2)
# for i in [0, 1, 2]:
# 	fig.add_subplot(3,1,i+1)
# 	plt.plot(x_y_z_phi_theta[:,i])
# 	# corr = np.correlate(x_y_z_phi_theta[:,i], x_y_z_phi_theta[:,i], "full")
# 	# print corr
# 	# plt.plot(corr)
# # plt.contourf(x_y_z_phi_theta[:,0], x_y_z_phi_theta[:,1], x_y_z_phi_theta[:,2])
# plt.show()

# def get_equidistant_positions(r, n, boxtype='linear', distortion=0.):
# 	"""Returns equidistant, symmetrically distributed coordinates

# 	Works in dimensions higher than One.
# 	The coordinates are taken such that they don't lie on the boundaries
# 	of the environment but instead half a lattice constant away on each
# 	side.
# 	Note: In the case of circular boxtype positions outside the cirlce
# 		are thrown away.

# 	Parameters
# 	----------
# 	r : array_like
# 		Dimensions of the box
# 		If `boxtype` is 'circular', then r can just be an integer, if it
# 		is an array the first entry is taken as the radius
# 	n : array_like
# 		Array of same shape as r, number of positions along each direction
# 	boxtype : string
# 		'linear': A quadratic arrangement of positions is returned
# 		'circular': A ciruclar arrangement instead 
# 	distortion : float or array_like
# 		Maximal length by which each lattice coordinate (x and y separately)
# 		is shifted randomly (uniformly)
# 	Returns
# 	-------
# 	(ndarray) of shape (m, len(n)), where m < np.prod(n) for boxtype
# 	'circular', because points at the edges are thrown away.
# 	"""
# 	r, n, distortion = np.asarray(r), np.asarray(n), np.asarray(distortion)
# 	# Get the distance from the boundaries
# 	d = 2*r/(2*n)
# 	# Get linspace for each dimension
# 	spaces = [np.linspace(-ra+da, ra-da, na) for (ra, na, da) in zip(r, n, d)]
# 	# Get multidimensional meshgrid
# 	Xs = np.meshgrid(*spaces)
# 	if boxtype == 'circular':
# 		distance = np.sqrt(np.sum([x**2 for x in Xs], axis=0))
# 		# Set grid values outside the circle to NaN. Note: This sets the x
# 		# and the y component (or higher dimensions) to NaN
# 		for x in Xs:
# 			x[distance>r[0]] = np.nan
# 	# Obtain positions file (shape: (n1*n2*..., dimensions)) from meshgrids
# 	positions = np.array(zip(*[x.flat for x in Xs]))
# 	# Remove any subarray which contains at least one NaN
# 	# You do this by keeping only those that do not contain NaN (negation: ~)
# 	positions = positions[~np.isnan(positions).any(axis=1)]
# 	dist = 2*distortion * np.random.random_sample(positions.shape) - distortion
# 	return positions + dist

# n_x = 20
# n_y = 10
# n_z = 6

# # n = np.array([n_x, n_y])
# n = np.array([n_x, n_y, n_z])
# r = np.array([0.5, 0.5, 0.5])
# # r = np.array([0.5, 0.1])
# boxtype = 'linear' 
# distortion = r/n
# # distortion = 0.0
# centers = get_equidistant_positions(r=r, n=n, boxtype=boxtype, distortion=distortion)
# print centers
# fig = plt.figure()

# if len(n) == 2:
# 	ax = fig.add_subplot(111)
# 	ax.scatter(centers[...,0], centers[...,1], marker='x')
# elif len(n) == 3:
# 	ax = fig.add_subplot(111, projection='3d')
# 	ax.scatter(centers[...,0], centers[...,1], centers[...,2], marker='x')

# ax.add_artist(plt.Circle((0,0), r[0], fc='none'))
# ax.add_artist(plt.Rectangle((-r[0],-r[0]), 2*r[0], 2*r[0], fc='none'))
# ax.set_aspect('equal')
# plt.show()

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