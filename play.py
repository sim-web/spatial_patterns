import numpy as np
import matplotlib as mpl
import initialization
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy import signal
import scipy
from general_utils import plotting
import cProfile
import pstats
from scipy.ndimage import filters
from initialization import get_equidistant_positions


radius = 7.0
linspace = np.linspace(-radius, radius, 8001)
sigma = 0.03
gp = initialization.get_gaussian_process(radius, sigma, linspace, rescale=True)
# plt.plot(linspace, gp)

# plt.subplots(2,1)
plt.subplot(3,1,1)
plt.plot(linspace, gp)
plt.xlim([-radius, radius])
print np.mean(gp)

plt.subplot(3,1,2)
gp_zero_mean = gp - np.mean(gp)
ac = np.correlate(gp_zero_mean, gp_zero_mean, mode='same')
plt.plot(linspace, ac, color='gray', alpha=0.5, lw=3)
plt.xlim([-0.5, 0.5])

gauss = scipy.stats.norm(loc=0, scale=sigma * np.sqrt(2)).pdf
gauss_scaling = np.amax(ac) * np.sqrt(2*np.pi*(sigma*np.sqrt(2))**2)
plt.plot(linspace, gauss_scaling * gauss(linspace), color='red', lw=3, alpha=0.5)
plt.xlim([-0.5, 0.5])

gauss = scipy.stats.norm(loc=0, scale=sigma).pdf
gauss_scaling = np.amax(ac) * np.sqrt(2*np.pi*(sigma)**2)
plt.plot(linspace, gauss_scaling * gauss(linspace), color='green', lw=3, alpha=0.5)
plt.xlim([-0.5, 0.5])

ac_gauss = np.correlate(gauss(linspace), gauss(linspace), mode='same')
plt.plot(linspace, np.amax(ac) * ac_gauss / np.amax(ac_gauss), color='blue', lw=3, alpha=0.5)
plt.xlim([-0.5, 0.5])
plt.show()

#
# np.random.seed(1000)
#
# radius = np.array([0.5, 0.5])
# sigma = 0.05
# center_overlap = 3 * sigma
# n = np.array([70, 70])
# fields_per_synapse = 1
# limit = radius + center_overlap
# distortion = radius / n
#
# centers = get_equidistant_positions(limit, n, distortion=distortion)
# N = centers.shape[0]
# centers = centers.reshape(N, 1, 2)
# print centers

# radius = 1.0
# pos = np.array([0.2, -1.2])
# test = (pos > radius) * 1 - (pos < -radius) * 1
# print test
# positions = initialization.get_equidistant_positions(
# 	r=[0.5, 0.5],
# 	n=[10, 10],
# 	boxtype='circular',
# 	distortion=[0.07142857,  0.07142857],
# 	on_boundary=False
# )

# positions = initialization.get_random_positions_within_circle(
# 	n=100,
# 	r=[0.5,0.5])
#
# plt.scatter(positions[:,0], positions[:,1])
# plt.show()

###########################################################################
########################## Numerical Integration ##########################
###########################################################################
from scipy.integrate import quad, dblquad
from scipy.special import erf

# norm_a = np.load('norm_a.npy')
# norm_new = np.load('norm_new.npy')
#
# # plt.plot(norm_a)
# print norm_a
# # plt.plot(norm_new)
# print norm_new
# plt.plot(norm_a/norm_new)
# plt.show()

# sigma = 0.1
# loc = 0.0
# limit = 0.2
#
# print np.sqrt(2*np.pi) * sigma
# # print quad(lambda x: 1. / (1 + ((x-loc)/sigma)**2), -limit, limit)
# print quad(lambda x: np.exp(-(x-loc)**2/(2*sigma**2)), -limit, limit)
#
# print - sigma * np.sqrt(np.pi/2) * (erf((-limit-loc)/(sigma*np.sqrt(2))) + erf((-limit+loc)/(sigma*np.sqrt(2))))
# print - sigma * (np.arctan((-limit-loc)/sigma) + np.arctan((-limit+loc)/sigma))

# def I(gammax, gammay, radius):
# 	return dblquad(lambda x, y: 1. / (np.power(1 + (x/gammax)**2 + (y/gammay)**2, 1.5)),
# 				   -radius, radius, lambda y: -radius, lambda y: radius)
#
#     # return dblquad(lambda t, x: np.exp(-x*t)/t**n, 0, np.inf, lambda x: 1, lambda x: np.inf)
#
# print I(0.05, 0.05, 0.5)

###########################################################################
######################### Get random number AGAIN #########################
###########################################################################
# def get_random_numbers(n, mean, spreading, distribution):
# 	"""Returns random numbers with specified distribution
#
# 	Parameters
# 	----------
# 	n: (tuple) shape of random numbers array to be returned
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
# 		rns = np.random.normal(mean, spreading['stdev'], 100*n)
# 		rns = rns[rns>spreading['left']]
# 		rns = rns[rns<spreading['right']]
# 		rns = rns[:n]
#
# 	elif distribution == 'cut_off_gaussian_with_standard_limits':
# 		rns = np.random.normal(mean, spreading, 100*n)
# 		left = 0.001
# 		right = 2 * mean - left
# 		rns = rns[rns>left]
# 		rns = rns[rns<right]
# 		rns = rns[:n]
#
# 	elif distribution == 'gamma':
# 		k = (mean/spreading)**2
# 		theta = spreading**2 / mean
# 		rns = np.random.gamma(k, theta, n)
#
# 	elif distribution == 'gamma_with_cut_off':
# 		k = (mean/spreading)**2
# 		theta = spreading**2 / mean
# 		rns = np.random.gamma(k, theta, n)
# 		rns[rns<0.01] = 0.01
#
# 	elif distribution == 'gaussian_peak':
# 		linspace = np.linspace(-1, 1, n[1])
# 		rns = stats.norm(loc=mean, scale=spreading).pdf(linspace).reshape(1, n[1])
#
#
# 	return rns
#
# nexc = 400
# n = (1, nexc)
# weights = get_random_numbers(n, mean=0.1, spreading=0.03, distribution='gaussian_peak')
# centers = np.linspace(-1, 1, nexc)
# plt.plot(centers, weights[0,:])
# plt.show()

##########################################################################
##################### 2 dimensional gaussian process #####################
##########################################################################

# def get_gaussian_process(radius, sigma, linspace, dimensions=1):
# 	"""
# 	Returns function with autocorrelation length sqrt(2)*sigma
#
# 	So the returned function has the same autocorrelation length like a
# 	gaussian of standard deviation sigma.
#
# 	Parameters
# 	----------
# 	radius : float
# 	sigma : float or ndarray
# 		The autocorrelation length of the resulting function will be the
# 		same as of a Gaussian with standard deviation of `sigma`
# 	linspace : ndarray
# 		Linear space on which the returned function should lie.
# 		Typically (-limit, limit, spacing), where `limit` either equals
# 		`radius` or is slightly larger if there's a chance that the
# 		rat moves outside the box. Typically limit = 1.1 * radius.
# 		Note: The limit must be <= 2*radius. If you ever want to change this,
# 		make the value of agp larger than 2.
# 	dimensions : int
# 		Number of dimensions of the gaussian process function
#
# 	Return
# 	------
# 	output : ndarray
# 		An interpolation of a random function with the same autocorrelation
# 		length as a Gaussian of std = sigma, interpolated to the
# 		discretization defined given in `linspace`.
# 	"""
# 	if dimensions == 1:
# 		# The bin width
# 		dx = sigma / 20.
# 		if sigma < radius/8.:
# 			# For small enough sigma it's enough to define the gaussian
# 			# in the range [-r, r]
# 			agauss = 1.0
# 		else:
# 			# For larger sigma we need the gaussian on a larger array,
# 			# to avoid the curve from being cut-off
# 			agauss = np.ceil(8*sigma)
# 			# We need to take a smaller discretization than given by the
# 			# large sigma.
# 			dx /= agauss
# 		# Maybe put agp as function argument with default 2
# 		agp = 2
# 		# The number of bins for each ocurring array
# 		bins_per_radius = np.ceil(radius / dx)
# 		bins_wn = (agauss + agp) * bins_per_radius
# 		bins_gauss = agauss * bins_per_radius
# 		bins_gp = agp * bins_per_radius
# 		white_noise = np.random.random(bins_wn)
# 		gauss_limit = agauss*radius
# 		gauss_space = np.linspace(-gauss_limit, gauss_limit, bins_gauss)
# 		conv_limit = agp*radius
# 		# Note: you need to add +1 to the number of bins
# 		conv_space = np.linspace(-conv_limit, conv_limit, bins_gp + 1)
# 		# Centered Gaussian on gauss_space
# 		gaussian = np.sqrt(2 * np.pi * sigma ** 2) * stats.norm(loc=0.0,
# 							scale=sigma).pdf(gauss_space)
# 		# Convolve the Gaussian with the white_noise
# 		# Note: in fft convolve the larger array must be the first argument
# 		convolution = signal.fftconvolve(white_noise, gaussian, mode='valid')
# 		# Rescale the result such that its maximum is 1.0 and its minimum 0.0
# 		gp = (convolution - np.amin(convolution)) / (np.amax(convolution) - np.amin(
# 			convolution))
# 		# Interpolate the outcome to the desired output discretization given
# 		# in `linspace`
# 		return np.interp(linspace, conv_space, gp)
#
# 	elif dimensions == 2:
# 		# Works like in 1D but we take a larger dx, for faster initialization
# 		dx = sigma / 10.
# 		# We choose 1.0 as the standard
# 		agauss = np.array([1.0, 1.0])
# 		# Only in the dimensions where sigma>radius/8 we change it
# 		agauss[sigma>radius/8.] = np.ceil(8*sigma)[sigma>radius/8.]
# 		# Change dx as in 1D case (unchanged values will just be divided by 1)
# 		dx /= agauss
# 		agp = np.array([2, 2])
# 		# The number of bins for each ocurring array
# 		bins_per_radius = np.ceil(radius / dx)
# 		bins_wn = (agauss + agp) * bins_per_radius
# 		bins_gauss = agauss * bins_per_radius
# 		bins_gp = agp * bins_per_radius
# 		white_noise = np.random.random(bins_wn)
# 		# Now we need to differentiate between x and y
# 		gauss_limit = agauss*radius
# 		gauss_space_x = np.linspace(-gauss_limit[0], gauss_limit[0], bins_gauss[0])
# 		gauss_space_y = np.linspace(-gauss_limit[1], gauss_limit[1], bins_gauss[1])
# 		conv_limit = agp*radius
# 		conv_space_x = np.linspace(-conv_limit[0], conv_limit[0], bins_gp[0] + 1)
# 		conv_space_y = np.linspace(-conv_limit[1], conv_limit[1], bins_gp[1] + 1)
# 		# Note: meshgrid leads to shape (len(gauss_space_y), len(gauss_space_x))
# 		X_gauss, Y_gauss = np.meshgrid(gauss_space_x, gauss_space_y)
# 		pos = np.empty(X_gauss.shape + (2,))
# 		pos[:, :, 0] = X_gauss
# 		pos[:, :, 1] = Y_gauss
# 		gaussian = (2*np.pi*sigma[0]**1) * stats.multivariate_normal(None, [[sigma[0]**2, 0.0], [0.0, sigma[1]**2]]).pdf(pos)
# 		# Since gaussian now has switched x and y we transpose it to make
# 		# it fit the shape of the white noise.
# 		# Note: now plotting the result with plt.contour shows switched x and y
# 		convolution = signal.fftconvolve(white_noise, gaussian.T, mode='valid')
# 		gp = (convolution - np.amin(convolution)) / (np.amax(convolution) - np.amin(
# 			convolution))
# 		interp_gp = scipy.interpolate.RectBivariateSpline(conv_space_x, conv_space_y, gp)(linspace, linspace)
# 		return interp_gp
#
# # np.random.seed(2)
# np.random.seed(6)
# dimensions = 1
# radius = 3.0
# sigma = np.array([0.1, 0.05])[:dimensions]
# spacing = 501
# factor = 1.0
# # nwn = 6e3
# # awn = max([1, 6*sigma[0]])
# # agp = 2.
# # nwn = awn * int(10*radius/sigma)
# # white_noise = np.random.random(2)
# # white_noise = np.random.random((4e2, 4e2))
# # Linspace of output from function
# linspace = np.linspace(-radius, radius, spacing)
# # plt.ylim([0.0, 1.0])
#
# gp = get_gaussian_process(radius, sigma, linspace, dimensions=dimensions)

# def create_some_gps(n=1000):
# 	for i in np.arange(n):
# 		# print i
# 		white_noise = np.random.random((5e2, 5e2))
# 		get_gaussian_process(radius, sigma, linspace, white_noise, factor=1.1,
# 							 dimensions=2)
#
# if __name__ == '__main__':
# 	cProfile.run('create_some_gps()', 'profile_gps')
# 	pstats.Stats('profile_gps').sort_stats('cumulative').print_stats(20)

# X, Y = np.meshgrid(linspace, linspace)
# V = np.linspace(0.0, 1.0, 80)
# plt.contourf(X, Y, gp, V)
# ticks = np.linspace(0.0, 1.0, 8)
# cb = plt.colorbar(format='%f', ticks=ticks)
# ax = plt.gca()
# ax.set_aspect('equal')

# plt.plot(linspace, gp, color='red')
# plt.xlim([-radius, radius])

# gaussian = np.sqrt(2 * np.pi * sigma ** 2) * stats.norm(loc=0.0, scale=sigma).pdf(linspace)
#
# # For sum of gaussians as input
# fps = 1
# gp = np.zeros_like(linspace)
# for i in np.arange(fps):
# 	gp += np.sqrt(2 * np.pi * sigma ** 2) * stats.norm(loc=(2*radius*np.random.random_sample()-radius), scale=sigma).pdf(linspace)
#
# def minimal_plot():
# 	ax = plt.gca()
# 	ax.axis('off')
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.margins(0.1)
#
# color = plotting.color_cycle_blue4[0]
# lw = 3
# fig = plt.figure(figsize=(5, 4))
# plt.xticks([])
# plt.yticks([])
# plt.subplot(3,1,1)
# # plt.plot(linspace, gaussian, color=color, lw=lw)
# plt.plot(linspace, np.random.random_sample(spacing), color=color)
# minimal_plot()
# # plt.subplots(2,1)
# plt.subplot(3,1,2)
# plt.plot(linspace, gp, color=color, lw=lw)
# minimal_plot()
# plt.subplot(3,1,3)
# gp_zero_mean = gp - np.mean(gp)
# ac = np.correlate(gp_zero_mean, gp_zero_mean, mode='same')
# plt.plot(linspace, ac, color=color, lw=lw)
# gauss = scipy.stats.norm(loc=0, scale=sigma * np.sqrt(2)).pdf
# gauss_scaling = np.amax(ac) * np.sqrt(2*np.pi*(sigma*np.sqrt(2))**2)
# plt.plot(linspace, gauss_scaling * gauss(linspace), color=plotting.color_cycle_blue4[2], linestyle='dashed', lw=lw)
# minimal_plot()
# save_path = '/Users/simonweber/doktor/TeX/learning_grids/1dim_input_tuning/white_noise.pdf'
# plt.savefig(save_path, dpi=2000, bbox_inches='tight', pad_inches=0.02,
# 				transparent=True)
# plt.show()



###########################################################################
########## Playing with gaussian process and its autocorrelation ##########
###########################################################################

# np.random.seed(1)
# radius = 5.0
# sigma = 0.03
# spacing = 1001
# linspace = np.linspace(-radius, radius, spacing)
# white_noise = np.random.random(1e3)
# gp = get_gaussian_process(radius, sigma, linspace, white_noise, factor=1.0)
# test = scipy.ndimage.filters.gaussian_filter(white_noise, sigma)
# plt.plot(test)

# def create_some_gps(n=10):
# 	for i in np.arange(n):
# 		print i
# 		white_noise = np.random.random(6e4)
# 		get_gaussian_process(radius, sigma, linspace, white_noise, factor=1.1)
#
# if __name__ == '__main__':
# 	cProfile.run('create_some_gps()', 'profile_gps')
# 	pstats.Stats('profile_gps').sort_stats('cumulative').print_stats(20)
#


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