import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##############################################
##########	Space discretization	##########
##############################################
# The goal is to get a method for two and three dimensions where given number
# of positions along each dimension is obtained. The resulting array should be
# indexable in a meaningful way


# This is the get_equidistant_positions function from initialization.py in
# learning grids. You want to make sure that this is the newest version, when
# you play around with the code below
def get_equidistant_positions(r, n, boxtype='linear', distortion=0., on_boundary=False):
	"""Returns equidistant, symmetrically distributed coordinates

	Works in dimensions higher than One.
	The coordinates are taken such that they don't lie on the boundaries
	of the environment but instead half a lattice constant away on each
	side (now optional).
	Note: In the case of circular boxtype, positions outside the cirlce
		are thrown away.

	Parameters
	----------
	r : array_like
		Dimensions of the box [Rx, Ry, Rz, ...]
		If `boxtype` is 'circular', then r can just be an integer, if it
		is an array the first entry is taken as the radius
	n : array_like
		Array of same shape as r, number of positions along each direction
	boxtype : string
		'linear': A quadratic arrangement of positions is returned
		'circular': A ciruclar arrangement instead
	distortion : float or array_like or string
		If float or array: Maximal length by which each lattice coordinate (x and y separately)
		is shifted randomly (uniformly)
		If string:
			'half_spacing': The distortion is taken as half the distance between
			two points along a perfectly symmetric lattice (along each dimension)
	on_boundary : bool
		If True, positions can also lie on the system boundaries
	Returns
	-------
	(ndarray) of shape (m, len(n)), where m < np.prod(n) for boxtype
	'circular', because points at the edges are thrown away. Note
	len(n) is the dimensionality.
	"""
	if distortion == 'half_spacing':
		distortion = r / (n-1)
	r, n, distortion = np.asarray(r), np.asarray(n), np.asarray(distortion)
	if not on_boundary:
		# Get the distance from the boundaries
		d = 2*r/(2*n)
	else:
		# Set the distance from the boundaries to zero
		d = np.zeros_like(r)
	# Get linspace for each dimension
	spaces = [np.linspace(-ra+da, ra-da, na) for (ra, na, da) in zip(r, n, d)]
	# Get multidimensional meshgrid
	Xs = np.meshgrid(*spaces)
	if boxtype == 'circular':
		distance = np.sqrt(np.sum([x**2 for x in Xs], axis=0))
		# Set grid values outside the circle to NaN. Note: This sets the x
		# and the y component (or higher dimensions) to NaN
		for x in Xs:
			x[distance>r[0]] = np.nan
	# Obtain positions file (shape: (n1*n2*..., dimensions)) from meshgrids
	positions = np.array(zip(*[x.flat for x in Xs]))
	# Remove any subarray which contains at least one NaN
	# You do this by keeping only those that do not contain NaN (negation: ~)
	positions = positions[~np.isnan(positions).any(axis=1)]
	dist = 2*distortion * np.random.random_sample(positions.shape) - distortion
	return positions + dist


dimensions = 1
r = np.array([0.5, 0.5, 0.5])[:dimensions]
n_x, n_y, n_z = 100, 77, 58
n = np.array([n_x, n_y, n_z])[:dimensions]
boxtype = 'linear' 
distortion = 0.1
discrete_positions = get_equidistant_positions(r=r, n=n, boxtype=boxtype,
									distortion=distortion, on_boundary=False)
print discrete_positions
# Now the discrete_positions have shape (prod(n), dimensions)

# The usage of meshgrid in get_equidistant_positions calls for careful reshaping
if dimensions == 2:
	discrete_positions = discrete_positions.reshape(n_y, n_x, dimensions)
if dimensions == 3:
	discrete_positions = discrete_positions.reshape(n_y, n_x, n_z, dimensions)

print 'discrete_positions.shape'
print discrete_positions.shape

position = np.array([-0.21, -0.42, 0.49])[:dimensions]
### GETTING THE INDEX ###
# Let's assume we have a position in continous space.
# From that position we want to get an index for the discrete_positions array
# which results in the discrete position which is closest to the continous
# position.
# Example in x direction: 
# The total length is 2*r. x varies between -r and r. For x=r we want the 
# index to be n_y-1. For x= (0 + tiny epsilon) we want it to be zero.
# We thus get the following mapping. Note that we treat the case x==0 separately
index = np.ceil((position + r)*n/(2*r)) - 1
index[index==-1] = 0
print 'index'
print index
# Plugging this index in the right order (same as in the shaping) in the
# discrete positions file results in an [x, y, z] array close to the
# continous one.
if dimensions == 2:
	print discrete_positions[tuple([index[1], index[0]])]
elif dimensions == 3:
	print discrete_positions[tuple([index[1], index[0], index[2]])]