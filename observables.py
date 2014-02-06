import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

##################################
##########	Gridness	##########
##################################
def get_correlation_2d(a, b, mode='full'):
	"""
	Determines array of Pearson correlation coefficients.

	Array b is shifted with respect to a and for each possible shift
	the Pearson correlation coefficient for the overlapping part of
	the two arrays is determined and written in the output rate.
	For a = b this results in the auto-correlogram.
	Note that erratic values (for example seeking the correlation in
	array of only zeros) result in np.nan values.

	See Evernote 'Pearson Correlation Coefficient for Auto Correlogram'
	to find an explanation of the algorithm
	
	Parameters
	----------
	a : array_like
		Shape needs to be (N x N), N should be an odd number.
	b : array_like
		Needs to be of same shape as a
	mode : string
		'full': Computes correlogram for twice the boxlenth
		'same': Computes correlogram only for the original scale of the box
	
	Returns
	-------
	output : ndarray
		Array of Pearson correlation coefficients for shifts of b
		with respect to a.
		The shape of the correlations array is (M, M) where M = 2*N + 1.
	"""
	
	if mode == 'full':
		spacing = a.shape[0] - 1
	if mode == 'same':
		# Note that spacing should be an odd number
		spacing = (a.shape[0] - 1) / 2

	space = np.arange(1, spacing+1)
	corr_spacing = 2*spacing+1
	correlations = np.zeros((corr_spacing, corr_spacing))
	
	# Center
	s1 = a.flatten()
	s2 = b.flatten()
	corrcoef = np.corrcoef(s1, s2)
	correlations[spacing, spacing] = corrcoef[0, 1]
	print corrcoef[0 , 1]

	# East line
	for nx in space:
		s1 = a[nx:, :].flatten()
		s2 = b[:-nx, :].flatten()
		corrcoef = np.corrcoef(s1, s2)
		correlations[nx+spacing, spacing] = corrcoef[0, 1]

	# North line
	for ny in space:
		s1 = a[:, ny:].flatten()
		s2 = b[:, :-ny].flatten()
		corrcoef = np.corrcoef(s1, s2)
		correlations[spacing, spacing+ny] = corrcoef[0, 1]

	# West line
	for nx in space:
		s1 = a[:-nx, :].flatten()
		s2 = b[nx:, :].flatten()
		corrcoef = np.corrcoef(s1, s2)
		correlations[spacing-nx, spacing] = corrcoef[0, 1]

	# South line
	for ny in space:
		s1 = a[:, :-ny].flatten()
		s2 = b[:, ny:].flatten()
		corrcoef = np.corrcoef(s1, s2)
		correlations[spacing, spacing-ny] = corrcoef[0, 1]

	# First quadrant
	for ny in space:
		for nx in space: 
			s1 = a[nx:, ny:].flatten()
			s2 = b[:-nx, :-ny].flatten()
			corrcoef = np.corrcoef(s1, s2)
			correlations[nx+spacing, ny+spacing] = corrcoef[0, 1]

	# Second quadrant
	for ny in space:
		for nx in space: 
			s1 = a[:-nx, ny:].flatten()
			s2 = b[nx:, :-ny].flatten()
			corrcoef = np.corrcoef(s1, s2)
			correlations[spacing-nx, ny+spacing] = corrcoef[0, 1]

	# Third quadrant
	for nx in space:
		for ny in space: 
			s1 = a[:-nx, :-ny].flatten()
			s2 = b[nx:, ny:].flatten()
			corrcoef = np.corrcoef(s1, s2)
			correlations[spacing-nx, spacing-ny] = corrcoef[0, 1]

	# Fourth quadrant
	for ny in space:
		for nx in space: 
			s1 = a[nx:, :-ny].flatten()
			s2 = b[:-nx, ny:].flatten()
			corrcoef = np.corrcoef(s1, s2)
			correlations[nx+spacing, spacing-ny] = corrcoef[0, 1]

	return corr_spacing, correlations

class Gridness():
	def __init__(self, a, radius=None, neighborhood_size=None,
					threshold_difference=None):
		self.a = a
		self.radius = radius
		self.neighborhood_size = neighborhood_size
		self.threshold_difference = threshold_difference
		self.spacing = a.shape[0]
		self.x_space = np.linspace(-self.radius, self.radius, self.spacing)
		self.y_space = np.linspace(-self.radius, self.radius, self.spacing)
		self.X, self.Y = np.meshgrid(self.x_space, self.y_space)
		self.distance = np.sqrt(self.X*self.X + self.Y*self.Y)

	def get_local_maxima_boolean_2d(self):
		# maximum_filter sets each point in area (neighborhood X neighborhood) the
		# maximal value occurring in this area
		data_max = filters.maximum_filter(self.a, self.neighborhood_size)
		# Create booelan array with True entries at the positions of the local
		# maxima.
		maxima = (self.a == data_max)
		# Find also local minima and remove maxima that are not distinct enough
		# from the maxima array by setting them False
		data_min = filters.minimum_filter(self.a, self.neighborhood_size)
		below_threshold = ((data_max - data_min) <= self.threshold_difference)
		maxima[below_threshold] = False
		return maxima

	def get_distance_to_center_of_n_most_central_peaks(self, n):
		maxima_boolean = self.get_local_maxima_boolean_2d()
		first_distances = np.sort(self.distance[maxima_boolean])[:7]
		return first_distances

	def get_rotated_arrays(self, angles):
		rotated_a = np.empty((len(angles), self.a.shape[0], self.a.shape[1]))
		for n, angle in enumerate(angles):
			rotated_a[n] = scipy.ndimage.interpolation.rotate(
										self.a, angle, reshape=False)
		return rotated_a

	def get_inner_radius(self, first_distances):
		inner_radius = 0.5 * np.mean(first_distances)
		return inner_radius

	def get_outer_radius(self,
		first_distances, inner_radius, multiple_radii=False):
		if multiple_radii:
			outer_radius = np.linspace(2*inner_radius, self.radius, 10)[::-1]
		else:
			outer_radius = max(first_distances) + inner_radius
		return outer_radius

	def set_inner_and_outer_radius(self, method='Weber'):
		first_distances = self.get_distance_to_center_of_n_most_central_peaks(7)
		if method == 'Weber':
			self.inner_radius = self.get_inner_radius(first_distances)
			self.outer_radius = self.get_outer_radius(first_distances, self.inner_radius)

	def get_cropped_flattened_arrays(self, arrays):
		cropped_flattened_arrays = []
		for a in arrays:
			# a[self.distance<self.inner_radius] = -5
			# a[self.distance>self.outer_radius] = -5
			indexL = self.distance<self.inner_radius
			indexG = self.distance>self.outer_radius
			a[np.logical_or(indexL, indexG)] = np.nan
			cfa = a.flatten()
			cfa = cfa[np.isfinite(cfa)]
			# cfa = [x for x in a.flatten() if x > -5]
			# print cfa
			cropped_flattened_arrays.append(cfa)
		return np.array(cropped_flattened_arrays)

	def get_correlation_vs_angle(self, angles=np.arange(0, 180, 2)):
		self.set_inner_and_outer_radius()
		a0 = self.get_cropped_flattened_arrays([self.a.copy()])[0]
		rotated_arrays = self.get_rotated_arrays(angles)
		cropped_rotated_arrays = self.get_cropped_flattened_arrays(rotated_arrays)
		correlations = []
		for cra in cropped_rotated_arrays:
			correlations.append(np.corrcoef(a0, cra)[0, 1])
		return angles, correlations

	def get_grid_score(self):
		correlation_60_120 = self.get_correlation_vs_angle(
								angles=[60, 120])[1]
		correlation_30_90_150 = self.get_correlation_vs_angle(
								angles=[30, 90, 150])[1]
		grid_score = min(correlation_60_120) - max(correlation_30_90_150)
		return grid_score

##############################################
##########	Measures for Learning	##########
##############################################
def sum_difference_squared(old_weights, new_weights):
	"""
	Sum of squared differences between old and new weights

	- old_weights, new_weights are numpy arrays
	"""
	return np.sum(np.square(new_weights - old_weights))
