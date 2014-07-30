import numpy as np
import general_utils.arrays
import scipy
import scipy.ndimage as ndimage
# import scipy.ndimage.filters as filters

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
	"""
	A class to get information on gridnes  of correlogram
	
	Parameters
	----------
	a : ndarray
		A square array containing correlogram data
	radius : float
		The radius of the correlogram
		This should either be 1 or 2 times the radius of the box from
		which the correlogram was obtained. 1 if mode is 'same' 2 if 
		mode is 'full'
	neighborhood_size : int
		The area for the filters used to determine local peaks in the
		correlogram.
	threshold_difference : float
		A local maximum needs to fulfill the condition that
	method : string 
		Gridness is defined in different ways. `method` selects one
		of the possibilites found in the literature.
	Returns
	-------
	
	"""
		
	def __init__(self, a, radius=None, neighborhood_size=5,
					threshold_difference=0.2, method='Weber'):
		self.a = a
		self.radius = radius
		self.method = method
		self.neighborhood_size = neighborhood_size
		self.threshold_difference = threshold_difference
		self.spacing = a.shape[0]
		self.x_space = np.linspace(-self.radius, self.radius, self.spacing)
		self.y_space = np.linspace(-self.radius, self.radius, self.spacing)
		self.X, self.Y = np.meshgrid(self.x_space, self.y_space)
		self.distance = np.sqrt(self.X*self.X + self.Y*self.Y)
		self.distance_1d = np.abs(self.x_space)

	def set_spacing_and_quality_of_1d_grid(self):
		# We pass a normalized version (center = 1) of the correlogram
		maxima_boolean = general_utils.arrays.get_local_maxima_boolean(
			self.a/self.a[self.spacing/2], self.neighborhood_size, self.threshold_difference)
		distances_from_center = np.abs(self.x_space[maxima_boolean])
		# The first maximum of the autocorrelogram gives the grid spacing
		try:
			self.grid_spacing = np.sort(distances_from_center)[1]
		except:
			self.grid_spacing = 0.0
		# The quality is taken as the coefficient of variation of the
		# inter-maxima distances 
		# You could define other methods. Use method = ... for this purpose.
		distances_between_peaks = (np.abs(distances_from_center[:-1] 
									- distances_from_center[1:]))
		self.std = np.std(distances_between_peaks)
		self.quality = (np.std(distances_between_peaks)
							/ np.mean(distances_between_peaks))


	def get_peak_center_distances(self, n):
		"""
		Distance to center of n most central peaks

		The center itself (distance 0.0) is excluded
		 
		Parameters
		----------
		n : int
			Only the `n` most central peaks are returned

		Returns
		-------
		output : ndarray
			Sorted distances of `n` most central peaks.
		
		"""
		maxima_boolean = general_utils.arrays.get_local_maxima_boolean(
							self.a, self.neighborhood_size, self.threshold_difference)
		first_distances = np.sort(self.distance[maxima_boolean])[1:n+1]
		return first_distances

	def set_inner_and_outer_radius(self):
		"""Set inner and outer radius for the cropping of the correlograms

		Different grid scores differ in their way of cropping the
		correlogram. 
		"""
		first_distances = self.get_peak_center_distances(6)
		closest_distance = first_distances[0]
		# We don't want distances outside 1.5*closest distance
		first_distances = [d for d in first_distances if d<1.5*closest_distance]
		self.grid_spacing = np.mean(first_distances)
		if self.method == 'Weber':
			self.inner_radius = 0.5 * np.mean(first_distances)
			self.outer_radius = max(first_distances) + 1.0*self.inner_radius

	def get_cropped_flattened_arrays(self, arrays):
		"""
		Crop arrays, keep only values where inner_radius<distance<outer_radius
		
		Parameters
		----------
		arrays : ndarray
			Array of shape (n, N, N) with `n` different array of shape `N`x`N`
		
		Returns
		-------
		output : ndarray
			Array of cropped and flattened arrays for further processing
		"""	
		cropped_flattened_arrays = []
		for a in arrays:
			index_i = self.distance<self.inner_radius
			index_o = self.distance>self.outer_radius
			# Set value outside the ring to nan
			a[np.logical_or(index_i, index_o)] = np.nan
			cfa = a.flatten()
			# Keep only finite, i.e. not nan values
			cfa = cfa[np.isfinite(cfa)]
			cropped_flattened_arrays.append(cfa)
		return np.array(cropped_flattened_arrays)

	def get_correlation_vs_angle(self, angles=np.arange(0, 180, 2)):
		"""
		Pearson correlation coefficient (PCC) for unrotated and rotated array.

		The PCC is determined from the unrotated array, compared with arrays
		rotated for every angle in `angles`.
		
		Parameters
		----------
		angles : ndarray
			Angle values

		Returns
		-------
		output : ndarray tuple
			`angles` and corresponding PCCs
		"""
		self.set_inner_and_outer_radius()
		# Unrotated array
		a0 = self.get_cropped_flattened_arrays([self.a.copy()])[0]
		rotated_arrays = general_utils.arrays.get_rotated_arrays(self.a, angles)
		cropped_rotated_arrays = self.get_cropped_flattened_arrays(rotated_arrays)
		correlations = []
		for cra in cropped_rotated_arrays:
			correlations.append(np.corrcoef(a0, cra)[0, 1])
		return angles, correlations

	def get_grid_score(self):
		"""
		Determine the grid score.
		
		Returns
		-------
		output : ndarray
			The grid score corresponding to the given method.
		"""
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
