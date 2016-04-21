import numpy as np
import general_utils.arrays
import scipy
import scipy.ndimage as ndimage
from pylab import *
# import scipy.ndimage.filters as filters

# #############################################
# #########	Head Direction Tuning	##########
##############################################
class Head_Direction_Tuning():
	"""Class to get value of Head Direction tuning"""

	def __init__(self, HD_firing_rates, spacing, n=10000, alpha=0.05):
		self.HD_firing_rates = HD_firing_rates
		self.spacing = spacing
		self.n = n
		self.alpha = alpha

	def uniques_ties_cumuls_relfreqs(self, A):
		"""One liner description

		Parameters
		----------
		A : corresponds to directions in example

		Returns
		-------

		"""
		a = unique(A)
		# t: frequencies
		t = zeros(len(a))
		# m:
		m = zeros(len(a))
		for jj in range(len(a)):
			t[jj] = len(A[A == a[jj]])
			m[jj] = sum(t[:jj + 1])
		n = m[-1]
		m_n = m / n
		return a, t, m, n, m_n

	def watson_u2(self, ang1, ang2, alpha):
		"""
		adapted from pierre.megevand@gmail.com

		Computes Watson's U2 statistic for nonparametric 2-sample testing of
		circular data, accommodating ties. The code is derived from eq. 27.17
		Zar (1999) and its performance was verified using the numerical examples
		27.10 and 27.11 from that reference.
		Inputs:
		A1, A2:   vectors containing angles (in degrees or radians, the unit does
		not matter)

		Outputs:
		U2:       Watson's U2 statistic

		Significance tables for U2 have been published, e.g. in Zar (1999).
		Alternatively, an ad hoc permutation test can be used.

		References:
		Zar JH. Biostatistical Analysis. 4th ed., 1999.
		Chapter 27: Circular Distributions: Hypothesis Testing.
		Upper Saddle River, NJ: Prentice Hall.
		"""

		a1, t1, m1, n1, m1_n1 = self.uniques_ties_cumuls_relfreqs(ang1)
		a2, t2, m2, n2, m2_n2 = self.uniques_ties_cumuls_relfreqs(ang2)

		n = n1 + n2

		k = len(unique(append(ang1, ang2)))
		table = zeros((k, 4))
		table[:, 0] = unique(append(ang1, ang2))
		for ii in range(len(table)):
			if table[ii, 0] in a1:
				loc1 = find(a1 == table[ii, 0])[0]
				table[ii, 1] = table[ii, 1] + m1_n1[loc1]
				table[ii, 3] = table[ii, 3] + t1[loc1]
			else:
				if ii > 0:
					table[ii, 1] = table[ii - 1, 1]

			if table[ii, 0] in a2:
				loc2 = find(a2 == table[ii, 0])[0]
				table[ii, 2] = table[ii, 2] + m2_n2[loc2]
				table[ii, 3] = table[ii, 3] + t2[loc2]
			else:
				if ii > 0:
					table[ii, 2] = table[ii - 1, 2]

		#print table
		d = table[:, 1] - table[:, 2]
		t = table[:, 3]
		td = sum(t * d)
		td2 = sum(t * d ** 2)
		U2 = ((n1 * n2) / (n ** 2)) * (td2 - (td ** 2 / n))

		k1 = min(n1, n2)
		k2 = max(n1, n2)
		if k1 > 10:
			k1 = inf
		if k2 > 12:
			k2 = inf

		# Critical values (from Kanji, 100 statistical tests, 1999)
		len1 = array(
			[5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8,
			 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, inf])
		len2 = array(
			[5, 6, 7, 8, 9, 10, 11, 12, 6, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11,
			 12, 8, 9, 10, 11, 12, 9, 10, 11, 12, 10, 11, 12, inf])
		if alpha == 0.01:
			a = array(
				[nan, nan, nan, nan, 0.28, 0.289, 0.297, 0.261, nan, 0.282,
				 0.298, 0.262, 0.248, 0.262, 0.259, 0.304, 0.272, 0.255, 0.262,
				 0.253, 0.252, 0.250, 0.258, 0.249, 0.252, 0.252, 0.266, 0.254,
				 0.255, 0.254, 0.255, 0.255, 0.255, 0.268])
		elif alpha == 0.05:
			a = array(
				[0.225, 0.242, 0.2, 0.215, 0.191, 0.196, 0.19, 0.186, 0.206,
				 0.194, 0.196, 0.193, 0.19, 0.187, 0.183, 0.199, 0.182, 0.182,
				 0.187, 0.184, 0.186, 0.184, 0.186, 0.185, 0.184, 0.185, 0.187,
				 0.186, 0.185, 0.185, 0.185, 0.186, 0.185, 0.187])
		else:
			print(
				'Watson U2: This test is implemented for alpha levels of 0.05 and 0.01 only.')

		# Find critical value
		i1 = (len1 == k1)
		i2 = (len2 == k2)
		value = a[i1 * i2]
		if isnan(value):
			print("error: watson u2 test cannot be computed for given input")
		h = (U2 >= value)[0]

		return U2, h

	def draw_from_head_direction_distribution(self):
		xk = np.arange(self.spacing)
		pk = self.HD_firing_rates / np.sum(self.HD_firing_rates)
		hd_dist = scipy.stats.rv_discrete(
			a=0, b=self.spacing, name='hd_dist', values=(xk, pk))
		hd_angles = hd_dist.rvs(size=self.n) * 180 / self.spacing
		return hd_angles

	def get_watson_U2_against_uniform(self):
		uniform_angles = np.random.uniform(0, 180, self.n)
		hd_angles = self.draw_from_head_direction_distribution()
		return self.watson_u2(uniform_angles, hd_angles, self.alpha)


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
		spacing = a.shape[0]
	if mode == 'same':
		# Note that spacing should be an odd number
		spacing = (a.shape[0] - 1) / 2

	space = np.arange(1, spacing + 1)
	corr_spacing = 2 * spacing + 1
	correlations = np.zeros((corr_spacing, corr_spacing))

	# Center
	s1 = a.flatten()
	s2 = b.flatten()
	corrcoef = np.corrcoef(s1, s2)
	correlations[spacing, spacing] = corrcoef[0, 1]

	# East line
	for nx in space:
		s1 = a[nx:, :].flatten()
		s2 = b[:-nx, :].flatten()
		corrcoef = np.corrcoef(s1, s2)
		correlations[nx + spacing, spacing] = corrcoef[0, 1]

	# North line
	for ny in space:
		s1 = a[:, ny:].flatten()
		s2 = b[:, :-ny].flatten()
		corrcoef = np.corrcoef(s1, s2)
		correlations[spacing, spacing + ny] = corrcoef[0, 1]

	# West line
	for nx in space:
		s1 = a[:-nx, :].flatten()
		s2 = b[nx:, :].flatten()
		corrcoef = np.corrcoef(s1, s2)
		correlations[spacing - nx, spacing] = corrcoef[0, 1]

	# South line
	for ny in space:
		s1 = a[:, :-ny].flatten()
		s2 = b[:, ny:].flatten()
		corrcoef = np.corrcoef(s1, s2)
		correlations[spacing, spacing - ny] = corrcoef[0, 1]

	# First quadrant
	for ny in space:
		for nx in space:
			s1 = a[nx:, ny:].flatten()
			s2 = b[:-nx, :-ny].flatten()
			corrcoef = np.corrcoef(s1, s2)
			correlations[nx + spacing, ny + spacing] = corrcoef[0, 1]

	# Second quadrant
	for ny in space:
		for nx in space:
			s1 = a[:-nx, ny:].flatten()
			s2 = b[nx:, :-ny].flatten()
			corrcoef = np.corrcoef(s1, s2)
			correlations[spacing - nx, ny + spacing] = corrcoef[0, 1]

	# Third quadrant
	for nx in space:
		for ny in space:
			s1 = a[:-nx, :-ny].flatten()
			s2 = b[nx:, ny:].flatten()
			corrcoef = np.corrcoef(s1, s2)
			correlations[spacing - nx, spacing - ny] = corrcoef[0, 1]

	# Fourth quadrant
	for ny in space:
		for nx in space:
			s1 = a[nx:, :-ny].flatten()
			s2 = b[:-nx, ny:].flatten()
			corrcoef = np.corrcoef(s1, s2)
			correlations[nx + spacing, spacing - ny] = corrcoef[0, 1]

	return corr_spacing, correlations


class Gridness():
	"""
	A class to get information on gridnes of correlogram

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
				 threshold_difference=0.1, method='Weber',
				 n_contiguous=16, type='hexagonal'):
		self.a = a
		self.radius = radius
		self.method = method
		self.neighborhood_size = neighborhood_size
		self.threshold_difference = threshold_difference
		self.spacing = a.shape[0]
		self.x_space = np.linspace(-self.radius, self.radius, self.spacing)
		self.y_space = np.linspace(-self.radius, self.radius, self.spacing)
		self.X, self.Y = np.meshgrid(self.x_space, self.y_space)
		self.distance = np.sqrt(self.X * self.X + self.Y * self.Y)
		self.distance_1d = np.abs(self.x_space)
		self.type = type
		if type == 'hexagonal':
			self.n_peaks = 7
		elif type == 'quadratic':
			self.n_peaks = 5
		if method == 'sargolini':
			self.set_labeled_array(n_contiguous, n_peaks=self.n_peaks)
		elif method == 'sargolini_extended':
			self.set_labeled_array(n_contiguous, extended=True,
								   n_peaks=self.n_peaks)
		self.set_center_to_inner_peaks_distances()
		self.set_inner_radius_outer_radius_grid_spacing()

	def set_spacing_and_quality_of_1d_grid(self):
		# We pass a normalized version (value at center = 1) of the correlogram
		maxima_boolean = general_utils.arrays.get_local_maxima_boolean(
			self.a / self.a[self.spacing / 2], self.neighborhood_size,
			self.threshold_difference)
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

	def set_labeled_array(self, n_contiguous, extended=False, n_peaks=7):
		"""
		Finds peaks in correlogram and labels them.

		Peaks are clusters (or features) of adjacent pixels (diagonally adjacent
		also counts) with values above threshold.
		Each cluster gets an integer label and each pixel of the same
		cluster gets the same label.
		We keep only the (6+1) most central peaks.
		Everthing else in the array is set to zero.

		Parameters
		----------
		n_contiguous : int
			Only clusters of `n_contiguous` particles ore more are kept
			and labeled (except for the central one, which can be arbitrarily
			small)

		Returns
		-------
		Nothing
		"""
		# Definition of neighborhood (diagonal neighborhood matters)
		structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
		clipped_a = self.a.copy()
		# Keep only pixels above threshold value (typically 0.1)
		clipped_a[clipped_a <= self.threshold_difference] = 0.
		# Get all clusters
		labeled_array, self.num_features = ndimage.measurements.label(
			clipped_a, structure=structure)
		# Keep only clusters of size > n_contiguous
		for n in np.arange(1, self.num_features + 1):
			if len(labeled_array[labeled_array == n]) < n_contiguous:
				labeled_array[labeled_array == n] = 0.
		# Keep only the 6+1 most central clusters
	 	# Note: the number of clusters can get smaller than 7
		labeled_array = self.keep_n_most_central_features(
													labeled_array, n=n_peaks)
		self.labeled_array = labeled_array
		if extended:
			self.labeled_array = self.keep_meaningful_central_features(
													self.labeled_array)

	def get_sorted_feature_distance_array(self, labeled_array):
		"""
		Returns structured array of labels and distances

		The structured array is sorted with increasing distance from the
		cluster (labeled with `label`) to the coordinate center

		Parameters
		----------
		see labeled_array

		Returns
		-------
		feature_distance_array : ndarray (structured)
		"""
		my_dtype = [('label', int), ('distance', float)]
		feature_distance = []
		# Get all the ocurring labels
		all_labels = np.unique(labeled_array[labeled_array != 0])
		for l in all_labels:
			d = self.distance[labeled_array == l]
			feature_distance.append(
				(l, np.mean(d))
			)
		feature_distance_arr = np.array(feature_distance, my_dtype)
		feature_distance_arr.sort(order='distance')
		return feature_distance_arr

	def keep_n_most_central_features(self, labeled_array, n=7):
		"""
		Keeps only the n most central features

		Parameters
		----------
		labeled_array : ndarray
			Array with labeled features
		n : int
			See above

		Returns
		-------
		reduced labeled_array : ndarray
			Array of same shape as labeled_array, where at most
			the 6+1 most central clusters are still labeled
		"""
		feature_distance_arr = self.get_sorted_feature_distance_array(
																labeled_array)
		labels_to_drop = feature_distance_arr['label'][n:]
		for l in labels_to_drop:
			labeled_array[labeled_array == l] = 0
		# return self.keep_meaningful_central_features(labeled_array)
		return labeled_array

	def keep_meaningful_central_features(self, labeled_array):
		"""
		Removes clusters that are too far outside

		For bad grids there are typically not six inner peaks around the
		center. If we keep the 6+1 most central peaks (as done in
		keep_n_most_central_peaks) this might lead to peaks of very different
		distances to the center. This does not make sense, because then the
		outer radius is drawn around the largest outlier. In this scenario
		we instead would like to draw the outer radius around the furthest
		outside inner peaks, although this leads to less than 6+1 peaks
		considered. We identify this scenario by comparing the distance of the
		outermost peak to the center with the distance of one of the inner
		peaks to the center (actually the third one, which should exist).
		If the outermost peak is more than 1.5 further away than the inner
		one it is rejected.
		This is not mentioned in Sargolini 2006, but it is desirable still.
		Note: You neeed to run keep_n_most_central_features first.

		Returns
		-------
		reduced labeled_array : ndarray
		"""
		feature_distance_arr = self.get_sorted_feature_distance_array(
																labeled_array)
		labels_all = feature_distance_arr['label']
		try:
			while (feature_distance_arr['distance'][-1]
					   > 1.5 * feature_distance_arr['distance'][2]):
				feature_distance_arr = np.delete(feature_distance_arr, -1, axis=0)
		except IndexError:
			pass
		labels_to_keep = feature_distance_arr['label']
		labels_to_drop = np.setdiff1d(labels_all, labels_to_keep)
		for l in labels_to_drop:
			labeled_array[labeled_array == l] = 0
		return labeled_array


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
		first_distances = np.sort(self.distance[maxima_boolean])[1:n + 1]
		return first_distances

	def get_central_cluster_bool(self):
		"""
		Returns the array with True at the location of the center peak pixels
		"""
		central_label = self.labeled_array[(self.spacing - 1) / 2.][
			(self.spacing - 1) / 2.]
		central_cluster_bool = (self.labeled_array == central_label)
		# Check if central cluster cool is True everywhere
		# If so, this means that the central cluster is considered to be
		# the entire box. If this occurs we manually say that the central
		# cluster is one off set from the absolute center.
		# This leads to a very small inner radius.
		# Note the double negation in the condition!
		if np.count_nonzero(~central_cluster_bool) == 0:
			central_cluster_bool = ~central_cluster_bool
			central_cluster_bool[np.floor((self.spacing - 2) / 2.)][
				((self.spacing - 1) / 2.)] = True
		return central_cluster_bool

	def get_inner_radius(self):
		"""
		Returns the inner radius for the correlogram cropping

		For method = 'sargolini', this is the outermost pixel of all the
		pixels of the most central peak

		For method = 'Weber', this is the mean of the distances to the
		first six peaks (excluding the center peak)

		Returns
		-------
		inner_radius : float
		"""
		if (self.method == 'sargolini' or self.method == 'sargolini_extended'):
			central_cluster_bool = self.get_central_cluster_bool()
			return np.amax(self.distance[central_cluster_bool])
		elif self.method == 'Weber':
			first_distances = self.center_to_inner_peaks_distances
			return 0.5 * np.mean(first_distances)

	def get_outer_radius(self):
		"""
		Returns the outer radius for the correlogram cropping

		For method = 'sargolini', this is the outermost pixel of the seven
		most central pixel clusters. If there are less than seven, than the
		outermost pixel of the remaing clusters is taken.

		For method = 'Weber', this is the maximal distances to the 6 closest
		non-center clusters + the inner radius.

		Returns
		-------
		outer_radius : float
		"""
		if (self.method == 'sargolini' or self.method == 'sargolini_extended'):
			valid_cluster_bool = np.nonzero(self.labeled_array)
			try:
				return np.amax(self.distance[valid_cluster_bool])
			except ValueError as e:
				print e
				print 'number of nonzero values in labeled array:'
				print np.count_nonzero(self.labeled_array)
				return self.radius
		elif self.method == 'Weber':
			first_distances = self.center_to_inner_peaks_distances
			return max(first_distances) + 1.0 * self.inner_radius

	def get_grid_spacing(self):
		"""
		Returns the grid spacing
		"""
		first_distances = self.center_to_inner_peaks_distances
		# If set_center_to_inner_peaks_distances cause an exception
		# because no clusters were found it sets
		# first_distances[1] = self.radius. Here we check for that.
		# set_center_to_inner_peaks_distances
		if first_distances[1] == self.radius:
			return np.nan
		else:
			return np.mean(first_distances)

	def set_inner_radius_outer_radius_grid_spacing(self):
		self.inner_radius = self.get_inner_radius()
		self.outer_radius = self.get_outer_radius()
		self.grid_spacing = self.get_grid_spacing()

	def set_center_to_inner_peaks_distances(self):
		"""
		The distance to the 6 most central peaks to the center
		"""
		first_distances = self.get_peak_center_distances(self.n_peaks-1)
		if len(first_distances) > 1:
			closest_distance = first_distances[0]
			# We don't want distances outside 1.5*closest distance
			first_distances = [d for d in first_distances if
							   d < 1.5 * closest_distance]
		# If before or after taking the inner most peaks, we only have one
		# or two peaks left, we set them artificially
		if len(first_distances) <= 1:
			first_distances = [0.01, self.radius]
		self.center_to_inner_peaks_distances = first_distances

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
			index_i = self.distance < self.inner_radius
			index_o = self.distance > self.outer_radius
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
		# Unrotated array
		a0 = self.get_cropped_flattened_arrays([self.a.copy()])[0]
		rotated_arrays = general_utils.arrays.get_rotated_arrays(self.a, angles)
		cropped_rotated_arrays = self.get_cropped_flattened_arrays(
			rotated_arrays)
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
		if self.type == 'hexagonal':
			correlation_good = self.get_correlation_vs_angle(
				angles=[60, 120])[1]
			correlation_bad = self.get_correlation_vs_angle(
				angles=[30, 90, 150])[1]
		elif self.type == 'quadratic':
			correlation_good = self.get_correlation_vs_angle(
				angles=[90])[1]
			correlation_bad = self.get_correlation_vs_angle(
				angles=[45, 135])[1]
		grid_score = min(correlation_good) - max(correlation_bad)
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
