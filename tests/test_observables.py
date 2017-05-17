__author__ = 'simonweber'
import unittest
import numpy as np
from learning_grids import observables
import matplotlib.pyplot as plt
import scipy.signal as signal
import time

class TestObservables(unittest.TestCase):

	def setUp(self):
		# unittest.TestCase.__init__()
		seven_by_seven = np.array([
			[0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
			[0.1, 0.3, 0.0, 0.0, 0.0, 0.4, 0.0],
			[0.2, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.9, 0.5, 0.0, 0.0],
			[0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.1],
			[0.3, 0.4, 0.4, 0.0, 0.2, 0.0, 0.1],
			[0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.1]
		])
		self.gridness = observables.Gridness(a=seven_by_seven, radius=0.5,
										threshold_difference=0.1,
										method='sargolini', n_contiguous=3)
		seven_by_seven_zeros = np.zeros_like(seven_by_seven)
		self.gridness_zeros = observables.Gridness(a=seven_by_seven_zeros,
										radius=0.5,
										threshold_difference=0.1,
										method='sargolini', n_contiguous=3)
		seven_by_seven_ones = np.ones_like(seven_by_seven)
		self.gridness_ones = observables.Gridness(a=seven_by_seven_ones,
										radius=0.5,
										threshold_difference=0.1,
										method='sargolini', n_contiguous=3)
	def test_set_labeled_array(self):
		la = self.gridness.labeled_array
		expected = np.array([
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		])
		result = np.zeros_like(la, dtype=float)
		# Set all the labels to 1.0, because you can't predict the
		# assignment of the numbers
		result[la > 0.0] = 1.0
		np.testing.assert_array_almost_equal(expected, result)

	def test_keep_n_most_central_features(self):
		la = np.array([
			[3, 3, 0, 4, 4, 0, 5],
			[3, 0, 0, 0, 4, 0, 5],
			[0, 0, 0, 0, 0, 0, 0],
			[1, 1, 0, 2, 2, 0, 0],
			[1, 1, 0, 2, 0, 0, 0],
			[0, 0, 0, 0, 0, 6, 6],
			[0, 0, 0, 0, 0, 6, 6],
		])
		expected = np.array([
			[3, 3, 0, 4, 4, 0, 0],
			[3, 0, 0, 0, 4, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[1, 1, 0, 2, 2, 0, 0],
			[1, 1, 0, 2, 0, 0, 0],
			[0, 0, 0, 0, 0, 6, 6],
			[0, 0, 0, 0, 0, 6, 6],
		])
		result = self.gridness.keep_n_most_central_features(la, n=5)
		np.testing.assert_array_equal(expected, result)


	def test_get_sorted_feature_distance_array(self):
		la = np.array([
			[3, 3, 0, 4, 4, 0, 5],
			[3, 0, 0, 0, 4, 0, 5],
			[0, 0, 0, 0, 0, 0, 0],
			[1, 1, 0, 2, 2, 0, 0],
			[1, 1, 0, 2, 0, 0, 0],
			[0, 0, 0, 0, 0, 6, 6],
			[0, 0, 0, 0, 0, 6, 6],
		])
		expected = np.array([2, 1, 4, 6, 3, 5])
		result = self.gridness.get_sorted_feature_distance_array(la)['label']
		np.testing.assert_array_equal(expected, result)

	def test_keep_meaningful_central_features(self):
		la = np.array([
			[0, 0, 0, 0, 0, 0, 0],
			[1, 0, 0, 4, 0, 0, 0],
			[0, 0, 0, 0, 0, 6, 0],
			[0, 3, 0, 2, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 5],
		])
		expected = np.array([
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 4, 0, 0, 0],
			[0, 0, 0, 0, 0, 6, 0],
			[0, 3, 0, 2, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
		])
		result = self.gridness.keep_meaningful_central_features(la)
		np.testing.assert_array_equal(expected, result)

	def test_get_central_cluster_bool(self):
		expected = np.array([
			[False, False, False, False, False, False, False],
			[False, False, False, False, False, False, False],
			[False, False, False, True, False, False, False],
			[False, False, False, True, True, False, False],
			[False, False, False, False, False, False, False],
			[False, False, False, False, False, False, False],
			[False, False, False, False, False, False, False]
		])
		result = self.gridness.get_central_cluster_bool()
		np.testing.assert_array_equal(expected, result)
		# Test the all zeros scenario
		expected = np.array([
			[False, False, False, False, False, False, False],
			[False, False, False, False, False, False, False],
			[False, False, False, True, False, False, False],
			[False, False, False, False, False, False, False],
			[False, False, False, False, False, False, False],
			[False, False, False, False, False, False, False],
			[False, False, False, False, False, False, False]
		])
		result = self.gridness_zeros.get_central_cluster_bool()
		np.testing.assert_array_equal(expected, result)

	def test_get_inner_radius(self):
		expected = 1./6
		result = self.gridness.get_inner_radius()
		self.assertAlmostEqual(expected, result, 5)
		# Test the all zeros scenario
		expected = 1./6
		result = self.gridness_zeros.get_inner_radius()
		self.assertAlmostEqual(expected, result, 5)

	def test_get_outer_radius(self):
		expected = np.sqrt((3./6)**2 + (2./6)**2)
		result = self.gridness.get_outer_radius()
		self.assertAlmostEqual(expected, result, 5)
		# Test the all zeros scenario
		expected = self.gridness_zeros.radius
		result = self.gridness_zeros.get_outer_radius()
		self.assertAlmostEqual(expected, result, 5)

	def test_get_grid_axes_angles(self):
		self.gridness_ones.labeled_array = np.array([
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 5, 0, 1, 0, 0],
			[0, 6, 0, 0, 3, 0, 0],
			[0, 6, 2, 4, 4, 3, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
		])
		angles = np.array(
			[-45, 135,
			 np.rad2deg(np.arctan2(0.5, 1.5)), np.rad2deg(np.arctan2(1, 0.5)),
			 -135, np.rad2deg(np.arctan2(0.5, -2))
			 ])
		# Axis 1 (0   degrees): label 3
		# Axis 2 (60  degrees): label 4
		# Axis 3 (-60 degrees): label 1
		# Note, you need to subtract 1 (because its indeces)
		expected = np.take(angles, [2,3,0])
		result = np.rad2deg(self.gridness_ones.get_grid_axes_angles())
		# self.plot_illustration_of_array_with_angles()
		np.testing.assert_array_almost_equal(expected, result)

	def plot_illustration_of_array_with_angles(self):
		# Note: the vertical axis is the x axis
		# You need to turn everthing counterclockwise by 90 degrees in order
		# to get the values for angles
		la = self.gridness_ones.labeled_array
		### Contour plot ###
		plt.subplot(211)
		plt.contourf(la)
		plt.gca().set_aspect('equal')
		### Imshow plot ###
		plt.subplot(212)
		plt.imshow(la, interpolation='none', origin='lower')
		ax = plt.gca()
		x_n = la.shape[1]
		y_n = la.shape[0]
		ax.set_xticks(np.linspace(0, x_n-1, x_n))
		ax.set_yticks(np.linspace(0, y_n-1, y_n))
		ax.set_xlim(-.5, x_n-.5)
		ax.set_ylim(-.5, y_n-.5)
		x_names = [str(x) for x in range(x_n)]
		y_names = [str(x) for x in range(y_n)]
		ax.set_xticklabels(x_names, rotation=45., minor=False)
		ax.set_yticklabels(y_names, rotation=45., minor=False)
		for angle in self.gridness_ones.get_grid_axes_angles():
			plt.plot(np.array([0, 3*np.cos(angle)])+3,
					 np.array([0, 3*np.sin(angle)])+3, marker='o', color='red')
		plt.show()

	def test_get_correlation_2d(self):
		"""
		Testing stuff associated with cross correlation and
		pearson correlation.
		NB: Double checked it on a note.
		The determined pearson corrleation is the same as here:
		https://en.wikipedia.org/wiki/Covariance_and_correlation
		"""
		a = np.array(
			[
				[0., 1., 2.],
				[3., 4., 5.],
				[1., 2., 1.]
			]
		)
		b = np.ones_like(a)
		print(signal.correlate2d(a, b, mode='same'))
		### Pure correlate2d ###
		# correlate2d simply integrates the products of all overlapping
		# fields, without normalizing by the number of entries.
		expected_correlate2d = np.array(
			[
				[0*4+1*5+3*2+4*1, 0*3+1*4+2*5+3*1+4*2+5*1, 1*3+2*4+4*1+5*2],
				[0*1+1*2+3*4+4*5+1*2+2*1, np.sum(a**2), 1*0+2*1+4*3+5*4+2*1+1*2],
				[3*1+4*2+1*4+2*5, 3*0+4*1+5*2+1*3+2*4+1*5, 4*0+5*1+2*3+1*4]
			]
		)
		result = signal.correlate2d(
			a, a, mode='same')
		np.testing.assert_array_equal(expected_correlate2d, result)

		### correlate2d normalized by number of overlapping fields ###
		expected_correlate2d_norm = (
			expected_correlate2d
			/ np.array([[4, 6, 4], [6, 9, 6], [4, 6, 4]]))
		ones = np.ones_like(expected_correlate2d)
		result = (expected_correlate2d
				  / signal.correlate2d(ones, ones, mode='same'))
		np.testing.assert_array_equal(expected_correlate2d_norm,
									  result)

		### Compare pearson_autocorrelate2d to get_correlation_2d
		for x in [a, np.random.random_sample((11, 11))]:
			corr_spacing, expected = observables.get_correlation_2d(
				x, x, mode='same')
			result = observables.pearson_correlate2d(
				x, x, mode='same', fft=True)
			np.testing.assert_array_almost_equal(expected, result, decimal=10)

		a = np.random.random_sample((11, 11))
		b = np.random.random_sample((11, 11))
		corr_spacing, expected = observables.get_correlation_2d(
			a, b, mode='same')
		result = observables.pearson_correlate2d(
			a, b, mode='same', fft=True)
		np.testing.assert_array_almost_equal(expected, result, decimal=10)

		### Compare to Daniel Wennbergs pearson autocorrelogram
		# # Import does not work
		# np.random.seed(0)
		# size = 11
		# a = np.random.random_sample((size, size))
		# b = np.random.random_sample((size, size))
		# t0 = time.time()
		# pc = observables.pearson_correlate2d(a, b, mode='same', fft=True)
		# print(pc)
		# print(pc[3, 4], pc[5, 1])
		# t1 = time.time()
		# print(t1 - t0)








	# def test_get_cylinder(self):
	# 	# (3, 3, 3) array with peak of 3 fields in x, y plane
	# 	# The z axis is homogeneously 1 except along the
	# 	# x0, y1 cylinder
	# 	a = np.array([
	# 		[
	# 			[3, 2, 3],  # Index in x, y plane: [0, 0]
	# 			[4, 7, 9],  # [0, 1]
	# 			[1, 1, 1],  # [0, 2]
	# 		],
	# 		[
	# 			[3, 1, 1],  # [1,0]
	# 			[1, 1, 1],
	# 			[1, 1, 1],
	# 		],
	# 		[
	# 			[1, 3, 100],
	# 			[1, 1, 1],  # [2, 1]
	# 			[1, 1, 1],
	# 		],
	# 	])
	# 	indices = np.array([
	# 		[0, 0], [0, 1], [1, 0]
	# 	])
	# 	expected = np.array([3, 2, 3], [4, 7, 9], [3, 1, 1])
	# 	result = self.gridness.get_cylinder(a, indices)
	# 	np.testing.assert_array_equal(expected, result)
	#

	#
	# def test_set_inner_radius_sargolini(self):
	#
	# 	expected = 1.5
	# 	result = self.gridness.inner_radius(method='Sargolini')
	# 	self.assertAlmostEqual(expected, result, places=7)