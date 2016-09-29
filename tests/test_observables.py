__author__ = 'simonweber'
import unittest
import numpy as np
from learning_grids import observables


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

	def test_get_grid_axis_angles(self):
		self.gridness_ones.labeled_array = np.array([
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 5, 0, 1, 0, 0],
			[0, 6, 0, 0, 3, 0, 0],
			[0, 6, 2, 4, 4, 3, 0],
			[0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0],
		])
		# Note: the vertical axis is the x axis
		# You need to turn everthing counterclockwise by 90 degrees in order
		# to get the values for angles
		angles = np.array(
			[135, -45,
			 np.rad2deg(np.arctan2(1.5, 0.5)), np.rad2deg(np.arctan2(0.5, 1)),
			 -135, np.rad2deg(np.arctan2(-2, 0.5))
			 ])
		# Axis 1 (0   degrees): label 4
		# Axis 2 (60  degrees): label 3
		# Axis 3 (-60 degrees): label 2
		# Note, you need to subtract 1 (because its indeces)
		expected = np.take(angles, [3,2,1])
		result = np.rad2deg(self.gridness_ones.get_grid_axis_angles())
		np.testing.assert_array_equal(expected, result)

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