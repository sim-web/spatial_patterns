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

	def test_get_inner_radius(self):
		expected = 1./6
		result = self.gridness.get_inner_radius()
		self.assertAlmostEqual(expected, result, 5)

	def test_get_outer_radius(self):
		expected = np.sqrt((3./6)**2 + (2./6)**2)
		result = self.gridness.get_outer_radius()
		self.assertAlmostEqual(expected, result, 5)


	#
	# def test_set_inner_radius_sargolini(self):
	#
	# 	expected = 1.5
	# 	result = self.gridness.inner_radius(method='Sargolini')
	# 	self.assertAlmostEqual(expected, result, places=7)