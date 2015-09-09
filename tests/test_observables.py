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