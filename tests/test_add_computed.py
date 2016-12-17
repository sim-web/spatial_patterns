__author__ = 'simonweber'
import unittest
import numpy as np
from learning_grids import add_computed

class TestObservables(unittest.TestCase):
	def setUp(self):
		self.add_computed = add_computed.Add_computed()
	def test_keep_only_entries_with_good_grid_score(self):
		minimum_grid_score = 0.5
		gs = np.array([
					[0.2, 0.45],
					[0.7, 0.9],
					[0.4, 1.2]
				])
		angles = np.array([
					[
						[0.11, np.nan, -0.89], [0.1, 1.1, -0.9]],
					[
						[np.nan, 0.87, -1.22], [0.2, 0.8, -1.2]],
					[
						[0.4, 1.4, -1.4], [0., 1, -1]]
				])
		expected = np.array([
					[
						[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
					[
						[np.nan, 0.87, -1.22], [0.2, 0.8, -1.2]],
					[
						[np.nan, np.nan, np.nan], [0., 1, -1]]
				])
		result = self.add_computed.keep_only_entries_with_good_grid_score(
			gs, angles, minimum_grid_score
		)
		np.testing.assert_array_almost_equal(expected, result)

	def test_keep_only_entries_with_good_grid_score_2(self):
		minimum_grid_score = 0.5
		gs = np.array([
					[0.2, 0.45],
					[0.7, 0.9],
					[0.4, 1.2]
				])
		peak_locations = np.array([
					[
						[0.11, -0.89], [0.1, -0.9]],
					[
						[np.nan, -1.22], [0.8, -1.2]],
					[
						[0.4,  -1.4], [0., -1]]
				])
		expected = np.array([
					[
						[np.nan, np.nan], [np.nan, np.nan]],
					[
						[np.nan, -1.22], [0.8, -1.2]],
					[
						[np.nan, np.nan], [0., -1]]
				])
		result = self.add_computed.keep_only_entries_with_good_grid_score(
			gs, peak_locations, minimum_grid_score, size=2
		)
		np.testing.assert_array_almost_equal(expected, result)