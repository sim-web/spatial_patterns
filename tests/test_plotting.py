__author__ = 'simonweber'
import unittest
import numpy as np
from learning_grids import plotting

class TestPlotting(unittest.TestCase):
	def setUp(self):
		self.plot = plotting.Plot()

	def test_get_spiketimes(self):
		firing_rates = np.array([0.4, 0.0, 8.0, 0.0, 0.0, 5.0])
		rate_factor = 20
		random_numbers = np.array([1.0, 0.8, 5.0, 0.3, 4.0, 3.0]) / rate_factor
		dt = 0.01
		expected = [0.02, 0.05]
		result = self.plot.get_spiketimes(firing_rates,
										  dt,
										  rate_factor,
										  random_numbers=random_numbers)
		np.testing.assert_array_equal(expected, result)

	def test_get_correlation_in_regions(self):
		a = np.array([
			[0.0, 1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0, 7.0],
			[8.0, 0.0, 10., 11.],
			[12., 13., 14., 15.]
		])
		b = np.array([
			[0.0, 0.0, 0.5, 0.6],
			[0.0, 0.0, 0.7, 0.8],
			[0.0, 0.0, 0.9, 1.0],
			[0.0, 0.0, 1.1, 2.0]
		])
		expected = np.array([
			[np.nan, 0.97618706],
			[np.nan, 0.80154549]
		])
		result = self.plot.get_correlation_in_regions(a, b, region_size=(2, 2))
		np.testing.assert_array_almost_equal(expected, result, decimal=7)

	def test_slices_into_subarrays(self):
		step = 3
		end = 12
		expected = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)]
		result = self.plot.slices_into_subarrays(step, end)
		self.assertEqual(expected, result)
