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
        # Symmetric region size
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
        # Asymmetric region size (2, 3)
        a = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 0.0, 10., 11.],
            [12., 13., 14., 15.],
            [16., 17., 18., 19.],
            [20., 21., 22., 23.]
        ])
        b = np.array([
            [0.0, 0.0, 0.5, 0.6],
            [0.0, 0.0, 0.7, 0.8],
            [0.0, 0.0, 0.9, 1.0],
            [0.0, 0.0, 1.1, 2.0],
            [0.0, 0.0, 2.1, 2.2],
            [0.0, 0.0, 3.1, 3.2]
        ])
        expected = np.array([
            [np.nan, 0.9894757],
            [np.nan, 0.94789182]
        ])
        result = self.plot.get_correlation_in_regions(a, b, region_size=(3, 2))
        np.testing.assert_array_almost_equal(expected, result, decimal=7)
        # Asymmetric region size (2, 3) from symmetric array
        a = np.array([
            [0.0, 1.0, 2.0, 3.0, 3.1, 3.2  ],
            [4.0, 5.0, 6.0, 7.0, 7.1, 7.2  ],
            [8.0, 0.0, 10., 11., 11.1, 11.2],
            [12., 13., 14., 15., 15.1, 15.2],
            [16., 17., 18., 19., 19.1, 19.2],
            [20., 21., 22., 23., 23.1, 23.2]
        ])
        b = np.array([
            [0.0, 0.0, 0.5, 0.6, 0.61, 0.62],
            [0.0, 0.0, 0.7, 0.8, 0.81, 0.82],
            [0.0, 0.0, 0.9, 1.0, 1.10, 1.20],
            [0.0, 0.0, 1.1, 2.0, 2.10, 2.20],
            [0.0, 0.0, 2.1, 2.2, 2.21, 2.22],
            [0.0, 0.0, 3.1, 3.2, 3.21, 3.22]
        ])
        expected = np.array([
            [np.nan, 0.9894757, 0.98229246],
            [np.nan, 0.94789182, 0.8908921]
        ])
        result = self.plot.get_correlation_in_regions(a, b, region_size=(3, 2))
        np.testing.assert_array_almost_equal(expected, result, decimal=7)

    def test_slices_into_subarrays(self):
        step = 3
        end = 12
        expected = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)]
        result = self.plot.slices_into_subarrays(step, end)
        self.assertEqual(expected, result)

    def test_concatenate_left_and_right_ratemap(self):
        # Symmetric region size
        a = np.array([
            [0.0, 1.0],
            [4.0, 5.0],
            [8.0, 0.0],
            [12., 13.]
        ])
        b = np.array([
            [0.5, 0.6],
            [0.7, 0.8],
            [0.9, 1.0],
            [1.1, 2.0]
        ])
        expected = np.array([
            [0.0, 1.0, 0.5, 0.6],
            [4.0, 5.0, 0.7, 0.8],
            [8.0, 0.0, 0.9, 1.0],
            [12., 13., 1.1, 2.0]
        ])
        result = self.plot.concatenate_left_and_right_ratemap(a, b)
        np.testing.assert_array_almost_equal(expected, result, decimal=8)