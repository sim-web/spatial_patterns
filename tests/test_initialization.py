__author__ = 'simonweber'
import unittest
import numpy as np
from learning_grids import initialization

class TestSynapses(initialization.Synapses):
    def __init__(self):
        pass


class TestRat(initialization.Rat):
    def __init__(self, load_trajectories=False):
        if load_trajectories:
            self.sargolini_data = np.load(
                '../data/sargolini_trajectories_concatenated.npy')
        else:
            self.sargolini_data = np.array([
                [0.5, 0.5],
                [0.5, 0.5]
            ])
        self.sargolini_norm = 51.6182218615
        self.radius = 0.5
        self.x = self.radius * self.sargolini_data[0][0] / self.sargolini_norm
        self.y = self.radius * self.sargolini_data[0][1] / self.sargolini_norm
        self.phi = 0.
        self.velocity_dt = 1e-2
        self.dimensions = 2
        self.persistence_length = 0.5
        self.angular_sigma = np.sqrt(
            2. * self.velocity_dt / self.persistence_length)


class TestInitialization(unittest.TestCase):

    def test_centers2gridcenters(self):
        synapses = TestSynapses()

        # Example with 3 inputs
        centers = np.array(
            [
                [[-0.4]],
                [[-0.1]],
                [[ 1.1]],
            ]
        )
        # Make it 4 fields with equidistant spacing add 2 to the left and 2
        # to the right. Spacing = 0.2
        expected = np.array(
            [
                [[-0.8], [-0.6], [-0.4], [-0.2], [0.0]],
                [[-0.5], [-0.3], [-0.1], [ 0.1], [0.3]],
                [[ 0.7], [ 0.9], [ 1.1], [ 1.3], [1.5]],
            ]
        )
        result = synapses.centers2gridcenters_1d(centers, gridspacing=0.2,
                                                 n_fields=5)
        np.testing.assert_array_almost_equal(expected, result, decimal=10)

    def test_combine_centers(self):
        """
        We test it for center arrays with 4 input neurons with 3 fields
        per synapse each.
        """
        idx = np.array([1, 3])
        centers1 = np.array(
            [
                [
                    [ 0.17,  0.94],
                    [ 0.32,  0.77],
                    [ 0.71,  0.05]
                ],
                [
                    [ 0.84,  0.34],
                    [ 0.83,  0.38],
                    [ 0.69,  0.27]
                ],
                [
                    [0.70, 0.98],
                    [0.93, 0.10],
                    [0.79, 0.18]],
                [
                    [0.57, 0.51],
                    [0.15, 0.57],
                    [0.85, 0.06]
                ]
            ]
        )

        centers2 = np.array(
            [
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6]
                ],
                [
                    [0.7, 0.8],
                    [0.9, 1.0],
                    [1.1, 1.2]
                ],
                [
                    [1.3, 1.4],
                    [1.5, 1.6],
                    [1.7, 1.8]
                ],
                [
                    [1.9, 2.0],
                    [2.1, 2.2],
                    [2.3, 2.4]
                ]
            ]
        )
        expected = np.array(
            [
                [
                    [0.17, 0.94],
                    [0.32, 0.77],
                    [0.71, 0.05]
                ],
                [
                    [0.7, 0.8],
                    [0.9, 1.0],
                    [1.1, 1.2]
                ],
                [
                    [0.70, 0.98],
                    [0.93, 0.10],
                    [0.79, 0.18]],
                [
                    [1.9, 2.0],
                    [2.1, 2.2],
                    [2.3, 2.4]
                ]
            ]
        )
        synapses = TestSynapses()
        result = synapses.combine_centers(centers1, centers2, alpha=0.5,
                                 idx=idx)
        np.testing.assert_array_almost_equal(expected, result, decimal=6)

    def test_vary_faction_of_field_locations_for_each_neuron(self):
        """
        We test it for center arrays with 4 input neurons with 3 fields
        per synapse each.
        """
        idx = np.array([1])
        centers1 = np.array(
            [
                [
                    [0.17, 0.94],
                    [0.32, 0.77],
                    [0.71, 0.05]
                ],
                [
                    [0.84, 0.34],
                    [0.83, 0.38],
                    [0.69, 0.27]
                ],
                [
                    [0.70, 0.98],
                    [0.93, 0.10],
                    [0.79, 0.18]],
                [
                    [0.57, 0.51],
                    [0.15, 0.57],
                    [0.85, 0.06]
                ]
            ]
        )

        centers2 = np.array(
            [
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6]
                ],
                [
                    [0.7, 0.8],
                    [0.9, 1.0],
                    [1.1, 1.2]
                ],
                [
                    [1.3, 1.4],
                    [1.5, 1.6],
                    [1.7, 1.8]
                ],
                [
                    [1.9, 2.0],
                    [2.1, 2.2],
                    [2.3, 2.4]
                ]
            ]
        )
        expected = np.array(
            [
                [
                    [0.17, 0.94],
                    [0.3, 0.4],
                    [0.71, 0.05]
                ],
                [
                    [0.84, 0.34],
                    [0.9, 1.0],
                    [0.69, 0.27]
                ],
                [
                    [0.70, 0.98],
                    [1.5, 1.6],
                    [0.79, 0.18]],
                [
                    [0.57, 0.51],
                    [2.1, 2.2],
                    [0.85, 0.06]
                ]
            ]
        )
        synapses = TestSynapses()
        result = synapses.vary_fraction_of_field_locations_for_each_neuron(
                centers1, centers2, alpha=0.5, idx=idx)
        np.testing.assert_array_almost_equal(expected, result, decimal=6)



    def test_move_sargolini_data(self):
        rat = TestRat()
        old_pos = np.array([rat.x, rat.y])
        old_compare = (rat.radius
                        * np.array([32.28176299, -30.4114696538])
                        / rat.sargolini_norm)
        np.testing.assert_array_almost_equal(old_pos, old_compare, 10)
        rat.step = 2
        rat.move_sargolini_data()
        new_pos = np.array([rat.x, rat.y])
        # new_compare = (rat.radius
        # 				* np.array([32.0411856817, -30.1708923456])
        # 				/ rat.sargolini_norm)
        # np.testing.assert_array_almost_equal(new_pos, new_compare, 10)

    def test_get_gaussian_process_in_one_dimension(self):
        """
        Tests if the gaussian random field inputs have the right ac-length.

        Note:
        - The gaussian random fields (GRF) should have an
        auto-correlation length of 2*sigma if we define the auto-correlation
        length as the value where the auto-correlation function has
        decayed to 1/e of its maximum value.
        - We take the mean of many randomly created GRFs, because there
        is quite some variance
        - We take a very large radius (much larger than in the simulations)
        because only then do we get good results.
        - If we take a smaler radius, we typically get larger values for
        the auto-correlation length
        """
        radius = 20.
        n = 100
        for sigma in [0.03, 0.1, 0.2, 0.3]:
            print('sigma: {0}'.format(sigma))
            # Typical resolution value in real simulations
            resolution = sigma / 8.
            # Linspace as in simulations
            linspace = np.arange(-radius+resolution, radius, resolution)
            auto_corr_lengths = self.get_auto_corr_lengths_from_n_grf(
                radius=radius, sigma=sigma, linspace=linspace, n=n)
            expected_auto_corr = 2. * sigma # Note: NOT np.sqrt(2) * sigma !!!
            self.assertAlmostEqual(
                np.mean(auto_corr_lengths), expected_auto_corr, delta=sigma/10.)
            self.assertAlmostEqual(
                np.std(auto_corr_lengths)/n, 0., delta=sigma/10.)

    def get_auto_corr_lengths_from_n_grf(self, radius, sigma, linspace, n=100):
        """
        Returns lists of auto-correlation lengths of many GRFs.

        Just a convenience function

        Parameters
        ----------
        n : int
            Number of auto-correlation lengths in the returned list

        Returns

        -------
        auto_corr_lengths : list
        """
        auto_corr_lengths = []
        for seed in np.arange(n):
            np.random.seed(seed)
            gp, gp_min, gp_max = initialization.get_gaussian_process(
                                    radius, sigma, linspace, rescale=True)
            gp_zero_mean = gp - np.mean(gp)
            auto_corr = np.correlate(gp_zero_mean, gp_zero_mean, mode='same')
            auto_corr_max_over_e = np.amax(auto_corr) / np.e
            # Index where autocorrelation decayed to 1/e (roughly)
            idx = (np.abs(auto_corr - auto_corr_max_over_e)).argmin()
            auto_corr_length = np.abs(linspace[idx])
            auto_corr_lengths.append(auto_corr_length)
        return auto_corr_lengths

    def test_variance_of_rates_of_each_input_neuron(self):
        rat = TestRat()

        # In one dimension (spacing 3), with 4 neurons
        input_rates = np.array(
            [
                [0, 1, 0.5, 1],
                [4, 5, 6, 3],
                [7, 7, 9, 7],
            ]
        )
        expected = np.array(
            [
                np.var([0, 4, 7]),
                np.var([1, 5, 7]),
                np.var([0.5, 6, 9]),
                np.var([1, 3, 7]),
            ]
        )
        input_norm = 1
        result = rat.variance_of_rates_of_each_input_neuron(
            input_norm, input_rates, dimensions=1)
        np.testing.assert_array_almost_equal(expected, result, decimal=8)
        # In two dimension (spacing 3x3), with 4 neurons
        input_rates = np.array(
            [
                [
                    [0, 1, 2, 3],
                    [0.5, 0.1, 0.2, 0.4],
                    [5, 5, 6, 5],
                ],
                [
                    [3, 3, 2, 3],
                    [1, 0, 1, 2],
                    [7, 5, 4, 1],
                ],
                [
                    [3, 3, 1, 3],
                    [1, 2, 1, 2],
                    [7, 6, 4, 1],
                ]
            ]
        )
        expected = np.array(
            [
                np.var([0, 0.5, 5, 3, 1, 7, 3, 1, 7]),
                np.var([1, 0.1, 5, 3, 0, 5, 3, 2, 6]),
                np.var([2, 0.2, 6, 2, 1, 4, 1, 1, 4]),
                np.var([3, 0.4, 5, 3, 2, 1, 3, 2, 1])
            ]
        )
        input_norm = 1
        result = rat.variance_of_rates_of_each_input_neuron(
            input_norm, input_rates, dimensions=2)
        np.testing.assert_array_almost_equal(expected, result, decimal=8)

    def test_get_random_numbers(self):
        # Single output neuron, 3**2 many inputs
        n = (1, 9)
        mean = 1
        spreading = 0.5
        distribution = 'single_weight'
        # Something like that is expected
        expected = np.array([
            # [0.95, 0.95, 1.05, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
            [mean, mean, 1.5*mean, mean, mean, mean, mean, mean, mean]
        ])
        ### Single weight ###
        result = initialization.get_random_numbers(n,
                                                   mean,
                                                   spreading,
                                                   distribution,
                                                   selected_weight=2)
        np.testing.assert_array_equal(expected, result)
