__author__ = 'simonweber'
import unittest
import numpy as np
from learning_grids import initialization


class TestRat(initialization.Rat):
	def __init__(self):
		self.sargolini_data = np.load(
			'../data/sargolini_trajectories_concatenated.npy')
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
		new_compare = (rat.radius
						* np.array([32.0411856817, -30.1708923456])
						/ rat.sargolini_norm)
		np.testing.assert_array_almost_equal(new_pos, new_compare, 10)

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
			print 'sigma: {0}'.format(sigma)
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
