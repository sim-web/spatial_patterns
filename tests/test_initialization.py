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
