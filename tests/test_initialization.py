__author__ = 'simonweber'
import unittest
import numpy as np
from learning_grids import initialization

class TestRat(initialization.Rat):
	def __init__(self):
		self.sargolini_data = np.load('../data/sargolini_trajectories_concatenated.npy')
		self.x, self.y = self.sargolini_data[0]
		self.phi = 0.
		self.velocity_dt = 1e-2
		self.dimensions = 2
		self.radius = 0.5
		self.persistence_length = 0.5
		self.angular_sigma = np.sqrt(2.*self.velocity_dt/self.persistence_length)

class TestInitialization(unittest.TestCase):

	def test_move_sargolini_data(self):
		rat = TestRat()
		old_pos = np.array([rat.x, rat.y])
		np.testing.assert_array_almost_equal(old_pos,
										  [32.28176299, -30.4114696538], 10)
		rat.step = 2
		rat.move_sargolini_data()
		new_pos = np.array([rat.x, rat.y])
		np.testing.assert_array_almost_equal(new_pos,
								np.array([32.0411856817, -30.1708923456]), 10)
