__author__ = 'simonweber'

import unittest
import numpy as np
from learning_grids import utils


class TestObservables(unittest.TestCase):

	# def setUp(self):
	# 	# unittest.TestCase.__init__()
	# 	seven_by_seven = np.array([
	# 		[0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
	# 		[0.1, 0.3, 0.0, 0.0, 0.0, 0.4, 0.0],
	# 		[0.2, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0],
	# 		[0.0, 0.0, 0.0, 0.9, 0.5, 0.0, 0.0],
	# 		[0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.1],
	# 		[0.3, 0.4, 0.4, 0.0, 0.2, 0.0, 0.1],
	# 		[0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.1]
	# 	])
	# 	self.gridness = observables.Gridness(a=seven_by_seven, radius=0.5,
	# 									threshold_difference=0.1,
	# 									method='sargolini', n_contiguous=3)

	def test_set_values_to_none(self):
		d = {'a0': {'a1': [1, 2, 3]}, 'b0': ['one', 'two']}
		key_lists = [['a0', 'a1'], ['b0']]
		utils.set_values_to_none(d, key_lists)
		self.assertIsNone(d['a0']['a1'])
		self.assertIsNone(d['b0'])