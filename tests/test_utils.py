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

	def test_get_boolian_of_positions_in_subquare(self):
		# Positions of shape (3, 3, 2)
		positions = np.array([
			[[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]],
			[[0.6, 0.7], [0.8, 0.9], [1.0, 1.1]],
			[[1.2, 1.3], [1.4, 1.5], [1.6, 1.7]],
		])
		x_limit = 0.8
		y_limit = None
		# Shape of expected array (3, 3)
		expected = np.array([
			[True, True, True],
			[True, True, False],
			[False, False, False],
		])
		result = utils.get_boolian_of_positions_in_subsquare(positions,
															 x_limit,
															 y_limit)
		np.testing.assert_array_equal(expected, result)

	def test_set_values_to_none(self):
		d = {'a0': {'a1': [1, 2, 3]}, 'b0': ['one', 'two']}
		key_lists = [['a0', 'a1'], ['b0']]
		utils.set_values_to_none(d, key_lists)
		self.assertIsNone(d['a0']['a1'])
		self.assertIsNone(d['b0'])

	def test_check_psp_condition(self):
		class Object(object):
			pass

		sim_seed_object = Object()
		sim_seed_object.quantity = 17
		test_object = Object()
		test_object.quantity = 'figure'
		exc_sigma_object = Object()
		exc_sigma_object.quantity = np.array([0.05, 2.0])
		p = {('sim', 'seed'): sim_seed_object, ('visual'): test_object,
			 ('exc', 'sigma'): exc_sigma_object}
		print np.equal('figure', 'asdf')
		condition_tuple1 = (('sim', 'seed'), 'lt', 18)
		condition_tuple2 = (('visual'), 'eq', 'figure')
		condition_tuple3 = (('exc', 'sigma'), 'eq', np.array([0.05, 2.0]))
		result1 = utils.check_conditions(p, condition_tuple1, condition_tuple2,
										 condition_tuple3)
		self.assertTrue(result1)

	def test_get_concatenate_10_minute_trajectories(self):
		order = np.arange(61)
		result = utils.get_concatenated_10_minute_trajectories(order)
		expected = np.load('../data/sargolini_trajectories_610min.npy')
		np.testing.assert_array_equal(result, expected)

	def test_idx2loc(self):
		radius = 0.5
		nbins = 3
		idx = 0
		expected = - radius
		result = utils.idx2loc(idx, radius, nbins)
		self.assertEqual(result, expected)
		idx = nbins - 1
		expected = radius
		result = utils.idx2loc(idx, radius, nbins)
		self.assertEqual(result, expected)
		idx = 1.01
		expected = 0.0
		result = utils.idx2loc(idx, radius, nbins)
		self.assertEqual(result, expected)