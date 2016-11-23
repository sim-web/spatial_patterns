import numpy as np
import sys
import operator
import os
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt

class Utilities:
	"""
	Currently we have just placed two function here

	get_output_rates_from_equation was copied from the Rat class in
	initialization

	get_rates_function was copied from the Synapses class in initialization

	The reason is that we use these functions again in plotting.py, but
	we want to prevent cycllic imports.
	In principle both of these functions need to be divided into subfunctions
	to increase readability.
	Also they should be changed such that they work on their own, which
	they currently don't.
	"""

	def get_output_rates_from_equation(self, frame, rawdata, spacing,
		positions_grid=False, input_rates=False, equilibration_steps=10000):
		"""	Return output rates at many positions

		***
		Note:
		This function used to be in plotting.py, but now it is used here
		to output arrays containing the output rates. This makes
		quick plotting and in particular time traces of Grid Scores feasible.
		***

		For normal plotting in 1D and for contour plotting in 2D.
		It is differentiated between cases with and without lateral inhibition.

		With lateral inhibition the output rate has to be determined via
		integration (but fixed weights).
		In 1 dimensions we start at one end of the box, integrate for a
		time specified by equilibration steps and than walk to the
		other end of the box.

		Parameters
		----------
		frame : int
			Frame at which the output rates are plotted
		rawdata : dict
			Contains the synaptic weights
		spacing : int
			The spacing, describing the detail richness of the plor or contour plot (spacing**2)
		positions_grid, input_rates : ndarray
			Arrays as described in get_X_Y_positions_grid_input_rates_tuple
		equilibration_steps : int
			Number of steps of integration to reach the correct
			value of the output rates for the case of lateral inhibition

		Returns
		-------
		output_rates : ndarray
			Array with output rates at several positions tiling the space
			For 1 dimension with shape (spacing)
			Fro 2 dimensions with shape (spacing, spacing)
		"""

		# plt.title('output_rates, t = %.1e' % (frame * self.every_nth_step_weights), fontsize=8)
		if self.dimensions == 1:
			linspace = np.linspace(-self.radius, self.radius, spacing)

			if self.lateral_inhibition:
				output_rates = np.empty((spacing, self.output_neurons))

				start_pos = -self.radius
				end_pos = self.radius
				r = np.zeros(self.output_neurons)
				dt_tau = self.dt / self.tau
				# tau = 0.011
				# dt = 0.01
				# dt_tau = 0.1
				x = start_pos
				for s in np.arange(equilibration_steps):
					r = (
							r*(1 - dt_tau)
							+ dt_tau * ((
							np.dot(rawdata['exc']['weights'][frame],
								input_rates['exc'][0]) -
							np.dot(rawdata['inh']['weights'][frame],
								input_rates['inh'][0])
							)
							- self.weight_lateral
							* (np.sum(r) - r)
							)
							)
					r[r<0] = 0
				start_r = r
				# output_rates = []
				for n, x in enumerate(linspace):
					for s in np.arange(200):
						r = (
								r*(1 - dt_tau)
								+ dt_tau * ((
								np.dot(rawdata['exc']['weights'][frame],
									input_rates['exc'][n]) -
								np.dot(rawdata['inh']['weights'][frame],
									input_rates['inh'][n])
								)
								- self.weight_lateral
								* (np.sum(r) - r)
								)
								)
						r[r<0] = 0
					output_rates[n] = r

			else:
				output_rates = (
					np.tensordot(rawdata['exc']['weights'][frame],
										input_rates['exc'], axes=([-1], [1]))
					- np.tensordot(rawdata['inh']['weights'][frame],
						 				input_rates['inh'], axes=([-1], [1]))
				)
				output_rates = output_rates
			output_rates[output_rates<0] = 0
			output_rates = output_rates.reshape(spacing, self.output_neurons)
			return output_rates

		if self.dimensions >= 2:
			if self.lateral_inhibition:
				output_rates = np.empty((spacing, spacing, self.output_neurons))
				start_pos = positions_grid[0, 0, 0, 0]
				r = np.zeros(self.output_neurons)
				dt_tau = self.dt / self.tau

				pos = start_pos
				for s in np.arange(equilibration_steps):
					r = (
							r*(1 - dt_tau)
							+ dt_tau * ((
							np.dot(rawdata['exc']['weights'][frame],
								input_rates['exc'][0][0]) -
							np.dot(rawdata['inh']['weights'][frame],
								input_rates['inh'][0][0])
							)
							- self.weight_lateral
							* (np.sum(r) - r)
							)
							)
					r[r<0] = 0
				# start_r = r
				# print r
				# output_rates = []

				for ny in np.arange(positions_grid.shape[1]):
					for nx in np.arange(positions_grid.shape[0]):
						pos = positions_grid[nx][ny]
						for s in np.arange(200):
							r = (
									r*(1 - dt_tau)
									+ dt_tau * ((
									np.dot(rawdata['exc']['weights'][frame],
										input_rates['exc'][nx][ny]) -
									np.dot(rawdata['inh']['weights'][frame],
										input_rates['inh'][nx][ny])
									)
									- self.weight_lateral
									* (np.sum(r) - r)
									)
									)
							r[r<0] = 0

						output_rates[nx][ny] = r

				# for i in np.arange(self.output_neurons):
				# 	output_rates[:,:,i] = np.transpose(output_rates[:,:,i])

			else:
				output_rates = (
					np.tensordot(rawdata['exc']['weights'][frame],
										input_rates['exc'], axes=([-1], [self.dimensions]))
					- np.tensordot(rawdata['inh']['weights'][frame],
						 				input_rates['inh'], axes=([-1], [self.dimensions]))
				)
				# Rectification
				output_rates[output_rates < 0] = 0.
				if self.dimensions == 2:
					output_rates = output_rates.reshape(
									spacing, spacing, self.output_neurons)
				elif self.dimensions == 3:
					output_rates = output_rates.reshape(
									spacing, spacing, spacing, self.output_neurons)
			return output_rates

	def get_rates_function(self, position, data=False, params=False):
		"""Returns function which computes values of place field Gaussians at <position>.

		Depending on the parameters and the desired simulation, the rates need to
		be set by a different function. To prevent the conditionals from ocurring
		in each time step, this function returns a function which is ready for later
		usage.

		Note that the outer most sum in all these functions is over the
		fields per synapse.

		If position is an array of positions (so of length > 2) and not a
		single position, an array of rates at all the given positions is
		returned. This is useful for plotting.

		Ocurring arrays (example in two dimensions for excitation):
		centers: shape = (n_exc, fields_per_synapse_exc, 2)
		twoSigma2: shape = (n_exc, field_per_synapse_exc, 2)


		Parameters
		----------
		position: (ndarray) [x, y]
			This can either be just [x,y] with x and y being floats,
			or a large array.
			For example in two dimensions it would be of shape
			(spacing, spacing, 1, 2)
			Note that spacing can either be the spacing of the
			output_rates_grid or the spacing of the space discretization.
		data: e.g. rawdata['exc']

		Returns
		-------
		get_rates : function
			Function which takes `position` as an argument and returns
			the input rate at either the specific location, if `position`
			is just [x,y] or the input rates at every location specified
			in a grid. Then the output has shape (spacing, spacing, n), where
			n is the number of input cells (either excitatory or inhibitory)
		"""
		# If data is given (used for plotting), then the values from the data are chosen
		# instead of the ones inside this class
		if data:
			for k, v in data.iteritems():
				setattr(self, k, v)
		if params:
			for k, v in params.iteritems():
				setattr(self, k, v)

		# Set booleans to choose the desired functions for the rates
		if self.dimensions == 2:
			# try:
			symmetric_fields = np.all(self.twoSigma2[..., 0] == self.twoSigma2[..., 1])
			# except AttributeError as e:
			# 	symmetric_fields = (
			# 		self.params['inh']['sigma'][0]
			# 		== self.params['inh']['sigma'][1])
			# 	print e
			# 	print 'WARNING: twoSigma2 was not stored. Thus symmetric fields' \
			# 		  ' are assumed!'
		# Nice way to ensure downward compatibility
		# The attribute 'tuning_function' is new and if it had existed before
		# it would have been 'gaussian' always, so we make this the default
		# in case it doesn't exist.
		self.tuning_function = getattr(self, 'tuning_function', 'gaussian')

		if self.dimensions == 1:
			if len(np.atleast_1d(position)) > 2:
				axis = 2
			else:
				axis = 1

			if self.tuning_function == 'gaussian':
				# The outer most sum is over the fields per synapse
				def get_rates(position):
					shape = (position.shape[0], self.number)
					rates = np.zeros(shape)
					for i in np.arange(self.fields_per_synapse):
						rates += (
								np.exp(
								-np.power(position[:,:,0] - self.centers[:,i,0], 2)
								*self.twoSigma2[:,i])
						)
					return self.input_norm * rates

			elif self.tuning_function == 'lorentzian':
				def get_rates(position):
					# gammas = np.sqrt(2*np.log(2)) * self.sigmas
					gammas = self.sigmas
					rates = (
						np.sum(
							1. / ((1 + ((position-self.centers)/gammas)**2)),
						axis=axis))
					return self.input_norm * rates

		if self.dimensions == 2:
			# For contour plots we pass grids with many positions
			# where len(position) > 2. For these grids we need to sum along axis 4.
			if len(position) > 2:
				axis = 3
			else:
				axis = 1

			if self.tuning_function == 'gaussian':
				if symmetric_fields:
					def get_rates(position):
						shape = (position.shape[0], position.shape[1], self.number)
						rates = np.zeros(shape)
						for i in np.arange(self.fields_per_synapse):
							rates += (
									np.exp(
									-np.sum(
										np.power(position - self.centers[:,i,:], 2),
									axis=axis)
									*self.twoSigma2[:, i, 0]))
						return self.input_norm * rates
				# For band cell simulations
				elif not symmetric_fields:
					def get_rates(position):
						shape = (position.shape[0], position.shape[1], self.number)
						rates = np.zeros(shape)
						for i in np.arange(self.fields_per_synapse):
							rates += (
									np.exp(
										-np.power(
											position[..., 0] - self.centers[:, i, 0], 2)
										*self.twoSigma2[:, i, 0]
										-np.power(
											position[..., 1] - self.centers[:, i, 1], 2)
										*self.twoSigma2[:, i, 1]
										)
									)
						return rates

			elif self.tuning_function == 'lorentzian':
				def get_rates(position):
					shape = (position.shape[0], position.shape[1], self.number)
					rates = np.zeros(shape)
					gammas = self.sigmas
					for i in np.arange(self.fields_per_synapse):
						rates += (
							1. / (
									np.power(1 +
									np.power((position[..., 0] - self.centers[:, i, 0]) / gammas[:, i, 0], 2) +
									np.power((position[..., 1] - self.centers[:, i, 1])/ gammas[:, i, 1], 2)
									, 1.5)
								 )
							)
					return self.input_norm * rates

			elif self.tuning_function == 'von_mises':
				def get_rates(position):
					shape = (position.shape[0], position.shape[1], self.number)
					rates = np.zeros(shape)
					for i in np.arange(self.fields_per_synapse):
						rates += (
								np.exp(
									-np.power(
										position[...,0] - self.centers[:, i, 0], 2)
									*self.twoSigma2[:, i, 0]
									)
								* self.norm_von_mises[..., i, 1]
								* np.exp(
									self.scaled_kappas[...,i, 1]
									* np.cos(
										self.pi_over_r*(position[...,1]
										- self.centers[:, i, 1]))
									)
						)
					return rates

			elif self.tuning_function == 'periodic':
				def get_rates(position):
					shape = (position.shape[0], position.shape[1], self.number)
					rates = np.zeros(shape)
					for i in np.arange(self.fields_per_synapse):
						rates += (
								self.norm_von_mises[..., i, 0]
								* np.exp(
									self.scaled_kappas[...,i, 0]
									* np.cos(
										self.pi_over_r*(position[...,0]
										- self.centers[:, i, 0]))
									)
								* self.norm_von_mises[..., i, 1]
								* np.exp(
									self.scaled_kappas[...,i, 1]
									* np.cos(
										self.pi_over_r*(position[...,1]
										- self.centers[:, i, 1]))
									)
						)
					return rates

		if self.dimensions == 3:
			def get_rates(position):
				shape = (position.shape[0], position.shape[1], position.shape[2],
						 self.number)
				rates = np.zeros(shape)
				for i in np.arange(self.fields_per_synapse):
					rates += (
							np.exp(
								-np.power(
									position[..., 0] - self.centers[:, i, 0], 2)
								*self.twoSigma2[:, i, 0]
								-np.power(
									position[..., 1] - self.centers[:, i, 1], 2)
								*self.twoSigma2[:, i, 1])
							* self.norm_von_mises[:, i, -1]
							* np.exp(
								self.scaled_kappas[:, i, -1]
								* np.cos(
									self.pi_over_r*(position[..., 2]
									- self.centers[:, i, 2]))
								)
					)
				return rates
		return get_rates


def psp2params(psp):
	params = {}
	for k, v in psp.iteritems():
		params[k[1]] = v
	return params

def rectify(value):
	"""
	Rectification of Firing Rates
	"""
	if value < 0:
		value = 0.
	return value

def rectify_array(array):
	"""
	Rectification of array entries using fancy indexing
	"""
	array[array < 0] = 0.
	return array

def set_values_to_none(d, key_lists):
	"""
	Sets selected values of a dictionary to None

	Parameters
	----------
	d : dict

	key_lists : list
		List of list of keys

	Example
	-------
	d = {'a0': {'a1': [1, 2, 3]}, 'b0': ['one', 'two']}
	key_lists = [['a0', 'a1'], ['b0']]
	set_values_to_none(d, key_lists)
	Results in: {'a0': {'a1': None}, 'b0': None}
	"""
	for kl in key_lists:
		if len(kl) == 1:
			d[kl[0]] = None
		elif len(kl) == 2:
			d[kl[0]][kl[1]] = None
		elif len(kl) == 3:
			d[kl[0]][kl[1]][kl[2]] = None
		elif len(kl) > 3:
			sys.exit('ERROR: function not defined for key_lists larger than 3')

def check_conditions(p, *condition_tuples):
	"""
	Check if paramspace point fullfils conditions

	Parameters
	----------
	p : paramspace point object (see snep)
	condition_tuples : tuple
		Example:
			('sim', 'seed_centers'), 'lt', 10)
			See also tests.test_utils

	Returns
	-------
	Bool
	"""
	for t in condition_tuples:
		parameter = t[0]
		given_operator = t[1]
		value_to_compare_with = np.atleast_1d(t[2])
		oper = getattr(operator, given_operator)
		# Loop over all elements of a parameter
		# Necessary for example for ('exc', 'sigma') which can have
		# 1, 2 or 3 elements.
		for i in np.arange(len(value_to_compare_with)):
			value = np.atleast_1d(p[parameter].quantity)
			if oper(value[i], value_to_compare_with[i]):
				pass
			else:
				return False
	return True

def real_trajectories_from_data(data,
								save_path,
								plot_n_steps=None):
	"""
	Extract the trajectories from experimental data

	The data is between -50 and 50 but sometimes exceed the
	boundaries. We normalize it beteween -0.5 and 0.5 along both
	dimensions.
	
	Parameters
	----------

	
	
	Returns
	-------
	"""

	main_data_dir = '/Users/simonweber/doktor/Data/'
	data_dir = os.path.join(main_data_dir,
				'Sargolini_2006/8F6BE356-3277-475C-87B1-C7A977632DA7_1/')
	if data == 'sargolini_70min':
		filenames = [
				'11084-03020501_t2c1.mat', '11084-10030502_t1c1.mat',
				'11138-11040509_t5c1.mat', '11207-11060502_t6c2.mat',
				'11207-16060501_t7c1.mat',
				# '11207-21060503_t8c1.mat', # Dropped because it contains outliers
				'11207-27060501_t1c3.mat', '11343-08120502_t8c2.mat']
	elif data == 'sargolini_all':
		# These are all the files from the sargolini data
		# Some of them contain NaNs. We skip thoes.
		filenames = [
				'11084-03020501_t2c1.mat',
				'11084-10030502_t1c1.mat',
				'11138-11040509_t5c1.mat',
				'11207-11060502_t6c2.mat',
				'11207-16060501_t7c1.mat',
				'11207-27060501_t1c3.mat',
				'11343-08120502_t8c2.mat',
				'all_data/10073-17010302_POS.mat',
				# 'all_data/10884-14070405_POS.mat',
				# 'all_data/11025-20050501_POS.mat',
				# 'all_data/11138-25040501_POS.mat',
				'all_data/11207-21060503_POS.mat',
				'all_data/10697-02030402_POS.mat',
				'all_data/10884-16070401_POS.mat',
				# 'all_data/11084-01030503_POS.mat',
				# 'all_data/11138-26040501_POS.mat',
				'all_data/11207-23060501_POS.mat',
				'all_data/10697-24020402_POS.mat',
				# 'all_data/10884-19070401_POS.mat',
				# 'all_data/11084-02030502_POS.mat',
				# 'all_data/11207-03060501+02_POS.mat',
				# 'all_data/11207-24060501+02_POS.mat',
				'all_data/10704-06070402_POS.mat',
				# 'all_data/10884-21070405_POS.mat',
				'all_data/11084-03020501_POS.mat',
				# 'all_data/11207-03060501_POS.mat',
				'all_data/11207-27060501_POS.mat',
				# 'all_data/10704-07070402_POS.mat',
				'all_data/10884-24070401_POS.mat',
				# 'all_data/11084-08030506_POS.mat',
				# 'all_data/11207-04070501+02_POS.mat',
				# 'all_data/11207-27060504+05_POS.mat',
				'all_data/10704-07070407_POS.mat',
				'all_data/10884-31070404_POS.mat',
				'all_data/11084-09030501_POS.mat',
				# 'all_data/11207-05070501_POS.mat',
				'all_data/11207-30060501_POS.mat',
				'all_data/10704-08070402_POS.mat',
				'all_data/10938-08100401_POS.mat',
				# 'all_data/11084-09030503_POS.mat',
				# 'all_data/11207-06070501+02_POS.mat',
				# 'all_data/11207-30060503+04_POS.mat',
				# 'all_data/10704-19070402_POS.mat',
				'all_data/10938-12100406_POS.mat',
				'all_data/11084-10030502_POS.mat',
				# 'all_data/11207-07070501+02_POS.mat',
				# 'all_data/11265-01020602_POS.mat',
				# 'all_data/10704-20060402_POS.mat',
				'all_data/10938-12100410_POS.mat',
				'all_data/11084-23020502_POS.mat',
				'all_data/11207-08060501_POS.mat',
				'all_data/11265-02020601_POS.mat',
				# 'all_data/10704-20070402_POS.mat',
				'all_data/10962-27110403_POS.mat',
				'all_data/11084-24020502_POS.mat',
				# 'all_data/11207-08070504+05_POS.mat',
				'all_data/11265-03020601_POS.mat',
				'all_data/10704-23060402_POS.mat',
				'all_data/10962-28110402_POS.mat',
				'all_data/11084-28020501_POS.mat',
				# 'all_data/11207-09060501+02_POS.mat',
				'all_data/11265-06020601_POS.mat',
				'all_data/10704-25060402_POS.mat',
				'all_data/10962-28110406_POS.mat',
				'all_data/11138-05040502_POS.mat',
				# 'all_data/11207-09070501_POS.mat',
				'all_data/11265-07020602_POS.mat',
				'all_data/10704-26060402_POS.mat',
				'all_data/10962-29110404_POS.mat',
				'all_data/11138-06040501_POS.mat',
				# 'all_data/11207-09070505+06_POS.mat',
				# 'all_data/11265-09020601_POS.mat',
				# 'all_data/10884-01080402_POS.mat',
				'all_data/11016-02020502_POS.mat',
				'all_data/11138-06040507_POS.mat',
				'all_data/11207-10070501_POS.mat',
				'all_data/11265-13020601_POS.mat',
				# 'all_data/10884-02080405_POS.mat',
				# 'all_data/11016-25010501_POS.mat',
				'all_data/11138-07040501_POS.mat',
				# 'all_data/11207-10070503+04_POS.mat',
				# 'all_data/11265-16030601+02_POS.mat',
				# 'all_data/10884-03080402_POS.mat',
				# 'all_data/11016-28010501_POS.mat',
				# 'all_data/11138-08040501_POS.mat',
				'all_data/11207-11060501_POS.mat',
				# 'all_data/11265-16030604+05_POS.mat',
				# 'all_data/10884-03080405_POS.mat',
				'all_data/11016-29010503_POS.mat',
				'all_data/11138-11040501_POS.mat',
				'all_data/11207-11060502_POS.mat',
				# 'all_data/11265-31010601_POS.mat',
				'all_data/10884-03080409_POS.mat',
				# 'all_data/11016-31010502_POS.mat',
				# 'all_data/11138-11040504+05_POS.mat',
				# 'all_data/11207-11060503+04_POS.mat',
				# 'all_data/11278-30080505_POS.mat',
				# 'all_data/10884-04080402_POS.mat',
				'all_data/11025-01060511_POS.mat',
				# 'all_data/11138-11040509_POS.mat',
				# 'all_data/11207-14060501_POS.mat',
				# 'all_data/11278-31080502_POS.mat',
				# 'all_data/10884-05080401_POS.mat',
				# 'all_data/11025-06050501+02_POS.mat',
				# 'all_data/11138-12110501_POS.mat',
				'all_data/11207-16060501_POS.mat',
				# 'all_data/11340-01120501_POS.mat',
				'all_data/10884-08070402_POS.mat',
				# 'all_data/11025-11040501+02_POS.mat',
				'all_data/11138-13040502_POS.mat' ,
				# 'all_data/11207-17060501+02_POS.mat',
				# 'all_data/11340-22110501_POS.mat',
				# 'all_data/10884-08070405_POS.mat',
				# 'all_data/11025-14050501+02_POS.mat',
				# 'all_data/11138-15040504+05_POS.mat',
				# 'all_data/11207-18060501+02_POS.mat',
				# 'all_data/11340-25110501_POS.mat',
				# 'all_data/10884-09080404_POS.mat',
				# 'all_data/11025-16050501+02_POS.mat',
				'all_data/11138-19040502_POS.mat',
				# 'all_data/11207-20060501_POS.mat',
				'all_data/11343-08120502_POS.mat',
				'all_data/10884-13070402_POS.mat',
				'all_data/11025-19050503_POS.mat',
				# 'all_data/11138-20040502_POS.mat',
				# 'all_data/11207-21060501+02_POS.mat',

		]
		x_label = 'posx'
		y_label = 'posy'
	print len(filenames)
	x_positions = []
	y_positions = []
	z_positions = []
	counter = 0
	for i, filename in enumerate(filenames):
		a = sio.loadmat(os.path.join(data_dir, filename))
		print len(a)
		# They named the coordinates differently in the selected
		# data and in the unselected data. Here we make this diffentiation
		# in the label
		if 'all_data' in filename:
			x_label = 'posx'
			y_label = 'posy'
			x2_label = 'posx2'
			y2_label = 'posy2'
		else:
			x_label = 'x1'
			y_label = 'y1'
			x2_label = 'x2'
			y2_label = 'y2'

		if len(a[x2_label]) > 0 and len(a[y2_label]) > 0:
			max_x_y_x2_y2_min_x_y_x2_y2 = np.array([
				np.amax(a[x_label]), np.amax(a[y_label]),
				np.amax(a[x2_label]), np.amax(a[y2_label]),
				np.amin(a[x_label]), np.amin(a[y_label]),
				np.amin(a[x2_label]), np.amin(a[y2_label])
			])
			abs_max = np.array([np.amax(np.abs(a[x_label])),
								np.amax(np.abs(a[y_label]))])


			# print filename
			# print max_x_y_min_x_y
			deviation_too_large = np.any(np.abs(max_x_y_x2_y2_min_x_y_x2_y2) > 52.)
			# This conditional is now not needed, because I manually outcommented the
			# erraneous data files in the list above
			# if not np.isnan(max_x_y_min_x_y).any() and not deviation_too_large:

			if not np.isnan(max_x_y_x2_y2_min_x_y_x2_y2).any():
				counter += 1
				x_positions_this_file = a[x_label][:, 0] * 0.5 / abs_max[0]
				y_positions_this_file = a[y_label][:, 0] * 0.5 / abs_max[1]
				x2_positions_this_file = a[x2_label][:, 0] * 0.5 / abs_max[0]
				y2_positions_this_file = a[y2_label][:, 0] * 0.5 / abs_max[1]
				z_positions_this_file =  np.arctan2(
					y2_positions_this_file-y_positions_this_file,
					x2_positions_this_file-x_positions_this_file) * 0.5 / np.pi
				x_positions.append(x_positions_this_file)
				y_positions.append(y_positions_this_file)
				z_positions.append(z_positions_this_file)


	print 'Trajectory duration: {0} minutes'.format(counter*10)

	x_positions = np.hstack(x_positions)
	y_positions = np.hstack(y_positions)
	z_positions = np.hstack(z_positions)
	test = np.zeros(len(x_positions), dtype=[('x', float), ('y', float), ('z', float)])
	test['x'] = x_positions
	test['y'] = y_positions
	test['z'] = z_positions

	np.save(save_path, test)

	if plot_n_steps:
		test = np.load(save_path)
		plt.scatter(test['x'][:plot_n_steps], test['y'][:plot_n_steps])
		plt.show()

# def get_noisy_array(value, noise, size):
# 	"""Returns array with uniformly distributed values in range [value-noise, value+noise]
	
# 	Parameters
# 	----------
	
# 	Returns
# 	-------
	
# 	"""
# 	retunr np.uniform(value-noise, value+noise, )
# 	return ((1 + noise * (2 * np.random.random_sample(size) - 1)) * value)


if __name__ == '__main__':
	real_trajectories_from_data(data='sargolini_all',
								save_path='data/sargolini_trajectories_610min_incl_head_direction.npy',
								plot_n_steps=10000)