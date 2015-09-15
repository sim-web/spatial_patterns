import numpy as np

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
			symmetric_fields = np.all(self.twoSigma2[..., 0] == self.twoSigma2[..., 1])

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
					rates = (
						np.sum(
							self.gaussian_height * np.exp(
								-np.power(
									position-self.centers, 2)
								*self.twoSigma2),
						axis=axis))

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
			# where len(position) > 2. For these grids we need to some along axis 4.
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
			if len(position) > 3:
				axis = 4
			else:
				axis = 1

			def get_rates(position):
				rates = (
					np.sum(
						np.exp(
							-np.power(
								position[...,0] - self.centers[...,0], 2)
							*self.twoSigma2[..., 0]
							-np.power(
								position[...,1] - self.centers[...,1], 2)
							*self.twoSigma2[..., 1])
						* self.norm_von_mises[...,-1]
						* np.exp(
							self.scaled_kappas[..., -1]
							* np.cos(
								self.pi_over_r*(position[...,2]
								- self.centers[...,2]))
							),
					axis=axis)
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

# def get_noisy_array(value, noise, size):
# 	"""Returns array with uniformly distributed values in range [value-noise, value+noise]
	
# 	Parameters
# 	----------
	
# 	Returns
# 	-------
	
# 	"""
# 	retunr np.uniform(value-noise, value+noise, )
# 	return ((1 + noise * (2 * np.random.random_sample(size) - 1)) * value)
