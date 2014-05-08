import numpy as np
import general_utils.arrays

class Attractor():
	"""docstring for Attractor"""
	def __init__(self, params):
		self.params = params
		for k, v in params['sim'].items():
			setattr(self, k, v)
		self.N = self.sidelength**2

	def set_distances(self):
		filename = 'dist' + self.sidelength + '.npy'
		try:
			self.distances = np.load(filename)
		except IOError:
			self.distances = general.utils.arrays.get_distances(self.sidelength)
			np.save(filename, self.distances)

	def set_weights(self, gamma, beta, model='Burak'):
		if model == 'Burak':
			self.W_EE = self.factor_GABA * (
					np.exp(- gamma * self.distances**2)
					- np.exp(- beta * self.distances**2)
					) 

	def get_increment(self, r_E, r_I=None, model='Burak'):
		if model=='Burak':
			W_EE_dot_r_E = np.dot(self.W_EE,r_E)
			W_EE_dot_r_E[W_EE_dot_r_E<(-self.external_current)] \
										= self.external_current
			return self.dt/self.tau * (- r_E + W_EE_dot_r_E
										+ self.external_current)

	def run(self):
		n_time_steps = int(self.simulation_time/self.dt)
		
		r_E = np.random(self.N)/10.
		# r_I = np.random(self.N)/10.
		# rates = []
		# activity = sum(r_E)

		rawdata = {}
		rawdata['rate_exc'] = np.empty(np.ceil(
								n_time_steps / self.every_nth_step), self.N)
		rawdata['rate_exc'][0] = r_E
		for step in np.arange(n_time_steps):
			r_E += self.get_increment(r_E)
			if step % self.every_nth_step == 0:
				index = step / self.every_nth_step
				print 'step %i out of %i' % (step, n_time_steps)
				rawdata['rate_exc'][index] = r_E

		return rawdata
			# NORMALIZATION
			# current_activity = sum(r_E)
			# Multiplicative normalization
			# r_E *= activity/current_activity
			# Substractive Multiplication
			# r_E -= (current_activity-activity)/N
			# r_E[r_E<0] = 0
			# r_E[r_E>1] = 1
			# rates.append(r_E.copy())
