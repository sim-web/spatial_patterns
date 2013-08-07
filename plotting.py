import matplotlib as mpl
import math
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import initialization
import utils
mpl.rcParams.update({'figure.autolayout': True})
# print mpl.rcParams.keys()
# mpl.rcParams['animation.frame_format'] = 'jpeg'
# print mpl.rcParams['animation.frame_format']

def plot_list(fig, plot_list):
	"""
	Takes a list of lambda forms of plot functions and plots them such that
	no more than four rows are used to assure readability
	"""
	n_plots = len(plot_list)
	for n, p in enumerate(plot_list, start=1):
		if n_plots < 4:
			fig.add_subplot(n_plots, 1, n)
		else:
			fig.add_subplot(math.ceil(n_plots/2.), 2, n)
		p()

# def set_rates(self, position):
# 	"""
# 	Computes the values of all place field Gaussians at <position>

# 	Future Tasks:
# 		- 	Maybe do not normalize because the normalization can be put into the
# 			weights anyway
# 		- 	Make it work for arbitrary dimensions
# 	"""
# 	if self.dimensions == 1:
# 		rates = self.norm*np.exp(-np.power(position - self.centers, 2)*self.twoSigma2)
# 		return rates

def set_current_output_rate(self):
	"""
	Sums exc_weights * exc_rates and substracts inh_weights * inh_rates
	"""
	rate = (
		np.dot(self.exc_syns.weights, self.exc_syns.rates) -
		np.dot(self.inh_syns.weights, self.inh_syns.rates)
	)
	self.output_rate = utils.rectify(rate)

def set_current_input_rates(self):
	"""
	Set the rates of the input neurons by using their place fields
	"""
	self.exc_syns.set_rates(self.x)
	self.inh_syns.set_rates(self.x)

class Plot:
	"""The Plotting Class"""
	def __init__(self, params, rawdata):
		for k, v in params.items():
			setattr(self, k, v)				
		for k, v in rawdata.items():
			setattr(self, k, v)
		self.box_linspace = np.linspace(0, params['boxlength'], 200)
		self.time = np.arange(0, self.simulation_time + self.dt, self.dt)
		self.colors = {'exc': 'g', 'inh': 'r'}
		# self.fig = plt.figure()

	def set_rates(self, position, norm, sigma, twoSigma2, centers):
		"""
		Computes the values of all place field Gaussians at <position>

		Future Tasks:
			- 	Maybe do not normalize because the normalization can be put into the
				weights anyway
			- 	NOTE: This is simply take from the Synapse class, but now you use
				it with an additional argument <syn_type> to make it easier to use here
		"""
		if self.dimensions == 1:
			return norm*np.exp(-np.power(position - centers, 2)*twoSigma2)
		if self.dimensions == 2:
			return norm*np.exp(-np.sum(np.power(position - centers, 2), axis=1)*twoSigma2)
			# return np.exp(-np.sum(np.power(position[0] - centers[:0], 2), axis=1)*twoSigma2) * np.exp(-np.sum(np.power(position[1] - centers[:1], 2), axis=1)*twoSigma2)

	def output_rates_vs_position(self, start_time=0):
		if self.dimensions == 1:
			_positions = self.positions[:,0][start_time:,]
			_output_rates = self.output_rates[start_time:,]
			plt.plot(_positions, _output_rates, linestyle='none', marker='o')
		if self.dimensions == 2:
			positions = self.positions[start_time:,]
			output_rates = self.output_rates[start_time:,]
			plt.xlim(0, self.boxlength)
			plt.ylim(0, self.boxlength)
			color_norm = mpl.colors.Normalize(np.amin(output_rates), np.amax(output_rates))			
			for p, r in zip(positions, output_rates):
				color = mpl.cm.bwr(color_norm(r))
				plt.plot(p[0], p[1], linestyle='none', marker='s', color=color, markersize=5, alpha=0.5)
	
	def output_rates_from_equation(self, time=-1):
		"""Plots the output rate R = w_E * E - w_I * I at time=time"""
		if self.dimensions == 1:
			n_values = 201  # Points for linspace
			linspace = np.linspace(0, self.boxlength, n_values)
			rates = {'exc': [], 'inh': []}
			for x in linspace:
				# Loop over synapse types
				for syn_type in ['exc', 'inh']:
					_sigma = getattr(self, 'sigma_' + syn_type)
					twoSigma2 = 1. / (2 * _sigma**2)
					norm = 1. / (_sigma * np.sqrt(2 * np.pi))
					centers = getattr(self, syn_type + '_centers')	
					rates[syn_type].append(self.set_rates(x, norm, _sigma, twoSigma2, centers))
			output_rates = np.zeros(n_values)
			for n, x in enumerate(linspace):
				output_rates[n] = (np.dot(self.exc_weights[time], rates['exc'][n]) 
								- np.dot(self.inh_weights[time], rates['inh'][n]))
			output_rates = utils.rectify_array(output_rates)
			plt.title('output_rates, Time = ' + str(time))
			plt.plot(linspace, output_rates)

		if self.dimensions == 2:
			n_values = 11
			x_space = np.linspace(0, self.boxlength, n_values)
			y_space = np.linspace(0, self.boxlength, n_values)
			X, Y = np.meshgrid(x_space, y_space)
			rates = {'exc': [], 'inh': []}
			for y in y_space:
				for x in x_space:
					for syn_type in ['exc', 'inh']:
						_sigma = getattr(self, 'sigma_' + syn_type)
						twoSigma2 = 1. / (2 * _sigma**2)
						norm = 1. / (_sigma**2 * 2 * np.pi)
						centers = getattr(self, syn_type + '_centers')	
						rates[syn_type].append(self.set_rates([x, y], norm, _sigma, twoSigma2, centers))					
			output_rates = np.zeros((n_values**2))
			for n in xrange(0, n_values**2):
				output_rates[n] = (np.dot(self.exc_weights[time], rates['exc'][n]) 
								- np.dot(self.inh_weights[time], rates['inh'][n]))
			output_rates = utils.rectify_array(output_rates)
			output_rates = output_rates.reshape(n_values, n_values)
			plt.title('output_rates, Time = ' + str(time))
			plt.contour(X, Y, output_rates)

	# def output_rate_as_function_of_fields_and_weights(self):
	# 	"""docstring"""
	# 	pass

	def fields_times_weights(self, time=-1, syn_type='exc', normalize_sum=True):
		"""
		Plots the Gaussian Fields multiplied with the corresponding weights

		Arguments:
		- time: default -1 takes weights at the last moment in time
				Warning: if time_step != 1.0 this doesn't work, because
				you take the array at index [time]
		- normalize_sum: If true the sum gets scaled such that it
			is comparable to the height of the weights*gaussians,
			this way it is possible to see the sum and the individual
			weights on the same plot. Otherwise the sum would be way larger.
		"""
		plt.title(syn_type + ' fields x weights')
		x = self.box_linspace
		t = syn_type
		# colors = {'exc': 'g', 'inh': 'r'}	
		summe = 0
		divisor = 1.0
		if normalize_sum:
			# divisor = 0.5 * len(rawdata[t + '_centers'])
			divisor = 0.5 * len(getattr(self, t + '_centers'))			
		for c, s, w in np.nditer([
						getattr(self, t + '_centers'),
						getattr(self, t + '_sigmas'),
						getattr(self, t + '_weights')[time]	]):
			gaussian = scipy.stats.norm(loc=c, scale=s).pdf
			l = plt.plot(x, w * gaussian(x), color=self.colors[syn_type])
			summe += w * gaussian(x)
		plt.plot(x, summe / divisor, color=self.colors[syn_type], linewidth=4)
		return l

	def fields(self, show_sum=False):
		"""
		Plotting of Gaussian Fields and their sum

		Note: The sum gets divided by a something that depends on the 
				number of cells of the specific type, to make it fit into
				the frame (see note in fields_times_weighs)
		"""
		x = self.box_linspace
		# Loop over different synapse types and color tuples
		for t, color in [('exc', 'g'), ('inh', 'r')]:
			summe = 0
			for c, s in np.nditer([getattr(self, t + '_centers'), getattr(self, t + '_sigmas')]):
				gaussian = scipy.stats.norm(loc=c, scale=s).pdf
				plt.plot(x, gaussian(x), color=color)
				summe += gaussian(x)
			if show_sum:
				plt.plot(x, 2*summe/(len(getattr(self, t + '_centers'))), color=color, linewidth=4)
		return

	def weights_vs_centers(self, syn_type='exc', time=-1):
		plt.title(syn_type + ' Weights vs Centers' + ', ' + 'Time = ' + str(time))	
		plt.xlim(0, self.boxlength)
		centers = getattr(self, syn_type + '_centers')
		weights = getattr(self, syn_type + '_weights')[time]
		plt.plot(centers, weights, linestyle='none', marker='o')

	def weight_evolution(self, syn_type='exc'):
		"""
		Plots the time evolution of each synaptic weight
		"""
		plt.title(syn_type + ' weight evolution')
		time = np.arange(0, len(self.exc_weights)) * self.every_nth_step
		for i in np.arange(0, getattr(self, 'n_' + syn_type), 100):
			# Create array of the i-th weight for all times
			weight = getattr(self, syn_type + '_weights')[:,i]
			center = getattr(self, syn_type + '_centers')[i]
			if self.dimensions == 2:
				center = center[0]
			# Specify the range of the colormap
			color_norm = mpl.colors.Normalize(0, self.boxlength)
			# Set the color from a color map
			color = mpl.cm.rainbow(color_norm(center))
			plt.plot(time, weight, color=color)

	def output_rate_distribution(self, start_time=0):
		n_bins = 100
		positions = self.positions[:,0][start_time:,]
		output_rates = self.output_rates[start_time:,]
		dx = self.boxlength / n_bins
		bin_centers = np.linspace(dx, self.boxlength-dx, num=n_bins)
		mean_output_rates = []
		for i in np.arange(0, n_bins):
			indexing = (positions >= i*dx) & (positions < (i+1)*dx)
			mean_output_rates.append(np.mean(output_rates[indexing]))
		plt.plot(bin_centers, mean_output_rates, marker='o')
		plt.axhline(y=self.target_rate, linewidth=3, linestyle='--', color='black')

	def position_distribution(self):
		x = self.positions[:,0]
		n, bins, patches = plt.hist(x, 50, normed=True, facecolor='green', alpha=0.75)