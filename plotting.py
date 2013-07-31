import matplotlib as mpl
import math
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
mpl.rcParams.update({'figure.autolayout': True})
# print mpl.rcParams.keys()
# mpl.rcParams['animation.frame_format'] = 'jpeg'
# print mpl.rcParams['animation.frame_format']

def plot_list(fig, plot_list):
	n_plots = len(plot_list)
	for n, p in enumerate(plot_list, start=1):
		if n_plots < 4:
			fig.add_subplot(n_plots, 1, n)
		else:
			fig.add_subplot(math.ceil(n_plots/2.), 2, n)
		p()

class Plot:
	"""docstring for Plot"""
	def __init__(self, params, rawdata):
		for k, v in params.items():
			setattr(self, k, v)				
		for k, v in rawdata.items():
			setattr(self, k, v)

		self.box_linspace = np.linspace(0, params['boxlength'], 200)
		self.time = np.arange(0, self.simulation_time + self.dt, self.dt)
		self.colors = {'exc': 'g', 'inh': 'r'}
		# self.fig = plt.figure()

	def fields_times_weights(self, time=-1, syn_type='exc', normalize_sum=True):
		"""docstring"""
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
			gaussian = norm(loc=c, scale=s).pdf
			l = plt.plot(x, w * gaussian(x), color=self.colors[syn_type])
			summe += w * gaussian(x)
		plt.plot(x, summe / divisor, color=self.colors[syn_type], linewidth=4)
		return l

	def fields(self, show_sum=False):
		"""
		Plotting of Gaussian Fields and their sum

		Note: The sum gets divided by a something that depends on the 
				number of cells of the specific type, to make it fit into
				the frame
		"""
		x = self.box_linspace
		# Loop over different synapse types and color tuples
		for t, color in [('exc', 'g'), ('inh', 'r')]:
			summe = 0
			for c, s in np.nditer([getattr(self, t + '_centers'), getattr(self, t + '_sigmas')]):
				gaussian = norm(loc=c, scale=s).pdf
				plt.plot(x, gaussian(x), color=color)
				summe += gaussian(x)
			if show_sum:
				plt.plot(x, 2*summe/(len(getattr(self, t + '_centers'))), color=color, linewidth=4)
		return

	def weights_vs_centers(self, syn_type='exc'):
		plt.xlim(0, self.boxlength)
		centers = getattr(self, syn_type + '_centers')
		weights = getattr(self, syn_type + '_weights')[-1]
		plt.plot(centers, weights, linestyle='none', marker='o')

	def weight_evolution(self, syn_type='exc'):
		# time = np.arange(0, len(rawdata['exc_weights']))
		for i in np.arange(0, getattr(self, 'n_' + syn_type)):
			weight = getattr(self, syn_type + '_weights')[:,i]
			plt.plot(self.time, weight)

	def output_rate_distribution(self, n_last_steps=10000):
		n_bins = 50
		positions = self.positions[:,0][-n_last_steps:,]
		output_rates = self.output_rates[-n_last_steps:,]
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