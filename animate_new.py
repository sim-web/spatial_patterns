__author__ = 'simonweber'

# open the tablefile
from snep.configuration import config

config['network_type'] = 'empty'
import snep.utils
import matplotlib as mpl
import matplotlib.animation as animation
import plotting
import initialization
import general_utils
import matplotlib.pyplot as plt
import time
import numpy as np
import os


os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'


# date_dir = '2014-12-12-12h43m56s_nice_grid_video'
date_dir = '2015-01-15-17h05m43s_boundary_effects_1d'

path = os.path.expanduser(
	'~/localfiles/itb_experiments/learning_grids/')

path_date_dir = os.path.join(path, date_dir)
path_visuals = os.path.join(path_date_dir, 'visuals/')

tables = snep.utils.make_tables_from_path(
	path + date_dir
	+ '/experiment.h5')
t0 = time.time()
tables.open_file(True)

print tables
psps = [p for p in tables.paramspace_pts()
		if p[('sim', 'seed_init_weights')].quantity == 0]

# Set the time space
# times = np.linspace(0, 1e7, 1001)
times = np.linspace(0, 400, 3)

class Animation(initialization.Synapses, initialization.Rat,
			general_utils.snep_plotting.Snep, plotting.Plot):

	def __init__(self, tables=None, psps=[None], params=None, rawdata=None,
				 path_all_videos=None):
		general_utils.snep_plotting.Snep.__init__(self, params, rawdata)
		self.tables = tables
		self.psps = psps
		self.path_all_videos = path_all_videos

	def animate(self, times, plot_function):
		print path_all_videos
		try:
			os.mkdir(path_all_videos)
		except OSError:
			pass
		for psp in self.psps:
			path_this_video = os.path.join(path_all_videos, tables.get_results_directory(psp) + '/')
			try:
				os.mkdir(path_this_video)
			except OSError:
				pass
			for n, t in enumerate(times):
				plot_function(time=t)
				# plt.plot(np.arange(10), (n+1)*np.arange(10))
				save_path_full = path_this_video + str(n) + '.png'
				print save_path_full
				plt.savefig(save_path_full, dpi=300, bbox_inches='tight',
							pad_inches=0.01)

	def test_plot(self, time):
		plt.subplot(111)
		self.plot_output_rates_from_equation(time=time, from_file=True)
		# plt.title('')
		# plt.subplot(312)
		# plt.ylim([0.87, 1.07])
		# plt.yticks([0.87, 1.07])
		# self.weights_vs_centers(time=0, populations=['exc'], marker='')
		# self.weights_vs_centers(time=t, populations=['exc'])
		# plt.subplot(313)
		# plt.ylim([0.29, 0.37])
		# plt.yticks([0.29, 0.37])
		# self.weights_vs_centers(time=0, populations=['inh'], marker='')
		# self.weights_vs_centers(time=t, populations=['inh'])
		#


for psp in psps:
	plot = plotting.Plot(tables, [psp])  # Initiating the Plot class
	params = tables.as_dictionary(psp, True)
	# If you want data from all preceding times to also appear on the plot
	# you create the figure outside the loop and don't close it at the end
	# fig = plt.figure(figsize=(3.5, 2.1))
	for n, t in enumerate(times):
		# If you want a new plot without the preceding plot, you need to
		# set a new figure and also close it at the end (see below)
		fig = plt.figure()
		print t
		# The function plot.positions returns a list of artists
		# p = plot.plot_output_rates_from_equation(time=t, from_file=True)
		# p = plot.plot_correlogram(time=t, from_file=True, mode='same', publishable=True)
		# plot.plot_time_evolution('grid_score', t_start=t, t_end=t+1)
		plt.subplot(311)
		plot.plot_output_rates_from_equation(time=t, from_file=True)
		plt.title('')
		plt.subplot(312)
		plt.ylim([0.87, 1.07])
		plt.yticks([0.87, 1.07])
		plot.weights_vs_centers(time=0, populations=['exc'], marker='')
		plot.weights_vs_centers(time=t, populations=['exc'])
		plt.subplot(313)
		plt.ylim([0.29, 0.37])
		plt.yticks([0.29, 0.37])
		plot.weights_vs_centers(time=0, populations=['inh'], marker='')
		plot.weights_vs_centers(time=t, populations=['inh'])
		# foldername = ('gridscore/')
		foldername = ('test/seed_%s/' % params['sim']['seed_init_weights'])
		file_path = os.path.join(path, date_dir, 'visuals/videos')
		dirname = os.path.join(file_path, foldername)
		try:
			os.mkdir(dirname)
		except OSError:
			pass
		save_path_full = dirname + str(n) + '.png'
		# print save_path_full
		print save_path_full
		plt.savefig(save_path_full, dpi=300, bbox_inches='tight',
					pad_inches=0.01)
		# Close the figure if you don't also want to plot data from preceding
		# times
		plt.clf()
		plt.close()


if __name__ == '__main__':
	path_all_videos = os.path.join(path_visuals, 'videos/test2/')
	print path_all_videos
	animation = Animation(tables, psps, path_all_videos=path_all_videos)
	animation.animate(times, plot_function=animation.test_plot)


