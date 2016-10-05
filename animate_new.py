__author__ = 'simonweber'

# open the tablefile
# from snep.configuration import config

# config['network_type'] = 'empty'
import snep.utils
import matplotlib as mpl
# import matplotlib.animation as animation
import plotting
import initialization
import general_utils
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from general_utils import scripts
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'


class Animation(initialization.Synapses, initialization.Rat,
			general_utils.snep_plotting.Snep, plotting.Plot):

	def __init__(self, tables=None, psps_video=[None], params=None, rawdata=None,
				 path_all_videos=None):
		"""
		NOTE:
		Be careful with the psps. To this class you pass all the psps you
		want to create videos from. For the Plot class to get properly
		initialized we pass these psps to its __init__ function.
		However, we will change self.psps, which is use in all plotting
		functions of Plot to a single psp in create_images, in order to
		only plot the time evolution of that single psp. To avoid confusion
		we therefore introduce psps_video, as the psps for which we create
		videos. In the future we might be interested in plotting videos
		with figure that contain multiple psps. Then this needs to be
		reconsidered.
		"""
		plotting.Plot.__init__(self, tables, psps_video, params, rawdata)
		self.psps_video = psps_video
		self.path_all_videos = path_all_videos

	def try_mkdir(self, path):
		try:
			os.mkdir(path)
		except OSError:
			pass

	def create_images(self, times, plot_function, show_preceding=False,
					  function_kwargs={}, extension='.png'):
		"""
		Creates images for given moments in time by plotting a function

		Parameters
		----------
		times : ndarray
			Array containing all the times at which to plot
		plot_function : function
			Functions that plots the desired plots into a single figure
		show_preceding : bool
			If True, every image also contains the data of all the preceding
			moments in time
		"""
		self.try_mkdir(path_all_videos)
		self.path_video_type = os.path.join(self.path_all_videos, plot_function.__name__)
		self.try_mkdir(self.path_video_type)
		for psp in self.psps_video:
			# Change self.psps so that it contains just the current psp (as a
			# list). Thus an image containing just one psp is created.
			# self.psps is used in the plotting functions of the Plot class.
			self.psps = [psp]
			path_psp = os.path.join(self.path_video_type, tables.get_results_directory(psp) + '/')
			self.try_mkdir(path_psp)
			if show_preceding:
				plt.figure(figsize=(5,5))
			for n, t in enumerate(times):
				print n
				if not show_preceding:
					plt.figure(figsize=(5,5))
				plot_function(time=t, **function_kwargs)
				save_path_full = path_psp + str(n) + extension
				plt.savefig(save_path_full, dpi=100, bbox_inches='tight',
							pad_inches=0.1)
				if not show_preceding:
					plt.cla()
					plt.clf()
					plt.close()


	def add_grid_to_axis(self):
		ax = plt.gca()
		ax.locator_params(axis='y', nbins=3)
		minorLocator = MultipleLocator(0.05)
		ax.xaxis.set_minor_locator(minorLocator)
		ax.grid(which='minor')

	def rates_currents_weights_1d(self, time):
		"""
		Plots output rates and exc. and inh. weights
		Parameters
		----------
		time : float
			The moment in time
		"""
		number_of_plots = 4

		plt.subplot(number_of_plots, 1, 1)
		self.plot_output_rates_from_equation(time=time, from_file=False, spacing=601)
		self.add_grid_to_axis()
		plt.title('')
		plt.xticks([])


		plt.subplot(number_of_plots, 1, 2)
		self.input_current(time=time, spacing=601)
		self.add_grid_to_axis()
		plt.xticks([])

		plt.subplot(number_of_plots, 1, 3)
		# Always shows initial weights in the background
		self.weights_vs_centers(time=0, populations=['exc'], marker='')
		self.weights_vs_centers(time=time, populations=['exc'])
		self.add_grid_to_axis()
		plt.xticks([])

		plt.subplot(number_of_plots, 1, 4)
		# Always shows initial weights in the background
		self.weights_vs_centers(time=0, populations=['inh'], marker='')
		self.weights_vs_centers(time=time, populations=['inh'])
		self.add_grid_to_axis()
		# plt.xticks([])



	def rates_correlogram_2d(self, time):
		plt.subplot(211)
		self.plot_output_rates_from_equation(time, from_file=True,
											 publishable=False,
											 maximal_rate=14.0,
											 show_colorbar=False)
		plt.subplot(212)
		self.plot_correlogram(time, from_file=True,
							  mode='same', method='Weber', publishable=False,
							  show_colorbar=False)

	def trajectory_rates(self, time, slice_length):
		"""
		Plots trajectory (or slice thereof) with color coded rate

		Plotting a slice avoids plotting the entire trajectory again and
		again in every frame of the video, because this would darken
		the entire trajectory.

		NOTE: We treat frame and time as the same here!!!

		Parameters
		----------
		frame : int
			Frame which marks the end point of the trajectory
		slice_length : int or None
			Determines the length of the trajoctory that is now additionally
			plotted.
			The trajectory starts at `time`-`slice_length` and
			ends at `time`
			If None, the entire trajectory is plotted.
		"""
		if slice_length:
			start_frame = time - slice_length
		else:
			start_frame = 0


		self.trajectory_with_firing(start_frame=start_frame,
									end_frame=time,
									symbol_size=20, show_title=True)


if __name__ == '__main__':
	# date_dir = '2014-12-12-12h43m56s_nice_grid_video'
	# date_dir = '2015-01-15-17h05m43s_boundary_effects_1d'
	# date_dir = '2015-01-05-17h44m42s_grid_score_stability'
	# date_dir = '2015-01-20-11h09m35s_grid_score_stability_faster_learning'
	# date_dir = '2015-01-22-14h31m24s_boundary_effects_1d_larger_time'
	# date_dir = '2015-03-10-17h14m00s_periodic_semiperiodic_nonperiodic'
	# date_dir = '2015-07-12-20h38m03s_trajectory_with_firing_video'
	# date_dir = '2015-09-22-22h46m44s_real_trajectory_grids'
	# date_dir = '2016-03-14-17h56m34s_GRF_2D_grid_stability'
	date_dir = '2016-10-05-14h42m39s_grid_stability_videos'
	# path = os.path.expanduser(
	# 	'~/localfiles/itb_experiments/learning_grids/')
	#
	# path_date_dir = os.path.join(path, date_dir)
	# path_visuals = os.path.join(path_date_dir, 'visuals/')

	path = general_utils.snep_plotting.get_path_to_hdf_file(date_dir)
	tables = snep.utils.make_tables_from_path(path)
	tables.open_file(True)
	path_visuals = os.path.join(os.path.dirname(path), 'visuals')


	psps_video = [p for p in tables.paramspace_pts()
			if p[('sim', 'seed_centers')].quantity == 9
			# and p[('exc', 'eta')].quantity == 4e-6
			]
	# times = np.linspace(0, 1e3, 101)
	extension = '.png'
	# slice_length = 300
	# times = np.arange(0, 1.5e5, slice_length)
	times = np.arange(0, 18e6, 18e6/500)
	print times
	path_all_videos = os.path.join(path_visuals, 'videos/')
	animation = Animation(tables, psps_video, path_all_videos=path_all_videos)
	try:
		# animation.create_images(times,
		# 						plot_function=animation.trajectory_rates,
		# 						show_preceding=False, extension=extension,
		# 						function_kwargs={'slice_length': None})
		animation.create_images(times,
								plot_function=animation.rates_correlogram_2d,
								show_preceding=False, extension=extension,
								function_kwargs={})
	except ValueError:
		pass
	scripts.images2movies(maindir=animation.path_video_type, framerate=20,
						  delete_images=True, overwrite=True,
						  scale_flag='-vf scale=584:584', extension=extension)

