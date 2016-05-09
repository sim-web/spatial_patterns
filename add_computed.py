# from snep.configuration import config
# config['network_type'] = 'empty'
import snep.utils
import numpy as np
import plotting
import general_utils
import os
import itertools


# def output_rate_grid(tables):

# 	for psp in tables.paramspace_pts():
# 		print psp
# 		params = tables.as_dictionary(psp, True)
# 		try:
# 			rawdata = tables.get_raw_data(psp)
# 		except tbls.exceptions.NoSuchNodeError:
# 			continue

# 		rat = initialization.Rat(params)

# 		output_rate_grid = rat.get_output_rates_from_equation(
# 			0, rawdata, rawdata['sim']['spacing'], )
# 		dirname = 'output_rate_grid '

# 		output_rate_grid = 
# 		# Set up the dictionary
# 		my_all_data = {
# 			dirname: output_rate_grid
# 		}

# 		tables.add_computed(paramspace_pt=psp, all_data=my_all_data)

class Add_computed(plotting.Plot):
	def __init__(self, tables=None, psps=[None], params=None, rawdata=None,
				 overwrite=False):
		general_utils.snep_plotting.Snep.__init__(self, params, rawdata)
		self.tables = tables
		self.psps = psps
		self.overwrite = overwrite

	def watson_u2(self):
		"""
		asdf
		"""

		# plot = plotting.Plot(tables, psps)
		for n, psp in enumerate(self.psps):
			print 'psp number: %i out of %i' % (n + 1, len(self.psps))
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			# frame = self.time2frame(time, weight=True)
			# if spacing is None:
			# 	spacing = self.spacing
			# # WATSON
			spacing = self.spacing
			U2_list = []
			for frame in np.arange(len(self.rawdata['exc']['weights'])):
				output_rates = self.get_output_rates(frame, spacing,
													 from_file=True)
				U2, h = self.get_watsonU2(spacing, output_rates)
				U2_list.append(U2)
			all_data = {'U2': np.array(U2_list)}
			if self.tables == None:
				return all_data
			else:
				self.tables.add_computed(psp, all_data,
										 overwrite=self.overwrite)

	def mean_inter_peak_distance(self):
		"""
		Mean inter peak distance (measure for grid spacing) at final time
		"""
		for n, psp in enumerate(self.psps):
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			output_rates = self.get_output_rates(-1, self.spacing,
												 from_file=True, squeeze=True)
			mipd = general_utils.arrays.get_mean_inter_peak_distance(
				output_rates, 2 * self.radius, 5, 0.1)
			all_data = {'mean_inter_peak_distance': np.array([mipd])}
			if self.tables == None:
				return all_data
			else:
				self.tables.add_computed(psp, all_data,
										 overwrite=self.overwrite)

	def grid_score_1d(self):
		"""
		As far as I remember I used this to get a 1d grid score from
		a 2d simulation (to see if it creates a Sargolini like Figure)
		"""

		# plot = plotting.Plot(tables, psps)
		for n, psp in enumerate(self.psps):
			print 'psp number: %i out of %i' % (n + 1, len(self.psps))
			self.set_params_rawdata_computed(psp, set_sim_params=True)

			spacing = self.spacing
			GS_list = []
			for frame in np.arange(len(self.rawdata['exc']['weights'])):
				output_rates = self.get_output_rates(frame, spacing,
													 from_file=True)
				spatial_tuning = self.get_spatial_tuning(output_rates)
				linspace = np.linspace(-self.radius, self.radius, spacing)
				grid_score = self.get_1d_grid_score(
					spatial_tuning, linspace, return_maxima_arrays=False)
				GS_list.append(grid_score)
			all_data = {'grid_score': np.array(GS_list)}
			if self.tables == None:
				return all_data
			else:
				self.tables.add_computed(psp, all_data,
										 overwrite=self.overwrite)

	def grid_score_2d(self, type='hexagonal'):
		suffix = self.get_grid_score_suffix(type)
		parent_group_str = 'grid_score' + suffix
		for n, psp in enumerate(self.psps):
			print 'psp number: %i out of %i' % (n + 1, len(self.psps))
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			all_data = {parent_group_str: {}}
			methods = ['Weber', 'sargolini', 'sargolini_extended']
			# methods = ['sargolini']
			for method in methods:
				all_data[parent_group_str][method] = {}
				for n_cum in [1, 3]:
					GS_list = []
					for frame in np.arange(len(self.rawdata['exc']['weights'])):
						time = self.frame2time(frame, weight=True)
						GS_list.append(
							self.get_grid_score(time, method=method,
												n_cumulative=n_cum,
												type=type)
						)
					all_data[parent_group_str][method][str(n_cum)] = np.array(
						GS_list)
			if self.tables == None:
				return all_data
			else:
				self.tables.add_computed(psp, all_data,
										 overwrite=self.overwrite)

	def grid_scores_for_all_times_and_seeds(self,
											methods=('Weber', 'sargolini',
													 'sargolini_extended'),
											n_cumulatives=(1, 3),
											types=('hexagonal', 'quadratic')):
		"""
		Adds and array of grid scores for all times and seeds

		This array is very useful for later plotting of grid score
		histograms and time evolutions

		Parameters
		----------
		methods : tuple
		n_cumulatives : tuple
		types : tuple


		Returns
		-------
		"""
		for m, n, t in itertools.product(methods, n_cumulatives, types):
			suffix = self.get_grid_score_suffix(type=t)
			l = self.get_list_of_grid_score_arrays_over_all_psps(
				method=m, n_cumulative=n, type=t, from_computed_full=False
			)
			all_data = {'grid_score' + suffix:
							{m:
								 {str(n): l
								  }}}
			self.tables.add_computed(paramspace_pt=None, all_data=all_data,
									 overwrite=True)


		# def inter_peak_distance(self):
		# 	"""
		# 	Inter peak distances for final frame
		#
		# 	Note: Since the number of peaks fluctuates you can't create a
		# 		homogeneous array including all times. We thus only take the
		# 		final time.
		# 	"""
		# 	for n, psp in enumerate(self.psps):
		# 		print 'psp number: %i out of %i' % (n, len(self.psps))
		# 		self.set_params_rawdata_computed(psp, set_sim_params=True)
		#
		# 		spacing = self.spacing
		# 		mylist = []
		# 		output_rates = self.get_output_rates(frame=-1, spacing=spacing,
		# 							from_file=True, squeeze=True)
		# 		maxima_boolean = general_utils.arrays.get_local_maxima_boolean(
		# 						output_rates, 5, 0.1)
		# 		x_space = np.linspace(-self.radius, self.radius, spacing)
		# 		peak_positions = x_space[maxima_boolean]
		# 		inter_peak_distance = (np.abs(peak_positions[:-1]
		# 										- peak_positions[1:]))
		# 		mylist.append(inter_peak_distance)
		# 		all_data = {'inter_peak_distance': np.array(mylist)}
		# 		print self.overwrite
		# 		if self.tables == None:
		# 			return all_data
		# 		else:
		# 			self.tables.add_computed(psp, all_data, overwrite=self.overwrite)


if __name__ == '__main__':
	# date_dir = '2015-01-05-17h44m42s_grid_score_stability'
	# date_dir = '2015-01-20-11h09m35s_grid_score_stability_faster_learning'
	# date_dir = '2015-07-04-10h57m42s_grid_spacing_VS_gaussian_height_inh'
	# date_dir = '2015-07-01-17h53m22s_grid_spacing_VS_eta_inh'
	# date_dir = '2015-07-02-15h08m01s_grid_spacing_VS_n_inh'
	# date_dir = '2015-09-14-16h03m44s'
	# date_dir = '2016-04-19-11h41m44s_20_fps'
	# date_dir = '2016-04-20-15h11m05s_20_fps_learning_rate_0.2'
	for date_dir in ['2016-05-04-15h30m04s_2D_GRF_sigma_inh_0.1_histogram']:
		tables = snep.utils.make_tables_from_path(
			general_utils.snep_plotting.get_path_to_hdf_file(date_dir))

		tables.open_file(False)
		tables.initialize()

		psps = tables.paramspace_pts()
		add_computed = Add_computed(tables, psps, overwrite=True)
		# add_computed.watson_u2()
		# add_computed.grid_score_1d()
		# add_computed.grid_score_2d(type='quadratic')
		# add_computed.mean_inter_peak_distance()
		add_computed.grid_scores_for_all_times_and_seeds()