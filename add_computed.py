# from snep.configuration import config
# config['network_type'] = 'empty'
from __future__ import print_function
import numpy as np
import plotting
import general_utils
import general_utils.misc
import os
import itertools
import initialization


class Add_computed(plotting.Plot):
	"""
	Class which helps adding post-processed rawdata to .h5 file

	The most important function here is add_computed from SNEP

	Parameters
	----------

	tables : SNEP tables
	psps : list
		List of parameter space points
	params : dict
	rawdata : dict
	"""
	def __init__(self, tables=None, psps=[None], params=None, rawdata=None,
				 overwrite=False):
		general_utils.snep_plotting.Snep.__init__(self, params, rawdata)
		self.tables = tables
		self.psps = psps
		self.overwrite = overwrite
		self.correlogram_of = 'rate_map'
		self.inner_square = False
		if tables:
			self.computed_full = self.tables.get_computed(None)
	def watson_u2(self):
		"""
		Add watson_u2 values
		"""
		u2s = []
		for n, psp in enumerate(self.psps):
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			spacing = self.spacing
			u2s_this_psp = []
			for frame in np.arange(len(self.rawdata['exc']['weights'])):
				output_rates = self.get_output_rates(frame, spacing,
													 from_file=True)
				u2, h = self.get_watsonU2(spacing, output_rates)
				u2s_this_psp.append(u2)
			u2s.append(u2s_this_psp)

		all_data = {'u2': np.array(u2s)}
		if self.tables == None:
			return all_data
		else:
			self.tables.add_computed(None, all_data,
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
			print('psp number: %i out of %i' % (n + 1, len(self.psps)))
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

	def grid_score_2d(self, type='hexagonal', inner_square=False):
		self.inner_square = inner_square
		suffix = self.get_grid_score_suffix(type)
		parent_group_str = 'grid_score' + suffix
		for n, psp in enumerate(self.psps):
			self.print_psp(n)
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			all_data = {parent_group_str: {}}
			# methods = ['Weber', 'sargolini', 'sargolini_extended']
			methods = ['langston']
			for method in methods:
				all_data[parent_group_str][method] = {}
				for n_cum in [1]:
					GS_list = []
					# for frame in np.arange(len(self.rawdata['exc']['weights'])):
					for frame in np.arange(self.rawdata[
											   'output_rate_grid'].shape[0]):
					# for frame in np.array([49, 50, 51]):
						print(frame)
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

	def print_psp(self, n):
		print('psp number: %i out of %i' % (n + 1, len(self.psps)))

	def grid_axes_angles(self):
		parent_group_str = 'grid_axes_angles'
		for n, psp in enumerate(self.psps):
			self.print_psp(n)
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			all_data = {}
			for n_cum in [1]:
				angles = []
				for frame in np.arange(len(self.rawdata['exc']['weights'])):
					time = self.frame2time(frame, weight=True)
					angles.append(
						self.get_grid_axes_angles(time, n_cumulative=n_cum)
					)
				all_data[parent_group_str] = np.array(angles)
			if self.tables == None:
				return all_data
			else:
				self.tables.add_computed(psp, all_data,
										 overwrite=self.overwrite)

	def grid_scores_for_all_times_and_seeds(self,
											methods=['Weber', 'sargolini',
													 'sargolini_extended'],
											n_cumulatives=[1, 3],
											types=['hexagonal', 'quadratic']):
		"""
		Adds and array of grid scores for all times and seeds

		This array is very useful for later plotting of grid score
		histograms and time evolutions

		Parameters
		----------
		methods : list
		n_cumulatives : list
		types : list


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

	def correlation_with_reference_grid_for_all_times_and_seeds(self,
											t_reference=None):
		"""
		Analgous to `grid_scores_for_all_times_and_seeds`
		"""
		l = self.get_list_of_correlation_with_reference_grid_over_all_psps(
			t_reference=t_reference, from_computed_full=False
		)
		all_data = {'correlation_with_reference_grid':
						{str(float(t_reference)): l}}
		self.tables.add_computed(paramspace_pt=None, all_data=all_data,
								 overwrite=True)

	def grid_angles_for_all_times_and_seeds(self, minimum_grid_score=None):
		"""
		NB: Need to run grid_axes_angles first!
		"""
		l_angles = self.get_list_of_grid_axes_angles_over_all_psps(
			from_computed_full=False)		
		if minimum_grid_score:
			l_grid_scores = self.get_list_of_grid_score_arrays_over_all_psps(
				method='sargolini', n_cumulative=1, from_computed_full=False)
			l_angles = self.keep_only_entries_with_good_grid_score(
				l_grid_scores, l_angles, minimum_grid_score
			)
		all_data = {'grid_axes_angles_' + str(minimum_grid_score): l_angles}
		self.tables.add_computed(paramspace_pt=None, all_data=all_data,
								 overwrite=True)

	def keep_only_entries_with_good_grid_score(self,
											   list_of_grid_scores,
											   list_of_entries,
											   minimum_grid_score,
											   size=3):
		"""
		Sets all angles that correspond to bad grids to NaN

		See also test_add_computed
		
		Parameters
		----------
		list_of_grid_scores : ndarray
			shape: (seeds, times)
		list_of_angles : ndarray
			shape: (seeds, times, axes)
		minimum_grid_score : float
			Only grid_scores larger or equal to `minimum_grid_score` are kept
		size : int
			The number of elements in the entry
			Defaults to 3, because this function was initially written for the
			axis angle, which has 3 elements
		Returns
		-------
		out : ndarray
			shape: (seeds, times, axes)
		"""
		idx_array_for_small_grid_score = np.where(
									list_of_grid_scores < minimum_grid_score)
		nan_array = np.ones((size)) * np.nan
		for idx_0, idx_1 in np.nditer(idx_array_for_small_grid_score):
			list_of_entries[idx_0, idx_1, :] = nan_array
		return list_of_entries


	def mean_correlogram(self):
		"""
		Adds mean correlogram (over all psps except for sigma_inh)

		For each sigma_inh the mean correlogram over all remaining psps
		is added to computed.
		-------
		"""
		### Creating tuples ###
		sigmas = []
		sigma_seed_correlogram_tuples = []
		for n, psp in enumerate(self.psps):
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			seed = self.params['sim']['seed_centers']
			sigma = self.params['inh']['sigma'][0]
			corr_linspace, correlogram = self.get_correlogram(
											time=-1, mode='same',
											from_file=True)
			sigmas.append(sigma)
			sigma_seed_correlogram_tuples.append(
				(sigma, seed, correlogram)
			)
		sigmas = list(set(sigmas))

		### Running over all sigma_inh and taking the mean ###
		for sigma in sigmas:
			correlograms = []
			for t in sigma_seed_correlogram_tuples:
				if general_utils.misc.approx_equal(t[0], sigma):
					correlograms.append(t[2])
			mean_correlogram = np.mean(correlograms, axis=0)
			all_data = {'mean_correlogram':
						{
							str(sigma): mean_correlogram
						}
					}
			if self.tables == None:
				return all_data
			else:
				self.tables.add_computed(paramspace_pt=None, all_data=all_data,
										 overwrite=True)

	def parameter_string_for_table(self):
		"""
		Adds string for excitatory, inhibitoyry and simulation parameters

		The string looks like a row in a LaTeX table.
		Note: Currenlty the strings are only printed and not stored
		"""
		for n, psp in enumerate(self.psps):
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			# s_exc = '$ & \widthexc$ & $\Nexc$ & $\lrexc$ & $\winitexc$ & $\Nfpsany{\exc}$'.format(
			#
			# )
			# s_inh = '$ & \widthinh$ & $\Ninh$ & $\lrinh$ & $\winitinh$ & $\Nfpsany{\inh}$'
			print('Excitatory:')
			se = self.get_string_from_parameters(population='exc')
			print('End excitatory')
			print('Inhibitory:')
			se = self.get_string_from_parameters(population='inh')
			print('End inhibitory')
			print('Simulation:')
			se = self.get_string_from_parameters(population='sim')

			si = 'asdfsi'
			ssim = 'blub'
			all_data = {'parameter_string_6': np.array([se,
														si,
														ssim])}
			# if self.tables == None:
			# 	return all_data
			# else:
			# 	self.tables.add_computed(psp, all_data,
			# 							 overwrite=self.overwrite)

	def get_string_from_parameters(self, population):
		"""
		Creates a string of table entries from parameters

		Note: currenlty the string is just printed and can be copied
		into a LaTeX table. It might be useful to store the string
		in the .h5 files.

		Parameters
		----------
		population : str
			Either 'exc' or 'inh' for the respective parameters, or
			'sim' for the relevant simulation parameters

		Returns
		-------
		A string that looks like a row in a LaTeX table
		"""
		prms = self.params[population]
		prms_sim = self.params['sim']
		prms_exc = self.params['exc']
		prms_inh = self.params['inh']
		if population == 'sim':
			s = 'Simulation time: {0}, Velocity: {1}, Radius: {2}'.format(
				prms['simulation_time'],
				prms['velocity'],
				prms['radius'],
			)

		else:
			prms.update(self.rawdata[population])
			try:
				init_weight = np.mean(prms['weights'], axis=2)[0][0]
			except KeyError:
				print('WARNING: Weights were not stored !!!')
				if population == 'exc':
					init_weight = prms['init_weight']
				elif population == 'inh':
					init_weight = initialization.get_fixed_point_initial_weights(
						dimensions=prms_sim['dimensions'],
						radius=prms_sim['radius'],
						center_overlap_exc=3*prms_exc['sigma'],
						center_overlap_inh=3*prms_inh['sigma'],
						target_rate=self.params['out']['target_rate'],
						init_weight_exc=prms_exc['init_weight'],
						n_exc=prms_exc['number'],
						n_inh=prms_inh['number'],
						sigma_exc=prms_exc['sigma'],
						sigma_inh=prms_inh['sigma'],
						fields_per_synapse_exc=prms_exc['fields_per_synapse'],
						fields_per_synapse_inh=prms_inh['fields_per_synapse'],
						tuning_function=prms_sim['tuning_function'],
						gaussian_height_exc=prms_exc['gaussian_height'],
						gaussian_height_inh=prms_inh['gaussian_height']
					)[0]
			print(init_weight)
			if self.params['sim']['gaussian_process']:
				fields_per_synapse_string = '$\infty$'
			else:
				fields_per_synapse_string = '{0}'.format(prms['fields_per_synapse'])
			try:
				if prms['untuned']:
					width_string = '$\infty$'
				else:
					width_string = '{0}'.format(prms['sigma'])
			except KeyError:
				width_string = '{0}'.format(prms['sigma'])

			s = '{0} & ' \
				'{1} & ' \
				'{2} & ' \
				'{3:.3} & ' \
				'{4}'.format(
				width_string,
				prms['number'][0],
				str('\\num{') + str(prms['eta']) + str('}'),
				init_weight,
				fields_per_synapse_string
			)
		print(s)
		return s

	def peak_locations_computed(self):
		self.inner_square = False
		parent_group_str = 'peak_locations'
		for n, psp in enumerate(self.psps):
			self.print_psp(n)
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			all_data = {}
			peak_locations = []
			for frame in np.arange(len(self.rawdata['output_rate_grid'])):
				time = self.frame2time(frame, weight=True)
				peak_locations.append(
					self.get_peak_locations(time, n_cumulative=1,
													   return_as_list=True)
				)
			all_data[parent_group_str] = np.array(peak_locations)
			if self.tables == None:
				return all_data
			else:
				self.tables.add_computed(psp, all_data,
										 overwrite=self.overwrite)
				print(all_data)

	def peak_locations_for_all_times_and_seeds(self, minimum_grid_score=None):
		"""
		asdf
		"""
		l_peak_locations = self.get_list_of_grid_axes_angles_over_all_psps(
			from_computed_full=False)
		if minimum_grid_score:
			l_grid_scores = self.get_list_of_grid_score_arrays_over_all_psps(
				method='sargolini', n_cumulative=1, from_computed_full=False)
			l_peak_locations = self.keep_only_entries_with_good_grid_score(
				l_grid_scores, l_peak_locations, minimum_grid_score
			)
		all_data = {'peak_locations_' + str(minimum_grid_score): l_peak_locations}
		self.tables.add_computed(paramspace_pt=None, all_data=all_data,
								 overwrite=True)

	def flattened_output_rate_grids(self, frame=-1):
		"""
		Adds an array that contains the flattened output rate grids.

		Each row in the resulting array corrsponds to a different paramspace
		point.
		"""
		l = []
		for psp in self.psps:
			try:
				self.set_params_rawdata_computed(psp, set_sim_params=True)
				array = self.rawdata['output_rate_grid'][frame, ...].flatten()
				l.append(array)
			except:
				pass
		l = np.asarray(l)
		all_data = {'flattened_output_rate_grids': l}
		self.tables.add_computed(paramspace_pt=None, all_data=all_data,
								 overwrite=self.overwrite)

	def cross_correlation_of_output_rates(self):
		flattened_rates = self.computed_full['flattened_output_rate_grids']
		psp_range = np.arange(flattened_rates.shape[0])
		corrcoeffs = []
		for i in psp_range:
			for j in psp_range:
				if j > i:
					corrcoeffs.append(np.corrcoef(
						flattened_rates[i, :], flattened_rates[j, :])[0, 1])
		corrcoeffs = np.asarray(corrcoeffs)
		all_data = {'cross_correlation_coefficients': corrcoeffs}
		self.tables.add_computed(paramspace_pt=None, all_data=all_data,
								 overwrite=self.overwrite)

	def spiketimes(self):
		rate_factors = [40]
		# all_data = {'spiketimes': {}}
		for n, psp in enumerate(self.psps):
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			rf_dict = {}
			firing_rates = self.rawdata['output_rates'][:, 0]
			for rf in rate_factors:
				print(rf)
				spiketimes = self.get_spiketimes(
					firing_rates=firing_rates,
					dt=0.02,
					rate_factor=rf
				)
				rf_dict[str(rf)] = spiketimes
			all_data = {'spiketimes': rf_dict}
			self.tables.add_computed(paramspace_pt=psp,
								all_data=all_data,
								overwrite=self.overwrite)

	def hd_tuning_direction(self, method='center_of_mass'):
		parent_group_str = 'hd_directions_' + method
		for n, psp in enumerate(self.psps):
			self.print_psp(n)
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			all_data = {}
			# for n_cum in [1]:
			directions = []
			for frame in np.arange(len(self.rawdata['exc']['weights'])):
				# time = self.frame2time(frame, weight=True)
				output_rates = self.get_output_rates(
					frame, spacing=None, from_file=True)
				hd_tuning = self.get_head_direction_tuning_from_output_rates(
						output_rates)
				angles = np.linspace(-np.pi, np.pi, self.spacing)
				direction = self.get_direction_of_hd_tuning(hd_tuning,
															angles,
															method)
				directions.append(direction)
			all_data[parent_group_str] = np.array(directions)
			if self.tables == None:
				return all_data
			else:
				self.tables.add_computed(psp, all_data,
										 overwrite=self.overwrite)

	def hd_tuning_directions_for_all_times_and_seeds(self,
													method='center_of_mass'):
		"""
		NB: Need to run above function firs
		"""
		l = self.get_list_of_hd_tuning_directions_over_all_psps(
			from_computed_full=False, method=method)
		all_data = {'hd_tuning_directions_' + str(method): l}
		self.tables.add_computed(paramspace_pt=None, all_data=all_data,
								 overwrite=True)

	def correlation_with_reference_grid(self, t_reference):
		"""
		Correlation of grids at all times with one grid at reference time
		
		Pearson correlation between firing pattern at time t (running over 
		all times) with firing pattern at time t_reference (fixed)
		
		Parameters
		----------
		t_reference : float
			All patterns are correlated with the pattern at `t_reference`.
			So for time = t_reference, the correlation is 1.
		"""
		for n, psp in enumerate(self.psps):
			self.print_psp(n)
			self.set_params_rawdata_computed(psp, set_sim_params=True)
			reference_frame = self.time2frame(t_reference, weight=True)
			reference_grid_flat = self.rawdata['output_rate_grid'][
				reference_frame, ...].flatten()
			correlations = []
			for frame in np.arange(self.rawdata['output_rate_grid'].shape[0]):
				print(frame)
				correlations.append(
					self.get_correlation_with_reference_grid(frame,
														reference_grid_flat)
				)
			correlations = np.array(correlations)
			all_data = {
				'correlation_with_reference_grid': {
							'{0}'.format(t_reference): correlations}
			}
			if self.tables == None:
				return all_data
			else:
				self.tables.add_computed(psp, all_data,
										 overwrite=self.overwrite)

if __name__ == '__main__':
	import snep.utils
	# date_dir = '2015-01-05-17h44m42s_grid_score_stability'
	# date_dir = '2015-01-20-11h09m35s_grid_score_stability_faster_learning'
	# date_dir = '2015-07-04-10h57m42s_grid_spacing_VS_gaussian_height_inh'
	# date_dir = '2015-07-01-17h53m22s_grid_spacing_VS_eta_inh'
	# date_dir = '2015-07-02-15h08m01s_grid_spacing_VS_n_inh'
	# date_dir = '2015-09-14-16h03m44s'
	# date_dir = '2016-04-19-11h41m44s_20_fps'
	# date_dir = '2016-04-20-15h11m05s_20_fps_learning_rate_0.2'
	for date_dir in ['2017-09-06-18h36m49s_test_run_fast_learning_4_seeds']:
		tables = snep.utils.make_tables_from_path(
			general_utils.snep_plotting.get_path_to_hdf_file(date_dir))

		tables.open_file(False)
		tables.initialize()

		psps = tables.paramspace_pts()
		all_psps = psps
		psps = [p for p in all_psps
				# if p[('sim', 'seed_centers')].quantity == 1
				]
		add_computed = Add_computed(tables, psps, overwrite=True)
		# add_computed.correlation_with_reference_grid(t_reference=4.9 * 36e4)
		# add_computed.correlation_with_reference_grid_for_all_times_and_seeds(
		# 	t_reference=9e5)
		add_computed.grid_scores_for_all_times_and_seeds(methods=['langston'],
													 types=['hexagonal'],
													 n_cumulatives=[1])
		# add_computed.grid_axes_angles()
		# add_computed.watson_u2()
		# add_computed.grid_score_1d()
		# add_computed.grid_score_2d(type='hexagonal')
		# add_computed.mean_inter_peak_distance()

		# add_computed.grid_angles_for_all_times_and_seeds(minimum_grid_score=0.7)
		# add_computed.grid_angles_for_all_times_and_seeds(minimum_grid_score=None)
		# add_computed.peak_locations()
		# add_computed.peak_locations_for_all_times_and_seeds(minimum_grid_score=0.7)
		# add_computed.parameter_string_for_table()
		# add_computed.flattened_output_rate_grids()
		# add_computed.cross_correlation_of_output_rates()
		# add_computed.mean_correlogram()
		# add_computed.spiketimes()
		# add_computed.hd_tuning_direction()
		# add_computed.hd_tuning_directions_for_all_times_and_seeds()
		# add_computed.hd_tuning_direction(method='maximum')
		# add_computed.hd_tuning_directions_for_all_times_and_seeds(method='maximum')
