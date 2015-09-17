__author__ = 'simonweber'
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import analytics.linear_stability_analysis as lsa
from general_utils.plotting import cm2inch
import scipy.stats
# open the tablefile
from snep.configuration import config
config['network_type'] = 'empty'
import snep.utils
import general_utils.arrays
import general_utils.plotting
import general_utils.misc
import itertools
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from general_utils.plotting import simpleaxis
from general_utils.plotting import adjust_spines
from matplotlib import gridspec
import plotting
import utils


os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
colors = {'exc': '#D7191C', 'inh': '#2C7BB6', 'diff': '0.4'}
legend = {'exc': 'Excitation', 'inh': 'Inhibition', 'diff': 'Difference'}
legend_short = {'exc': 'Exc.', 'inh': 'Inh.', 'diff': 'Difference'}
signs = {'exc': 1, 'inh': -1}
# sigma = {'exc': 0.025, 'inh': 0.075}
sigma = {'exc': 0.1, 'inh': 0.2}
marker = {'exc': '^', 'inh': 'o'}
populations = ['exc', 'inh']
scaling_factor = {'exc': 1.0, 'inh': 0.5}
color_cycle_blue3 = general_utils.plotting.color_cycle_blue3

def get_tables(date_dir):
	tables = snep.utils.make_tables_from_path(
		'/Users/simonweber/localfiles/itb_experiments/learning_grids/'
		+ date_dir
		+ '/experiment.h5')
	tables.open_file(True)
	return tables


def get_plot_class(date_dir, *condition_tuples):
	"""
	Creates tables object, paramspace points and plot class

	Parameters
	----------
	date_dir : see somwhere else
	condition_tuples : unpacked list of tuples
		arbitrarily many tuples setting conditions for the
						paramspace points (psps)
		Example:
			(('sim', 'seed_centers'), 'lt', 10)
		See also utils.check_conditions
	Returns
	-------
	Plot class instance
	"""
	tables = get_tables(date_dir=date_dir)
	psps = []
	for p in tables.paramspace_pts():
		if utils.check_conditions(p, *condition_tuples):
			psps.append(p)

	plot = plotting.Plot(tables, psps)
	plot.set_params_rawdata_computed(psps[0], set_sim_params=True)
	return plot

def grid_score_histogram():
	mpl.style.use('ggplot')
	fig = plt.figure(figsize=(14, 10))
	gs = gridspec.GridSpec(3, 4)

	date_dirs = ['2015-09-16-15h08m32s_fast_grids_200',
				 '2015-09-16-18h53m53s_slower_grids_200',
				 '2015-09-16-18h58m05s_fast_grids_200_5_times_longer',]
	for ndd, date_dir in enumerate(date_dirs):
		condition_tuples = [(('sim', 'seed_centers'), 'lt', 200)]
		plot = get_plot_class(date_dir, *condition_tuples)

		gs_dict = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
		methods = ['Weber', 'sargolini']
		hist_kwargs = {'alpha': 0.5, 'bins': 20}
		for i, method in enumerate(methods):
			# fig.add_subplot(3, 1, ndd+1)
			for k, ncum in enumerate([1, 10]):
				column_index = gs_dict[(i, k)]
				plt.subplot(gs[ndd, column_index])
				grid_scores = plot.get_list_of_grid_score_arrays_over_all_psps(method=method,
																	   n_cumulative=ncum)
				initial_grid_scores = grid_scores[:, 0]
				final_grid_scores = grid_scores[:, -1]
				plt.hist(initial_grid_scores[~np.isnan(initial_grid_scores)], **hist_kwargs)
				plt.hist(final_grid_scores[~np.isnan(final_grid_scores)], **hist_kwargs)
				if ndd == 0:
					plt.title('{0}, nc = {1}'.format(method, ncum))
				if column_index == 0:
					plt.ylabel('t_sim = {0:.1e}'.format(plot.params['sim']['simulation_time']))

def mean_grid_score_time_evolution():
	mpl.style.use('ggplot')
	fig = plt.figure(figsize=(14, 10))
	gs = gridspec.GridSpec(3, 4)

	date_dirs = ['2015-09-16-15h08m32s_fast_grids_200',
				 '2015-09-16-18h53m53s_slower_grids_200',
				 '2015-09-16-18h58m05s_fast_grids_200_5_times_longer',]
	for ndd, date_dir in enumerate(date_dirs):
		condition_tuples = [(('sim', 'seed_centers'), 'lt', 200)]
		plot = get_plot_class(date_dir, *condition_tuples)

		gs_dict = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
		methods = ['Weber', 'sargolini']
		for i, method in enumerate(methods):
			# fig.add_subplot(3, 1, ndd+1)
			for k, ncum in enumerate([1, 10]):
				plt.subplot(gs[ndd, gs_dict[(i, k)]])
				grid_scores = plot.get_list_of_grid_score_arrays_over_all_psps(method=method,
																	   n_cumulative=ncum)
				grid_score_mean = np.nanmean(grid_scores, axis=0)
				grid_score_std = np.nanstd(grid_scores, axis=0)
				time = (np.arange(0, len(grid_scores[0]))
						* plot.params['sim']['every_nth_step_weights']
						* plot.params['sim']['dt'])
				plt.plot(time, grid_score_mean)
				plt.fill_between(time, grid_score_mean + grid_score_std, grid_score_mean - grid_score_std,
								 alpha=0.5)
				for j in np.arange(4):
					plt.plot(time, grid_scores[j])
				plt.ylim([-0.5, 1.0])
				plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
				if ndd == 0:
					plt.title('{0}, nc = {1}'.format(method, ncum))
				if column_index == 0:
					plt.ylabel('t_sim = {0:.1e}'.format(plot.params['sim']['simulation_time']))

def one_dimensional_input_tuning(syn_type='exc', n_centers=3, perturbed=False,
				 highlighting=True, one_population=True,
				 decreased_inhibition=False,
				 plot_difference=False, perturbed_exc=False, perturbed_inh=False):
	"""
	Plots 1 dimensional input tuning

	Parameters
	----------
	perturbed : bool

	"""
	figsize = (2.5, 2.0)
	plt.figure(figsize=figsize)
	lw = 1.8
	radius = 1.0
	x = np.linspace(-radius, radius, 2001)
	gaussian = {}
	plt.ylim([-2, 2])
	# plt.plot(x, np.ones_like(x)*0.3, color='black', lw=lw)
	# plt.plot(x, np.sqrt(2*np.pi*sigma[p]**2) * gaussian[p](x), color=colors[p], lw=2)
	plt.margins(0.01)
	plt.ylim([-2, 2])
	plt.xlim([-radius, radius])
	plt.xticks([])
	plt.yticks([])
	plt.axis('off')

	if one_population:
		c_2 = -0.33
		if n_centers == 1:
			centers = np.array([c_2])
		elif n_centers == 3:
			centers = np.array([-0.73, c_2, 0.73])
		else:
			# centers = np.linspace(-1.0, 1.0, n_centers)
			centers = np.arange(-1.73, 1.73, 0.10)

		for c in centers:
			for p, s in sigma.iteritems():
				gaussian[p] = scipy.stats.norm(loc=c, scale=sigma[p]).pdf
				if p == syn_type:
					p_out = p
					print c
					print c_2
					if general_utils.misc.approx_equal(c, c_2, 0.01) and highlighting:
						alpha = 1.0
						scaling_factor = 1.0 if not perturbed else 1.5
					else:
						alpha = 0.2
						scaling_factor = 1.0
					plt.plot(x, scaling_factor* np.sqrt(2*np.pi*sigma[p]**2)
							 * gaussian[p](x), color=colors[p], lw=lw, alpha=alpha)
	else:
		# plt.plot(x, np.ones_like(x)*2, linestyle='none')
		# plt.plot(x, -np.ones_like(x)*2, linestyle='none')

		c_2 = -0.33
		if n_centers == 1:
			centers = np.array([c_2])
		elif n_centers == 3:
			centers = np.array([-0.73, c_2, 0.73])
		else:
			# centers = np.linspace(-1.0, 1.0, n_centers)
			centers = np.arange(-1.73, 1.73, 0.10)

		summe_exc = np.zeros_like(x)
		for c in centers:
			for p, s in sigma.iteritems():
				gaussian[p] = scipy.stats.norm(loc=c, scale=sigma[p]).pdf
				if p == 'exc':
					if general_utils.misc.approx_equal(c, c_2, 0.01) and highlighting:
						alpha = 1.0
						scaling_factor = 1.0 if not perturbed_exc else 1.5
					else:
						alpha = 0.2
						scaling_factor = 1.0
					plt.plot(x, scaling_factor* np.sqrt(2*np.pi*sigma[p]**2)
							 * gaussian[p](x), color=colors[p], lw=lw, alpha=alpha)
					summe_exc += scaling_factor * np.sqrt(2*np.pi*sigma[p]**2) * gaussian[p](x)
				elif p == 'inh':
					if general_utils.misc.approx_equal(c, c_2, 0.01) and highlighting:
						alpha = 1.0
						scaling_factor = 0.7 if not perturbed_inh else 1.5*0.7
					else:
						alpha = 0.2
						scaling_factor = 1.0
						if decreased_inhibition:
							scaling_factor = 0.7
					plt.plot(x, -scaling_factor* np.sqrt(2*np.pi*sigma[p]**2)
							 * gaussian[p](x), color=colors[p], lw=lw, alpha=alpha)


			if plot_difference:
				gaussian_exc = scipy.stats.norm(loc=c_2, scale=sigma['exc']).pdf
				gaussian_inh = scipy.stats.norm(loc=c_2, scale=sigma['inh']).pdf
				plt.plot(x,
						(1.5*np.sqrt(2*np.pi*sigma['exc']**2) * gaussian_exc(x)
						- 1.5*0.7*np.sqrt(2*np.pi*sigma['inh']**2) * gaussian_inh(x)),
						color='black', lw=lw
				)
		# print summe_exc
		# plt.plot(x, summe_exc / len(summe_exc), color=colors['exc'], lw=lw, alpha=alpha)

def eigenvalues():
	"""
	Plots the analytical results of the eigenvalues

	The non zero eigenvalue is obtained from the high density limit

	Note:

	"""
	mpl.rc('font', size=18)
	mpl.rc('legend', fontsize=18)
	params = lsa.params
	fig = plt.figure()
	fig.set_size_inches(4, 2.5)
	k = np.linspace(0, 100, 1000)
	plt.plot(k, lsa.lambda_p_high_density_limit(k, params),
			 color='#01665e', label=r'$\lambda_+$', lw=2)
	plt.plot(k, np.zeros_like(k),
			 color='#d8b365', label=r'$\lambda_-$', lw=2)
	plt.legend()
	ax = plt.gca()
	kmax = 2 * np.pi / lsa.grid_spacing_high_density_limit(params)
	plt.vlines(kmax, -1, 1, linestyle='dashed')
	ax.set(ylim=[-1e-6, 2.5e-6], xticks=[kmax], yticks=[0],
			xticklabels=[r'$k_{\mathrm{max}}$'])
	plt.xlabel(r'Spatial frequency $k$', fontsize=18)
	plt.ylabel(r'Eigenvalue', fontsize=18)


def plot_output_rates_and_gridspacing_vs_parameter(plot_list):
	"""
	Plots gridspacing vs. parameter together with two output rate examples

	Illustrates:
	Using gridspec with a combination of slicing AND height_ratios.

	NOTE, The order of function in plot.py should be something like that:

	('plot_output_rates_from_equation', {'time':  4e7, 'from_file': True,
										 'maximal_rate': False,
										 'publishable': True}),
	('plot_output_rates_from_equation', {'time':  4e7, 'from_file': True,
										 'maximal_rate': False,
										 'publishable': True}),
	 ('plot_grid_spacing_vs_parameter',
			{	'from_file': True,
				'parameter_name': 'sigma_inh',
				'parameter_range': np.linspace(0.08, 0.38, 201),
				# 'parameter_range': np.linspace(0.08, 0.36, 201),
				# 'parameter_range': np.linspace(0.015, 0.055, 200),
				'plot_mean_inter_peak_distance': True,
				'computed_data': False})
	"""
	gs = gridspec.GridSpec(2, 2, height_ratios=[5,1])
	# Output rates small gridspacing
	plt.subplot(gs[1, 0])
	plot_list[0](xlim=np.array([-1.0, 1.0]), selected_psp=0, sigma_inh_label=False)
	# Output rates large gridspacing
	plt.subplot(gs[1, 1])
	plot_list[1](xlim=np.array([-1.0, 1.0]), selected_psp=-1, no_ylabel=True, indicate_gridspacing=False)
	# Gridspacing vs parameter
	plt.subplot(gs[0, :])
	plot_list[2](sigma_corr=True)
	fig = plt.gcf()
	# fig.set_size_inches(2.4, 2.4)
	fig.set_size_inches(2.2, 2.0)
	gs.tight_layout(fig, rect=[0, 0, 1, 1], pad=0.2)


def grid_spacing_vs_gamma():
	"""
	Plots grid spacing vs. gamma = eta_inh * n_inh * alpha_inh**2

	One symbol corresponds to varying either eta, n or sqrt(alpha)
	by the factor given on the x-axis.

	Reentry: Try to use a loop for different ways to change gamma

	Parameters
	----------

	"""
	fig = plt.gcf()
	# fig.set_size_inches(4.6, 4.1)
	fig.set_size_inches(3.2, 3.0)

	tables_dict = {	'eta': get_tables(
					date_dir='2015-07-01-17h53m22s_grid_spacing_VS_eta_inh'),
					'number_per_dimension': get_tables(
					date_dir='2015-07-02-15h08m01s_grid_spacing_VS_n_inh'),
					'gaussian_height': get_tables(
					date_dir='2015-07-04-10h57m42s_grid_spacing_VS_gaussian_height_inh')
	}

	varied_parameters = ['eta', 'number_per_dimension', 'gaussian_height']
	markers = {'eta': 'o',
			   'number_per_dimension': 's',
			   'gaussian_height': 'd'}
	colors = {'eta': color_cycle_blue3[0],
			  'number_per_dimension': color_cycle_blue3[1],
			  'gaussian_height': color_cycle_blue3[2]}
	init_values = {'eta':  2e-3 / (2*3),
				   'number_per_dimension': 200,
				   'gaussian_height': 1.0}
	labels = {'eta': general_utils.plotting.lrinh,
			 'number_per_dimension': general_utils.plotting.ninh,
			 'gaussian_height': general_utils.plotting.ghinh_sq
			 }
	plot_kwargs = {'alpha': 1.0, 'linestyle': 'none', 'markerfacecolor': 'none',
				   'markeredgewidth': 1.0, 'lw': 1
				}

	for vp in varied_parameters:
		tables = tables_dict[vp]
		psps = [p for p in tables.paramspace_pts() if p[('sim', 'initial_x')].quantity < 0.6]
		for psp in psps:
			params = tables.as_dictionary(psp, True)
			# We need this later to plot the theory curve
			if vp == 'eta':
				params_eta = params
			computed = tables.get_computed(psp)
			grid_spacing = computed['mean_inter_peak_distance']
			x = params['inh'][vp] / init_values[vp]
			if vp == 'gaussian_height':
				x = x**2
			plt.plot(x, grid_spacing, marker=markers[vp], color=colors[vp],
								markeredgecolor=colors[vp], **plot_kwargs)
		# Plot it again outside the loop (now with labels)
		# This ensures that the legend appears only once
		plt.plot(x, grid_spacing, marker=markers[vp], color=colors[vp],
				markeredgecolor=colors[vp],
				label='Varied: ' + labels[vp],  **plot_kwargs)
	###########################################################################
	########################## The analytical result ##########################
	###########################################################################
	# Plotting the analytical curve (here I vary eta_inh, but it doesn't
	# matter which one I vary to vary gamma)
	parameter_range = np.linspace(1-0.3, 8+0.3, 201) * 2e-3 / (2*3)
	# Set gaussian height for downward compatibility
	for syn_type in ['exc', 'inh']:
		params_eta[syn_type]['gaussian_height'] = 1
	grid_spacing_theory = (
		lsa.grid_spacing_high_density_limit(
		params=params_eta, varied_parameter=('inh', 'eta'),
		parameter_range=parameter_range))
	plt.plot(parameter_range / init_values['eta'], grid_spacing_theory, lw=2,
					 color='gray', label=r'Theory')
	plt.legend(loc='best', numpoints=1, fontsize=12, frameon=False)
	plt.title(r'$\gamma = \eta_{\mathrm{I}} N_{\mathrm{I}} \alpha_{\mathrm{I}}^2$')
	plt.xlabel(r'$\eta_{\mathrm{I}} N_{\mathrm{I}} \alpha_{\mathrm{I}}^2 / \gamma_0$')
	plt.ylabel(r'$\ell (m)$')
	plt.xlim((0.7, 8.3))


def grid_spacing_vs_sigmainh_and_two_outputrates(indicate_grid_spacing=True,
		analytical_result=True, gaussian_process_inputs=False):
	"""
	Plots grid spacing vs sigma_inh and two output rate examples

	Parameters
	----------
	indicate_grid_spacing : bool
		If True the grid spacing (ell) is indicated in the example
		output rate plot with the higher spacing
	analytical_result : bool
		If True the high density limit analytical result is plotted
		behind the data
	gaussian_process_inputs : bool
		If True the data with GRF inputs is used
		If False, perfect place field input data is used
		NOTE: The value of `gaussian_process_inputs` resets the values
		for `indicate_grid_spacing` and `analytical_result`
	"""

	if gaussian_process_inputs:
		indicate_grid_spacing=False
		analytical_result=False
		from_file=True
		date_dir = '2014-11-24-14h08m24s_gridspacing_vs_sigmainh_GP_input_NEW'
		spacing = 601
	else:
		date_dir = '2014-08-05-11h01m40s_grid_spacing_vs_sigma_inh'
		from_file=False
		spacing = 3001


	tables = get_tables(date_dir=date_dir)
	if gaussian_process_inputs:
		psps = [p for p in tables.paramspace_pts()
				# if p[('sim', 'initial_x')].quantity > 0.6
				if p[('sim', 'seed_centers')].quantity == 0
				# if (p[('inh', 'sigma')].quantity == 0.08 or approx_equal(p[('inh', 'sigma')].quantity, 0.3, 0.001))
				# and p[('inh', 'sigma')].quantity < 0.31
		]
	else:
		psps = [p for p in tables.paramspace_pts()
				if p[('sim', 'initial_x')].quantity > 0.6
				# and (p[('inh', 'sigma')].quantity == 0.08 or approx_equal(p[('inh', 'sigma')].quantity, 0.3, 0.001))
				and p[('inh', 'sigma')].quantity < 0.31
				]

	plot = plotting.Plot(tables, psps)
	mpl.rcParams['legend.handlelength'] = 1.0

	gs = gridspec.GridSpec(2, 2, height_ratios=[5,1])
	###########################################################################
	######################## Grid spacing VS sigma inh ########################
	###########################################################################
	for psp in psps:
		plot.set_params_rawdata_computed(psp, set_sim_params=True)
		output_rates = plot.get_output_rates(-1, spacing, from_file=from_file,
													squeeze=True)
		limit = plot.radius
		linspace = np.linspace(-limit, limit, spacing)
		plt.subplot(gs[0, :])
		grid_spacing = general_utils.arrays.get_mean_inter_peak_distance(
			output_rates, 2*plot.radius, 5, 0.1)
		grid_spacing_vs_param_kwargs = {'marker': 'o',
										'color': color_cycle_blue3[0],
										'linestyle': 'none',
										'markeredgewidth': 0.0,
										'lw': 1}
		plt.plot(plot.params['inh']['sigma'], grid_spacing,
				 **grid_spacing_vs_param_kwargs)

		xlim=[0.00, 0.31]
		ax = plt.gca()
		ax.set(	xlim=xlim, ylim=[0.0, 0.63],
				xticks=[0, 0.03, 0.3], yticks=[0, 0.6],
				xticklabels=['0', general_utils.plotting.width_exc, '0.3'],
				yticklabels=['0', '0.6']
		)
		general_utils.plotting.simpleaxis(ax)
		ax.tick_params(axis='both', direction='out')
		ax.tick_params(axis='', direction='out')
		ax.tick_params(axis='y', direction='out')

		plt.xlabel(general_utils.plotting.width_inh_m, labelpad=-10.0)
		plt.ylabel(r'$\ell [m]$', labelpad=-10.0)

	# Plot it again to get the legend label only once
	plt.plot(plot.params['inh']['sigma'], grid_spacing, label=r'Simulation',
			**grid_spacing_vs_param_kwargs)

	###########################################################################
	############################ Analytical Result ############################
	###########################################################################
	if analytical_result:
		sigma_inh_range = np.linspace(plot.params['exc']['sigma'], xlim[1], 301)
		grid_spacing_theory = (
					lsa.grid_spacing_high_density_limit(
					params=plot.params,
					varied_parameter=('inh', 'sigma'),
					parameter_range=sigma_inh_range,
					sigma_corr=False))
		plt.plot(sigma_inh_range, grid_spacing_theory, color='gray',
				 label=r'Theory', lw=2, zorder=100)
	plt.legend(loc='upper left', numpoints=1, frameon=False)

	###########################################################################
	######################## Two firing rate examples #########################
	###########################################################################
	for psp in psps:
		plot.set_params_rawdata_computed(psp, set_sim_params=True)
		sigma_location = [(0.08, 0), (0.3, 1)]
		for sl in sigma_location:
			if general_utils.misc.approx_equal(plot.params['inh']['sigma'], sl[0], 0.001):
				plt.subplot(gs[1, sl[1]])
				color = 'black'
				output_rates = plot.get_output_rates(-1, spacing, squeeze=True,
													 from_file=from_file)
				plt.plot(linspace, output_rates, color=color, lw=1)
				ax = plt.gca()
				ymax = output_rates.max()
				# ymax = 9.0
				plt.ylim((0, ymax))
				general_utils.plotting.simpleaxis(ax)
				ax.set(
					xlim=[-1, 1],
					xticks=[], yticks=[0, int(ymax)],
					ylabel='', title='')
				plt.margins(0.0, 0.1)
				if sl[1] == 0:
					plt.ylabel('Hz')
				plt.axhline([plot.params['out']['target_rate']],
							dashes=(0.7, 0.7),
							color='gray', lw=0.6, zorder=100)

				if indicate_grid_spacing and sl[1] == 1:
					maxima_positions, maxima_values, grid_score = (
							plot.get_1d_grid_score(output_rates, linspace,
							neighborhood_size=7))
					plot.indicate_grid_spacing(maxima_positions, 4)

	fig = plt.gcf()
	fig.set_size_inches(2.6, 2.15)
	gs.tight_layout(fig, rect=[0, 0, 1, 1], pad=0.2)


def inputs_rates_heatmap(grf_input=False, invariance=False):
	if grf_input:
		end_time = 5e4
		date_dir = '2014-11-20-21h29m41s_heat_map_GP_shorter_time'
		tables = get_tables(date_dir=date_dir)
		psps = [p for p in tables.paramspace_pts()
			if p[('inh', 'weight_factor')].quantity == 1.03]
	else:
		end_time = 15e4
		if invariance:
			date_dir = '2015-07-15-14h39m10s_heat_map_invariance'
		else:
			date_dir = '2015-07-12-17h01m24s_heat_map'
		tables = get_tables(date_dir=date_dir)
		psps = [p for p in tables.paramspace_pts()]
	psp = psps[0]
	plot = plotting.Plot(tables, [psp])
	plot.set_params_rawdata_computed(psp, set_sim_params=True)
	end_frame = plot.time2frame(end_time, weight=True)

	# The meta grid spec (the distribute the two following grid specs
	# on a vertical array of length 5)
	gs0 = gridspec.GridSpec(4, 1)
	# Along the x axis we take the same number of array points for both
	# gridspecs in order to align the axes horizontally
	nx = 50
	# Room for colorbar
	n_cb = 3
	# The number of vertical array points can be chose differently for the
	# two inner grid specs and is used to adjust the vertical distance
	# between plots withing a gridspec
	ny = 102
	n_plots = 2 # Number of plots in the the first gridspec
	# A 'sub' gridspec place on the first fifth of the meta gridspec
	gs00 = gridspec.GridSpecFromSubplotSpec(ny, nx, subplot_spec=gs0[0])

	spacing = plot.params['sim']['spacing']
	limit = plot.params['sim']['radius']
	linspace = np.linspace(-limit, limit, spacing)

	def _input_tuning(syn_type):
		if grf_input:
			alpha = 1.0
			for n in [1, 3]:
				input_rates = plot.rawdata[syn_type]['input_rates'][:, n]
				positions = plot.rawdata['positions_grid']
				plt.plot(positions, input_rates, color=colors[syn_type], alpha=alpha)
				alpha = 0.3
		else:
			# neuron = 100 if syn_type == 'exc' else 50
			for n in np.arange(0, plot.rawdata[syn_type]['number']-1, 5):
				alpha = 0.2
			# plot.fields(neuron=neuron, show_each_field=False, show_sum=True,
			# 			populations=[syn_type], publishable=True)
			# plot.fields(neuron=10, show_each_field=False, show_sum=True,
			# 			populations=[syn_type], publishable=True)
				if syn_type == 'exc':
					if n == 80:
						alpha = 1.0
				else:
					if invariance:
						if n == 20:
							alpha = 1.0
					else:
						if n == 25:
							alpha = 1.0

				plot.fields(neuron=n, show_each_field=False, show_sum=True,
							populations=[syn_type], publishable=True, alpha=alpha)

		ylabel = {'exc': 'Exc', 'inh': 'Inh'}
		ax = plt.gca()
		plt.setp(ax, xticks=[], yticks=[0], ylim=[0, 1.6])
		ax.spines['right'].set_color('none')
		ax.spines['top'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.yaxis.set_label_position("right")
		plt.ylabel(ylabel[syn_type], color=colors[syn_type],
					rotation='horizontal', labelpad=12.0)

	def _output_rate(frame):
		max_rate = 9 if grf_input else 5
		output_rates = plot.get_output_rates(frame, spacing=spacing, from_file=True)
		plt.plot(linspace, output_rates, color='black', lw=1)
		ax = plt.gca()
		general_utils.plotting.simpleaxis(ax)
		ax.set( xticks=[], yticks=[0, max_rate],
				xlabel='', ylabel='Hz')
		plt.axhline([plot.params['out']['target_rate']],
					dashes=(0.7, 0.7),
					color='gray', lw=0.6, zorder=100)
	###########################################################################
	############################ Excitatory input #############################
	###########################################################################
	plt.subplot(gs00[0:ny/n_plots-1, :-n_cb])
	_input_tuning('exc')

	# positions = plot.rawdata['positions_grid']
	# input_rates = plot.rawdata['exc']['input_rates'][:, 0]
	# plt.plot(positions, input_rates, color=colors['exc'])
	# ax = plt.gca()
	# general_utils.plotting.simpleaxis(ax)
	# ax.set(xticks=[], yticks=[0])

	###########################################################################
	############################ Inhibitory input #############################
	###########################################################################
	plt.subplot(gs00[1+ny/n_plots:, :-n_cb])
	_input_tuning('inh')
	# input_rates = plot.rawdata['inh']['input_rates'][:, 0]
	# plt.plot(positions, input_rates, color=colors['inh'])
	# ax = plt.gca()


	###########################################################################
	########################### Initial output rate ###########################
	###########################################################################
	# Now we choose a different number of vertical array points in the
	# gridspec, to allow for independent adjustment of vertical distances
	# within the two sub-gridspecs
	ny = 40
	gs01 = gridspec.GridSpecFromSubplotSpec(ny, nx, subplot_spec=gs0[1:])
	plt.subplot(gs01[0:ny/8, :-n_cb])
	_output_rate(0)
	###########################################################################
	################################ Heat map #################################
	###########################################################################
	vrange = [5+ny/8, 7*ny/8-3]
	plt.subplot(gs01[vrange[0]:vrange[1], :-n_cb])
	vmax = 5
	if grf_input:
		vmax=None
	heat_map_rates = plot.output_rate_heat_map(from_file=True,
											   end_time=end_time,
											   publishable=True,
											   maximal_rate=vmax)
	ax = plt.gca()
	ax.set( xticks=[], yticks=[],
			xlabel='', ylabel='')
	plt.ylabel('Time', labelpad=12.0)
	###########################################################################
	############################ Final output rate ############################
	###########################################################################
	plt.subplot(gs01[7*ny/8:, :-n_cb])
	_output_rate(end_frame)
	###########################################################################
	######################### Color bar for heat map ##########################
	###########################################################################
	# The colorbar is plotted right next to the heat map
	plt.subplot(gs01[vrange[0]:vrange[1], nx-2:])
	vmin = 0.0
	if not invariance:
		vmax = np.amax(heat_map_rates)
	ax1 = plt.gca()
	cm = mpl.cm.gnuplot_r
	norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
	cb = mpl.colorbar.ColorbarBase(ax1, cmap=cm, norm=norm, ticks=[int(vmin), int(vmax)])
	# Negative labelpad puts the label further inwards
	# For some reason labelpad needs to be manually adjustes for different
	# simulations. No idea why!
	# Take -7.0 (for general inputs) and -1.0 for ideal inputs
	labelpad = -7.0 if grf_input else -1.0
	# labelpad=-1.0
	cb.set_label('Hz', rotation='horizontal', labelpad=labelpad)
	fig = plt.gcf()
	fig.set_size_inches(2.2, 2.6)
	gs0.tight_layout(fig, rect=[0, 0, 1, 1], pad=0.2)

def different_grid_spacings_in_line():
	date_dir = '2015-07-09-16h10m55s_different_grid_spacings'
	tables = get_tables(date_dir=date_dir)
	psps = [p for p in tables.paramspace_pts()]
	plot = plotting.Plot(tables=tables, psps=psps)
	gs = gridspec.GridSpec(2, 4, height_ratios=[1,1])
	for n, psp in enumerate(psps):
		plot.set_params_rawdata_computed(psp, set_sim_params=True)
		plt.subplot(gs[0, n])
		# max_rate = 9 if grf_input else 5
		if (plot.params['inh']['sigma'] == 0.3 or plot.params['inh']['sigma'] == 4.0):
			end_frame = plot.time2frame(1e5, weight=True)
		else:
			end_frame = -1
		output_rates = plot.get_output_rates(end_frame, spacing=None,
											 from_file=True)
		plt.plot(output_rates, color='gray', lw=2)
		ax = plt.gca()
		general_utils.plotting.simpleaxis(ax)
		ax.spines['left'].set_color('none')
		ax.set(
			xticks=[], yticks=[],
			ylim=[0, 16.5]
		)
		plt.axhline([plot.params['out']['target_rate']],
					dashes=(4, 4),
					color='black', lw=2, zorder=100, label=r'Target rate')
		if n == 0:
			plt.plot(output_rates, color='gray', lw=2, label='Output rate')
			plt.legend(loc='center left', numpoints=1, frameon=False, fontsize=12)
			plt.yticks([0])

	fig = plt.gcf()
	fig.set_size_inches(6, 5)
	gs.tight_layout(fig, rect=[0, 0, 1, 1], pad=0.2)

def sigma_x_sigma_y_matrix(to_plot='rate_map', time=-1):
	"""
	A matrix with firing rates or correlograms vs sigma_x, sigma_y of inhibition

	Parameters
	----------
	to_plot : str
		'rate_map': The firing rate map is plotted
		'correlogram': The auto correlogram of the firing rate map is plotted
	time : float
		Time at which the data is plotted (take -1 for final time)
	-------
	"""
	date_dir = '2015-07-11-11h54m34s_sigmax_sigmay_matrix'
	tables = get_tables(date_dir=date_dir)
	psps = [p for p in tables.paramspace_pts()
			# if p[('sim', 'seed_centers')].quantity == 0
			# if np.array_equal(p[('inh', 'sigma')].quantity, [0.10, 0.10])
	]
	plot = plotting.Plot(tables=tables, psps=psps)
	m = 4
	gs = gridspec.GridSpec(m, m)

	seed_sigmax_sigmay_gsrow_gscolumn = [
		# No tuning
		(2, 0.049, 0.049, -1, 0),
		# Grid cell small spacing
		(0, 0.1, 0.1, -1, 1),
		# Grid cell large spacing
		(0, 0.2, 0.2, -1, 2),
		# Place cell
		(3, 2.0, 2.0, -1, 3),
		# Vertical band cell small spacing
		# (3, 0.1, 0.049, -1, 1),
		# Vertical band cell large spacing
		# (0, 0.20, 0.049, -1, 2),
		# Vertical band cell single stripe
		# (2, 2.0, 0.049, -1, 3),
		# Horizontal band cell small spacing
		(2, 0.049, 0.1, -2, 0),
		# Horizontal band cell large spacing
		(1, 0.049, 0.2, -3, 0),
		# Horizontal band cell single stripe
		(1, 0.049, 2.0, -4, 0),
		### The weird types ###
		(3, 0.1, 0.2, -3, 1),
		(3, 0.1, 2.0, -4, 1),
		(2, 0.2, 0.1, -2, 2),
		(2, 0.2, 2.0, -4, 2),
		(2, 2.0, 0.1, -2, 3),
		(2, 2.0, 0.2, -3, 3),
	]

	for psp in psps:
		seed_sigmax_sigmay = (
		psp['sim', 'seed_centers'].quantity,
		psp['inh', 'sigma'].quantity[0],
		psp['inh', 'sigma'].quantity[1])

		for t in seed_sigmax_sigmay_gsrow_gscolumn:
			# Only do something if the (seed, sigma_x, sigma_y) tuple is
			# supposed to be plotted
			if seed_sigmax_sigmay == t[:3]:
				gsrow, gscolumn = t[3], t[4]
				plt.subplot(gs[gsrow, gscolumn])
				plot.set_params_rawdata_computed(psp, set_sim_params=True)
				if to_plot == 'rate_map':
					linspace = np.linspace(-plot.radius , plot.radius, plot.spacing)
					X, Y = np.meshgrid(linspace, linspace)
					output_rates = plot.get_output_rates(time, plot.spacing, from_file=True)
					maximal_rate = int(np.ceil(np.amax(output_rates)))
					V = np.linspace(0, maximal_rate, 40)
					plt.contourf(X, Y, output_rates[...,0], V)
				elif to_plot == 'correlogram':
					corr_linspace, correlogram = plot.get_correlogram(
											time=time, spacing=plot.spacing,
											mode='same', from_file=True)
					X_corr, Y_corr = np.meshgrid(corr_linspace, corr_linspace)
					V = np.linspace(-1.0, 1.0, 40)
					plt.contourf(X_corr, Y_corr, correlogram, V)
				ax = plt.gca()
				plt.setp(ax, xticks=[], yticks=[])

	fig = plt.gcf()
	fig.set_size_inches(6, 6)
	gs.tight_layout(fig, pad=0.7)


def single_output_stuff(to_plot='rate_map', time=-1):
	"""
	Output rate single version
	-------
	"""
	date_dir = '2015-07-11-11h54m34s_sigmax_sigmay_matrix'
	tables = get_tables(date_dir=date_dir)
	psps = [p for p in tables.paramspace_pts()
			# if p[('sim', 'seed_centers')].quantity == 0
			# if np.array_equal(p[('inh', 'sigma')].quantity, [0.10, 0.10])
	]
	plot = plotting.Plot(tables=tables, psps=psps)
	m = 4
	gs = gridspec.GridSpec(m, m)

	seed_sigmax_sigmay_gsrow_gscolumn = [
		# No tuning
		(2, 0.049, 0.049, -1, 0),
		# Grid cell small spacing
		(0, 0.1, 0.1, -1, 1),
		# Grid cell large spacing
		(0, 0.2, 0.2, -1, 2),
		# Place cell
		(3, 2.0, 2.0, -1, 3),
		# Vertical band cell small spacing
		# (3, 0.1, 0.049, -1, 1),
		# Vertical band cell large spacing
		# (0, 0.20, 0.049, -1, 2),
		# Vertical band cell single stripe
		# (2, 2.0, 0.049, -1, 3),
		# Horizontal band cell small spacing
		(2, 0.049, 0.1, -2, 0),
		# Horizontal band cell large spacing
		(1, 0.049, 0.2, -3, 0),
		# Horizontal band cell single stripe
		(1, 0.049, 2.0, -4, 0),
		### The weird types ###
		(3, 0.1, 0.2, -3, 1),
		(3, 0.1, 2.0, -4, 1),
		(2, 0.2, 0.1, -2, 2),
		(2, 0.2, 2.0, -4, 2),
		(2, 2.0, 0.1, -2, 3),
		(2, 2.0, 0.2, -3, 3),
	]

	for psp in psps:
		seed_sigmax_sigmay = (
		psp['sim', 'seed_centers'].quantity,
		psp['inh', 'sigma'].quantity[0],
		psp['inh', 'sigma'].quantity[1])

		for t in seed_sigmax_sigmay_gsrow_gscolumn:
			# Only do something if the (seed, sigma_x, sigma_y) tuple is
			# supposed to be plotted
			if seed_sigmax_sigmay == t[:3]:
				gsrow, gscolumn = t[3], t[4]
				plt.subplot(gs[gsrow, gscolumn])
				plot.set_params_rawdata_computed(psp, set_sim_params=True)
				if to_plot == 'rate_map':
					linspace = np.linspace(-plot.radius , plot.radius, plot.spacing)
					X, Y = np.meshgrid(linspace, linspace)
					output_rates = plot.get_output_rates(time, plot.spacing, from_file=True)
					maximal_rate = int(np.ceil(np.amax(output_rates)))
					V = np.linspace(0, maximal_rate, 40)
					plt.contourf(X, Y, output_rates[...,0], V)
				elif to_plot == 'correlogram':
					corr_linspace, correlogram = plot.get_correlogram(
											time=time, spacing=plot.spacing,
											mode='same', from_file=True)
					X_corr, Y_corr = np.meshgrid(corr_linspace, corr_linspace)
					V = np.linspace(-1.0, 1.0, 40)
					plt.contourf(X_corr, Y_corr, correlogram, V)
				ax = plt.gca()
				plt.setp(ax, xticks=[], yticks=[])

	fig = plt.gcf()
	fig.set_size_inches(6, 6)
	gs.tight_layout(fig, pad=0.7)

def two_dimensional_input_tuning():
	x, y = np.mgrid[-1:1:.01, -1:1:.01]
	pos = np.dstack((x, y))
	gaussian = scipy.stats.multivariate_normal([0., 0.], np.power([0.05, 0.1], 2))
	vanishing_value = 1e-1
	# fields = field(positions, location, sigma_exc).reshape((n_x, n_x))
	cm = mpl.cm.Reds
	my_masked_array = np.ma.masked_less(gaussian.pdf(pos), vanishing_value)
	plt.contourf(x, y, my_masked_array, cmap=cm)
	ax = plt.gca()
	# ax.set_aspect('equal')
	plt.setp(ax, aspect='equal', xticks=[], yticks=[])
	plt.axis('off')


if __name__ == '__main__':
	# If you comment this out, then everything works, but in matplotlib fonts
	# mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
	# mpl.rc('text', usetex=True)

	# plot_function = one_dimensional_input_tuning
	# plot_function = two_dimensional_input_tuning
	# plot_function = sigma_x_sigma_y_matrix
	# plot_function = inputs_rates_heatmap
	# plot_function = one_dimensional_input_tuning
	# plot_function = mean_grid_score_time_evolution
	plot_function = grid_score_histogram
	# syn_type = 'inh'
	# plot_function(syn_type=syn_type, n_centers=20, highlighting=True,
	# 			  perturbed=False, one_population=False, decreased_inhibition=True,
	# 			  perturbed_exc=True, perturbed_inh=True, plot_difference=True)
	# plot_function(time=-1, to_plot='correlogram')
	# plot_function(syn_type='inh')
	plot_function()
	prefix = ''
	sufix =''
	save_path = '/Users/simonweber/doktor/TeX/learning_grids/figs/' \
				+ prefix + '_' + plot_function.__name__ + '_' + sufix + '.png'
	plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0.015,
				transparent=True)
	# plt.show()