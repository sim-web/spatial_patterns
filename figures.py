__author__ = 'simonweber'
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import analytics.linear_stability_analysis as lsa
# open the tablefile
# from snep.configuration import config
# config['network_type'] = 'empty'
import snep.utils
import general_utils.arrays
import general_utils.plotting
import general_utils.misc
import general_utils.snep_plotting
from matplotlib.patches import ConnectionPatch
from matplotlib import gridspec
import plotting
import utils
import time
import matplotlib.mlab as mlab


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
color_cycle_blue4 = general_utils.plotting.color_cycle_blue4

def get_tables(date_dir):
	tables = snep.utils.make_tables_from_path(
	general_utils.snep_plotting.get_path_to_hdf_file(date_dir))
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
			(('sim', 'seed_centers'), 'lt', 10),
			(('exc', 'sigma'), 'eq', np.array([0.05, 1.0]))
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
		# date_dir = '2014-11-24-14h08m24s_gridspacing_vs_sigmainh_GP_input_NEW'
		date_dir = '2015-12-16-11h19m42s_grid_spacing_vs_sigma_inh_GP_less_inh_cells'
		spacing = 601
	else:
		# date_dir = '2014-08-05-11h01m40s_grid_spacing_vs_sigma_inh'
		date_dir = '2015-12-09-11h30m08s_grid_spacing_vs_sigma_inh_less_inhibitory_inputs'
		from_file=False
		spacing = 3001


	tables = get_tables(date_dir=date_dir)
	if gaussian_process_inputs:
		psps = [p for p in tables.paramspace_pts()
				# if p[('sim', 'initial_x')].quantity > 0.6
				if p[('sim', 'seed_centers')].quantity == 1
				# if (p[('inh', 'sigma')].quantity == 0.08 or approx_equal(p[('inh', 'sigma')].quantity, 0.3, 0.001))
				# and p[('inh', 'sigma')].quantity < 0.31
		]
	else:
		psps = [p for p in tables.paramspace_pts()
				# if p[('sim', 'initial_x')].quantity > 0.6
				if p[('sim', 'seed_centers')].quantity == 2
				# and (p[('inh', 'sigma')].quantity == 0.08 or approx_equal(p[('inh', 'sigma')].quantity, 0.3, 0.001))
				and p[('inh', 'sigma')].quantity < 0.31
				]

	plot = plotting.Plot(tables, psps)
	mpl.rcParams['legend.handlelength'] = 1.0

	gs = gridspec.GridSpec(2, 2, height_ratios=[5,1])
	sigma_location = [(0.08, 0), (0.3, 1)]
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
		sigma_inh = plot.params['inh']['sigma']
		plt.plot(sigma_inh, grid_spacing,
				 **grid_spacing_vs_param_kwargs)

		### Add stars to annotate the parameters that are plotted below ***
		symbol_sigma = [('*', sigma_location[0][0]),
						('**', sigma_location[1][0])]
		for symbol, sigma in symbol_sigma:
			if general_utils.misc.approx_equal(sigma_inh, sigma):
				plt.annotate(symbol, (sigma_inh, grid_spacing+0.04),
							 va='center', ha='center', color='black')

		xlim=[0.00, 0.31]
		ax = plt.gca()
		ax.set(	xlim=xlim, ylim=[0.0, 0.71],
				xticks=[0, 0.03, 0.3], yticks=[0, 0.7],
				xticklabels=['0', general_utils.plotting.width_exc, '0.3'],
				yticklabels=['0', '0.7']
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
		for n, sl in enumerate(sigma_location):
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
					ylabel='', title=symbol_sigma[n][0])
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


def inputs_rates_heatmap(input='grf', colormap='viridis'):
	"""
	Plots input examples init firing rate, heat map and final firing rate

	Parameters
	----------
	input : str
		'grf': For gaussian random field inputs
		'precise': For precise inhibition
		'gaussian': For Gaussian inputs

	Returns
	-------
	"""
	if input == 'grf':
		end_time = 5e4
		date_dir = '2014-11-20-21h29m41s_heat_map_GP_shorter_time'
		tables = get_tables(date_dir=date_dir)
		psps = [p for p in tables.paramspace_pts()
			if p[('inh', 'weight_factor')].quantity == 1.03]
	elif input == 'gaussian':
		end_time = 15e4
		if input == 'precise':
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
		if input == 'grf':
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
					if input == 'precise':
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
		max_rate = 9 if (input == 'grf') else 5
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
	# arrow_kwargs = dict(head_width=0.2,
	# 	head_length=0.05,
	# 	color='black',
	# 	length_includes_head=True,
	# 	lw=1)
	# plt.arrow(0, 1.4, 0.97, 0,
	# 		  **arrow_kwargs)
	# plt.arrow(0, 1.4, -0.97, 0,
	# 		  **arrow_kwargs)
	# plt.xlabel('2 m', fontsize=12, labelpad=0.)
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
	if input == 'grf':
		vmax= 9
	elif input == 'gaussian':
		vmax = 4
	elif input == 'precise':
		vmax = 5

	heat_map_rates = plot.output_rate_heat_map(from_file=True,
											   end_time=end_time,
											   publishable=True,
											   maximal_rate=vmax,
											   colormap=colormap)
	# if input == 'gaussian':
	# 	vmax = np.amax(heat_map_rates)

	ax = plt.gca()
	ax.set( xticks=[], yticks=[],
			xlabel='', ylabel='')
	plt.ylabel('Time [a.u.]', labelpad=12.0)
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
	ax1 = plt.gca()
	cm = getattr(mpl.cm, colormap)
	norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
	cb = mpl.colorbar.ColorbarBase(ax1, cmap=cm, norm=norm, ticks=[int(vmin), int(vmax)])
	# Negative labelpad puts the label further inwards
	# For some reason labelpad needs to be manually adjustes for different
	# simulations. No idea why!
	# Take -7.0 (for general inputs) and -1.0 for ideal inputs
	# labelpad = -7.0 if (input == 'grf') else -1.0
	labelpad = -1.5
	cb.set_label('Hz', rotation='horizontal', labelpad=labelpad)
	fig = plt.gcf()
	fig.set_size_inches(2.3, 2.7)
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
		# max_rate = 9 if (input == 'grf') else 5
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

def sigma_x_sigma_y_matrix(to_plot='rate_map', time=-1, colormap='viridis'):
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
	cm = getattr(mpl.cm, colormap)

	seed_sigmax_sigmay_gsrow_gscolumn = [
		# No tuning
		(2, 0.049, 0.049, -1, 0),
		# Grid cell small spacing
		(0, 0.1, 0.1, -2, 1),
		# Grid cell large spacing
		(0, 0.2, 0.2, -3, 2),
		# Place cell
		(3, 2.0, 2.0, -4, 3),
		# Vertical band cell small spacing
		(3, 0.1, 0.049, -1, 1),
		# Vertical band cell large spacing
		(0, 0.20, 0.049, -1, 2),
		# Vertical band cell single stripe
		(2, 2.0, 0.049, -1, 3),
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
					# maximal_rate = 10.0
					V = np.linspace(0, maximal_rate, 40)
					plt.contourf(X, Y, output_rates[...,0], V, cmap=cm, extend='max')
				elif to_plot == 'correlogram':
					corr_linspace, correlogram = plot.get_correlogram(
											time=time, spacing=plot.spacing,
											mode='same', from_file=True)
					X_corr, Y_corr = np.meshgrid(corr_linspace, corr_linspace)
					V = np.linspace(-1.0, 1.0, 40)
					plt.contourf(X_corr, Y_corr, correlogram, V, cmap=cm)
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


# def plot_grid_score_histogram(grid_scores, start_frame=0, end_frame=-1):
# 	"""
# 	Grid score histogram
#
# 	Parameters
# 	----------
#
#
# 	Returns
# 	-------
# 	"""
# 	# date_dir = '2016-04-01-10h24m43s_600_minutes_very_fast_learning'
# 	# tables = get_tables(date_dir=date_dir)
# 	# plot = plotting.Plot(tables=tables, psps=None)
# 	# grid_score = plot.tables.get_computed(None)['grid_score']
# 	# print grid_score['Weber']['1'].shape
# 	hist_kwargs = {'alpha': 0.5, 'bins': 20}
# 	grid_scores = grid_score['Weber']['1'][:, end_frame]
# 	grid_scores = grid_scores[~np.isnan(grid_scores)]
# 	plt.hist(grid_scores, **hist_kwargs)
# 	print grid_scores

def dummy_plot(aspect_ratio_equal=False, contour=False):
	if contour:
		delta = 0.025
		x = np.arange(-3.0, 3.0, delta)
		y = np.arange(-3.0, 3.0, delta)
		X, Y = np.meshgrid(x, y)
		Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
		Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
		# difference of Gaussians
		Z = 10.0 * (Z2 - Z1)
		plt.contourf(X, Y, Z)
	else:
		plt.plot([1,2,3,4])
	if aspect_ratio_equal:
		ax = plt.gca()
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])
	return plt.gca()

def _grid_score_histogram(
		grid_spec, plot_class, grid_scores, seed=0, end_frame=-1, dummy=False,
		grid_score_marker=False, show_number_of_simulations=True):
	ax = plt.gcf().add_subplot(grid_spec)
	if not dummy:
		plot_class.plot_grid_score_histogram(grid_scores, end_frame=end_frame)
		if grid_score_marker:
			init_grid_score = grid_scores[seed, :][0]
			final_grid_score = grid_scores[seed, :][-1]
			colors = {'init': color_cycle_blue4[2], 'final': color_cycle_blue4[0]}
			grid_score_arrow(init_grid_score, color=colors['init'])
			grid_score_arrow(final_grid_score, color=colors['final'])
		if show_number_of_simulations:
			### WARNING: shows on how many simulations the histogram is based on
			plt.text(0.5, 0.5, str(len(grid_scores)),
					 horizontalalignment='center',
					 verticalalignment='center',
					 transform=plt.gca().transAxes, color='red')

	else:
		dummy_plot()
	return ax

def grid_score_arrow(grid_score, color):
	if not np.isnan(grid_score):
		ax = plt.gca()
		trans = mpl.transforms.blended_transform_factory(
							ax.transData, ax.transAxes)
		plt.annotate(
			 '', xy=(grid_score, 0.2), xycoords=trans,
			xytext=(grid_score, 0), textcoords=trans,
			arrowprops={'arrowstyle': '<-', 'shrinkA': 1, 'shrinkB': 1, 'lw':1.5,
						'mutation_scale': 10., 'color': color})
	else:
		pass

def _grid_score_evolution_with_individual_traces(
		grid_spec, date_dir, seed, end_frame=None, dummy=False):
	plt.subplot(grid_spec)
	if not dummy:
		seed_centers = [seed, 1, 2]
		plot = get_plot_class(
			date_dir,
			(('sim', 'seed_centers'), 'eq', seed)
		)
		grid_scores_fast_learning = plot.computed_full['grid_score']['sargolini']['1']
		plot.plot_grid_score_evolution(grid_scores_fast_learning,
									   end_frame=end_frame,
									   seed_centers=seed_centers)
	else:
		plot = None
		grid_scores_fast_learning = None
		dummy_plot()
	return plot, grid_scores_fast_learning

def trajectories_time_evolution_and_histogram(seed=140):
	plt.figure(figsize=(13,5))
	gs = gridspec.GridSpec(1, 2)
	###########################################################################
	############################ The trajectories ############################
	###########################################################################
	gs_trajectories = gridspec.GridSpecFromSubplotSpec(2,4,gs[0,0], wspace=0.1, hspace=1.0)
	plot_fast_learning = get_plot_class(
		'2016-04-19-12h32m07s_180_minutes_trajectories_fast_learning',
		(('sim', 'seed_centers'), 'eq', seed))
	plot_slow_learning = get_plot_class(
		'2016-04-19-12h32m57s_180_minutes_trajectories_one_third_learning',
		(('sim', 'seed_centers'), 'eq', seed)
	)
	def trajectory_plotting(grid_spec, start_frame, end_frame, plot_class):
		plt.subplot(grid_spec)
		plot_class.trajectory_with_firing(
			start_frame=start_frame, end_frame=end_frame, show_title=False,
			symbol_size=1, max_rate_for_colormap=None)
		# dummy_plot(aspect_ratio_equal=True)

	def plot_title(t1, t2):
		time_1 = int(t1 / 3000.)
		time_2 = int(t2 / 3000.)
		title = '{0} - {1} min'.format(time_1, time_2)
		plt.title(title, fontsize=12)

	t0, t1, t2, t3, t4, t5 = 0., 3e4, 9e4, 18e4, 45e4, 54e4
	time_tuples = [(t0, t1), (t0, t2), (t2, t3), (t4, t5)]
	for n, tt in enumerate(time_tuples):
		trajectory_plotting(gs_trajectories[0,n], tt[0], tt[1], plot_fast_learning)
		plot_title(tt[0], tt[1])
		trajectory_plotting(gs_trajectories[1,n], tt[0], tt[1], plot_slow_learning)

	###########################################################################
	######################## The grid score evolution ########################
	###########################################################################
	gs_evo_hist = gridspec.GridSpecFromSubplotSpec(2,2,gs[0,1], wspace=0.3, hspace=1.0)
	date_dir = '2016-04-01-10h24m43s_600_minutes_very_fast_learning'
	plot_fast_learning, grid_scores_fast_learning = (
		_grid_score_evolution_with_individual_traces(
							grid_spec=gs_evo_hist[0, 0],
							date_dir=date_dir,
							seed=seed,
							dummy=False)
	)
	plt.title('Time course')

	date_dir = '2016-05-10-16h51m48s_600_minutes_500_simulations_1_fps'
	plot_slow_learning, grid_scores_slow_learning = (
		_grid_score_evolution_with_individual_traces(
							grid_spec=gs_evo_hist[1, 0],
							date_dir=date_dir,
							seed=seed,
							dummy=False)
	)

	###########################################################################
	######################## The grid score histograms ########################
	###########################################################################
	_grid_score_histogram(
		grid_spec=gs_evo_hist[0, 1],
		plot_class=plot_fast_learning, grid_scores=grid_scores_fast_learning,
		seed=seed, dummy=False, grid_score_marker=True
	)
	plt.title('Grid score histogram')

	_grid_score_histogram(
		grid_spec=gs_evo_hist[1, 1],
		plot_class=plot_slow_learning, grid_scores=grid_scores_slow_learning,
		seed=seed, dummy=False, grid_score_marker=True
	)


class Figure():
	"""
	Convenience class for the newer plotting functions
	
	Only introduced to make it easier to divide the plotting of a
	single figure into multiple functions.
	"""
	def __init__(self, colormap='viridis'):
		self.colormap = colormap

	def figure_2_grids(self, colormap='viridis'):
		"""
		Plots input examples, initial and final rate map and correlogram ...

		The figure is aranged in 2 grid_specs.
		gs_main: Specifies how many rows we have. We take one row
					for a simulation
		gs_one_row: The grid spec of a single row. See comment below. This a
				subgrid of gs_main.
		Another grid spec is defined within gs_one_row.

		NB: This used to be done in plotting.py

		Parameters
		----------



		Returns
		-------
		"""
		self.time_init = 0
		self.time_final = 18e5
		self.show_initial_correlogram = True
		# All the different simulations that are plotted.
		plot_classes = [
			get_plot_class(
			'2016-05-09-16h39m38s_600_minutes_examples_good_and_bad',
			(('sim', 'seed_centers'), 'eq', 9)),
			get_plot_class(
			'2016-04-25-14h42m02s_100_fps_examples',
			(('sim', 'seed_centers'), 'eq', 333)),
			get_plot_class(
			'2016-05-10-12h55m32s_600_minutes_GRF_examples_BEST',
			(('sim', 'seed_centers'), 'eq', 287)),
			# get_plot_class(
			# '2016-04-26-10h55m00s_100_fps_random_centers',
			# (('sim', 'seed_centers'), 'eq', 92)),
			# get_plot_class(
			# '2016-04-19-12h32m07s_180_minutes_trajectories_fast_learning',
			# (('sim', 'seed_centers'), 'eq', 105)),
			# get_plot_class(
			# '2016-04-19-12h32m07s_180_minutes_trajectories_fast_learning',
			# (('sim', 'seed_centers'), 'eq', 124)),
			# get_plot_class(
			# '2016-04-19-12h32m07s_180_minutes_trajectories_fast_learning',
			# (('sim', 'seed_centers'), 'eq', 141)),
			# get_plot_class(
			# '2016-04-19-12h32m07s_180_minutes_trajectories_fast_learning',
			# (('sim', 'seed_centers'), 'eq', 442)),
			]

		n_simulations = len(plot_classes)
		# Grid spec for all the rows:
		gs_main = gridspec.GridSpec(n_simulations, 1)

		self.plot_the_rows_of_input_examples_rate_maps_and_correlograms(
			grid_spec = gs_main, plot_classes=plot_classes)

		# It's crucial that the figure is not too high, because then the smaller
		# squares move to the top and bottom. It is a bit trick to work with
		# equal aspect ratio in a gridspec
		# NB: The figure width is the best way to justify the wspace, because
		# the function of wspace is limited since we use figures with equal
		# aspect ratios.
		fig = plt.gcf()
		fig.set_size_inches(6.6, 1.1*n_simulations)
		gs_main.tight_layout(fig, pad=0.2, w_pad=0.0)

	def figure_4_cell_types(self, show_initial_correlogram=False):
		self.show_initial_correlogram = show_initial_correlogram
		self.time_init = 0
		self.time_final = 2e7
		# All the different simulations that are plotted.
		plot_classes = [
			get_plot_class(
			'2015-07-13-22h35m10s_GRF_all_cell_types',
				(('sim', 'seed_centers'), 'eq', 6),
				(('inh', 'sigma'), 'eq', np.array([2.0, 2.0]))
			),
			get_plot_class(
			'2015-08-05-17h06m08s_2D_GRF_invariant',
				(('sim', 'seed_centers'), 'eq', 4),
				(('inh', 'sigma'), 'eq', np.array([0.049, 0.049]))
			),
			get_plot_class(
			'2015-07-13-22h35m10s_GRF_all_cell_types',
				(('sim', 'seed_centers'), 'eq', 0),
				(('inh', 'sigma'), 'eq', np.array([0.3, 0.049]))
			),
		]

		n_simulations = len(plot_classes)
		# Grid spec for all the rows:
		gs_main = gridspec.GridSpec(n_simulations, 1)
		self.plot_the_rows_of_input_examples_rate_maps_and_correlograms(
			grid_spec = gs_main, plot_classes=plot_classes)
		# It's crucial that the figure is not too high, because then the smaller
		# squares move to the top and bottom. It is a bit trick to work with
		# equal aspect ratio in a gridspec
		# NB: The figure width is the best way to justify the wspace, because
		# the function of wspace is limited since we use figures with equal
		# aspect ratios.
		fig = plt.gcf()
		fig.set_size_inches(6.6, 1.1*n_simulations)
		gs_main.tight_layout(fig, pad=0.2, w_pad=0.0)

	def plot_the_rows_of_input_examples_rate_maps_and_correlograms(self,
																   grid_spec,
																   plot_classes):
		"""
		Plots all the rows in a input, ratemaps and correlograms plot.

		Convenience function.

		Parameters
		----------
		grid_spec : gridspec object
			The main grid in which the rows are embedded
		plot_classes : list of classes
			All the plot classes from the different input files from which
			data is plotted

		Returns
		-------
		"""
		# Taking a width ratio with 6 entries always is intentional!
		# This way the resulting figures has exactly the same proportions
		# with and without the initial correlogram
		# I have no idea why!
		width_ratios = [0.001, 0.7, 1, 1, 1, 1]
		n_columns = 6 if self.show_initial_correlogram else 5
		for row, plot in enumerate(plot_classes):
			# Grid Spec for inputs, init rates, final rates, correlogram
			# NB 1: we actually create a grid spec of shape (1,6) even though
			# we only need one of shape (1,5). For some reason the colorbar
			# in the first rate map is only plotted if this plot is not the
			# first (starting from the left) element in a grid spec that is
			# not a subgrid. we make this first element vanishingly small.
			# NB2 2: By adjusting the width ratio of the input example subgrid,
			# we can modify the wspace between the excitatory and the
			# inhibitory inputs. Using wpace in the gridspec doesn't work,
			# because we use an equal apsect ratio.
			top_row = True if row == 0 else False
			gs_one_row = gridspec.GridSpecFromSubplotSpec(1, n_columns, grid_spec[row, 0],
														wspace=0.0,
														hspace=0.1,
										width_ratios=width_ratios)
			self.plot_row_of_input_examples_rate_maps_and_correlograms(
							gs_one_row=gs_one_row,
							plot=plot, top_row=top_row)

	def plot_row_of_input_examples_rate_maps_and_correlograms(self,
															  gs_one_row,
															  plot,
															  top_row=False):

		# settings for the rate maps and correlograms
		rate_map_kwargs = dict(from_file=True, maximal_rate=False,
							   show_colorbar=True, show_title=False,
							   publishable=True, colormap=self.colormap,
							   firing_rate_title=top_row,
							   colorbar_label=top_row)
		correlogram_kwargs = dict(from_file=True, mode='same', method=None,
								  publishable=True, colormap=self.colormap,
								  correlogram_title=top_row)

		# Gridspec for the two input examples of each kind (so four in total)
		gs_input_examples = gridspec.GridSpecFromSubplotSpec(2,2, gs_one_row[0, 1],
															 wspace=0.0, hspace=0.1)
		### INPUT ###
		# If there is only 1 field per synapse, you can't plot the input
		# example from the data, because the first three examples (and only
		# those are saved) lie outside the box
		# Therefore you create the Gaussians manually.
		if plot.params['exc']['fields_per_synapse'] == 1 and not \
		plot.params['sim']['gaussian_process']:
			neuron0 = [-0.2, 0.3]
			neuron1 = [0.1, -0.05]
			neuron2 = [0.32, 0.2]
			neuron3 = [-0.05, -0.05]
		else:
			neuron0, neuron1, neuron2, neuron3 = 0, 1, 0, 1
		# Excitation
		plt.subplot(gs_input_examples[0, 0])
		plot.input_tuning(neuron=neuron0, populations=['exc'], publishable=True,
						  plot_title=top_row)
		# dummy_plot(aspect_ratio_equal=True, contour=True)
		plt.subplot(gs_input_examples[1, 0])
		plot.input_tuning(neuron=neuron1, populations=['exc'], publishable=True)
		# dummy_plot(aspect_ratio_equal=True)
		# Inhibition
		plt.subplot(gs_input_examples[0, 1])
		plot.input_tuning(neuron=neuron2, populations=['inh'], publishable=True,
						  plot_title=top_row)
		# dummy_plot(aspect_ratio_equal=True)
		plt.subplot(gs_input_examples[1, 1])
		plot.input_tuning(neuron=neuron3, populations=['inh'], publishable=True)
		# dummy_plot(aspect_ratio_equal=True)

		# Initial rate map
		plt.subplot(gs_one_row[0, 2])
		plot.plot_output_rates_from_equation(time=self.time_init,
											 **rate_map_kwargs)
		# dummy_plot(aspect_ratio_equal=True, contour=True)
		if self.show_initial_correlogram:
			# Initial correlogram
			plt.subplot(gs_one_row[0, -3])
			plot.plot_correlogram(time=self.time_init, **correlogram_kwargs)
			# dummy_plot(aspect_ratio_equal=True)
			if top_row:
				plt.title('Correlogram')

		# Final rate map
		plt.subplot(gs_one_row[0, -2])
		plot.plot_output_rates_from_equation(time=self.time_final,
											 **rate_map_kwargs)

		# dummy_plot(aspect_ratio_equal=True)
		# Final correlogram
		plt.subplot(gs_one_row[0, -1])
		plot.plot_correlogram(time=self.time_final, **correlogram_kwargs)
		# dummy_plot(aspect_ratio_equal=True)

	def histogram_with_rate_map_examples(self, seed_good_example=4,
										 seed_bad_example=19):
		gs_main = gridspec.GridSpec(1, 2, width_ratios=[1, 0.5])
		fig = plt.gcf()


		#####################################################################
		########################### The histogram ###########################
		#####################################################################
		plot_1_fps = get_plot_class(
		'2016-05-10-16h51m48s_600_minutes_500_simulations_1_fps',
		(('sim', 'seed_centers'), 'eq', seed_good_example))
		grid_scores = plot_1_fps.computed_full['grid_score']['sargolini']['1']
		ax_histogram = _grid_score_histogram(gs_main[0, 0], plot_1_fps,
											 grid_scores, dummy=False)
		# fig.add_subplot(gs_main[0, 0])
		# ax_histogram = dummy_plot()

		#####################################################################
		########################### The rate maps ###########################
		#####################################################################
		gs_rate_maps = gridspec.GridSpecFromSubplotSpec(2,1, gs_main[0, 1],
														wspace=0.0,
														hspace=0.1)
		self.rate_map_with_connection_path(grid_spec=gs_rate_maps[0, 0],
										   ax_histogram=ax_histogram,
										   seed=seed_bad_example,
										   time=18e5,
										   dummy=False)

		self.rate_map_with_connection_path(grid_spec=gs_rate_maps[1, 0],
										   ax_histogram=ax_histogram,
										   seed=seed_good_example,
										   time=18e5,
										   dummy=False)

		#####################################################################
		######################## The other histogram ########################
		#####################################################################
		# _grid_score_histogram(gs_main[0, 2], plot_1_fps,
		# 							 grid_scores, dummy=False)
		#
		# _grid_score_histogram(gs_main[0, 3], plot_1_fps,
		# 							 grid_scores, dummy=False)

		# plt.gcf().add_subplot(gs_main[0, 2])
		# dummy_plot()
		#
		# plt.gcf().add_subplot(gs_main[0, 3])
		# dummy_plot()

		fig.set_size_inches(2, 1.1)
		gs_main.tight_layout(fig, pad=0.0, w_pad=0.0)



	def rate_map_with_connection_path(self, grid_spec, ax_histogram,
									  seed=0, time=18e5, dummy=False):
		"""
		Plots a rate map with arrow to shown grid score in histogram.

		Parameters
		----------
		grid_spec : gridspec
			The part of a gridspe in which the rate map is supposed to be
			plotted.
		ax_histogram : axis
			The axis of the histogram to which an arrow is pointed
		seed : int
			The seed_centers value for which the rate map is drawn
		time : float
			Time at which the rate maps is drawn.

		Returns
		-------
		"""
		rate_map_kwargs = dict(from_file=True, maximal_rate=False,
							   show_colorbar=False, show_title=False,
							   publishable=True, colormap=self.colormap,
							   firing_rate_title=False,
							   colorbar_label=False)

		ax = plt.gcf().add_subplot(grid_spec)
		if not dummy:
			plot = get_plot_class(
			'2016-05-09-16h39m38s_600_minutes_examples_good_and_bad',
			(('sim', 'seed_centers'), 'eq', seed))
			plot.plot_output_rates_from_equation(time, **rate_map_kwargs)
			frame = plot.time2frame(time)
			# Grid score of the shown rate map
			grid_score = plot.computed['grid_score']['sargolini']['1'][frame]
			### Drawing the arrow ###
			con = ConnectionPatch(
				xyA=(grid_score, 0.0), xyB=(0, 0.5),
				coordsA='data', coordsB='axes fraction',
				axesA=ax_histogram, axesB=ax,
				arrowstyle='<-',
				shrinkA=1,
				shrinkB=1,
				mutation_scale=10.,
				lw=1.5,
				)
			ax_histogram.add_artist(con)
		else:
			dummy_plot(aspect_ratio_equal=True, contour=True)

	def grid_score_histogram_general_input(self):
		gs_main = gridspec.GridSpec(1, 2)
		fig = plt.gcf()

		#####################################################################
		########################### The histograms ##########################
		#####################################################################
		for n, date_dir in enumerate(['2016-04-22-13h15m01s_100_fps_0.03_best',
					'2016-05-10-16h20m57s_600_minutes_500_simulations_GRF']):
			plot = get_plot_class(
					date_dir, (('sim', 'seed_centers'), 'eq', 0))
			grid_scores = plot.computed_full['grid_score']['sargolini']['1']
			_grid_score_histogram(gs_main[0, n], plot, grid_scores, dummy=False)

		fig.set_size_inches(3.2, 1.1)
		gs_main.tight_layout(fig, pad=0.0, w_pad=0.0)

if __name__ == '__main__':
	t1 = time.time()
	# If you comment this out, then everything works, but in matplotlib fonts
	# mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
	# mpl.rc('text', usetex=True)
	figure = Figure()
	# plot_function = figure.figure_4_cell_types
	# plot_function = figure.figure_2_grids
	# plot_function = figure.histogram_with_rate_map_examples
	plot_function = figure.grid_score_histogram_general_input
	# plot_function = trajectories_time_evolution_and_histogram
	# plot_function = one_dimensional_input_tuning
	# plot_function = two_dimensional_input_tuning
	# plot_function = sigma_x_sigma_y_matrix
	# plot_function = inputs_rates_heatmap
	# plot_function = one_dimensional_input_tuning
	# plot_function = mean_grid_score_time_evolution
	# plot_function = grid_score_histogram
	# plot_function = grid_spacing_vs_sigmainh_and_two_outputrates
	# plot_function = grid_score_histogram
	# syn_type = 'inh'
	# plot_function(syn_type=syn_type, n_centers=20, highlighting=True,
	# 			  perturbed=False, one_population=False, d         ecreased_inhibition=True,
	# 			  perturbed_exc=True, perturbed_inh=True, plot_difference=True)
	# plot_function(time=-1, to_plot='correlogram')
	# plot_function(syn_type='inh')
	# plot_function(indicate_grid_spacing=False, gaussian_process_inputs=True)
	# input = 'grf'
	# plot_function(input=input)
	# for seed in [140, 124, 105, 141, 442]:
	# seed = 140
	plot_function()
	# prefix = input
	prefix = 'test_'
	# sufix = str(seed)
	sufix = ''
	save_path = '/Users/simonweber/doktor/TeX/learning_grids/figs/' \
				+ prefix + '_' + plot_function.__name__ + '_' + sufix + '.png'
	plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.015,
				transparent=False)
	t2 = time.time()
	print 'Plotting took % seconds' % (t2 - t1)
	# plt.show()
