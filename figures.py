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


os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
colors = {'exc': '#D7191C', 'inh': '#2C7BB6', 'diff': '0.4'}
legend = {'exc': 'Excitation', 'inh': 'Inhibition', 'diff': 'Difference'}
legend_short = {'exc': 'Exc.', 'inh': 'Inh.', 'diff': 'Difference'}
signs = {'exc': 1, 'inh': -1}
# sigma = {'exc': 0.025, 'inh': 0.075}
sigma = {'exc': 0.03, 'inh': 0.1}
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

def approx_equal(x, y, tolerance=0.001):
	return abs(x-y) <= 0.5 * tolerance * (abs(x) + abs(y))

def input_tuning(syn_type='exc', n_centers=3, perturbed=False,
				 highlighting=True):
	"""
	Plots 1 dimensional input tuning

	Parameters
	----------
	perturbed : bool

	"""
	# figsize = (1.6, 0.3)
	figsize = (cm2inch(3.), cm2inch(0.5625))
	plt.figure(figsize=figsize)

	radius = 1.0
	x = np.linspace(-radius, radius, 501)
	gaussian = {}

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
				if approx_equal(c, c_2, 0.01) and highlighting:
					alpha = 1.0
					scaling_factor = 1.0 if not perturbed else 1.5
				else:
					alpha = 0.2
					scaling_factor = 1.0
				plt.plot(x, scaling_factor* np.sqrt(2*np.pi*sigma[p]**2)
						 * gaussian[p](x), color=colors[p], lw=0.8, alpha=alpha)

	# plt.plot(x, np.sqrt(2*np.pi*sigma[p]**2) * gaussian[p](x), color=colors[p], lw=2)
	plt.margins(0.05)
	# plt.ylim([-0.03, 1.03])
	# plt.xlim([0.3, 0.7])
	plt.xlim([-radius, radius])
	plt.xticks([])
	plt.yticks([])
	plt.axis('off')

	### OLD PLOTTING OF GAUSSIANS WITH ARROWS INDICATING THE WIDTH ###
	# ax = plt.gca()
	# Set y position for arrow to half the gaussian height
	# y_for_arrow = 0.5
	# Draw an arrow between at height y_for_arrow and between mu-sigma and
	# mu+sigma
	# ax.annotate('', xy=(c-sigma[p], y_for_arrow),  xycoords='data',
	#                 xytext=(c+sigma[p], y_for_arrow), textcoords='data',
	#                 arrowprops=dict(arrowstyle='<->',
	#                 				lw=1,
	#                                 connectionstyle="arc3",
	#                                 shrinkA=0, shrinkB=0)
	#                 )


	# arrowopts = {'shape': 'full', 'lw':1, 'length_includes_head':True,
	# 			'head_length':0.01, 'head_width':0.04, 'color':'black'}
	#
	# arrowlength = sigma[p] - 0.003
	# plt.arrow(0.5, 0.5, arrowlength, 0, **arrowopts)
	# plt.arrow(0.5, 0.5, -arrowlength, 0, **arrowopts)
	# # Put the sigma underneath the arrow
	# sigma_string = {'exc': r'$2 \sigma_{\mathrm{E}}$', 'inh': r'$2 \sigma_{\mathrm{I}}$'}
	# ax.annotate(sigma_string[p], xy=(c, y_for_arrow-0.2), va='top', ha='center')
	#
	# # plt.autoscale(tight=True)
	# # plt.tight_layout()
	# name = 'input_tuning' + '_' + p + '_center_' + str(c).replace('.', 'p') + '.pdf'
	# plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/input_tuning/' + name,
	# 	bbox_inches='tight', pad_inches=0.001)


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

if __name__ == '__main__':
	# If you comment this out, then everything works, but in matplotlib fonts
	# mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
	# mpl.rc('text', usetex=True)

	plot_function = grid_spacing_vs_gamma
	# syn_type = 'exc'
	# plot_function(syn_type=syn_type, n_centers=20, highlighting=False)
	plot_function()
	# sufix = 'many_' + syn_type
	sufix = ''
	save_path = '/Users/simonweber/doktor/TeX/learning_grids/figs/' \
				+ plot_function.__name__ + sufix + '.pdf'
	plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.015,
				transparent=True)
	# plt.show()