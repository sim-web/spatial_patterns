__author__ = 'simonweber'
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import analytics.linear_stability_analysis as lsa

os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

def eigenvalues():
	"""
	Plots the analytical resutls of the eigenvalues

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

if __name__ == '__main__':
	# If you comment this out, then everything works, but in matplotlib fonts
	# mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
	# mpl.rc('text', usetex=True)

	plot_function = eigenvalues
	plot_function()
	save_path = '/Users/simonweber/doktor/TeX/learning_grids/figs/' \
				+ plot_function.__name__ + '.pdf'
	plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.015,
				transparent=True)
	# plt.show()