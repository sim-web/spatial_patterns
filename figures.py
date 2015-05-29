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