import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os

##############################################
##########	General Plotting Stuff	##########
##############################################
mpl.rc('font', size=12)
mpl.rc('legend', fontsize=12)
mpl.rcParams.update({'figure.autolayout': True})

# If you comment this out, then everything works, but in matplotlib fonts
# mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
# mpl.rc('text', usetex=True)

colors = {'exc': '#D7191C', 'inh': '#2C7BB6', 'diff': '0.4'}
legend = {'exc': 'Excitation', 'inh': 'Inhibition', 'diff': 'Difference'}
legend_short = {'exc': 'Exc.', 'inh': 'Inh.', 'diff': 'Difference'}
signs = {'exc': 1, 'inh': -1}
sigma = {'exc': 0.025, 'inh': 0.075}
marker = {'exc': '^', 'inh': 'o'}
populations = ['exc', 'inh']
scaling_factor = {'exc': 1.0, 'inh': 0.5}

fig = plt.figure(figsize=(2.2, 1.5))

sigma_inh = np.linspace(0.08, 0.4, 201)
grid_spacing = np.linspace(0.2, 0.8, 201)
plt.plot(sigma_inh, grid_spacing)
# plt.autoscale(tight=True)
plt.margins(0.02)
mpl.rcParams.update({'figure.autolayout': True})
plt.xlim([0.05, 0.41])
plt.ylim([0.15, 0.84])
plt.xticks([0.1, 0.4])
plt.yticks([0.2, 0.8])
plt.xlabel('s_I (cm)')
# plt.ylabel('Grid spacing')
plt.ylabel('a (cm)')

plt.savefig('/Users/simonweber/doktor/TeX/learning_grids'
			'/grid_spacing_vs_sigma_inh'
			'/test'
			'.pdf',
	bbox_inches='tight', pad_inches=0.001)