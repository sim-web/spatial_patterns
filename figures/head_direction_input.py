import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os
from learning_grids import initialization
import itertools

os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

##############################################
##########	General Plotting Stuff	##########
##############################################
mpl.rc('font', size=18)
mpl.rc('legend', fontsize=18)
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

def von_mises_height_one(theta, sigma, center):
	kappa = 1. / sigma**2
	return np.exp(kappa * np.cos(theta - center) - kappa)


fig = plt.figure(figsize=(3.0, 1.4))
# ax = fig.add_subplot(111, polar=True, axisbg=general_utils.plotting.color_cycle_blue4[3])
# ax = fig.add_subplot(111, polar=True)
ax = fig.add_subplot(111, polar=False)

# ax.set_yticks([-0.95, 0.])
# ax.set_yticklabels(['', 0])
# ax.set_ylim([-1, 1])
ax.set_xlim([0, 2*np.pi])
theta = np.linspace(0, 2*np.pi, 501)


tuning_in_degrees = np.array([0.2, 1.5])*360

sigma = tuning_in_degrees/360. * 2 * np.pi / 2.

r_exc = von_mises_height_one(theta, sigma[0], np.pi)
r_inh = von_mises_height_one(theta, sigma[1], np.pi)

diff = r_exc - 0.5*r_inh
positive_diff = diff.copy()
negative_diff = diff.copy()
positive_diff[positive_diff<0] = np.nan
negative_diff[negative_diff>0] = np.nan
# ax.plot(theta, positive_diff, color=colors['exc'], lw=3)
# ax.plot(theta, negative_diff, color=colors['inh'], lw=3)
# ax.plot(theta, r_exc-0.5*r_inh, color='black', lw=3)

ax.plot(theta, positive_diff, color=colors['exc'], lw=3)
ax.plot(theta, negative_diff, color=colors['inh'], lw=3)

plt.axhline(y=0., linewidth=2, linestyle='--', color='black')

thetaticks = np.arange(0,360,360)
# ax.set_thetagrids(thetaticks, frac=1.4)
# ax.set_thetagrids(thetaticks, frac=1.4)
ax.axis('off')

# plt.ylabel('test')
plt.yticks([])
plt.xticks([])
# plt.xticks([0, 2*np.pi])
# ax.set_xticklabels([r'0$^{\circ}$', '360$^{\circ}$'])

# plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2])
# plt.axis('off')
# plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/2dim_cell_types/head_direction_input.pdf',
# 	bbox_inches='tight', pad_inches=0.02, transparent=True)
plt.show()