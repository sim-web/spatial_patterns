import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os
from learning_grids import initialization
import itertools
import general_utils.plotting

os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

##############################################
##########	General Plotting Stuff	##########
##############################################
mpl.rc('font', size=12)
mpl.rc('legend', fontsize=18)
mpl.rcParams.update({'figure.autolayout': True})
mpl.rc('grid', color='black', linewidth=1, linestyle=':')
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


fig = plt.figure(figsize=(2.5, 2.5), frameon=False)
# ax = fig.add_subplot(111, polar=True, axisbg=general_utils.plotting.color_cycle_blue4[3])
ax = fig.add_subplot(111, polar=True)

# For the example case
radius = 0.5
theta = np.linspace(-np.pi, np.pi, 501)
# r = scipy.stats.norm(loc=0.0, scale=0.2 * np.pi /radius).pdf(theta)
r = scipy.stats.vonmises(loc=-0.43*np.pi/radius, kappa=1/(0.5*np.pi/radius)**2).pdf(theta)

# For the head direction cell from data
# theta = np.linspace(0, 2*np.pi, 31)
# r = np.load('../temp_data/head_direction_cell.npy')

ax.plot(theta, r, color='black', lw=1)
thetaticks = np.arange(0,360,90)
# ax.set_thetagrids(thetaticks, frac=1.4)
ax.set_thetagrids(thetaticks, labels=[])
# ax.set_thetagrids([])
ax.set_aspect('equal')
# plt.ylabel('test')
# plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2], [])
r_max = ax.get_ylim()[-1]
plt.ylim([0, r_max*1.1])
plt.yticks([])
# plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2])
# plt.axis('off')
# plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/2dim_cell_types/conjunctive_cell_from_data.pdf',
# 	bbox_inches='tight', pad_inches=0.02)
ax.spines['polar'].set_visible(False)
ax.set_axis_bgcolor('0.9')
plt.show()