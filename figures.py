import utils
import matplotlib as mpl
# mpl.use('TkAgg')
# import plotting
# import animating
from matplotlib import gridspec
import matplotlib.pyplot as plt
import time
import general_utils.arrays
import numpy as np
import string
import scipy.stats
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

##############################################
##########	General Plotting Stuff	##########
##############################################
mpl.rc('font', size=18)
mpl.rc('legend', fontsize=16)
mpl.rcParams.update({'figure.autolayout': True})

# If you comment this out, then everything works, but in matplotlib fonts
# mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
# mpl.rc('text', usetex=True)

colors = {'exc': '#D7191C', 'inh': '#2C7BB6', 'diff': '0.4'}
legend = {'exc': 'Excitation', 'inh': 'Inhibition', 'diff': 'Difference'}
legend_short = {'exc': 'Exc.', 'inh': 'Inh.', 'diff': 'Difference'}
signs = {'exc': 1, 'inh': -1}
sigma = {'exc': 0.005, 'inh': 0.005}
marker = {'exc': '^', 'inh': 'o'}
mpl.rcParams['legend.handlelength'] = 0

##############################################
##########	Inhibitory Plasticity	##########
##############################################
np.random.seed(20)
fig = plt.figure(figsize=(4.4, 7.6))
gs = gridspec.GridSpec(3, 1, height_ratios=[3,3,1]) 
populations = ['exc', 'inh']
target_rate_distance = 0.55
target_rate = 1.0


plt.axis('off')
plt.xticks([])
plt.yticks([])

n = 12
current = {} 
general_settings = {'xticks': [], 'yticks': [], 'xmargin': 0.025,
								'ymargin': 0.23}
membrane_current_settings = {'ylabel': 'Membrane current'}
membrane_current_settings.update(general_settings)
### Before Learning ###
plt.subplot(gs[0], title='Before learning', **membrane_current_settings)
legend_loc = 'lower left'

# shift = np.pi/2.3
shift = 0.
x = np.linspace(shift, 5*np.pi+shift, n)
current['exc'] = np.sin(x) + 1.0*(2*np.random.random_sample(n) - 1)
current['inh'] = np.zeros_like(current['exc'])
inputs = np.linspace(0., 1., len(current['exc']))

for p in populations:
	label = legend_short[p]
	plt.plot(inputs, current[p], marker=marker[p], color=colors[p], lw=1, markeredgewidth=0.0, label=label, markersize=10)

ax = plt.gca()
ylim_before = ax.get_ylim()
plt.legend(loc=legend_loc, numpoints=1).draw_frame(False)
width = 0.002
distances = current['exc'] - current['inh']
firing_rates = target_rate/target_rate_distance * distances
firing_rates[firing_rates<0] = 0.
print firing_rates
for n, i in enumerate(inputs):
	sign = np.sign(distances[n])
	if abs(distances[n]) < 0.25:
		arrow_length = 0.
	else:
		arrow_length = distances[n] - sign*0.25
	plt.arrow(inputs[n], 0, 0, arrow_length, color=colors['inh'], transform=ax.transData, width=width, head_width=8*width, head_length=30*width)

### Inset 1 ###
axins = inset_axes(ax,
                   width="30%",
                   height="30%",
                   loc=1,
                   )

plt.xticks([])
# plt.yticks([0], fontsize=12)
ax = plt.gca()
ax.set_yticks((0, target_rate))
ax.set_yticklabels((r'$0$', r'$\rho_0$'), fontsize=12)
ax.text(-0.15, 0.9, r'$v$', transform = ax.transAxes, fontsize=12)
plt.hlines(0., 0, 1)
plt.hlines(target_rate, 0, 1, lw=2, linestyle='dotted', label=r'$\rho_0$')
# plt.legend(fontsize=12).draw_frame(False)
plt.margins(0., 0.2)
plt.plot(inputs, firing_rates, color='black', lw=2)
ylim_inset1 = ax.get_ylim()


### Tuning ###
plt.subplot(gs[2], frameon=False, **general_settings)

# plt.plot(np.arange(40), np.arange(40))
for p in populations:
	for n, i in enumerate(inputs):
		gauss = scipy.stats.norm(loc=inputs[n], scale=sigma[p]).pdf
		x = np.linspace(0, 1, 1001)
		plt.plot(x, signs[p] * np.sqrt(2*np.pi*sigma[p]**2) * gauss(x),
					color=colors[p])
plt.ylabel('Tuning')

### After Learning ###
plt.subplot(gs[1], title='After learning', **membrane_current_settings)
plt.ylim(*ylim_before)
current['inh'] = current['exc'] + ((0.04*np.random.random_sample(current['exc'].shape) - 0.02) - target_rate_distance)
distances = current['exc'] - current['inh']
firing_rates = target_rate/target_rate_distance * distances
firing_rates[firing_rates<0] = 0.
for p in populations:
	label = legend_short[p]
	plt.plot(inputs, current[p], marker=marker[p], color=colors[p], lw=1, markeredgewidth=0.0, label=label, markersize=10)
plt.legend(loc=legend_loc, numpoints=1).draw_frame(False)

### Inset 2 ###
ax = plt.gca()
axins = inset_axes(ax,
                   width="30%",
                   height="30%",
                   loc=1,
                   )

plt.xticks([])
# plt.yticks([0], fontsize=12)
ax = plt.gca()
ax.set_yticks((0, target_rate))
ax.set_yticklabels((r'$0$', r'$\rho_0$'), fontsize=12)
ax.text(-0.15, 0.9, r'$v$', transform = ax.transAxes, fontsize=12)
plt.plot(inputs, firing_rates, color='black', lw=2)
plt.ylim(*ylim_inset1)

# plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/inhibitory_plasticity/inhibitory_plasticity.pdf')
plt.show()

##################################
##########	Playground	##########
##################################

# output_rate = np.array([[1.1, 1.2], [0.8, 0.6], [0.9, 1.0]])
# linspace = np.linspace(-0.5, 0.5, 2)
# time = np.array([0, 1000, 2000])
# X, Y = np.meshgrid(linspace, time)
# # plt.contourf(output_rate)

# plt.contourf(X, Y, output_rate)
# print np.arange(11, 15)


##########################################
##########	1D Center Surround	##########
##########################################
# radius = 0.5
# x = np.linspace(-radius, radius, 200)
# plt.xlim([-radius, radius])
# plt.ylim([-4.5, 14])

# c = 0.0
# sigma = {'exc': 0.03, 'inh': 0.1}
# gaussian = {}
# for t, s in sigma.iteritems():
# 	gaussian[t] = scipy.stats.norm(loc=c, scale=s).pdf

# lw = {'exc': 2, 'inh': 2, 'diff': 4}
# plots = {'exc': gaussian['exc'](x), 'inh': gaussian['inh'](x), 'diff':gaussian['exc'](x) - gaussian['inh'](x)}
# for t, p in sorted(plots.iteritems(), reverse=True):
# 	plt.plot(x, p, color=colors[t], lw=lw[t], label=legend[t])

# plt.xlabel('position')
# plt.ylabel('firing rate')
# plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

# ax = plt.gca()
# ax.get_xaxis().set_ticklabels([])
# ax.get_yaxis().set_ticklabels([0.0, 0])
# # rectangle1=plt.Rectangle((-radius, 0),
# # 						radius, radius, ec='none', fc='r', lw=2)
# alpha = 0.3
# factor=2
# root = 0.04887
# ax.axvspan(-root, root, color=colors['exc'], alpha=alpha)
# ax.axvspan(-factor*sigma['inh'], -root, color=colors['inh'], alpha=alpha)
# ax.axvspan(root, factor*sigma['inh'], color=colors['inh'], alpha=alpha)


# plt.show()