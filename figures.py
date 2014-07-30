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
mpl.rc('legend', fontsize=18)
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

excitatory_plasticity = True
if excitatory_plasticity:
	# sigma = {'exc': 0.06, 'inh': 0.18}
	sigma = {'exc': 0.05, 'inh': 0.15}
##############################################
##########	Inhibitory Plasticity	##########
##############################################
np.random.seed(20)
fig = plt.figure(figsize=(4.7, 8.0))
# fig = plt.figure(figsize=(9.5, 8.6))

gs = gridspec.GridSpec(3, 1, height_ratios=[3,3,1.2]) 
populations = ['exc', 'inh']
target_rate_distance = 0.4
target_rate = 1.0
legend_loc = 'lower left'
inset_loc = 4

# plt.axis('off')
# plt.xticks([])
# plt.yticks([0])

n = 12
current = {} 
general_settings = {'xticks': [], 'yticks': [], 'xmargin': 0.025,
								'ymargin': 0.23}
membrane_current_settings = {'ylabel': 'Membrane current'}
membrane_current_settings = {'xlabel': 'Stimulus'}
membrane_current_settings.update(general_settings)
### Before Learning ###
plt.subplot(gs[0], title='Before Learning', **membrane_current_settings)


# shift = np.pi/2.3
shift = 0.
x = np.linspace(shift, 5*np.pi+shift, n)
# Currents Before Learning
# current['exc'] = np.sin(x) + 1.0*(2*np.random.random_sample(n) - 1) + 1
# current['exc'] = [0.1762616, 1.7852489, 1.06479402, -0.27795704, -1.46886165,
# 	1.13926474, 0.51311146, -0.50361893, -0.59372906, -0.33056701, 0.53445425, 
# 	0.43721187]
current['exc'] = np.array([0.1762616, 1.2, 0.8, -0.27795704, -1.46886165,
	-0.8, -0.9, -0.50361893, -0.59372906, -0.33056701, 0.53445425, 
	0.43721187]) + 1
current['inh'] = np.zeros_like(current['exc']) + 1
inputs = np.linspace(0., 1., len(current['exc']))

for p in populations:
	label = legend_short[p]
	plt.plot(inputs, current[p], marker=marker[p], color=colors[p], lw=1, markeredgewidth=0.0, label=label, markersize=10)

ax = plt.gca()
ybottom, ytop = ax.get_ylim()
ax.set_ylim((ybottom - 0.9, ytop + 0.2))
print ax.get_ylim()
ylim_before = ax.get_ylim()
plt.legend(loc=legend_loc, numpoints=1).draw_frame(False)
width = 0.002
distances = current['exc'] - current['inh']
firing_rates = target_rate/target_rate_distance * distances
firing_rates[firing_rates<0] = 0.
print firing_rates
# for n, i in enumerate(inputs):
# 	sign = np.sign(distances[n])
# 	if abs(distances[n]) < 0.25:
# 		arrow_length = 0.
# 	else:
# 		arrow_length = distances[n] - sign*0.25
# 	plt.arrow(inputs[n], 1, 0, arrow_length, color=colors['inh'], transform=ax.transData, width=width, head_width=8*width, head_length=30*width)

### Inset 1 ###
axins = inset_axes(ax,
                   width="30%",
                   height="30%",
                   loc=inset_loc,
                   )

plt.xticks([])
# plt.yticks([0], fontsize=12)
ax = plt.gca()
ax.set_yticks((0, target_rate))
ax.set_yticklabels((r'$0$', r'$\rho_0$'), fontsize=14)
ax.text(-0.15, 0.9, r'$v$', transform = ax.transAxes, fontsize=14)
plt.hlines(0., 0, 1)
plt.hlines(target_rate, 0, 1, lw=2, linestyle='dotted', label=r'$\rho_0$')
# plt.legend(fontsize=12).draw_frame(False)
plt.margins(0., 0.2)
plt.plot(inputs, firing_rates, color='black', lw=2)
y0, y1 = ax.get_ylim()
ax.set_ylim((y0, y1-0.4))
ylim_inset1 = ax.get_ylim()


### Tuning ###
plt.subplot(gs[2], frameon=False, **general_settings)

# plt.plot(np.arange(40), np.arange(40))
for p in populations:
	for n, i in enumerate(inputs):
		gauss = scipy.stats.norm(loc=inputs[n], scale=sigma[p]).pdf
		x = np.linspace(0, 1, 1001)
		# Shift Gaussians slightyl (looks nicer)
		plt.plot(x, signs[p] * (np.sqrt(2*np.pi*sigma[p]**2) * gauss(x) + 0.05),
					color=colors[p])
plt.ylabel('Tuning')
ax = plt.gca()
# plt.hlines(0., 0, 1, lw=1, zorder=100, color='black')
ax.text(0.9, 1.0, 'Exc.', color=colors['exc'], transform = ax.transAxes, fontsize=14)
ax.text(0.9, -0.15, 'Inh.', color=colors['inh'], transform = ax.transAxes, fontsize=14)

### After Learning ###
plt.subplot(gs[1], title='After Learning', **membrane_current_settings)
plt.ylim(*ylim_before)
# Currents After Learning
# Only inhibitory plasticity
current['inh'] = current['exc'] + ((0.04*np.random.random_sample(current['exc'].shape) - 0.02) - target_rate_distance)
# Both plasticities
print current['exc']
# current['exc'] = [0.1762616, 1.7852489, 1.06479402, -0.27795704, -1.46886165,
# 	1.13926474, 0.51311146, -0.50361893, -0.59372906, -0.33056701, 0.53445425, 
# 	0.43721187]

if excitatory_plasticity:
	vpeak = 2.8
	vmin = 0.2
	vint = 0.7
	current['exc'] = [vint, vpeak, vint, vmin, vint, vpeak, vint, vmin, vint,
			vpeak, vint, vmin]

	vpeak -= 0.8
	vmin += 0.5
	vint += 0.3
	current['inh'] = np.array([vint, vpeak, vint, vmin, vint, vpeak, vint, vmin, vint,
			vpeak, vint, vmin])

distances = current['exc'] - current['inh']
firing_rates = target_rate/target_rate_distance * distances
firing_rates[firing_rates<0] = 0.
for p in populations:
	label = legend_short[p]
	plt.plot(inputs, current[p], marker=marker[p], color=colors[p], lw=1, markeredgewidth=0.0, label=label, markersize=10)
plt.legend(loc=legend_loc, numpoints=1).draw_frame(False)


# plt.subplot(gs[2], **general_settings)
### Inset 2 ###
ax = plt.gca()
axins = inset_axes(ax,
                   width="30%",
                   height="30%",
                   loc=inset_loc,
                   )

plt.xticks([])
# plt.yticks([0], fontsize=12)
ax = plt.gca()
ax.set_yticks((0, target_rate))
ax.set_yticklabels((r'$0$', r'$\rho_0$'), fontsize=14)
plt.hlines(target_rate, 0, 1, lw=2, linestyle='dotted', label=r'$\rho_0$')
plt.hlines(0., 0, 1)
ax.text(-0.15, 0.9, r'$v$', transform = ax.transAxes, fontsize=14)
# plt.ylabel('Firing rate', fontsize=18)
plt.plot(inputs, firing_rates, color='black', lw=2)
plt.ylim(*ylim_inset1)

name = 'inhibitory_plasticity.pdf'
if excitatory_plasticity:
	name = 'with_excitatory_plasticity.pdf'
plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/inhibitory_plasticity/' + name)
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