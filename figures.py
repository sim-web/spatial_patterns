import utils
import matplotlib as mpl
# mpl.use('TkAgg')
# import plotting
# import animating
import matplotlib.pyplot as plt
import time
import general_utils.arrays
import numpy as np
import string
import scipy.stats
import os
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

##############################################
##########	General Plotting Stuff	##########
##############################################
mpl.rc('font', size=18)
mpl.rc('legend', fontsize=16)
mpl.rcParams.update({'figure.autolayout': True})

# If you comment this out, then everything works, but in matplotlib fonts
mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
mpl.rc('text', usetex=True)

colors = {'exc': '#D7191C', 'inh': '#2C7BB6', 'diff': '0.4'}
legend = {'exc': 'Excitation', 'inh': 'Inhibition', 'diff': 'Difference'}
legend_short = {'exc': 'Exc.', 'inh': 'Inh.', 'diff': 'Difference'}
marker = {'exc': '^', 'inh': 'o'}
mpl.rcParams['legend.handlelength'] = 0

##############################################
##########	Inhibitory Plasticity	##########
##############################################
np.random.seed(2)
fig = plt.figure(figsize=(4.3, 3.3))
# fig.patch.set_visible(False)
# plt.axis('off')
plt.xlim([-0.02, 1.02])
plt.ylim([-1.5, 1.2])
n = 12
current = {} 
# current['exc'] = np.array([0.1, 0.3, 0.75, 0.85, 0.9, 0.75, 0.4, 0.2, 0.1])
# current['exc'] = np.array([0.1, 0.55, 0.75, 0.25, 0., 0.8, 0.7, 0.2, 0.1])
shift = np.pi/2.3
x = np.linspace(shift, 2*np.pi+shift, n)
current['exc'] = np.sin(x) + 0.2*(2*np.random.random_sample(n) - 1)
current['inh'] = np.zeros_like(current['exc'])
# current['inh'] = current['exc'] + (0.00*np.random.random_sample(current['exc'].shape) - 0.25)

inputs = np.linspace(0., 1., len(current['exc']))
for p in ['exc', 'inh']:
	label = legend_short[p]
	plt.plot(inputs, current[p], marker=marker[p], color=colors[p], lw=1, markeredgewidth=0.0, label=label, markersize=10)


ax = plt.gca()
# ax.autoscale_view()
plt.legend(loc='best', numpoints=1).draw_frame(False)
width = 0.002
distances = current['exc'] - current['inh']
print distances
for n, i in enumerate(inputs):
	sign = np.sign(distances[n])
	if abs(distances[n]) < 0.25:
		arrow_length = 0.
	else:
		arrow_length = distances[n] - sign*0.25
	print inputs[n]
	plt.arrow(inputs[n], 0, 0, arrow_length, color=colors['inh'], transform=ax.transData, width=width, head_width=8*width, head_length=30*width)

# plt.arrow(inputs[1], 0.2, 0, 0.3 , color=colors['inh'], transform=ax.transData, width=width, head_width=8*width, head_length=30*width)
# plt.arrow(inputs[6], -0.2, 0, -0.3 , color=colors['inh'], transform=ax.transData, width=width, head_width=8*width, head_length=30*width)
# plt.arrow(inputs[10], 0.2, 0, 0.3 , color=colors['inh'], transform=ax.transData, width=width, head_width=8*width, head_length=30*width)

# plt.arrow(0.1, 0.45, 0, -0.15, color=colors['inh'])
# plt.arrow(0.9, 0.45, 0, -0.15, color=colors['inh'])
plt.xticks([])
# plt.xticks(inputs, np.arange(1, 10))
plt.yticks([])
plt.title('Before Learning')
plt.xlabel('Stimulus') 
plt.ylabel('Membrane Current')
# plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/inhibitory_plasticity/before_learning.pdf')
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