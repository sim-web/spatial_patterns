import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os
from general_utils.plotting import cm2inch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

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
sigma = {'exc': 0.03, 'inh': 0.1}
marker = {'exc': '^', 'inh': 'o'}
populations = ['exc', 'inh']
scaling_factor = {'exc': 1.0, 'inh': 0.5}

radius = 1.0
# mpl.rcParams['legend.handlelength'] = 0
current = {} 
# current['exc'] = np.array([0.1762616, 1.2, 0.8, -0.27795704, -1.46886165,
# 	-0.8, -0.9, -0.50361893, -0.59372906, -0.33056701, 0.53445425, 
# 	0.43721187]) + 1
# current['exc'] = np.arange(500)
# inputs = np.linspace(0., 1., len(current['exc']))
# inputs = np.linspace(-radius, radius, 10)
inputs = np.linspace(-1.03, 1.03, 33)
print inputs

# figsize = (cm2inch(3.), cm2inch(0.5625))

figsize = (3, 1)
# fig = plt.figure(figsize=(0.8*4.7, 2.5))
fig = plt.figure(figsize=figsize)


x = np.linspace(-radius, radius, 1001)
plt.xlim([-radius, radius])
# plt.xlim([0.1, 0.9])
# plt.xlim([0.25, 0.75])
# plt.margins(0.02)
# plt.ylim([-4.5, 14])

c = 0.0
gaussian = {}
for t, s in sigma.iteritems():
	gaussian[t] = scipy.stats.norm(loc=c, scale=s).pdf

lw = {'exc': 1, 'inh': 1, 'diff': 1}
# lw = {'exc': 0.5, 'inh': 0.5, 'diff': 0.5}

for p in populations:
	for n, i in enumerate(inputs):
		gauss = scipy.stats.norm(loc=inputs[n], scale=sigma[p]).pdf
		# Shift Gaussians slightyl (looks nicer)
		alpha = 0.2
		# if n == 5:
		# if n == 250:
		scaling2 = 1.0
		if n == 11:
			scaling2 = 1.3
			alpha = 1.0
			difference = (scaling_factor['exc'] * np.sqrt(2*np.pi*sigma['exc']**2)
					* scipy.stats.norm(loc=inputs[n], scale=sigma['exc']).pdf(x)
					-
					scaling_factor['inh'] * np.sqrt(2*np.pi*sigma['inh']**2)
					* scipy.stats.norm(loc=inputs[n], scale=sigma['inh']).pdf(x)
					)
			# plt.plot(x, difference,
			# 		color='black',
			# 		lw=lw['diff'])

			# positve_difference = difference.copy()
			# negative_difference = difference.copy()
			# positve_difference[positve_difference<0] = np.nan
			# negative_difference[negative_difference>0] = np.nan
			# plt.plot(x, positve_difference, color=colors['exc'], lw=4)
			# plt.plot(x, negative_difference, color=colors['inh'], lw=4)
			plt.plot(x, difference, color='black', lw=lw['diff'], alpha=0.0)

		sf = scaling_factor[p]
		# alpha = 1.0
		plt.plot(x, scaling2 * signs[p] * (sf * np.sqrt(2*np.pi*sigma[p]**2) *
									   gauss(x)),
					color=colors[p], alpha=alpha, lw=lw[p])



# plt.xlabel('Stimulus')
# plt.ylabel('Firing rate')
# plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

# plt.ylim([-0.03, 1.03])

plt.margins(0.05)
ax = plt.gca()
ax.axis('off')
plt.xticks([])
plt.yticks([])

### The Inset ###
# Zoom factor and location of inset
axins = zoomed_inset_axes(ax, 1.2, loc=2)
plt.sca(axins)
# Specifiy the location and size of the zoom window
x, y = -0.2, 0.0
xlim = np.array([x, x+0.5])
ylim = np.array([y, y+1])
plt.setp(axins, xlim=xlim, ylim=ylim, xticks=[], yticks=[])
# You need to plot this again, to actually make it appear in the zoom
# window
# for t in ['particle', 'dm']:
# 	for x, y in zip(positions[t][:,0], positions[t][:,1]):
# 		circle = plt.Circle((x, y), radius=radius[t], facecolor=colors[t],
# 					edgecolor='none', fill=True, lw=0.0)
# 		axins.add_patch(circle)
# Nicely connects the two rectangles
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", color='0.7')


# ax.get_xaxis().set_ticklabels([])
# ax.get_yaxis().set_ticklabels([])

# ax.get_yaxis().set_ticklabels([0.0, 0])

# rectangle1=plt.Rectangle((-radius, 0),
# 						radius, radius, ec='none', fc='r', lw=2)

# alpha = 0.3
# factor=2
# root = 0.04887
# ax.axvspan(-root, root, color=colors['exc'], alpha=alpha)
# ax.axvspan(-factor*sigma['inh'], -root, color=colors['inh'], alpha=alpha)
# ax.axvspan(root, factor*sigma['inh'], color=colors['inh'], alpha=alpha)

# plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/1dim_Gaussians'
# 			'/center_surround_new2.pdf',  transparent=True,
# 			bbox_inches='tight', pad_inches=0.001)
plt.show()