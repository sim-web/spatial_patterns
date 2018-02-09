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
current = {}
inputs = np.linspace(-1.03, 1.03, 33)

figsize = (3, 3)
fig = plt.figure(figsize=figsize)
x = np.linspace(-radius, radius, 1001)

c = 0.0
gaussian = {}
for t, s in sigma.items():
	gaussian[t] = scipy.stats.norm(loc=c, scale=s).pdf

lw = {'exc': 1, 'inh': 1, 'diff': 1}
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
			print(inputs[n])
			alpha = 1.0
			difference = (scaling_factor['exc'] * np.sqrt(2*np.pi*sigma['exc']**2)
					* scipy.stats.norm(loc=inputs[n], scale=sigma['exc']).pdf(x)
					-
					scaling_factor['inh'] * np.sqrt(2*np.pi*sigma['inh']**2)
					* scipy.stats.norm(loc=inputs[n], scale=sigma['inh']).pdf(x)
					)
			plt.plot(x, difference, color='black', lw=lw['diff'], alpha=0.0)

		sf = scaling_factor[p]
		# alpha = 1.0
		plt.plot(x, scaling2 * signs[p] * (sf * np.sqrt(2*np.pi*sigma[p]**2) *
									   gauss(x)),
					color=colors[p], alpha=alpha, lw=lw[p])


plt.margins(0.05)
ax = plt.gca()
ax.axis('off')
plt.ylim([-8.0, 1.3*1.1])
plt.xlim([-1.0, 1.0])
plt.xticks([])
plt.yticks([])

### The Inset ###
# Zoom factor and location of inset
delta_x = 0.35
axins = zoomed_inset_axes(ax, 0.95*2*radius/(2*delta_x), loc=3)
# axins = inset_axes(ax, width="100%", height="200%", loc=3)
plt.sca(axins)
# Specifiy the location and size of the zoom window
xins, yins = -0.321875, -0.5*1.3*1.1
delta_y = 1.5*1.3*1.06
xlim = np.array([xins-delta_x, xins+delta_x])
ylim = np.array([yins, yins+delta_y])
plt.setp(axins, xlim=xlim, ylim=ylim, xticks=[], yticks=[])
# You need to plot this again, to actually make it appear in the zoom
# window
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
			plt.plot(x, difference, color='black', lw=2, alpha=1.0)
			zero_crossing = np.array([x[difference[difference>0]][0], x[difference[difference>0]][-1]])
		sf = scaling_factor[p]
		plt.plot(x, scaling2 * signs[p] * (sf * np.sqrt(2*np.pi*sigma[p]**2) *
									   gauss(x)),
					color=colors[p], alpha=alpha, lw=2)

print(zero_crossing)
mark_inset(ax, axins, loc1=2, loc2=1, fc="none", color='black')
plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/1dim_Gaussians'
			'/center_surround_zoomed.pdf',  transparent=True,
			bbox_inches='tight', pad_inches=0.001)
# plt.show()