import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os

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
sigma = {'exc': 0.03, 'inh': 0.1}
marker = {'exc': '^', 'inh': 'o'}
populations = ['exc', 'inh']
scaling_factor = {'exc': 1.0, 'inh': 0.5}

# mpl.rcParams['legend.handlelength'] = 0
current = {} 
# current['exc'] = np.array([0.1762616, 1.2, 0.8, -0.27795704, -1.46886165,
# 	-0.8, -0.9, -0.50361893, -0.59372906, -0.33056701, 0.53445425, 
# 	0.43721187]) + 1
current['exc'] = np.arange(500)
inputs = np.linspace(0., 1., len(current['exc']))


fig = plt.figure(figsize=(0.8*4.7, 2.5))

x = np.linspace(0, 1, 501)
# plt.xlim([0.1, 0.8])
# plt.xlim([0.1, 0.9])
plt.xlim([0.25, 0.75])
plt.margins(0.02)
# plt.ylim([-4.5, 14])

c = 0.0
gaussian = {}
for t, s in sigma.iteritems():
	gaussian[t] = scipy.stats.norm(loc=c, scale=s).pdf

lw = {'exc': 2, 'inh': 2, 'diff': 2}

for p in populations:
	for n, i in enumerate(inputs):
		gauss = scipy.stats.norm(loc=inputs[n], scale=sigma[p]).pdf
		x = np.linspace(0, 1, 1001)
		# Shift Gaussians slightyl (looks nicer)
		alpha = 0.0
		# if n == 5:
		if n == 250:
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

			positve_difference = difference.copy()
			negative_difference = difference.copy()
			positve_difference[positve_difference<0] = np.nan
			negative_difference[negative_difference>0] = np.nan
			plt.plot(x, positve_difference, color=colors['exc'], lw=4)
			plt.plot(x, negative_difference, color=colors['inh'], lw=4)

			


		sf = scaling_factor[p]
		# plt.plot(x, signs[p] * (sf * np.sqrt(2*np.pi*sigma[p]**2) * gauss(x)),
		# 			color=colors[p], alpha=alpha, lw=lw[p])



# plt.xlabel('Stimulus')
# plt.ylabel('Firing rate')
# plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

ax = plt.gca()
ax.axis('off')
plt.xticks([])
plt.yticks([])

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

# plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/1dim_Gaussians/center_surround_different.pdf', transparent=True)
plt.show()