import utils
import matplotlib as mpl
mpl.use('TkAgg')
# import plotting
# import animating
import matplotlib.pyplot as plt
import time
import general_utils.arrays
import numpy as np
import string
import scipy.stats
import os

##############################################
##########	General Plotting Stuff	##########
##############################################
mpl.rc('font', size=18)
mpl.rc('legend', fontsize=18)
# If you comment this out, then everything works, but in matplotlib fonts
# mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
# mpl.rc('text', usetex=True)
colors = {'exc': '#D7191C', 'inh': '#2C7BB6', 'diff': '0.4'}
legend = {'exc': 'excitation', 'inh': 'inhibition', 'diff': 'difference'}

##########################################
##########	1D Center Surround	##########
##########################################
radius = 0.5
x = np.linspace(-radius, radius, 200)
plt.xlim([-radius, radius])
plt.ylim([-4.5, 14])

c = 0.0
sigma = {'exc': 0.03, 'inh': 0.1}
gaussian = {}
for t, s in sigma.iteritems():
	gaussian[t] = scipy.stats.norm(loc=c, scale=s).pdf

lw = {'exc': 2, 'inh': 2, 'diff': 4}
plots = {'exc': gaussian['exc'](x), 'inh': -gaussian['inh'](x), 'diff':gaussian['exc'](x) - gaussian['inh'](x)}
for t, p in sorted(plots.iteritems(), reverse=True):
	plt.plot(x, p, color=colors[t], lw=lw[t], label=legend[t])

plt.xlabel('position')
plt.ylabel('firing rate')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

ax = plt.gca()
ax.get_xaxis().set_ticklabels([])
ax.get_yaxis().set_ticklabels([0.0, 0])
# rectangle1=plt.Rectangle((-radius, 0),
# 						radius, radius, ec='none', fc='r', lw=2)
alpha = 0.3
factor=2
root = 0.04887
ax.axvspan(-root, root, color=colors['exc'], alpha=alpha)
ax.axvspan(-factor*sigma['inh'], -root, color=colors['inh'], alpha=alpha)
ax.axvspan(root, factor*sigma['inh'], color=colors['inh'], alpha=alpha)

# ax.axvspan(+sigma['exc'], radius, color=colors['exc'], alpha=alpha)

# path = '/Users/simonweber/doktor/talks/figures'
# figname = 'bla.pdf'
# save_path = os.path.join(path, figname)
# print save_path
# plt.savefig(save_path='Users/simonweber/test.png')
plt.show()