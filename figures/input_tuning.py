import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os
from general_utils.plotting import cm2inch

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
# sigma = {'exc': 0.025, 'inh': 0.075}
sigma = {'exc': 0.03, 'inh': 0.1}
marker = {'exc': '^', 'inh': 'o'}
populations = ['exc', 'inh']
scaling_factor = {'exc': 1.0, 'inh': 0.5}


# fig = plt.figure(figsize=(1.6, 0.3))
figsize = (cm2inch(3.), cm2inch(0.5625))
fig = plt.figure(figsize=figsize)

radius = 1.0
x = np.linspace(-radius, radius, 501)
gaussian = {}

c_2 = -0.33
for c in [-0.73, c_2, 0.73]:
	for p, s in sigma.iteritems():
		gaussian[p] = scipy.stats.norm(loc=c, scale=sigma[p]).pdf
		if p == 'inh':
			p_out = p
			alpha = 1.0 if c == c_2 else 0.2
			plt.plot(x, np.sqrt(2*np.pi*sigma[p]**2) * gaussian[p](x),
					 color=colors[p], lw=0.8, alpha=alpha)


# plt.plot(x, np.sqrt(2*np.pi*sigma[p]**2) * gaussian[p](x), color=colors[p], lw=2)


plt.margins(0.05)
# plt.ylim([-0.03, 1.03])
# plt.xlim([0.3, 0.7])
plt.xlim([-radius, radius])
plt.xticks([])
plt.yticks([])
plt.axis('off')

# ax = plt.gca()
# Set y position for arrow to half the gaussian height
# y_for_arrow = 0.5
# Draw an arrow between at height y_for_arrow and between mu-sigma and
# mu+sigma
# ax.annotate('', xy=(c-sigma[p], y_for_arrow),  xycoords='data',
#                 xytext=(c+sigma[p], y_for_arrow), textcoords='data',
#                 arrowprops=dict(arrowstyle='<->',
#                 				lw=1,
#                                 connectionstyle="arc3",
#                                 shrinkA=0, shrinkB=0)
#                 )


# arrowopts = {'shape': 'full', 'lw':1, 'length_includes_head':True,
# 			'head_length':0.01, 'head_width':0.04, 'color':'black'}
#
# arrowlength = sigma[p] - 0.003
# plt.arrow(0.5, 0.5, arrowlength, 0, **arrowopts)
# plt.arrow(0.5, 0.5, -arrowlength, 0, **arrowopts)
# # Put the sigma underneath the arrow
# sigma_string = {'exc': r'$2 \sigma_{\mathrm{E}}$', 'inh': r'$2 \sigma_{\mathrm{I}}$'}
# ax.annotate(sigma_string[p], xy=(c, y_for_arrow-0.2), va='top', ha='center')
#
# # plt.autoscale(tight=True)
# # plt.tight_layout()
# name = 'input_tuning' + '_' + p + '_center_' + str(c).replace('.', 'p') + '.pdf'
# plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/input_tuning/' + name,
# 	bbox_inches='tight', pad_inches=0.001)
plt.savefig('/Users/simonweber/doktor/TeX/learning_grids/input_tuning'
			'/three_fields_' + p_out + '.pdf',
	bbox_inches='tight', pad_inches=0.001)
# plt.show()