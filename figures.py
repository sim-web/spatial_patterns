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


radius = 0.5
x = np.linspace(-radius, radius, 200)
plt.xlim([-radius, radius])
c = 0.0
sigma = {'exc': 0.03, 'inh': 0.1}
gaussian = {}
for t, s in sigma.iteritems():
	gaussian[t] = scipy.stats.norm(loc=c, scale=s).pdf
	
plt.plot(x, gaussian['exc'](x))
plt.plot(x, -gaussian['inh'](x))
plt.plot(x, gaussian['exc'](x) - gaussian['inh'](x))

plt.show()