import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def fields(params, synapses):
	x = np.linspace(0, params['boxlength'], 200)
	summe = 0
	for s in synapses:
		plt.plot(x, s.field(x))
		summe += s.field(x)
	plt.plot(x, 2*summe/(len(synapses)))
	return

# def centers(params, centers_array):
# 	x = np.linspace(0, params['boxlength'], 200)
# 	s = 0
# 	for c in centers_array:
# 		y = norm(loc=c, scale=params['sigma_exc']).pdf
# 		s += y(x)
# 		plt.plot(x, y(x))
# 	plt.plot(x, 2*s/(len(centers_array)))
# 	return
