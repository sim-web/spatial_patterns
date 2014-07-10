import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt



def plot_polar(radius, spacing, a):
	def __str__():
		return 'bla'
	linspace = np.linspace(-radius , radius, spacing)
	X, Y = np.meshgrid(linspace, linspace)
	b = a[...,0].T
	ax = plt.gca()
	plt.xlim([-radius, radius])
	# ax.set_aspect('equal')
	theta = np.linspace(0, 2*np.pi, spacing)
	r = np.mean(b, axis=1)
	plt.polar(theta, r)
	# plt.plot(linspace, np.mean(b, axis=0))
	# plt.show()

def plot_linear(radius, spacing, a):
	linspace = np.linspace(-radius , radius, spacing)
	X, Y = np.meshgrid(linspace, linspace)
	b = a[...,0].T
	ax = plt.gca()
	plt.xlim([-radius, radius])
	plt.plot(linspace, np.mean(b, axis=0))
	# plt.show()


radius = 0.5
spacing = 51
a = np.load('/Users/simonweber/programming/workspace/learning_grids/test_output_rates.npy')

# plt.contourf(X, Y, a[...,0].T)

# plt.plot(linspace, np.mean(b, axis=1))

# # Now axis 1 of b corresponds to angle (y direction)
plot_linear(radius, spacing, a)
plt.show()