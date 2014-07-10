import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

radius = 0.5
spacing = 51
linspace = np.linspace(-radius , radius, spacing)
X, Y = np.meshgrid(linspace, linspace)
a = np.load('/Users/simonweber/programming/workspace/learning_grids/test_output_rates.npy')

# plt.contourf(X, Y, a[...,0].T)

b = a[...,0].T

# plt.plot(linspace, np.mean(b, axis=1))

# # Now axis 1 of b corresponds to angle (y direction)
ax = plt.gca()
plt.xlim([-radius, radius])
# ax.set_aspect('equal')

theta = np.linspace(0, 2*np.pi, spacing)
# theta = (linspace + radius) * 2 * np.pi
# r = np.linspace(0., 0.5, 51)
r = np.mean(b, axis=1)
print r.shape
print theta
print r
# plt.polar(theta, r)
plt.plot(linspace, np.mean(b, axis=0))

plt.show()
