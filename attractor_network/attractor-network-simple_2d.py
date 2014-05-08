from pylab import *
import numpy as np

length = 128
N = length**2

# positions_grid = np.empty((length, length, 2))
# x_space = np.arange(length)
# y_space = np.arange(length)
# X1, Y1 = np.meshgrid(x_space, y_space)

# distances = np.empty((N, length, length))
# for i in np.arange(N):
# 	print i
# 	X, Y = np.meshgrid(	x_space - X1.T.reshape(N)[i],
# 					y_space - Y1.T.reshape(N)[i])
# 	# Periodic boundary conditions
# 	X[X > length/2] = length - X[X > length/2] 
# 	X[X < -length/2] = length + X[X < -length/2] 
# 	Y[Y > length/2] = length - Y[Y > length/2] 
# 	Y[Y < -length/2] = length + Y[Y < -length/2] 
# 	distances[i] = np.sqrt(X*X + Y*Y)

# X2 = X1.reshape(N)
# Y2 = Y1.reshape(N)
# # # np.save('distances30', distances)

# # # distances = np.load('distances' + str(length) + '.npy')

# def distance_between_neurons(neuron1, neuron2):
# 	return distances[neuron1][X2[neuron2], Y2[neuron2]]

# dist = np.empty((N, N))
# for i in range(N):
# 	print i
# 	for j in range(N):
# 		dist[i, j] = distance_between_neurons(i,j)
# 		# W_EE[i,j] = 1. * exp(- gamma * distance_between_neurons(i,j)**2)
# np.save('dist128', dist)

# seed(3)

T = 3000.0
dt = 0.5
NT = int(T/dt)
tau = 10.0
Ntau = int(tau/dt)

I = 1.0

factor_GABA = 1.0

# W_EE = zeros((N,N))
# W_EI = zeros((N,N))
# W_IE = zeros((N,N))
# W_II = zeros((N,N))

# # sigma_exc = 7.
# # sigma_inh = 1.5 * sigma_exc
# # sigma_EE = sigma_exc
# # sigma_II = sigma_inh
# # sigma_EI = sigma_inh
# # sigma_IE = sigma_exc
# # beta = 3./((length/10.)**2)
beta = 3./(13**2)
gamma = 1.05 * beta

print 'loading'
dist = np.load('dist128.npy')
print 'functioning'
W_EE = factor_GABA * (exp(- gamma * dist**2) - exp(- beta * dist**2)) 
print 'done'
# # for i in range(N):
# # 	print i
# # 	for j in range(N):
# # 		# W_EE[i,j] = 1. * exp(- gamma * distance_between_neurons(i,j)**2)
# # 		W_EE[i,j] = exp(- gamma * distance_between_neurons(i,j)**2) - exp(- beta * distance_between_neurons(i,j)**2)
		
# # 		# W_EI[i,j] = 128./length * exp(- beta * distance_between_neurons(i,j)**2)
# # 		# W_EI[i,j] = 0.0
# # 		# W_IE[i,j] = 1. * exp(-gamma * distance_between_neurons(i,j)**2)
# # 		# W_II[i,j] = 128./length * exp(- beta * distance_between_neurons(i,j)**2)
# # 		# W_II[i,j] = 0.0
# # 	# W_EE[i,i] = 0
# # 	# W_EI[i,i] = 0
# # 	# W_IE[i,i] = 0
# # 	# W_II[i,i] = 0

# # W_EE = factor_GABA * np.load('WEE64.npy')
	

r_E = random(N)/10.
# r_E[N/2] = 1.0
r_I = random(N)/10.

rates = []
activity = sum(r_E)
print activity

for t in range(0,NT):
	if t % 10 == 0:
		print t
	# r_E += dt/tau * (- r_E + dot(W_EE,r_E) - factor_GABA*dot(W_EI, r_I) + I)
	
	dot(W_EE,r_E)[dot(W_EE,r_E)<(-I)] = I
	r_E += dt/tau * (- r_E + dot(W_EE,r_E) + I)

	# NORMALIZATION
	# current_activity = sum(r_E)
	# Multiplicative normalization
	# r_E *= activity/current_activity
	# Substractive Multiplication
	# r_E -= (current_activity-activity)/N
	r_E[r_E<0] = 0
	# r_E[r_E>1] = 1
	rates.append(r_E.copy())

	# r_I += dt/tau * (- r_I + dot(W_IE,r_E) - factor_GABA*dot(W_II, r_I) + I)
	# r_I[r_I<0] = 0
	#r_I[r_I>1] = 1

# np.save('r_E', r_E)
# np.save('rates', rates)
print r_E
figure()
subplot(211)
title('GABA factor: %.2f' % factor_GABA)
contourf(r_E.reshape(length, length), 30)
colorbar()
ax = gca()
ax.set_aspect('equal')

subplot(212)
plot(array(rates)[:,::10])

show()
		

