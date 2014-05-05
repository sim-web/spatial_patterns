from pylab import *

# Simply simulation from Henning
# Several models can be tested:
	# Couey Model: only inhibitory connections
		# Put take negative W_EE (let them be inhibitory).
		# Comment out any other weights
		# Don't use normalization
		# Effect of reducing the weights: grid spacing goes down
		# Use input I = 0.1
	# Classical Head Direction cell in attractor model (see McNaughton 2006):
		# Take positive W_EE, Comment out any other weights
		# Take substractive normalization
		# Use input I = 0.0
	# Mexican Hat:
		# Don't use normalization
		# Use all the inibitory weights and change the equation accordingly
		# Use input I = 0.1


N = 40

def d(i,j):
	if abs(i-j)<N/2: 
		return abs(i-j)
	else:
		return N - abs(i-j)

seed(1)

T = 200
dt = 0.01
NT = int(T/dt)
tau = 1.
Ntau = int(tau/dt)

I = 0.1

factor_GABA = 1.0

W_EE = zeros((N,N))
W_EI = zeros((N,N))
W_IE = zeros((N,N))
W_II = zeros((N,N))

sigma_EE = 1.
sigma_II = 4.
sigma_EI = 4.
sigma_IE = 4.

for i in range(N):
	for j in range(N):
		W_EE[i,j] =  1. * exp(-d(i,j)**2/(2*sigma_EE**2))#/sigma_EE
		W_EI[i,j] = 3. * exp(-d(i,j)**2/(2*sigma_EI**2))#/sigma_EI
		W_IE[i,j] = 3. * exp(-d(i,j)**2/(2*sigma_IE**2))#/sigma_IE
		W_II[i,j] = 3. * exp(-d(i,j)**2/(2*sigma_II**2))#/sigma_II
	W_EE[i,i] = 0
	W_EI[i,i] = 0
	W_IE[i,i] = 0
	W_II[i,i] = 0
	
r_E = random(N)
# r_E[N/2] = 1.0
r_I = random(N)

rates = []
activity = sum(r_E)
print activity
for t in range(0,NT):
	r_E += dt/tau * (- r_E + dot(W_EE,r_E) - factor_GABA*dot(W_EI, r_I) + I)
	
	# NORMALIZATION
	# current_activity = sum(r_E)
	# Multiplicative normalization
	# r_E *= activity/current_activity
	# Substractive Multiplication
	# r_E -= (current_activity-activity)/N

	r_E[r_E<0] = 0
	# r_E[r_E>1] = 1
	rates.append(r_E.copy())

	r_I += dt/tau * (- r_I + dot(W_IE,r_E) - factor_GABA*dot(W_II, r_I) + I)
	r_I[r_I<0] = 0
	#r_I[r_I>1] = 1

figure()
subplot(311)
plot(r_E, marker='o')
#plot(W_EE[:,50])
subplot(312)
plot(array(rates)[:NT/10,:])
print r_E
print r_I

subplot(313)
# plot(-dot(W_EE,r_E))
#plot(dot(W_EI, r_I))


factor_GABA = 0.5
#r_E = zeros(N)
#r_I = zeros(N)
for t in range(0,1):
	r_E += dt/tau * (- r_E + dot(W_EE,r_E) - factor_GABA*dot(W_EI, r_I) + I)
	r_E[r_E<0] = 0
	r_I += dt/tau * (- r_I + dot(W_IE,r_E) - factor_GABA*dot(W_II, r_I) + I)
	r_I[r_I<0] = 0	
	
#figure()
plot(r_E)
show()
		

