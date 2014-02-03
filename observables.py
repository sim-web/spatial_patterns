import numpy as np

##################################
##########	Gridness	##########
##################################
def get_correlation_2d(a, b):
	"""
	Determines array of Pearson correlation coefficients.

	Array b is shifted with respect to a and for each possible shift
	the Pearson correlation coefficient for the overlapping part of
	the two arrays is determined and written in the output rate.
	For a = b this results in the auto-correlogram.
	Note that erratic values (for example seeking the correlation in
	array of only zeros) result in np.nan values.

	See Evernote 'Pearson Correlation Coefficient for Auto Correlogram'
	to find an explanation of the algorithm
	
	Parameters
	----------
	a : array_like
		Shape needs to be (N x N) 
	b : array_like
		Needs to be of same shape as a
	
	Returns
	-------
	output : ndarray
		Array of Pearson correlation coefficients for shifts of b
		with respect to a.
		The shape of the correlations array is (M, M) where M = 2*N + 1.
	"""
		
	spacing = a.shape[0]
	correlations = np.zeros((2*spacing+1, 2*spacing+1))

	# Center
	s1 = a.flatten()
	s2 = b.flatten()
	corrcoef = np.corrcoef(s1, s2)
	correlations[spacing, spacing] = corrcoef[0, 1]
	print corrcoef[0 , 1]

	space = np.arange(1, spacing)

	# East line
	for nx in space:
		s1 = a[nx:, :].flatten()
		s2 = b[:-nx, :].flatten()
		corrcoef = np.corrcoef(s1, s2)
		correlations[nx+spacing, spacing] = corrcoef[0, 1]

	# North line
	for ny in space:
		s1 = a[:, ny:].flatten()
		s2 = b[:, :-ny].flatten()
		corrcoef = np.corrcoef(s1, s2)
		correlations[spacing, spacing+ny] = corrcoef[0, 1]

	# West line
	for nx in space:
		s1 = a[:-nx, :].flatten()
		s2 = b[nx:, :].flatten()
		corrcoef = np.corrcoef(s1, s2)
		correlations[spacing-nx, spacing] = corrcoef[0, 1]

	# South line
	for ny in space:
		s1 = a[:, :-ny].flatten()
		s2 = b[:, ny:].flatten()
		corrcoef = np.corrcoef(s1, s2)
		correlations[spacing, spacing-ny] = corrcoef[0, 1]

	# First quadrant
	for ny in space:
		for nx in space: 
			s1 = a[nx:, ny:].flatten()
			s2 = b[:-nx, :-ny].flatten()
			corrcoef = np.corrcoef(s1, s2)
			correlations[nx+spacing, ny+spacing] = corrcoef[0, 1]

	# Second quadrant
	for ny in space:
		for nx in space: 
			s1 = a[:-nx, ny:].flatten()
			s2 = b[nx:, :-ny].flatten()
			corrcoef = np.corrcoef(s1, s2)
			correlations[spacing-nx, ny+spacing] = corrcoef[0, 1]

	# Third quadrant
	for nx in space:
		for ny in space: 
			s1 = a[:-nx, :-ny].flatten()
			s2 = b[nx:, ny:].flatten()
			corrcoef = np.corrcoef(s1, s2)
			correlations[spacing-nx, spacing-ny] = corrcoef[0, 1]

	# Fourth quadrant
	for ny in space:
		for nx in space: 
			s1 = a[nx:, :-ny].flatten()
			s2 = b[:-nx, ny:].flatten()
			corrcoef = np.corrcoef(s1, s2)
			correlations[nx+spacing, spacing-ny] = corrcoef[0, 1]

	return correlations

##############################################
##########	Measures for Learning	##########
##############################################
def sum_difference_squared(old_weights, new_weights):
	"""
	Sum of squared differences between old and new weights

	- old_weights, new_weights are numpy arrays
	"""
	return np.sum(np.square(new_weights - old_weights))
