import tables
from tables import IsDescription, Float64Col
import time

name_size = 32
eqs_size = 256
units_size = 12

# params = {
# 	'dimensions': 1,
# 	'boxtype': 'line',
# 	'boxlength': 1.0,
# 	'n_exc': 100,
# 	'n_inh': 100,
# 	'sigma_exc': 0.03,
# 	'sigma_inh': 0.1,
# 	'init_weight_noise_exc': 0.05,  # Noise on weights (as relative value)
# 	'init_weight_noise_inh': 0.05,  # 0.05 corresponds to +- 5 percent
# 	'simulation_time': 100.0,
# 	'target_rate': 5.0,
# 	'diff_const': 0.01,
# 	'dt': 1.0,
# 	'eta_exc': 0.000001,	
# 	'eta_inh': 0.002,
# 	'normalization': 'quadratic_multiplicative',
# 	# 'normalization': 'linear_multiplicative',
# 	# 'normalization': 'quadratic_multiplicative',
# 	'every_nth_step': 1,
# 	'seed': 2
# }

# class Rawdata(IsDescription):
# 	# def __init__(self, params):
# 	time = Float64Col()
# 	position = Float64Col(shape=2)
# 	exc_weights = Float64Col(shape=params['n_exc'])
# 		# inh_weights = Float64Col(shape=params['n_inh'])
# 		# output_rate = Float64Col()

class Output():
	"""
	docstring
	"""
	def __init__(self, params):
		self.params = params
		class Rawdata(IsDescription):
			time = Float64Col()
			position = Float64Col(shape=2)
			exc_weights = Float64Col(shape=params['n_exc'])
			inh_weights = Float64Col(shape=params['n_inh'])
			output_rate = Float64Col()

		class Configuration(IsDescription):
			if params['dimensions'] == 1:
				exc_centers = Float64Col(shape=(params['n_exc']))
				inh_centers = Float64Col(shape=(params['n_inh']))
			if params['dimensions'] == 2:
				exc_centers = Float64Col(shape=(params['n_exc'], 2))
				inh_centers = Float64Col(shape=(params['n_inh'], 2))
			exc_sigmas = Float64Col(shape=params['n_exc'])
			inh_sigmas = Float64Col(shape=params['n_inh'])	

		datetime = time.strftime('%Y-%m-%d-%Hh%Mm%Ss')
		file_path = '/Users/simonweber/doktor/learning_grids/data/' + datetime + '.h5'
		self.h5file = tables.openFile(file_path, mode='w', title='')
		self.t_rawdata = self.h5file.createTable('/', 'rawdata', Rawdata, 'rawdata')
		for k, v in params.items():
			setattr(self.t_rawdata.attrs, k, v)
		self.t_configuration = self.h5file.createTable('/', 'configuration', Configuration, 'configuration')