import matplotlib as mpl
mpl.use('TkAgg')
import initialization
import plotting
import animating
import observables
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import output
import tables

t0 = time.time()
##################################
##########	Parameters	##########
##################################
params = {
	'dimensions': 1,
	'boxtype': 'line',
	'boxlength': 1.0,
	'n_exc': 100,
	'n_inh': 100,
	'sigma_exc': 0.03,
	'sigma_inh': 0.1,
	'init_weight_noise_exc': 0.05,  # Noise on weights (as relative value)
	'init_weight_noise_inh': 0.05,  # 0.05 corresponds to +- 5 percent
	# 'simulation_time': 10000000.0,
	'simulation_time': 50000.0,
	'target_rate': 5.0,
	'diff_const': 0.01,
	'dt': 1.0,
	'eta_exc': 0.00000001,
	'eta_inh': 0.00002,
	'normalization': 'quadratic_multiplicative',
	'velocity': 0.01,
	'persistence_length': 0.1,
	'motion': 'diffusive',
	'boundary_conditions': 'periodic',
	# 'normalization': 'linear_multiplicative',
	# 'normalization': 'quadratic_multiplicative',
	'every_nth_step': 10,
	# 'seed': 1
}

params['init_weight_exc'] = 20. * params['target_rate'] / params['n_exc']
params['init_weight_inh'] = 5.0 * params['target_rate'] / params['n_inh']

params['initial_x'] = params['boxlength']/2.0
params['initial_y'] = params['boxlength']/2.0

# np.random.seed(params['seed'])
##############################################
##########	Configure the output	##########
##############################################

# o = output.Output(params)

# # datetime = time.strftime('%Y-%m-%d-%Hh%Mm%Ss')
# # file_path = '/Users/simonweber/doktor/learning_grids/data/' + datetime
# # h5file = tables.open_file(file_path, mode='w', title='')
# # table = h5file.create_table('/', 'rawdata', Rawdata(params), 'rawdata')
# # row = table.row

######################################
##########	Initialize Rat	##########
######################################
rat = initialization.Rat(params)

##################################
#########	Move Rat	##########
##################################
# rawdata = rat.run(output=False, rawdata_table=o.t_rawdata, configuration_table=o.t_configuration)
rawdata = rat.run(output=True)
# o.t_configuration.flush()
# o.t_rawdata.flush()
# o.h5file.close()

t1 = time.time()
print 'Time for Simulating: %f Seconds' %(t1 - t0)
# # print rawdata['exc_centers']
# print rawdata
##################################
##########	Plotting	##########
##################################

# Setting up the rawdata from file
# file_path = '/Users/simonweber/doktor/learning_grids/data/2013-08-09-16h41m09s.h5'
# f = tables.openFile(file_path, mode='r')
# t_rawdata = f.getNode('/rawdata')
# file_path_old = '/Users/simonweber/doktor/learning_grids/data/2013-08-08-23h59m43s.h5'
# f_old = tables.openFile(file_path_old, mode='r')
# t_configuration = f_old.getNode('/configuration')
# t_sparsified_weights = f.getNode('/sparsified_weights/sparse_factor')
# t_last_rows = f.getNode('/last_rows')
# rawdata = {}

# rawdata['positions'] = t_rawdata.col('position')
# rawdata['output_rates'] = t_rawdata.col('output_rate')

# rawdata['exc_centers'] = t_configuration.col('exc_centers')[0]
# rawdata['inh_centers'] = t_configuration.col('inh_centers')[0]
# rawdata['exc_sigmas'] = t_configuration.col('exc_sigmas')[0]
# rawdata['inh_sigmas'] = t_configuration.col('inh_sigmas')[0]

# For the weight_evolution Plot
# rawdata['n_exc'] = 100
# rawdata['n_inh'] = 100
# rawdata['exc_weights'] = t_sparsified_weights.col('exc_weights')
# rawdata['inh_weights'] = t_sparsified_weights.col('inh_weights')

# For the output rate from equation plot
# rawdata['exc_weights'] = t_last_rows.col('exc_weights')
# rawdata['inh_weights'] = t_last_rows.col('inh_weights')
# rawdata['positions'] = t_last_rows.col('position')
# rawdata['output_rates'] = t_last_rows.col('output_rate')

print 'data read'

plot = plotting.Plot(params, rawdata)
fig = plt.figure()
# Create a list of lambda forms
# If you put multiple plot function calls into parentheses and in one
# lambda form, they get plotted on the same axis
plot_list = [
	# lambda: [plot.fields_times_weights(syn_type='exc'), 
	# 			plot.weights_vs_centers(syn_type='exc')],
	# lambda: [plot.fields_times_weights(syn_type='inh'), 
	# 			plot.weights_vs_centers(syn_type='inh')],
	# lambda: plot.weights_vs_centers(syn_type='exc', time=-1),	
	# lambda: plot.weights_vs_centers(syn_type='inh', time=-1),	
	# # lambda: plot.weights_vs_centers(syn_type='exc'),	
	# # lambda: plot.weights_vs_centers(syn_type='inh'),			
	# lambda: plot.weight_evolution(syn_type='exc'),
	# lambda: plot.weight_evolution(syn_type='inh'),
	# lambda: plot.output_rate_distribution(start_time=(params['simulation_time']-10000)/params['every_nth_step']),
	# # lambda: plot.output_rate_as_function_of_fields_and_weights(time=1000/params['every_nth_step']),
	# # lambda: plot.output_rate_as_function_of_fields_and_weights(time=2000/params['every_nth_step']),
	# lambda: plot.output_rates_from_equation(time=-5),
	# lambda:	plot.output_rates_vs_position(start_time=(params['simulation_time']-9000000)/params['every_nth_step']),
	lambda:	plot.output_rates_vs_position(start_time=200),

]
plotting.plot_list(fig, plot_list)

##################################
##########	Animating	##########
##################################
# save_path = '/Users/simonweber/Desktop/ani.mp4'
# animation = animating.Animation(rat, rawdata, start_time=0)
# # animation.animate_all_synapses(interval=200)
# animation.animate_positions(interval=10)
# twoSigma2_exc = 
# print rawdata['exc_weights']
print 'Time for Plotting: %f Seconds' % (time.time() - t1)
print 'Total Time: %f Seconds' % (time.time() - t0)
plt.show()
