# open the tablefile
from snep.configuration import config
config['network_type'] = 'empty'
import snep.utils
import utils
import plotting
import matplotlib.pyplot as plt
import time
import general_utils.arrays
import numpy as np
# import IPython
# tables = snep.utils.make_tables_from_path('/Users/simonweber/localfiles/itb_experiments/learning_grids/2013-08-20-11h48m27s/experiment.h5')
# tables = snep.utils.make_tables_from_path('/Users/simonweber/localfiles/itb_experiments/learning_grids/2013-08-19-19h55m42s/experiment.h5')
tables = snep.utils.make_tables_from_path('/Users/simonweber/localfiles/itb_experiments/learning_grids/2013-08-23-18h47m27s/experiment.h5')
t0 = time.time()
tables.open_file(True)
print tables
# iterate over all the paramspacepoints
for psp in tables.paramspace_pts():
	params = tables.as_dictionary(psp, True)
	# my_params = {}
	# rawdata = {}
	# for k, v in params.iteritems():
	# 	my_params[k[1]] = v
	# for d in params:
	# 	if isinstance(params[d], dict):
	# 		for k in params[d]:
	# 			my_params[k] = params[d][k]
	if params['inh']['eta'] == 0.00002 and params['sim']['motion'] == 'persistent':
		rawdata = tables.get_raw_data(psp)
		# print tables.get_raw_data(psp, 'exc/weights')
	  	# rawdata.update({'positions': tables.get_raw_data(psp, 'positions')})
		# rawdata.update({'output_rates': tables.get_raw_data(psp, 'output_rates')})
		# # rawdata.update({'exc_weights': take_every_nth(tables.get_raw_data(psp, 'exc_weights'), 100)})
		# # rawdata.update({'exc_weights': tables.get_raw_data(psp, 'exc_weights')})	
		# # rawdata.update({'exc_weights': tables.get_computed(psp, 'exc_weights_sparse')})
		# rawdata.update({'exc_centers': tables.get_raw_data(psp, 'exc_centers')})
		# # rawdata.update({'inh_weights': tables.get_computed(psp, 'inh_weights_sparse')})
		# # rawdata.update({'inh_weights': tables.get_raw_data(psp, 'inh_weights')})		
		# # rawdata.update({'inh_centers': tables.get_raw_data(psp, 'inh_centers')})
		# # rawdata.update({'inh_weights': general_utils.arrays.take_every_nth(tables.get_raw_data(psp, 'inh_weights'), 100)})
		# # rawdata.update({'inh_centers': tables.get_raw_data(psp, 'inh_centers')})
		
		plot = plotting.Plot(params, rawdata)
tables.close_file()

t1 = time.time()
print 'Reading Data took %f seconds' % (t1-t0)
# <demo> --- stop ---
# IPython.embed()
fig = plt.figure()
plot_list = [
  # lambda: [plot.fields_times_weights(syn_type='exc'), 
  #           plot.weights_vs_centers(syn_type='exc')],
  # lambda: [plot.fields_times_weights(syn_type='inh'), 
  #           plot.weights_vs_centers(syn_type='inh')],
  # lambda: plot.weights_vs_centers(syn_type='exc', time=-1), 
  # lambda: plot.weights_vs_centers(syn_type='inh', time=-1), 
  # # lambda: plot.weights_vs_centers(syn_type='exc'),    
  # # lambda: plot.weights_vs_centers(syn_type='inh'),            
  # lambda: plot.weight_evolution(syn_type='exc', time_sparsification=10, weight_sparsification=500),
  # lambda: plot.weight_evolution(syn_type='inh', time_sparsification=10, weight_sparsification=500),

  lambda: plot.weight_evolution(syn_type='inh', time_sparsification=10, weight_sparsification=1000),
  # lambda: plot.output_rate_distribution(start_time=(params['simulation_time']-10000)/params['every_nth_step']),
  # # lambda: plot.output_rate_as_function_of_fields_and_weights(time=1000/params['every_nth_step']),
  # # lambda: plot.output_rate_as_function_of_fields_and_weights(time=2000/params['every_nth_step']),
  # lambda: plot.output_rates_from_equation(time=-5),
  # lambda:   plot.output_rates_vs_position(start_time=(params['simulation_time']-9000000)/params['every_nth_step']),
  # lambda: plot.output_rates_vs_position(start_time=200, clipping=True),

]
plotting.plot_list(fig, plot_list)
t2 = time.time()
print 'Plotting took %f seconds' % (t2-t1)
plt.show()
