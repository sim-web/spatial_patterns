# open the tablefile
from snep.configuration import config
config['network_type'] = 'empty'
import snep.utils
import utils
import plotting
import matplotlib.pyplot as plt
import time
import numpy as np
# tables = snep.utils.make_tables_from_path('/Users/simonweber/localfiles/itb_experiments/learning_grids/2013-08-20-11h48m27s/experiment.h5')
tables = snep.utils.make_tables_from_path('/Users/simonweber/localfiles/itb_experiments/learning_grids/2013-08-19-19h55m42s/experiment.h5')

def take_every_nth(array, n):
	"""
	Returns an array containing the first and every subsequent
	n-th array of the input array

	--------
	Arguments:
	- array: numpy array
	- n: step size for sparsification

	--------
	Returns:
	- The sparsified array
	"""
	array_length = len(array)
	index_array = np.zeros(array_length)
	for i in np.arange(array_length):
		if i % n == 0:
			index_array[i] = 1
	index_array = index_array.astype(bool)
	return array[index_array]

t0 = time.time()
tables.open_file(True)
print tables
# iterate over all the paramspacepoints
for psp in tables.iter_paramspace_pts():
	params = tables.as_dictionary(psp, True)
	my_params = {}
	rawdata = {}
	for k, v in params.iteritems():
		my_params[k[1]] = v
	if my_params['eta_inh'] == 0.00002:
	  	rawdata.update({'positions': tables.get_raw_data(psp, 'positions')})
		rawdata.update({'output_rates': tables.get_raw_data(psp, 'output_rates')})
		rawdata.update({'exc_weights': take_every_nth(tables.get_raw_data(psp, 'exc_weights'), 100)})
		rawdata.update({'exc_centers': tables.get_raw_data(psp, 'exc_centers')})
		rawdata.update({'inh_weights': take_every_nth(tables.get_raw_data(psp, 'inh_weights'), 100)})
		rawdata.update({'inh_centers': tables.get_raw_data(psp, 'inh_centers')})
		plot = plotting.Plot(my_params, rawdata)
tables.close_file()

t1 = time.time()
print 'Reading Data took %f seconds' % (t1-t0)
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
  lambda: plot.weight_evolution(syn_type='exc'),
  lambda: plot.weight_evolution(syn_type='inh'),
  # lambda: plot.output_rate_distribution(start_time=(params['simulation_time']-10000)/params['every_nth_step']),
  # # lambda: plot.output_rate_as_function_of_fields_and_weights(time=1000/params['every_nth_step']),
  # # lambda: plot.output_rate_as_function_of_fields_and_weights(time=2000/params['every_nth_step']),
  # lambda: plot.output_rates_from_equation(time=-5),
  # lambda:   plot.output_rates_vs_position(start_time=(params['simulation_time']-9000000)/params['every_nth_step']),
  lambda: plot.output_rates_vs_position(start_time=3000),

]
plotting.plot_list(fig, plot_list)
t2 = time.time()
print 'Plotting took %f seconds' % (t2-t1)
plt.show()
