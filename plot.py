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

# date_dir = 'plot_test'
date_dir = '2013-09-04-14h05m51s'
#date_dir = '2013-09-02-19h49m48s'
tables = snep.utils.make_tables_from_path('/Users/simonweber/localfiles/itb_experiments/learning_grids/' + date_dir + '/experiment.h5')
t0 = time.time()
tables.open_file(True)
print tables
# iterate over all the paramspacepoints
for psp in tables.paramspace_pts():
	params = tables.as_dictionary(psp, True)

	if params['inh']['eta'] == 2e-3 and params['sim']['motion'] == 'diffusive':
		print params['inh']['eta']
		rawdata = tables.get_raw_data(psp)
		plot = plotting.Plot(params, rawdata)
		break

		
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

  #lambda: plot.weight_evolution(syn_type='inh', time_sparsification=10, weight_sparsification=1000),
  # lambda: plot.output_rate_distribution(start_time=(params['simulation_time']-10000)/params['every_nth_step']),
  # # lambda: plot.output_rate_as_function_of_fields_and_weights(time=1000/params['every_nth_step']),
  # # lambda: plot.output_rate_as_function_of_fields_and_weights(time=2000/params['every_nth_step']),
  lambda: plot.plot_output_rates_from_equation(frame=1003, spacing=11),
  # lambda:   plot.output_rates_vs_position(start_time=(params['simulation_time']-9000000)/params['every_nth_step']),
  # lambda: plot.output_rates_vs_position(start_time=1000, clipping=False),

]
plotting.plot_list(fig, plot_list)
t2 = time.time()
print 'Plotting took %f seconds' % (t2-t1)
plt.show()
