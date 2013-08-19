# open the tablefile
from snep.configuration import config
config['network_type'] = 'empty'
import snep.utils
import utils
import plotting
import matplotlib.pyplot as plt

tables = snep.utils.make_tables_from_path('/Users/simonweber/localfiles/itb_experiments/learning_grids/2013-08-19-16h57m02s/experiment.h5')

tables.open_file(True)
print tables
# iterate over all the paramspacepoints
for psp in tables.iter_paramspace_pts():
    params = tables.as_dictionary(psp, True)
    my_params = {}
    for k, v in params.iteritems():
        my_params[k[1]] = v
    print my_params
    if my_params['sigma_exc'] == 0.03:
        rawdata = tables.get_raw_data(psp)
        print rawdata
        plot = plotting.Plot(my_params, rawdata)
tables.close_file()

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
  # lambda: plot.weight_evolution(syn_type='exc'),
  # lambda: plot.weight_evolution(syn_type='inh'),
  # lambda: plot.output_rate_distribution(start_time=(params['simulation_time']-10000)/params['every_nth_step']),
  # # lambda: plot.output_rate_as_function_of_fields_and_weights(time=1000/params['every_nth_step']),
  # # lambda: plot.output_rate_as_function_of_fields_and_weights(time=2000/params['every_nth_step']),
  # lambda: plot.output_rates_from_equation(time=-5),
  # lambda:   plot.output_rates_vs_position(start_time=(params['simulation_time']-9000000)/params['every_nth_step']),
  lambda: plot.output_rates_vs_position(start_time=0),

]
plotting.plot_list(fig, plot_list)

plt.show()
