import matplotlib.pyplot as plt
from matplotlib import gridspec
from initialization import Rat
import plotting
import parameters


##############################################
########## Select a set of parameters ########
##############################################
### One dimension ###
# 1 dimension, place cells to grid cell
# params = parameters.params_1d_place2grid
# # 1 dimension, non-localized input to grid cell
# params = parameters.params_1d_non_localized2grid
# # 1 dimension, non-localized input to place cell
# params = parameters.params_1d_non_localized2place
# # 1 dimension, non-localized input to invariant pattern
# params = parameters.params_1d_non_localized2invarant
### Two dimensions ###
params = parameters.params_2d_place2grid
# params = parameters.params_2d_place2grid_fast
### Fast 1D Test ###
# params = parameters.params_test
### Fast 2D Test ###
# params = parameters.params_test_2d

###########################################################################
############################## Run the code ###############################
###########################################################################
# The code should return all the rawdata as a nested dictionary whose
# final leaves are arrays
# See initialization.py for the run function
rat = Rat(params)
rawdata = rat.run()
# rawdata is a dictionary of dictionaries (arbitrarily nested) with
# keys (strings) and values (arrays or deeper dictionaries)
# snep creates a group for each dictionary key and finally an array for
# the deepest value. you can do this for raw_data or computed
# whenever you wish

plot = plotting.Plot(params=params, rawdata=rawdata)
# plt.subplot(211)
# plot.plot_output_rates_from_equation(time=0)
# plt.subplot(212)
# plot.plot_output_rates_from_equation(time=params['sim']['simulation_time'])
# fig = plt.figure(figsize=(5,5))
if params['sim']['dimensions'] == 1:
    plot.output_rate_heat_map(start_time=0,
                              end_time=params['sim']['simulation_time'],
                              from_file=True, maximal_rate=7)
    plt.colorbar()

elif params['sim']['dimensions'] == 2:
    t_sim = params['sim']['simulation_time']
    gs = gridspec.GridSpec(1, 3)
    for n, time in enumerate([0, t_sim/2, t_sim]):
        plt.subplot(gs[n])
        plot.plot_output_rates_from_equation(time=time, from_file=True)

plt.show()
# plt.savefig(path='figure.png', dpi=300)