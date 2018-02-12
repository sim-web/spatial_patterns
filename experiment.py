import matplotlib.pyplot as plt
from matplotlib import gridspec
from spatial_patterns.initialization import Rat
from spatial_patterns import plotting
from spatial_patterns import parameters

##############################################
########## Select a set of parameters ########
##############################################
### One dimension ###
# 1 dimension, place cells to grid cell
params = parameters.params_1d_place2grid
# # 1 dimension, non-localized input to grid cell
# params = parameters.params_1d_non_localized2grid
# # 1 dimension, non-localized input to place cell
# params = parameters.params_1d_non_localized2place
# # 1 dimension, non-localized input to invariant pattern
# params = parameters.params_1d_non_localized2invariant

### Two dimensions ###
# 2 dimensions, place cells to grid cell
# params = parameters.params_2d_place2grid
# 2 dimensions, non-localized to grid cell
# params = parameters.params_2d_non_localized2grid

### Fast 1D Test ##s#
# params = parameters.params_test
### Fast 2D Test ###
# params = parameters.params_test_2d

###########################################################################
############################## Run the code ###############################
###########################################################################
# Instantiate a rat
rat = Rat(params)
# Let the rat run and learn
rawdata = rat.run()

###############################################
################ Plot the data ################
###############################################
# Instantiate the Plot class.
# You can select any plotting function from the Plot class
plot = plotting.Plot(params=params, rawdata=rawdata)

dim = params['sim']['dimensions']
# In one dimension, show a heat map of the time evolution
if dim == 1:
    fig = plt.figure(figsize=(7, 3))
    plot.output_rate_heat_map(start_time=0,
                              end_time=params['sim']['simulation_time'],
                              from_file=True, maximal_rate=7)
    plt.colorbar()
# In two dimensions, show 4 example screenshots
elif dim == 2:
    plt.figure(figsize=(6, 1.5))
    t_sim = params['sim']['simulation_time']
    gs = gridspec.GridSpec(1, 3)
    for n, time in enumerate([0, t_sim/2, t_sim]):
        plt.subplot(gs[n])
        plot.plot_output_rates_from_equation(time=time, from_file=True)
        # Plot the correlogram instead
        # plot.plot_correlogram(time=time, from_file=True)

# Show the plot. NB: You might need a different backend.
plt.show()
# Save the figure to disk
# plt.savefig(path='figure.png', dpi=300)