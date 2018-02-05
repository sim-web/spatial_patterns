import numpy as np
import matplotlib.pyplot as plt
import initialization
import plotting

sigma_exc = np.array([0.04])
sigma_inh = np.array([0.13])
simulation_time = 4e4

#######################################################################
############################## Parameters##############################
#######################################################################
# Here we define single parameters for the simulation
# The structure is:
# 'exc' / 'inh': For excitatoyr and inhibitory synapses
# 'sim': For main simulation parameters
# 'out':  For parameters that have to do with the output neurons
params = {
			'subdimension': 'none',
			'sim':
				{
                    'simulation_time': simulation_time,
                    'gaussian_process': False,
					'gaussian_process_rescale': 'fixed_mean',
					'take_fixed_point_weights': True,
					'input_space_resolution': sigma_exc / 10.,
					'spacing': 201,
					'equilibration_steps': 10000,
					'stationary_rat': False,
					'same_centers': False,
					'first_center_at_zero': False,
					'lateral_inhibition': False,
					'output_neurons': 1,
					'weight_lateral': 0.0,
					'tau': 10.,
					'symmetric_centers': True,
					'dimensions': 1,
					'boxtype': 'linear',
					'radius': 1,
					'diff_const': 0.01,
					'every_nth_step': simulation_time / 4,
					'every_nth_step_weights': simulation_time / 4,
					'seed_trajectory': 1,
					'seed_init_weights': 1,
					'seed_centers': 1,
					'seed_sigmas': 1,
					'seed_motion': 0,
					'dt': 1,
					'initial_x': 0.1,
					'initial_y': 0.2,
					'initial_z': 0.15,
					'velocity': 1e-2,
					'persistence_length': 1,
					'motion': 'persistent',
                    #################################################
                    ### Parameters that are typically not changed ###
                    #################################################
                    # Type of elementary tuning function
                    'tuning_function': 'gaussian',
                    # Store 2 sigma**2 array (not important)
                    'store_twoSigma2': False,
                    # Numper of input tuning examples that are stored
                    'save_n_input_rates': 3,
                    # Normalize each input rate
                    'input_normalization': 'none',
                    # Noisy alignment of head direction with running direction
                    'head_direction_sigma': np.pi / 6.,
                    # Discretize space for efficiancy
                    'discretize_space': True,
                    # Convolution dx (not important)
					'fixed_convolution_dx': False,
                    # Scaling the excitatory weights with the variance of the
                    #  tuning functions.
                    'scale_exc_weights_with_input_rate_variance': False,
                    #######################################################
                    ### Switching box side (wall experiment) parameters ###
                    #######################################################
                    'boxside_independent_centers': False,
                    # The boxside in which the rat learns first, for the
                    # boxside switch experiments.
                    'boxside_initial_side': 'left',
                    # Time at which the rat can explore the entire arena
                    # Set to False, if no wall experiment is conducted.
                    'explore_all_time': False,
                    # Time at which the rat should switch to the right side
                    # of the box on move only in the right side.
                    # Set to False, if no wall experiment is conducted.
                    'boxside_switch_time': False,
                    # We typically do not start in room2, so default is False
                    'in_room2': False,
                    # Correlation between the inputs
                    'alpha_room1': 1,
                    'alpha_room2': 0.5,
                    'room_switch_method': 'some_inputs_identical',
                    'room_switch_time': False,
				},
			'out':
				{
					'target_rate': 1,
					'normalization': 'quadratic_multiplicative',
				},
			'exc':
				{
					'grid_input_sidelength': 10,
					'grid_input_spacing_noise': 6*sigma_exc[0] / 6,
					'save_n_input_rates': 3,
					'gp_stretch_factor': 1.0,
					'gp_extremum': 'none',
					'center_overlap_factor': 3.,
					'number_per_dimension': np.array([160]),
					'distortion': 'half_spacing',
					'eta': 1e-3,
					'sigma': sigma_exc,
					'sigma_spreading': np.array([0]),
					'sigma_distribution': np.array(['uniform']),
					'fields_per_synapse': 1,
					'init_weight': 1,
					'init_weight_spreading': 5e-1,
					'init_weight_distribution': 'uniform',
					'gaussian_height': 1,
					'real_gaussian_height': 1,
					'untuned': False,
				},
			'inh':
				{
					'grid_input_sidelength': 1,
					'grid_input_spacing_noise': 0.,
					'save_n_input_rates': 3,
					'gp_stretch_factor': 1.0,
					'gp_extremum': 'none',
					'center_overlap_factor': 3.,
					'weight_factor': 1,
					'number_per_dimension': np.array([40]),
					'distortion': 'half_spacing',
					'eta': 1e-2,
					'sigma': sigma_inh,
					'sigma_spreading': np.array([0]),
					'sigma_distribution': np.array(['uniform']),
					'fields_per_synapse': 1,
					'init_weight': 1.0,
					'init_weight_spreading': 5e-2,
					'init_weight_distribution': 'uniform',
					'gaussian_height': 1,
					'real_gaussian_height': 1,
					'untuned': False,
				}
		}

if __name__ == '__main__':
    ###########################################################################
    ############################## Run the code ###############################
    ###########################################################################
    # The code should return all the rawdata as a nested dictionary whose
    # final leaves are arrays
    # See initialization.py for the run function
    rat = initialization.Rat(params)
    rawdata = rat.run()
    # rawdata is a dictionary of dictionaries (arbitrarily nested) with
    # keys (strings) and values (arrays or deeper dictionaries)
    # snep creates a group for each dictionary key and finally an array for
    # the deepest value. you can do this for raw_data or computed
    # whenever you wish
    plot = plotting.Plot(params=params, rawdata=rawdata)
    plt.subplot(211)
    plot.plot_output_rates_from_equation(time=0)
    plt.subplot(212)
    plot.plot_output_rates_from_equation(time=simulation_time)
    plt.show()