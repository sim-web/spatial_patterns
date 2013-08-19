# import pdb; pdb.set_trace()
import numpy as np
import os
import matplotlib as mpl
mpl.use('TkAgg')
import initialization
import plotting
import animating
import observables
import matplotlib.pyplot as plt
import math
import time
import output
# import tables

# sys.path.insert(1, os.path.expanduser('~/.local/lib/python2.7/site-packages/'))
path = os.path.expanduser('~/localfiles/itb_experiments/learning_grids/')

from snep.configuration import config
config['network_type'] = 'empty'

def main():
    from snep.utils import Parameter, ParameterArray
    from snep.experiment import Experiment

    # Note that runnet gets assigned to a function "run"
    exp = Experiment(path,runnet=run)
    tables = exp.tables

    target_rate = 5.0
    n_exc = np.array([100])
    n_inh = np.array([100])
   
   # Note: Maybe you don't need to use Parameter() if you don't have units
    param_ranges = {
        ('net','n_exc'):ParameterArray(n_exc),
        ('net','n_inh'):ParameterArray(n_inh),
        ('net','sigma_exc'):ParameterArray([0.03, 0.05]),
        ('net', 'init_weight_exc'):ParameterArray(20. * target_rate / n_exc),
        ('net', 'init_weight_inh'):ParameterArray(5.0 * target_rate / n_inh),   
        # TEST: only one parameter in range
        ('net','sigma_inh'):ParameterArray([0.03]),        
    }
    
    boxlength = 1.0
    sim_params = {
        ('sim', 'dimensions'):1,
        ('sim', 'boxtype'):'line',
        ('sim', 'boxlength'):boxlength,
        ('sim', 'diff_const'):0.01,
        ('sim', 'every_nth_step'):1,
        ('sim', 'seed'):1,
        ('sim', 'simulation_time'):10.0,
        ('sim', 'dt'):1.0,                                                                       
        ('sim', 'initial_x'):boxlength/2.0,                                                                       
        ('sim', 'initial_y'):boxlength/2.0,
    }
    
    params = {
        ('net', 'target_rate'):target_rate,
        ('net', 'init_weight_noise_exc'):0.05,
        ('net', 'init_weight_noise_inh'):0.05,
        ('net', 'eta_exc'):0.000001,  
        ('net', 'eta_inh'):0.002,  
        ('net', 'normalization'):'quadratic_multiplicative',          
    }

    tables.add_parameter_ranges(param_ranges)
    tables.add_parameters(sim_params)
    tables.add_parameters(params)

    # Note: maybe change population to empty string
    linked_params_tuples = [
        ('population', 'net', 'n_exc'),
        ('population', 'net', 'n_inh'),
        ('population', 'net', 'init_weight_exc'),
        ('population', 'net', 'init_weight_inh')
    ]

    tables.link_parameter_ranges(linked_params_tuples)
    exp.process()
    
    # Working on a table after it has been stored to disk
    # Note: here snep still knows which table you're working on
    # if that weren't the case you use make_tables_from_path(path) (in utils.py)

    # # open the tablefile
    # tables.open_file(True)
    # # iterate over all the paramspasepoints
    # for psp in tables.iter_paramspace_pts():
    #     # raw0 = tables.get_raw_data(psp,'something')
    #     # raw1 = tables.get_raw_data(psp,'something/simonsucks')
    #     # com = tables.get_computed(psp)
    #     # Note: a psp is a dictionary like in params (I think)
    #     # You can specify a path (here 'exc_sigmas') if you just want this 
    #     # specific part of it
    #     raw0 = tables.get_raw_data(psp, 'exc_sigmas')
    # print raw0
    # tables.close_file()

# Code from Owen:
    # tables.open_file(readonly)
    # for psp in tables.iter_paramspace_pts():
    #   tables.get_raw_data(psp)

# def run(params, all_network_objects, monitor_objs):
#     rawdata = {'raw_data':{'something':{'simonsucks':np.arange(50),
#                                         'owenrules':np.arange(5)
#                                         }}, 
#                'computed':{'poop':np.arange(20.)}}
#     return rawdata

def run(params, all_network_objects, monitor_objs):
    my_params = {}

    # Construct old style params file
    for k, v in params.iteritems():
        my_params[k[1]] = v

    rat = initialization.Rat(my_params)
    my_rawdata = rat.run(output=True)
    # rawdata is a dictionary of dictionaries (arbitrarily nested) with
    # keys (strings) and values (arrays or deeper dictionaries)
    # snep creates a group for each dictionary key and finally an array for
    # the deepest value. you can do this for raw_data or computed whenever you wish
    rawdata = {'raw_data': my_rawdata}
    return rawdata

if __name__ == '__main__':
    tables = main()
