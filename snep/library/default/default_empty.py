def make(params):
    '''
    Here we construct the network from the parameters, and return it in a dictionary.
    '''
    all_network_objects = {}
    return all_network_objects

def preproc(params, all_network_objects, monitors):
    '''
    This is an opportunity to construct the requested monitors
    and complete other pre-processing tasks prior to running.
    '''
    mon_objs = {}
    return all_network_objects, mon_objs

def run(params, net, monitor_objs):
    '''
    Now we simulate the network, extract any raw data from the simulation
    and return it in a dictionary. The rawdata dictionary must have
    a very particular structure. See snep.utils.monitors_to_rawdata
    for the format.
    '''
    rawdata = {}
    return rawdata

def postproc(params, rawdata):
    '''
    Here we can perform any data processing tasks we want on the rawdata
    and potentially remove items from rawdata if we don't want them
    stored in the hdf5 file.
    '''
    return rawdata
