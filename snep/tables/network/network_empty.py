from network_base import *

class NetworkTables(NetworkTablesReaderBase,NetworkTablesBase,object):
    __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to create a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the network and simulation that is part of the experiment.
    '''
    _parameter_ranges = property(fget = lambda self: self.network.parameter_ranges)
    _parameters = property(fget = lambda self: self.network.parameters)
    def initialize(self, h5f, parentgroup):
        '''
        Given a parent group this function creates all the necessary default groups and
        tables that this class is responsible for.
        '''
        NetworkTablesBase.initialize(self, h5f, parentgroup)

    def as_dictionary(self, paramspace_pt, brian=True):
        '''
        This returns the entire table of parameters, but with any value
        defined in the paramspace_pt overwritten.
        '''
        params = self.get_all_fixed_experiment_params(brian)
        if brian:
            paramspace_pt = {k:v.quantity for k,v in paramspace_pt.iteritems()}
        update_params_at_point(params,paramspace_pt)
        return params

    def _read_param_ranges(self):
        '''
        This is called from paramspace.ParameterSpaceTables.build_parameter_space
        
        Returns a dictionary mapping from parameter identifiers (a tuple containing
        the parameter owner and parameter name) to a ParameterArray of values.
        '''
        group = self._parameter_ranges
        table = self._parameter_ranges.all
        param_ranges = self._read_named_variant_table(group,table,False)
        param_ranges = flatten_params_to_point(param_ranges)
        return param_ranges
    
    def _read_param_links(self):
        '''
        This is called from paramspace.ParameterSpaceTables.build_parameter_space
        
        Returns a list of linked variables
        '''
        param_links = self._user_defined_links()
        return param_links
    