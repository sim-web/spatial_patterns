import tables
import logging
from lock import LockAllFunctionsMeta
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
from snep.utils import csr_make_ints

class ResultsTablesReader(object):
    __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to read a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the simulation results from the experiment.
    '''

    '''
    These internal properties are defined so that we can restructure the hdf5 file without
    modifying the code that accesses it anywhere but here.
    '''
    _raw_data = property(fget=lambda self: self.results.raw_data)
    _computed = property(fget=lambda self: self.results.computed)
    _logs = property(fget=lambda self: self.results.log_files)

    def set_root(self, results):
        ''' 
        results - Specifies the root group to read from.
        '''
        self.results = results

    def log_info(self, msg):
        #print(msg)
        logging.getLogger('snep.experiment').info(msg)
    
    def _build_full_path(self, paramspace_pt, path):
        if paramspace_pt is None:
            full_path = path
        elif path is None:
            full_path = self.get_results_group_path(paramspace_pt)
        else:
            psp_path = self.get_results_group_path(paramspace_pt)
            full_path = '/'.join((psp_path,path))
        return full_path
        
    def get_raw_data(self, paramspace_pt=None, path=None):
        full_path = self._build_full_path(paramspace_pt, path)
        node = self.h5f.getNode(self._raw_data,full_path)
        all_data = self._read_node(node)
        return all_data
    
    def get_computed(self, paramspace_pt=None, path=None):
        full_path = self._build_full_path(paramspace_pt, path)
        node = self.h5f.getNode(self._computed,full_path)
        all_data = self._read_node(node)
        return all_data
    
    def _read_node(self, node):
        if isinstance(node, tables.Group):
            data = self._read_group(node)
        elif isinstance(node, tables.CArray):
            data = node.read()
        elif isinstance(node, tables.VLArray):
            data = self._read_VLArray(node)
        return data

    def _read_group(self, group):
        if 'issparse' in group._v_attrs._f_list():
            data = self._read_sparse(group)
        else:
            data = {node._v_name:self._read_node(node)
                        for node in group._f_iterNodes()}
        return data

    def _read_VLArray(self, vlarray, asdictionary=True):
        vla = vlarray.read()
        if asdictionary:
            data = {i:values for i,values in enumerate(vla)}
        else:
            data = [(i,v) for i,values in enumerate(vla) for v in values]
        return data

    def _read_sparse(self, group):
        data   = group.data.read()
        indices= group.indices.read()
        indptr = group.indptr.read()
        shape  = group.shape.read()
        indptr, indices = csr_make_ints(indptr, indices)
        arr = csr_matrix((data,indices,indptr),shape=shape)
        return arr

#    spike_times_as = 'VLArray'
#    def get_spike_times(self, paramspace_pt, asdictionary=False):
#        ''' 
#        Gets spike times for every neuron in the network, in the formats
#        normally used by Brian. That is, either a dictionary from neuron index
#        to arrays of spike times or as a list of tuples (neuron index, spike time)
#        '''
#        import numpy as np
#        results_group_path = self.get_results_group_path(paramspace_pt)
#        self.log_info('-> Get spike times for ' + results_group_path)
#        spikes = {}
#        try:
#            group = self._spikes._f_getChild(results_group_path)
#            if ResultsTablesReader.spike_times_as == 'Array':
#                for source in group._f_iterNodes():
#                    # Single Array and indices
#                    idx_slices = source.idx_slices.read()
#                    spiketimes = source.spiketimes.read()
#                    if asdictionary:
#                        src_spikes = {i:np.copy(spiketimes[l:r]) 
#                                        for i,(l,r) in enumerate(zip(idx_slices[:-1],idx_slices[1:]))}
#                    else:
#                        src_spikes = [(i,spk) 
#                                        for i,(l,r) in enumerate(zip(idx_slices[:-1],idx_slices[1:]))
#                                            for spk in spiketimes[l:r]]
#                    spikes[source._v_name] = src_spikes
#            elif ResultsTablesReader.spike_times_as == 'CArrays':
#                # Many CArrays
#                for source in group._f_iterNodes():
#                    source_name = source._v_name
#                    if asdictionary:
#                        spikes[source_name] = {int(neuron_i_spikes._v_name[1:]):neuron_i_spikes.read()
#                                                    for neuron_i_spikes in source._f_iterNodes()}
#                    else:
#                        spikes[source_name] = [(int(neuron_i_spikes._v_name[1:]),spk)
#                                                    for neuron_i_spikes in source._f_iterNodes()
#                                                        for spk in neuron_i_spikes.read()]
#            elif ResultsTablesReader.spike_times_as == 'VLArray':
#                # Single VLArray
#                for source in group._f_iterNodes():
#                    vlarray = source.read()
#                    if asdictionary:
#                        spikes[source._v_name] = {i:spks for i,spks in enumerate(vlarray)}
#                    else:
#                        spikes[source._v_name] = [(i,spk) for i,spks in enumerate(vlarray)
#                                                                for spk in spks]
#        except tables.exceptions.NoSuchNodeError:
#            print('Spike times result does not exist: ' + results_group_path)
#        self.log_info('<- Got spike times for ' + results_group_path)
#        return spikes
#
#    def get_state_variables(self, paramspace_pt):
#        results_group_path = self.get_results_group_path(paramspace_pt)
#        self.log_info('-> Get state variables for ' + results_group_path)
#        statevars = {}
#        try:
#            group = self._state_variables._f_getChild(results_group_path)
#            for source in group._f_iterNodes():
#                statevars[source._v_name] = {}
#                for var in source._f_iterNodes():
#                    times = var.times.read()
#                    values = var.values.read()
#                    statevars[source._v_name][var._v_name] = (times, values)
#        except tables.exceptions.NoSuchNodeError:
#            print('State variable result does not exist: ' + results_group_path)
#        self.log_info('<- Got state variables for ' + results_group_path)
#        return statevars
#
#    def get_population_rates(self, paramspace_pt):
#        results_group_path = self.get_results_group_path(paramspace_pt)
#        self.log_info('-> Get population rates for ' + results_group_path)
#        poprates = {}
#        try:
#            group = self._population_rates._f_getChild(results_group_path)
#            for rategroup in group._f_iterNodes():
#                times = rategroup.times.read()
#                rates = rategroup.rates.read()
#                poprates[rategroup._v_name] = (times, rates)
#        except tables.exceptions.NoSuchNodeError:
#            print('Population rates result does not exist: ' + results_group_path)
#        self.log_info('<- Got population rates for ' + results_group_path)
#        return poprates
#
#    def get_connection_weights(self, paramspace_pt):
#        results_group_path = self.get_results_group_path(paramspace_pt)
#        self.log_info('-> Get connection weights for ' + results_group_path)
#        weight_info = {}
#        try:
#            group = self._weights._f_getChild(results_group_path)
#            for wgroup in group._f_iterNodes():
#                times = wgroup.times.read()
#                
#                try:
#                    bins  = wgroup.bins.read()
#                    hists = wgroup.histograms.read()
#                except:
#                    bins, hists = None, None
#                    
#                try:
#                    csrgroup = wgroup._f_getChild('all_weights')
#                    data   = csrgroup.data.read()
#                    indices= csrgroup.indices.read()
#                    indptr = csrgroup.indptr.read()
#                    shape = csrgroup.shape.read()
#                    # The next few lines make sure the index arrays are integer types
#                    # because numpy doesn't like it when they're floats.
#                    indptr = indptr if indptr.dtype == np.int32 \
#                                            or indptr.dtype == np.int64 \
#                                        else indptr.astype(np.int32)
#                    indices= indices if indices.dtype == np.int32 \
#                                            or indices.dtype == np.int64 \
#                                        else indices.astype(np.int32)
#                    weights = csr_matrix((data,indices,indptr),shape=shape)
#                except:
#                    weights = None
#                
#                weight_info[wgroup._v_name] = (times, bins, hists, weights)
#        except tables.exceptions.NoSuchNodeError:
#            print('Connection weights result does not exist: ' + results_group_path)
#        self.log_info('<- Got connection weights for ' + results_group_path)
#        return weight_info
#
#    def get_computed_results(self, paramspace_pt, computation_name):
#        results_group_path = self.get_results_group_path(paramspace_pt)
#        self.log_info('-> Get computed results for ' + results_group_path)
#        computed = {}
#        try:
#            pspgroup = self._computed._f_getChild(results_group_path)
#            group = pspgroup._f_getChild(computation_name)
#            for resultgroup in group._f_iterNodes():
#                computed[resultgroup._v_name] = {}
#                for result in resultgroup._f_iterNodes():
#                    values = result.read()
#                    computed[resultgroup._v_name][result._v_name] = values
#        except tables.exceptions.NoSuchNodeError:
#            print('Computed result does not exist ['+results_group_path+']: '+computation_name)
#        self.log_info('<- Got computed results for ' + results_group_path)
#        return computed
    
class ResultsTables(ResultsTablesReader):
    __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to write a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the simulation results from the experiment.
    
    The two most important functions from a user perspective are add_computed
    and add_raw_data. They both store data using the same basic structure. Any
    data to be stored in the HDF5 file should be passed in the all_data 
    dictionary parameter. That dictionary will recursively map to structures 
    in the file as follows:
    - Any value in a dictionary which is a dense ndarray will be 
      stored as a CArray, whose name is the corresponding key.
    - Any value in a dictionary which is a sparse array will be stored
      as the data structures underlying the equivalent csr_matrix representation.
    - Any value in a dictionary which is another dictionary whose keys are all
      integers and values are all ndarrays will be stored as a VLArray.
    - Any value in a dictionary which is another dictionary whose keys are
      strings (or is empty) will be stored as a group, whose values will be
      defined as above.
      
    all_data = {
    'nameofdense':numpy.array,
    'nameofsparse':scipy.sparse,
    'nameofvlarray':{0:ndarray, 1:ndarray, ... n:ndarray},
    'nameofgroup':{'subgroup':{}, 'somearray':ndarray, etc.}
    }
     
    
    See those functions for further details on how to use them.
    '''
    def __init__(self):
        '''
        The default compression has not really been tested, I don't know if this works as expected.
        '''
        ResultsTablesReader.__init__(self)
        self.filters = tables.Filters(complevel=5, complib='zlib')

    def initialize(self, h5f, parentgroup):
        '''
        Once the ExperimentTables class has created a new hdf5 file it passes the root
        group into this function as the parentgroup. Here we then create the necessary
        default groups and tables that this class is responsible for.
        '''
        try:
            results = parentgroup.results
        except tables.NoSuchNodeError:
            results =  h5f.createGroup(parentgroup,'results')
        ResultsTablesReader.set_root(self,results)
        
        try:
            self.results.log_files
        except tables.NoSuchNodeError:
            h5f.createGroup(self.results,'log_files')

        try:
            self.results.raw_data
        except tables.NoSuchNodeError:
            raw_data = h5f.createGroup(self.results,'raw_data')

        try:
            self.results.computed
        except tables.NoSuchNodeError:
            raw_data = h5f.createGroup(self.results,'computed')

        self.h5f = h5f

    def add_computed(self, paramspace_pt, all_data, overwrite=False):
        '''
        Adds data contained in all_data (as described in the main class doc string.
        
        If paramspace_pt is None, will add it directly in results.computed
        in groups defined by the all_data dictionary.
        If paramspace_pt is not None, the groups defined by all_data
        will be added in results.computed.<paramspace_pt_group>
        
        overwrite determines how to handle existing groups.
        True - Any data with the same path in both all_data and the file
                will be deleted from the file before the new data is stored.
        False - If any group defined by all_data exists, then the new data
                will be stored in those groups. If any data (not groups) of 
                the same name already exists, PyTables will throw an exception.
        '''
        if paramspace_pt is None:
            group = self._computed
        else:
            path = self.get_results_group_path(paramspace_pt)
            group = self._nested_get_or_createGroups(self._computed, path)
        self._store_data(group, all_data, overwrite)

    def add_raw_data(self, paramspace_pt, all_data):
        '''
        Same behaviour as add_computed, except that paramspace_pt is
        not optional and overwrite is not available (since raw data should
        not be overwritable). 
        See comment above for more details.
        '''
        path = self.get_results_group_path(paramspace_pt)
        group = self._nested_get_or_createGroups(self._raw_data, path)
        self._store_data(group, all_data, False)

    def remove_computed_results(self, paramspace_pt, computation_name):
        results_group_path = self.get_results_group_path(paramspace_pt)
        ident = '{0}, {1}'.format(results_group_path, computation_name)
        self.log_info('-> Removing computed results for ' + ident)
        try:
            pspgroup = self._computed._f_getChild(results_group_path)
            compgroup = pspgroup._f_getChild(computation_name)
            compgroup._f_remove(recursive=True)
            self.h5f.flush()
        except tables.exceptions.NoSuchNodeError:
            print('Computed result does not exist ['+results_group_path+']: '+computation_name)
        self.log_info('<- Removed computed results for '+ident)

#    def copy_spiketimes(self, paramspace_pt, rtr_tables):
#        '''
#        Copy the ResultTableReader spikes from the subprocess into this ResultsTable.
#        '''
#        results_group = self.get_results_group_path(paramspace_pt)
#        self.log_info('-> Copying spike times for ' + results_group)
#        if rtr_tables._spikes._f_listNodes():
#            parent = self._nested_get_or_createGroups(self._spikes, results_group)
#            try:
#                rtr_tables._spikes._f_copyChildren(parent, recursive=True)
#            except tables.exceptions.HDF5ExtError as ex:
#                print('Copy spikes exception for '+results_group)
#                print(ex)
#        self.log_info('<- Copied spike times for ' + results_group)
#
#    def copy_population_rates(self, paramspace_pt, rtr_tables):
#        '''
#        Copy the ResultTableReader population rates from the subprocess into this ResultsTable.
#        '''
#        results_group = self.get_results_group_path(paramspace_pt)
#        self.log_info('-> Copying population rates for ' + results_group)
#        if rtr_tables._population_rates._f_listNodes():
#            parent = self._nested_get_or_createGroups(self._population_rates, results_group)
#            rtr_tables._population_rates._f_copyChildren(parent, recursive=True)
#        self.log_info('<- Copied population rates for ' + results_group)
#
#    def copy_state_variables(self, paramspace_pt, rtr_tables):
#        '''
#        Copy the ResultTableReader state variables from the subprocess into this ResultsTable.
#        '''
#        results_group = self.get_results_group_path(paramspace_pt)
#        self.log_info('-> Copying state variables for ' + results_group)
#        if rtr_tables._state_variables._f_listNodes():
#            parent = self._nested_get_or_createGroups(self._state_variables, results_group)
#            rtr_tables._state_variables._f_copyChildren(parent, recursive=True)
#        self.log_info('<- Copied state variables for ' + results_group)
#    
#    def copy_computed_results(self, paramspace_pt, rtr_tables):
#        '''
#        Copy the ResultTableReader computed results from the subprocess into this ResultsTable.
#        '''
#        results_group = self.get_results_group_path(paramspace_pt)
#        self.log_info('-> Copying computed results for ' + results_group)
#        if rtr_tables._computed._f_listNodes():
#            parent = self._nested_get_or_createGroups(self._computed, results_group)
#            rtr_tables._computed._f_copyChildren(parent, recursive=True)
#        self.log_info('<- Copied computed results for ' + results_group)
#
#    def add_spiketimes(self, paramspace_pt, source_name, spiketimes):
#        resultsgroup_str = self.get_results_group_path(paramspace_pt)
#        resultsgroup = self._nested_get_or_createGroups(self._spikes, resultsgroup_str)
#        self._add_spiketimes(resultsgroup_str, resultsgroup, 
#                             source_name, spiketimes)
#
#    def _add_spiketimes(self, resultsgroup_str, resultsgroup, 
#                        source_name, spiketimes):
#        '''
#        Store spike times for a single population into a variable length array, where the row
#        corresponds to the neuron index.
#        '''
#        import numpy as np
#        ident = '{0}, {1}'.format(resultsgroup_str, source_name)
#        self.log_info('-> Adding spike times for '+ident)
#
#        hasspikes = False
#        for spks in spiketimes.itervalues():
#            hasspikes = spks.size > 0
#            if hasspikes: break
#        if hasspikes:
#            if ResultsTablesReader.spike_times_as == 'Array':
#                # Single Arrays and indices
#                group = self.h5f.createGroup(resultsgroup, source_name)
#                nspikes = sum(spks.size for spks in spiketimes.itervalues())
#                num_neurons = len(spiketimes)
#                idx_slices = np.empty(num_neurons+1,dtype=np.int32)
#                all_spiketimes = np.empty(nspikes)
#                idx_slices[0] = 0
#                for i in xrange(num_neurons):
#                    spks = spiketimes[i]
#                    idx_slices[i+1] = idx_slices[i] + spks.size
#                    all_spiketimes[idx_slices[i]:idx_slices[i+1]] = spks
#                _idx = self.h5f.createArray(group, 'idx_slices', idx_slices)
#                _spks = self.h5f.createArray(group, 'spiketimes', all_spiketimes)
#            elif ResultsTablesReader.spike_times_as == 'CArrays':
#                # Many CArrays
#                group = self.h5f.createGroup(resultsgroup, source_name)
#                for i,spks in spiketimes.iteritems():
#                    if spks.size > 0:
#                        _spks = self.h5f.createCArray(group, 'n{0}'.format(i), tables.FloatAtom(),
#                                                  spks.shape, filters=self.filters)
#                        _spks[:] = spks
#            elif ResultsTablesReader.spike_times_as == 'VLArray':
#                # VLArray
#                spks = self.h5f.createVLArray(resultsgroup, source_name, 
#                                              tables.FloatAtom(shape=()))#, filters=self.filters)
#                for i in xrange(len(spiketimes)):
#                    spks.append(spiketimes[i])
#
#            self.h5f.flush()
#        else:
#            self.log_info('No spikes for {0}'.format(ident))
#        self.log_info('<- Added spike times for '+ident)
#

    @staticmethod
    def _maps_int_to_ndarray(data):
        return all([isinstance(k,int) and isinstance(v,np.ndarray)
                                    for k,v in data.iteritems()])

    def _store_data(self, group, all_data, overwrite=False):
        for name,value in all_data.iteritems():
            if isinstance(value,dict):
                if self._maps_int_to_ndarray(value):
                    self._create_VLArray(group, name, value)
                else:
                    # If overwrite is enabled, we want to provide a list
                    # of keys that should be deleted. This means any key
                    # that maps to a non-dictionary (e.g. an array), or a 
                    # dictionary that stores a VLArray.
                    todelete = [k for k,v in value.iteritems() 
                                    if not isinstance(v,dict) or
                                        self._maps_int_to_ndarray(v)]
                    subgroup = self._single_get_or_createGroup(group, name, 
                                                               overwrite, 
                                                               todelete)
                    self._store_data(subgroup, value)
            elif sparse.issparse(value):
                self._store_sparse(group, name, value)
            elif isinstance(value,np.ndarray):
                self._create_CArray(group, name, value)
    
    def _single_get_or_createGroup(self, parent, name, overwrite, todelete):
        '''
        It's necessary to have both this function and the below because if
        we combine them, the todelete list would not work correctly since
        names would have to be unique across all layers of the hierarchy.
        '''
        try:
            group = parent._f_getChild(name)
        except tables.NoSuchNodeError:
            group = self.h5f.createGroup(parent, name)
        else:
            if overwrite:
                ident = '/'.join((parent._v_name, name))
                self.log_info('!!! OVERWRITING group '+ident)
                for node in group._f_iterNodes():
                    if node._v_name in todelete:
                        node._f_remove(recursive=True)
        return group

    def _nested_get_or_createGroups(self, parent, path):
        for name in path.split('/'):
            try:
                parent = parent._f_getChild(name)
            except tables.NoSuchNodeError:
                parent = self.h5f.createGroup(parent, name)
        return parent

    def _store_sparse(self, group, name, arr):
        if not sparse.isspmatrix_csr(arr):
            arr = arr.tocsr()

        csrgroup = self.h5f.createGroup(group, name)
        csrgroup._v_attrs.issparse = True
        if arr is not None and arr.nnz > 0:
            indptr, indices = csr_make_ints(arr.indptr, arr.indices)
            self.h5f.createArray(csrgroup, 'data',   arr.data)
            self.h5f.createArray(csrgroup, 'indptr', indptr)
            self.h5f.createArray(csrgroup, 'indices',indices)
            self.h5f.createArray(csrgroup, 'shape',  arr.shape)
        self.h5f.flush()

    def _create_CArray(self, group, name, data):
        atom = tables.Atom.from_dtype(data.dtype)
        _d = self.h5f.createCArray(group, name, atom, data.shape, filters=self.filters)
        _d[:] = data
        self.h5f.flush()
        
    def _create_VLArray(self, group, name, data):
        _d = self.h5f.createVLArray(group, name, tables.FloatAtom(), 
                                    filters=self.filters)
        for i in xrange(len(data)):
            _d.append(data[i])
        self.h5f.flush()

    def add_spiketimes(self, paramspace_pt, source_name, spiketimes):
        resultsgroup_str = self.get_results_group_path(paramspace_pt)
        psp_spks_str = '/'.join((resultsgroup_str,'spikes'))
        group = self._nested_get_or_createGroups(self._raw_data, psp_spks_str)

        ident = '{0}, {1}'.format(resultsgroup_str, source_name)
        self.log_info('-> Adding spike times for '+ident)
        all_data = {source_name:spiketimes}
        self._store_data(group, all_data, False)
        self.log_info('<- Added spike times for ' +ident)

    def add_population_rates(self, paramspace_pt, source_name, times, rates):
        resultsgroup_str = self.get_results_group_path(paramspace_pt)
        psp_rates_str = '/'.join((resultsgroup_str,'population_rates'))
        group = self._nested_get_or_createGroups(self._raw_data, psp_rates_str)

        ident = '{0}, {1}'.format(resultsgroup_str, source_name)
        self.log_info('-> Adding population rates for '+ident)
        all_data = {source_name: {'times':times, 'rates':rates}}
        self._store_data(group, all_data, False)
        self.log_info('<- Added population rates for '+ident)

    def add_connection_weights(self, paramspace_pt, connection_name,
                               w_bins, w_times, w_values):
        resultsgroup_str = self.get_results_group_path(paramspace_pt)
        psp_weights_str = '/'.join((resultsgroup_str,'weights'))
        group = self._nested_get_or_createGroups(self._raw_data, psp_weights_str)

        weights, hist = w_values
        all_data = {connection_name:
                    {'bins':w_bins,'times':w_times,'all_weights':weights,'histograms':hist}}

        ident = '{0}, {1}'.format(resultsgroup_str, connection_name)
        self.log_info('-> Adding connection weights for '+ident)
        self._store_data(group, all_data, False)
        self.log_info('<- Added connection weights for ' +ident)

    def add_state_variables(self, paramspace_pt, source_name, variable_name, 
                            times, values):
        resultsgroup_str = self.get_results_group_path(paramspace_pt)
        psp_statevar_str = '/'.join((resultsgroup_str,'state_variables'))
        group = self._nested_get_or_createGroups(self._raw_data, psp_statevar_str)

        ident = '{0}, {1}, {2}'.format(resultsgroup_str, source_name, variable_name)

        self.log_info('-> Adding state variables for '+ident)
        all_data = {source_name:{variable_name:{'times':times,'values':values}}}
        self._store_data(group, all_data, False)
        self.log_info('<- Added state variables for ' +ident)

#    def add_computed_results(self, paramspace_pt, computation_name, result_name, 
#                             names_values, overwrite=False):
#        resultsgroup_str = self.get_results_group_path(paramspace_pt)
#        compgroup_str = '/'.join((resultsgroup_str, computation_name))
#        compgroup = self._nested_get_or_createGroups(self._computed, compgroup_str)
#
#        ident = '{0}, {1}, {2}'.format(resultsgroup_str, computation_name, result_name)
#        self.log_info('-> Adding computed results for '+ident)
#        if overwrite:
#            try:
#                resgroup = compgroup._f_getChild(result_name)
#            except tables.NoSuchNodeError:
#                resgroup = self.h5f.createGroup(compgroup, result_name)
#            else:
#                self.log_info('!!! OVERWRITING computed results for '+ident)
#                for c in resgroup._f_iterNodes():
#                    c._f_remove(recursive=True)
#        else:
#            resgroup = self.h5f.createGroup(compgroup, result_name)
#        
#        self._store_data(resgroup, names_values)
#        self.log_info('<- Added computed results for '+ident)

    def add_log_file(self, paramspace_pt, filename, filetext):
        '''
        Stores filetext in an Array called filename. The filetext parameter
        can be anything storeable as an Array, including a list of strings.
        '''
        resultsgroup_str = self.get_results_group_path(paramspace_pt)
        resultsgroup = self._nested_get_or_createGroups(self._logs, resultsgroup_str)
        ident = '{0}, {1}'.format(resultsgroup_str, filename)
        self.log_info('-> Adding log file for '+ident)
        self.h5f.createArray(resultsgroup, filename, filetext)
        self.h5f.flush()
        self.log_info('<- Added log file for '+ident)
