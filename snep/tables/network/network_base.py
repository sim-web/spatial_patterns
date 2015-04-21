import tables
from snep.utils import Parameter, ParameterArray, ParametersNamed, \
                        update_params_at_point, flatten_params_to_point, \
                        write_sparse, read_sparse, write_named_param
from ..lock import LockAllFunctionsMeta
from ..rows import NamedVariantType, TimedVariantType, AliasedParameters, \
                    Population, Connection, LinkedRanges, Subpopulation

class NetworkTablesReaderBase(object):
    __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to read a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the network and simulation that is part of the experiment.
    '''

    _linked = property(fget = lambda self: self.network.linked_parameter_ranges)

    def set_root(self, network):
        '''
        network - a group in the ExperimentTables hdf5 file.
        '''
        self.network = network

    def get_all_fixed_experiment_params(self,brian):
        '''
        Returns a dictionary of dictionaries containing the parameters
        common to all simulations in this experiment. In other words,
        everything not defined by the parameter ranges. 
        '''
        return self.get_general_params(brian)

    def get_general_params(self,brian):
        '''
        Returns parameters defined by the "parameters" table hierarchy. For
        the generic 'empty' type of network, this is all the fixed parameters. 
        For 'brian' networks this excludes the tables that actually define the network.
        '''
        group = self._parameters
        table = self._parameters.all
        params = self._read_named_variant_table(group,table,brian)
        return params

    def _user_defined_links(self):
        '''
        Returns a list of all the user defined linked variables as tuples such 
        as (('exc', 'C'), ('inh', 'g_l')) which means that those parameters should be varied
        together over their ranges.
        '''
        all_links = []
        for l in self._linked:
            coord_a, coord_b = l['coord_a'], l['coord_b']
            link = (self._get_tuple_for_coord(coord_a), # defined in tables.paramspace
                    self._get_tuple_for_coord(coord_b))
            all_links.append(link)
        return all_links

    def _read_named_variant_table(self, group, table, brian):
        ''' Returns a dictionary that maps parameter names to parameter values from a given table '''
        return {row['name']:self._get_variant_type(group, row, 'vartype', brian)
                    for row in table}

    def _read_parameters_named(self,group, units):
        table = group._f_getChild('aliasedparameters')
        allparams = [(r['name'],read_sparse(group, r['name']) 
                                    if r['issparse'] else 
                                        group._f_getChild(r['name']).read())
                        for r in table]
        variant = ParametersNamed(allparams, units)
        return variant

    def _get_variant_type(self, group, row, field, brian):
        '''
        This reads a special variant type of data from a table. The data can be a short
        string, a single Parameter or a ParameterArray. This permits explicit support for
        a Brian feature which allows the user to define certain network parameters in all
        of those ways (see how the Brian Synapse class defines delays for an example).
        '''
        vartype = row[field+'/vartype']
        if vartype == 'strcode':
            variant = row[field+'/varstring']
        elif vartype == 'table':
            subgroup = group._f_getChild(row[field+'/varstring'])
            subtable = subgroup._f_getChild('all')
            variant = self._read_named_variant_table(subgroup, subtable, brian)
        elif vartype == 'named':
            subgroup = group._f_getChild(row[field+'/varstring'])
            variant = self._read_parameters_named(subgroup,row[field+'/units'])
        else:
            if vartype == 'array':
                variant = ParameterArray(group._f_getChild(row[field+'/varstring']).read(), 
                                       row[field+'/units'])
            elif vartype == 'sparse':
                csr = read_sparse(group,row[field+'/varstring'])
#                data   = csrgroup.data.read()
#                indices= csrgroup.indices.read()
#                indptr = csrgroup.indptr.read()
#                shape = csrgroup.shape.read()
#                # The next few lines make sure the index arrays are integer types
#                # because numpy doesn't like it when they're floats.
#                indptr = indptr if indptr.dtype == np.int32 \
#                                        or indptr.dtype == np.int64 \
#                                    else indptr.astype(np.int32)
#                indices= indices if indices.dtype == np.int32 \
#                                        or indices.dtype == np.int64 \
#                                    else indices.astype(np.int32)
#                csr = csr_matrix((data,indices,indptr),shape=shape)
                variant = ParameterArray(csr, row[field+'/units'])
            elif vartype == 'float':
                variant = Parameter(row[field+'/varflt'],row[field+'/units'])
            elif vartype == 'integer':
                variant = Parameter(row[field+'/varint'],row[field+'/units'])
            else:
                raise Exception('Unknown type was stored in variant table: {0}'.format(vartype))
            if brian:
                variant = variant.quantity
        return variant

class NetworkTablesBase(object):
    __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to create a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the network and simulation that is part of the experiment.
    '''
    
    def initialize(self, h5f, parentgroup):
        '''
        Given a parent group this function creates all the necessary default groups and
        tables that this class is responsible for.
        '''
        self.h5f = h5f
        try:
            network = parentgroup.network
        except tables.exceptions.NoSuchNodeError:
            network = self.h5f.createGroup(parentgroup,'network')
        self.set_root(network)

        try:
            self._parameters
        except tables.exceptions.NoSuchNodeError:
            parameters = self.h5f.createGroup(self.network,'parameters')
            self.h5f.createTable(parameters, 'all', NamedVariantType, "General Parameters")
        try:
            rangegroup = self._parameter_ranges
        except tables.exceptions.NoSuchNodeError:
            rangegroup = self.h5f.createGroup(self.network, 'parameter_ranges')
            self.h5f.createTable(rangegroup, 'all', NamedVariantType, "General Parameter Ranges")
        try:
            self.network.linked_parameter_ranges
        except tables.exceptions.NoSuchNodeError:
            self.h5f.createTable(self.network, 'linked_parameter_ranges', 
                                 LinkedRanges, "Linked parameter ranges")

    def copy_network(self, destination_tables):
        self.network._f_copyChildren(destination_tables.network, 
                                     recursive=True, overwrite=True)

    def add_parameters(self, params):
        '''
        Adds the provided dictionary of dictionaries to the global parameters
        table. For brian networks, this is currently only used for a few
        things like simulation time step, etc.
        '''
        paramsgroup = self._parameters
        paramstable = self._parameters.all
        self._write_named_variant_table(paramsgroup, paramstable, params)

    def add_parameter_ranges(self, param_ranges):
        '''
        Add a range of values for each parameter listed in the provided
        dictionary of dictionaries. The cartesian product of these ranges
        form the parameter space. If two ranges should not form a subspace,
        but rather co-vary, they can be linked using link_parameter_ranges.
        '''
        prgroup = self._parameter_ranges
        prtable = prgroup.all
        self._write_named_variant_table(prgroup, prtable, param_ranges)

    def add_parameter_range(self, param_id, param_values):
        '''
        Singular of add_parameter_ranges
        '''
        self.add_parameter_ranges({param_id:param_values})

    def _write_named_variant_table(self, group, table, variants):
        '''
        Any table of variant types needs to be defined in its own group, so that we can
        freely save ParameterArrays as array nodes without worrying about name collisions.
        The variant type table must be called 'all'.
        '''
        row = table.row
        for name, value in variants.iteritems():
            row['name'] = name
            self._set_variant_type(group, row, 'vartype', value, name)
            if isinstance(value, dict):
                subgroup = self.h5f.createGroup(group, name)
                subtable = self.h5f.createTable(subgroup, 'all', NamedVariantType)
                self._write_named_variant_table(subgroup, subtable, value)
            row.append()
        table.flush()

    def _write_timed_variant_table(self, group, timedvars):
        '''
        Same as _write_named_variant_table but with TimeVariantType rows
        rather than NamedVariantType rows.
        The timedvars parameter is a dictionary mapping from a variable name to either
        values, strings, arrays, or a tuple containing an array and a Parameter with units
        of time. This last case corresponds to the clock dt for a time varying variable.
        '''
        row = group.all.row
        for varname, values in timedvars.iteritems():
            if isinstance(values,tuple):
                values, dt = values
            else:
                dt = Parameter(0)
            row['dt/value'] = dt.value
            row['dt/units'] = dt.units
            row['namedvartype/name'] = varname
            self._set_variant_type(group, row, 'namedvartype/vartype', values, varname)
            row.append()
        group.all.flush()

    def _set_variant_type(self, group, row, field, variant, name=None):
        '''
        See the description of _get_variant_type on NetworkTablesReader class for an explanation
        of variant types.
        '''
        from scipy.sparse import lil_matrix, csr_matrix
        if isinstance(variant,str):
            row[field+'/vartype'] = 'strcode'
            row[field+'/varstring'] = variant
        elif isinstance(variant,ParametersNamed):
            row[field+'/vartype'] = 'named'
            arrayname = 'named_' + name
            row[field+'/varstring'] = arrayname
            row[field+'/units'] = variant.units
            subgroup = self.h5f.createGroup(group, arrayname)
            self._write_parameters_named(subgroup, variant)
        elif isinstance(variant,ParameterArray):
            lil = isinstance(variant.value, lil_matrix)
            if lil or isinstance(variant.value, csr_matrix):
                row[field+'/vartype'] = 'sparse'
                arrayname = 'sparse_' + name
                row[field+'/varstring'] = arrayname
                row[field+'/units'] = variant.units
                try:
                    group._f_getChild(arrayname)
                except:
                    write_sparse(self.h5f, group, arrayname, variant)
            else:
                row[field+'/vartype'] = 'array'
                arrayname = 'ndarray_' + name
                row[field+'/varstring'] = arrayname
                row[field+'/units'] = variant.units
                try:
                    group._f_getChild(arrayname)
                except:
                    self.h5f.createArray(group, arrayname, variant.value)
        elif isinstance(variant,dict):
            row[field+'/vartype'] = 'table'
            row[field+'/varstring'] = name
        else:
            if isinstance(variant,Parameter):
                value = variant.value
                units = variant.units
            else:
                value = variant
                units = '1'
            if isinstance(value,float):
                row[field+'/vartype'] = 'float'
                row[field+'/varflt'] = value
            elif isinstance(value,int):
                row[field+'/vartype'] = 'integer'
                row[field+'/varint'] = value
            else:
                raise Exception('Unhandled type passed as Parameter.')
            row[field+'/units'] = units

    def _write_parameters_named(self, group, variant):
        table = self.h5f.createTable(group, 'aliasedparameters', AliasedParameters)
        row = table.row
        for name, param in variant.iteritems():
            row['name'] = name
            try:
                child = group._f_getChild(name)
                issparse = isinstance(child,tables.Group)
            except:
                issparse = write_named_param(self.h5f, group, name, param.value)
            row['issparse'] = issparse
            row.append()
        table.flush()
        
    def link_parameter_ranges(self, linked_params):
        '''
        linked_params - a list of parameters to be linked to each other. Each
                    element should be a tuple comprised of:
                    type, name, param
                    (type an be 'model' or 'synapse', 'population' or 'connection')
        '''
        r = self._linked.row
        for a,b in zip(linked_params[:-1],linked_params[1:]):
            r['coord_a'] = self.make_column_from_coord(a)
            r['coord_b'] = self.make_column_from_coord(b)
            r.append()
        self._linked.flush()

    def _read_param_ranges(self):
        raise Exception('NetworkTables._read_param_ranges needs to be implemented')
        # Example of what return should look like:
        param_ranges = {('a',):[1,2,3],
                        ('exc','b'):[4.,5.,6.,7.],
                        ('inh','z'):['q','r','s']}
        return param_ranges

    def _read_param_links(self):
        raise Exception('NetworkTables._read_param_links needs to be implemented')
        # Example of what return should look like:
        param_links = [(('exc','a'),('inh','z'))]
        return param_links
        
    def _build_parameter_space(self):
        raise Exception('NetworkTables._build_parameter_space is deprecated')
