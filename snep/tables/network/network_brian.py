from network_base import *

class NetworkTablesReader(NetworkTablesReaderBase):
    __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to read a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the network and simulation that is part of the experiment.
    '''
    
    '''
    These internal properties are defined so that we can restructure the hdf5 file without
    modifying the code that accesses it anywhere but here.
    '''
    _parameters = property(fget = lambda self: self.network.parameters)
    _parameter_ranges = property(fget = lambda self: self.network.parameter_ranges)
    _models = property(fget = lambda self: self.network.models)
    _populations = property(fget = lambda self: self.network.populations)
    _synapses = property(fget = lambda self: self.network.synapses)
    _connections = property(fget = lambda self: self.network.connections)

    def __init__(self):
        self._all_param_ranges = None
        self._all_param_links = None

    def as_dictionary(self, paramspace_pt, brian=True):
        '''
        This is used by the snep.experiment framework to convert the network definition as
        it is stored in the hdf5 file into a dictionary of Python objects and, if brian=True
        also Brian.Quantity objects where appropriate. This dictionary is then passed to
        the subprocess to construct and simulate the network.
        
        IMPORTANT: the paramspace_pt value overrides any values also specified on the populations
        and connections. In other words: the network can be blindly constructed in the
        subprocess without any knowledge of the paramspace_pt. 
        '''
        params = self.get_all_fixed_experiment_params(brian, paramspace_pt)
        # we only will write psp items that have a path length of 1, because
        # otherwise we should assume that they're actually defined elsewhere
        # in the network.
        tmp_psp = {k:v for k,v in paramspace_pt.iteritems() if len(k) == 1}
        update_params_at_point(params,tmp_psp,brian)
        return params

    def get_all_fixed_experiment_params(self,brian,paramspace_pt=None):
        if not paramspace_pt:
            paramspace_pt = {}
        params = dict(
                    models = self.get_models_as_dictionary(),
                    synapses = self.get_synapses_as_dictionary(),
                    populations = self.get_populations(paramspace_pt, brian),
                    connections = self.get_connections(paramspace_pt, brian),
                    subpopulations = self.get_subpopulations(paramspace_pt, brian),
                    )

        generalparams = self.get_general_params(brian)
        params.update(generalparams)
        return params

    def get_models_as_dictionary(self):
        model_equations = self.get_model_equations()
        reset_strings = self.get_model_reset_strings()
        synaptic_curr_names = self.get_model_synaptic_curr_names()
        models = {modelname:{'equations':eqs,
                             'reset_str':reset_strings[modelname],
                             'synaptic_curr_name':synaptic_curr_names[modelname]}
                    for modelname, eqs in model_equations.iteritems()}
        return models
    
    def get_synapses_as_dictionary(self):
        synapse_equations = self.get_synapse_equations()
        synapse_output_vars = self.get_synapse_output_vars()
        synapses = {synname:{'equations':eqs,
                             'output_var':synapse_output_vars[synname]}
                        for synname, eqs in synapse_equations.iteritems()}
        return synapses

    def populations_with_model(self, modelname):
        ''' All populations that specify modelname as their model'''
        return [('population',p['name']) for p in self._populations.all
                                if p['model'] == modelname]

    def subpopulations_with_super(self, superpopname):
        ''' All subpopulations that specify superpopname as their super population'''
        return [('subpopulation',sp['name']) for sp in self._populations.subpops
                                if sp['super'] == superpopname]
        
    def pops_or_subpops_with_model(self, modelname):
        import operator
        pops = self.populations_with_model(modelname)
        spop = [self.subpopulations_with_super(pname) for (_,pname) in pops]
        pops = reduce(operator.add, spop, pops)
        return pops 

    def connections_with_synapse(self, synapsename):
        ''' All connections that specify synapsename as their synapse'''
        return [('connection',c['name']) for c in self._connections.all 
                                            if c['synapse'] == synapsename]

    def _read_all_parameters(self, child_name, types, child_group, parent_group, 
                             paramspace_pt, brian):
        '''
        This builds a dictionary that maps parameter names to parameter values for a given
        population or connection as specified by child_name.
        
        Parent here refers to either a model or synapse.
        Child here refers to a population or connection.
        
        Any parameter can be specified on parent, child or as part of the parameter space point, 
        including any combination thereof. The rule of precendence is that paramspace_pt 
        supercedes child and child supercedes parent.
        '''
        params = {}
        parent_type,child_type = types
        parentname = [p[parent_type] for p in child_group.all if p['name'] == child_name][0]

        try:
            # This try/except is here to handle PoissonInput & PoissonGroup
            # populations which do not have a parent in the model tables.
            parent_instance_group = parent_group._f_getChild(parentname)
            parent_table = parent_instance_group.parameters
            parent_params = self._read_named_variant_table(parent_instance_group,parent_table,brian)
        except tables.exceptions.NoSuchNodeError:
            parent_params = {}
        
        child_instance_group = child_group._f_getChild(child_name)
        child_table = child_instance_group.parameters
        child_params = self._read_named_variant_table(child_instance_group,child_table,brian)

        pspace_params = {pspkey[2]:value.quantity if brian else value
                            for pspkey,value in paramspace_pt.iteritems() 
                                        if len(pspkey)>2 and pspkey[0] == child_type
                                                         and pspkey[1] == child_name}

        params.update(parent_params)
        params.update(child_params)
        params.update(pspace_params)
        return params

    def population_parameters(self, popname, pspace_pt, brian):
        ''' Returns a dictionary of population parameters, see _read_all_parameters for more '''
        parent_group = self._models
        child_group = self._populations
        return self._read_all_parameters(popname, ('model','population'), 
                                         child_group, parent_group, pspace_pt, brian)

    def subpopulation_parameters(self, subpopname, pspace_pt, brian):
        ''' Separate function than for populations because we don't need the
        fancy inheritance rules (model->population->paramspacept) that true
        populations have. '''
        params = {}
        subpop_instance_group = self._populations._f_getChild(subpopname)
        params_table = subpop_instance_group.parameters
        subpop_params = self._read_named_variant_table(subpop_instance_group,params_table,brian)
        pspace_params = {pspkey[2]:value.quantity if brian else value 
                            for pspkey,value in pspace_pt.iteritems() 
                                        if len(pspkey)>2 and pspkey[0] == 'subpopulation'
                                                         and pspkey[1] == subpopname}
        params.update(subpop_params)
        params.update(pspace_params)
        return params

    def connection_parameters(self, conname, pspace_pt, brian):
        ''' Returns a dictionary of connection parameters, see _read_all_parameters for more '''
        parent_group = self._synapses
        child_group = self._connections
        return self._read_all_parameters(conname, ('synapse','connection'), 
                                         child_group, parent_group, pspace_pt, brian)

    def get_populations(self, pspace_pt, brian):
        '''
        Returns a dictionary that maps from population names to a dictionary that specifies
        everything about that population, including model, size, parameters and initial conditions.
        '''
        return {pop['name']:{'model':pop['model'], 
                              'size':pop['size'],
                              'params':self.population_parameters(pop['name'], pspace_pt, brian),
                              'svs':self.population_state_variable_setters(pop['name'], brian)}
                    for pop in self._populations.all}
    
    def get_subpopulations(self, pspace_pt, brian):
        '''
        Returns a dictionary that maps from subpopulation names to a dictionary that specifies
        everything about that population, including super and size.
        '''
        return {subpop['name']:{'super':subpop['super'], 
                              'size':subpop['size'],
                              'params':self.subpopulation_parameters(subpop['name'], pspace_pt, brian),
                              'svs':self.population_state_variable_setters(subpop['name'], brian)}
                    for subpop in self._populations.subpops}
    
    def _read_state_variable_setters(self, svsgroup, brian):
        all_svs = {}
        for sv in svsgroup.all:
            sv_vals = self._get_variant_type(svsgroup, sv, 'namedvartype/vartype', brian)
            dt = Parameter(sv['dt/value'],sv['dt/units'])
            if dt.units != '1':
                # If the clock step dt has units, this is a timed array, so we make
                # a tuple of the values and the clock step so that a TimedArray object
                # can be constructed in the subprocess (TimedArray cannot be properly pickled
                # so we can't construct here).
                sv_vals = (sv_vals, dt.quantity if brian else dt)
            all_svs[sv['namedvartype/name']] = sv_vals
        return all_svs

    def population_state_variable_setters(self, popname, brian):
        ''' Returns the initial conditions for a population '''
        svsgroup = self._populations._f_getChild(popname).state_variable_setters
        return self._read_state_variable_setters(svsgroup, brian)
    
    def connection_state_variable_setters(self, conname, brian):
        ''' Returns the initial conditions for a connection '''
        svsgroup = self._connections._f_getChild(conname).state_variable_setters
        return self._read_state_variable_setters(svsgroup, brian)

    def _get_connection(self, row, pspace_pt, brian):
        conn = {}
        conngroup = self._connections._f_getChild(row['name'])
        conn['synapse'] = row['synapse'] 
        conn['popname_pre'] = row['popname_pre']
        conn['popname_post'] = row['popname_post']
        conn['params'] = self.connection_parameters(row['name'], pspace_pt, brian)
        conn['svs'] = self.connection_state_variable_setters(row['name'], brian)

        conn['connectivity'] = self._get_variant_type(conngroup, row,'connectivity', brian)
        conn['delays'] = self._get_variant_type(conngroup, row, 'delays', brian)

        return conn

    def get_connections(self, pspace_pt, brian):
        '''
        Returns a dictionary that maps from connection names to a dictionary that specifies
        everything about that connection, including synapse, source population, target population,
        connectivity, delays, parameters and initial conditions.
        '''
        return {conn['name']:self._get_connection(conn, pspace_pt, brian)
                    for conn in self._connections.all}

    def get_model_reset_strings(self):
        ''' Returns the reset string if it is defined, otherwise an empty string '''
        reset_strs = {}
        for modelgroup in self._models._f_iterNodes('Group'):
            #modelgroup = self._models._f_getChild(modelname)
            try:
                reset_str = modelgroup._f_getChild('reset').read()
            except tables.exceptions.NoSuchNodeError:
                reset_str = ''
            reset_strs[modelgroup._v_name] = reset_str
        return reset_strs

    def get_model_equations(self):
        ''' Returns all model equations '''
        return {mg._v_name: mg.equations.read() 
                    for mg in self._models._f_iterNodes('Group')}
    
    def get_model_synaptic_curr_names(self):
        ''' Returns all model model synaptic current names '''
        return {mg._v_name: mg.synaptic_curr_name.read() 
                    for mg in self._models._f_iterNodes('Group')}

    def get_synapse_equations(self):
        ''' Returns all synapse equations, including pre, post, model and postsynaptic neuron'''
        return {sg._v_name: {   'eqs_model':sg.eqs_model.read(),
                                'eqs_pre':  sg.eqs_pre.read(),  
                                'eqs_post': sg.eqs_post.read(), 
                                'eqs_neuron': sg.eqs_neuron.read(),
                             }
                    for sg in self._synapses._f_iterNodes('Group')}

    def get_synapse_output_vars(self):
        return {sg._v_name:sg.output_var.read()
                    for sg in self._synapses._f_iterNodes('Group')}

    def _get_ranges(self, group, tablename=None):
        '''
        Returns a dictionary of parameter range tables where keys 
        are model/population/synapse/connection names. 
        '''
        ranges = {}
        if tablename:
            # Handles populations and connections
            table = group._f_getChild(tablename)
            for r in table:
                ranges[r['name']] = group._f_getChild(r['name']).parameter_ranges
        else:
            # Handles models and synapses
            for n in group._f_iterNodes():
                ranges[n._v_name] = n.parameter_ranges
        return ranges

    def _build_param_range_dict(self, ranges, descendents):
        '''
        Builds a dictionary of parameter ranges where the keys are tuples comprised
        of a population or connection name, plus a parameter name. For example,
        keys might be ('exc','C') or ('inh_exc', 'tau_i'). Any parameter range that was
        defined on a model or synapse is inherited by the derived populations and connections.
        '''
        all_params = {}
        linked = []
        for name, rangegroup in ranges.iteritems():
            params = {}
            for r in rangegroup.all:
                param_name = r['name']
                # If this is for synapses or models, descendants are connections or populations
                # respectively. This means they inherent the parameter range from this parent.
                descendent_names = descendents(self,name)
                pdns = tuple(dn+(param_name,) for dn in descendent_names)
                for pdn in pdns:
                    params[pdn] = self._get_variant_type(rangegroup, r, 
                                                         'vartype', False)
                # If there is more than one descendant of a model or synapse, then
                # the variable inherited on each of those descendants is linked.
                if len(descendent_names) > 1:
                    linked.append(pdns)
            all_params.update(params)

        return all_params, linked

    def _read_param_ranges(self):
        if self._all_param_ranges is None:
            all_params, all_linked = self._build_parameter_space()
            self._all_param_ranges = all_params
            self._all_param_links = all_linked
        return self._all_param_ranges

    def _read_param_links(self):
        ''' these functions are ugly ugly hacks due to old design of
        SNEP and need to be gutted and redesigned'''
        new_ud_links = self._user_defined_links()
        for udl in new_ud_links:
            if udl not in self._all_param_links:
                self._all_param_links.append(udl)
        return self._all_param_links

    def _read_general_param_ranges(self):
        '''
        This is called from network_brian.NetworkTablesReader._build_parameter_space
        
        Returns a dictionary mapping from parameter identifiers (a tuple containing
        the parameter owner and parameter name) to a ParameterArray of values.
        '''
        group = self._parameter_ranges
        table = self._parameter_ranges.all
        param_ranges = self._read_named_variant_table(group,table,False)
        param_ranges = flatten_params_to_point(param_ranges)
        return param_ranges
        
    def _build_parameter_space(self):
        '''
        This is called from paramspace.ParameterSpaceTables.build_parameter_space
        
        IMPORTANT: The models and synaptic parameter range dictionaries are specified in terms of
        the populations and connections that inherit them. This is done so that any parameter ranges
        defined explicitly on the populations and connections automatically supercede (overwrite) 
        the model and synapse ranges when adding them to the all_params dictionary.
        '''
        modlranges = self._get_ranges(self._models)
        popsranges = self._get_ranges(self._populations, 'all')
        spopranges = self._get_ranges(self._populations, 'subpops')
        synsranges = self._get_ranges(self._synapses)
        consranges = self._get_ranges(self._connections, 'all')
        pop_and_subpops = lambda self, popname: [('population',popname)] + NetworkTables.subpopulations_with_super(self,popname)

        model_params, ml = self._build_param_range_dict(modlranges, NetworkTables.pops_or_subpops_with_model)
        pops_params,  pl = self._build_param_range_dict(popsranges, pop_and_subpops)
        spop_params, spl = self._build_param_range_dict(spopranges,lambda self, name: [('subpopulation',name)])
        syn_params,   sl = self._build_param_range_dict(synsranges, NetworkTables.connections_with_synapse)
        conn_params,  cl = self._build_param_range_dict(consranges,lambda self, name: [('connection',name)])

        general_params = self._read_general_param_ranges()

        all_params = {}
        # The order in which the all_params dictionary is updated is ESSENTIAL. Population
        # ranges *must* overwrite model ranges and connection ranges *must* overwrite synaptic ranges.
        for p in [general_params, model_params, pops_params, spop_params, syn_params, conn_params]:
            all_params.update(p)

        all_linked = ml + pl + spl + sl + cl
        #all_linked += self._user_defined_links()

        return all_params, all_linked

    def _user_defined_links(self):
        '''
        Should use the linked variables returned by the base class to find
        inherited links and add those. 
        '''
        baselinks = NetworkTablesReaderBase._user_defined_links(self)
        all_links = []
        for coord_a, coord_b in baselinks:
            links = []
            for ab in [coord_a, coord_b]:
                if len(ab) == 3:
                    type_ab, name, param = ab
                    if type_ab == 'model':
                        names = self.pops_or_subpops_with_model(name)
                    elif type_ab == 'population':
                        names = [(type_ab, name)] + self.subpopulations_with_super(name)
                    elif type_ab == 'synapse':
                        names = self.connections_with_synapse(name)
                    else:
                        names = [(type_ab, name)]
                    links += [name+(param,) for name in names]
                else:
                    links += [ab]
            all_links.append(tuple(links))
        return all_links


class NetworkTables(NetworkTablesReader,NetworkTablesBase):
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
        NetworkTablesBase.initialize(self, h5f, parentgroup)

        try:
            self._models
        except tables.exceptions.NoSuchNodeError:
            self.h5f.createGroup(self.network,'models')
        try:
            self._synapses
        except tables.exceptions.NoSuchNodeError:
            self.h5f.createGroup(self.network,'synapses')
        try:
            populations = self._populations
        except tables.exceptions.NoSuchNodeError:
            populations = self.h5f.createGroup(self.network, 'populations')
            self.h5f.createTable(populations, 'all', Population)
            self.h5f.createTable(populations, 'subpops', Subpopulation)
        try:
            connections = self._connections
        except tables.exceptions.NoSuchNodeError:
            connections = self.h5f.createGroup(self.network, 'connections')
            self.h5f.createTable(connections, 'all', Connection, "All Connections")
        
    def set_simulation(self, solver, dt, runtime, experimental_synapses):
        ''' 
        Set the simulation parameters such as the numeric integration method (solver), step
        size (dt) and total simulated time (runtime)
        '''
        params = {'solver':solver,
                   'dt':dt,
                   'runtime':runtime,
                   'experimental_synapses':experimental_synapses}
        self.add_parameters(params)
    
    def add_model(self, name, eqs, default_params, reset_str=None, synaptic_curr_name=''):
        '''
        name - unique name for the model
        eqs - set of equations that define the model
        default_params - a dictionary which contains default parameter values for any population 
                        that uses this model
        synaptic_curr_name - the variable which represents the sum of all synaptic currents
        '''
        group = self.h5f.createGroup(self._models, name)
        self.h5f.createArray(group, 'equations', eqs, name+' equations')
        self.h5f.createArray(group, 'synaptic_curr_name', synaptic_curr_name,
                             'variable name for synaptic currents')
        prgroup = self.h5f.createGroup(group, 'parameter_ranges')
        self.h5f.createTable(prgroup, 'all', NamedVariantType)
        paramtable = self.h5f.createTable(group, 'parameters', NamedVariantType, name+" default params")
        self._write_named_variant_table(group, paramtable, default_params)
        if reset_str:
            self.h5f.createArray(group, 'reset', reset_str, name+' reset')
    
    def add_synapse(self, name, eqs, default_params, output_var):
        '''
        name - unique name for the synapse
        eqs - a dictionary containing all the equations that define the synapse. Possible
                keys in the dictionary are:
                eqs_model/neuron - the dynamics of this synapse. If the dynamics are linear
                                    then use eqs_neuron, otherwise use eqs_model
                eqs_pre/post - the expression that should be executed when the pre/post-synaptic
                                neuron fires.
                For more details see the example code.
        default_params - a dictionary which contains default parameter values for any connection 
                        that uses this synapse
        output_var - variable that should be added to the model's synaptic current equation
        '''
        syngroup = self.h5f.createGroup(self._synapses, name)
        prgroup = self.h5f.createGroup(syngroup, 'parameter_ranges')
        self.h5f.createTable(prgroup, 'all', NamedVariantType)

        self.h5f.createArray(syngroup, 'eqs_neuron', eqs['eqs_neuron']  if 'eqs_neuron' in eqs else '')
        self.h5f.createArray(syngroup, 'eqs_model',  eqs['eqs_model']   if 'eqs_model' in eqs else '')
        self.h5f.createArray(syngroup, 'eqs_pre',    eqs['eqs_pre']     if 'eqs_pre' in eqs else '')
        self.h5f.createArray(syngroup, 'eqs_post',   eqs['eqs_post']    if 'eqs_post' in eqs else '')

        self.h5f.createArray(syngroup, 'output_var', output_var)
        
        table = self.h5f.createTable(syngroup, 'parameters', NamedVariantType, name+" default params")
        self._write_named_variant_table(syngroup, table, default_params)

    def add_connection(self, name, popname_pre, popname_post, synapse, 
                       connectivity, delays, non_default_params={}):

        conngroup = self.h5f.createGroup(self._connections, name)
        svsgroup = self.h5f.createGroup(conngroup, 'state_variable_setters')
        self.h5f.createTable(svsgroup, 'all', TimedVariantType)
        prgroup = self.h5f.createGroup(conngroup, 'parameter_ranges')
        self.h5f.createTable(prgroup, 'all', NamedVariantType)

        conn = self._connections.all.row
        conn['name'] = name
        conn['popname_pre'] = popname_pre
        conn['popname_post'] = popname_post
        conn['synapse'] = synapse

        self._set_variant_type(conngroup, conn, 'connectivity', connectivity, 'connectivity')

        self._set_variant_type(conngroup, conn, 'delays', delays, 'delays')
        conn.append()
        self._connections.all.flush()

        ptable = self.h5f.createTable(conngroup, 'parameters', NamedVariantType, 'Connection params')
        self._write_named_variant_table(conngroup, ptable, non_default_params)

    def _add_pop_groups(self, name, non_default_params):
        popgroup = self.h5f.createGroup(self._populations, name)
        svsgroup = self.h5f.createGroup(popgroup, 'state_variable_setters')
        self.h5f.createTable(svsgroup, 'all', TimedVariantType)
        prgroup = self.h5f.createGroup(popgroup, 'parameter_ranges')
        self.h5f.createTable(prgroup, 'all', NamedVariantType)

        ptable = self.h5f.createTable(popgroup, 'parameters', NamedVariantType, 'Population params')
        self._write_named_variant_table(popgroup, ptable, non_default_params)
        
    def add_population(self, name, model, size, non_default_params={}):
        pop = self._populations.all.row
        pop['name'] = name
        pop['model'] = model
        pop['size'] = size
        pop.append()
        self._populations.all.flush()
        self._add_pop_groups(name, non_default_params)

    def add_subpopulation(self, name, size, superpop, non_default_params={}):
        subpop = self._populations.subpops.row
        subpop['super'] = superpop
        subpop['name'] = name
        subpop['size'] = size
        subpop.append()
        self._populations.subpops.flush()
        self._add_pop_groups(name, non_default_params)

    def add_population_state_variable_setters(self, popname, svs):
        svsgroup = self._populations._f_getChild(popname).state_variable_setters
        self._write_timed_variant_table(svsgroup, svs)

    def add_connection_state_variable_setters(self, conname, svs):
        svsgroup = self._connections._f_getChild(conname).state_variable_setters
        self._write_timed_variant_table(svsgroup, svs)

    def add_model_parameter_range(self, modelname, paramranges):
        parameter_ranges = self._models._f_getChild(modelname).parameter_ranges
        self._write_named_variant_table(parameter_ranges, parameter_ranges.all, paramranges)

    def add_population_parameter_range(self, popname, paramranges):
        parameter_ranges = self._populations._f_getChild(popname).parameter_ranges
        self._write_named_variant_table(parameter_ranges, parameter_ranges.all, paramranges)

    def add_synapse_parameter_range(self, synname, paramranges):
        parameter_ranges = self._synapses._f_getChild(synname).parameter_ranges
        self._write_named_variant_table(parameter_ranges, parameter_ranges.all, paramranges)

    def add_connection_parameter_range(self, conname, paramranges):
        parameter_ranges = self._connections._f_getChild(conname).parameter_ranges
        self._write_named_variant_table(parameter_ranges, parameter_ranges.all, paramranges)

    def _reset_parameter_range_groups(self, parentgroup):
        for childgroup in parentgroup._f_iterNodes('Group'):
            self.h5f.removeNode(childgroup,'parameter_ranges',recursive=True)
            prgroup = self.h5f.createGroup(childgroup, 'parameter_ranges')
            self.h5f.createTable(prgroup, 'all', NamedVariantType)
        self.h5f.flush()
        
    def delete_all_parameter_ranges(self):
        self._reset_parameter_range_groups(self._models)
        self._reset_parameter_range_groups(self._populations)
        self._reset_parameter_range_groups(self._synapses)
        self._reset_parameter_range_groups(self._connections)
        self.h5f.removeNode(self.network, 'linked_parameter_ranges')
        self.h5f.createTable(self.network, 'linked_parameter_ranges', 
                     LinkedRanges, "Linked parameter ranges")
        self.h5f.flush()

    def link_parameter_ranges(self, linked_params):
        # when we add linked params here, we need to define the coordinates
        # for those params (see paramspace.define_coordinates), because
        # when the paramspace is built, only the coordinates for the
        # derived types will appear, while the manual link may be on the
        # base type.
        NetworkTablesBase.link_parameter_ranges(self, linked_params)
        self.define_coordinates(linked_params)
        
