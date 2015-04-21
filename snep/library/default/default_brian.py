from ...utils import set_brian_var, monitors_to_rawdata
from ...decorators import defaultmonitors

@defaultmonitors
def preproc(params, neuron_groups, monitors):
    '''
    Add additional monitors to the network, network operators or 
    even additional neuron groups and connections.

    neuron_groups - dictionary of Brian NeuronGroup and Synapse objects.
    monitors - dictionary describing which dictionaries are to be built.

    Returns the original neuron_groups (including any new groups added) and
    a dictionary of Brian Monitor objects.
    
    The @defaultmonitors decorator will construct any standard Brian monitors: 
    PopulationRateMonitor, SpikeMonitor, StateMonitor.
    '''
    return neuron_groups, {}

#def run(params, neuron_groups, monitor_objs):
#    '''
#    
#    See run_max or run_phases for a slightly more complicated version.
#    '''
#    return run_phases(params, neuron_groups, monitor_objs,[])

def run(params, neuron_groups, monitor_objs, phases=[]):
    '''
    The most basic run implementation possible: it simply constructors a Network, runs it
    for the requested time and then extracts the raw data from all monitors.

    Also allows the user to call custom functions at different time points
    during the run. This can be used to add or remove network objects,
    or maybe enable/disable plasticity.
    '''
    from ...utils import filter_network_objs_for_run
    from brian.network import Network, clear as brian_clear

    runnable_objs = filter_network_objs_for_run(neuron_groups)
    network_monitors = filter_network_objs_for_run(monitor_objs)
    fornetwork = runnable_objs + network_monitors
    net = Network(*fornetwork)
    net.prepare()
    for t,next_phase in phases:
        net.run(t - net.clock.t)
        next_phase(net, params, neuron_groups, monitor_objs)
        net.prepare()
    net.run(params['runtime'] - net.clock.t)
    rawdata = monitors_to_rawdata(monitor_objs)
    brian_clear(True,all=True)
    return rawdata

def run_max(params, neuron_groups, monitor_objs):
    '''
    A run implementation that regularly checks population firing rates. If the
    firing rate of any population with an attached PopulationRateMonitor
    exceeds 500 Hz, the simulation is stopped early. This function requires at least
    one PopulationRateMonitor in the monitor_objs['poprate'] dictionary or it will fail.
    '''
    from ...utils import filter_network_objs_for_run
    from brian.network import Network, clear as brian_clear
    from brian.stdunits import Hz, ms
    from brian.units import second
    import numpy as np

    runnable_objs = filter_network_objs_for_run(neuron_groups)
    network_monitors = filter_network_objs_for_run(monitor_objs)
    fornetwork = runnable_objs + network_monitors
    net = Network(*fornetwork)

    maxrate_allowed = 500*Hz
    min_run_per_iter = 100*ms

    timestep = params['dt']
    half_timestep = timestep/2.

    popratemons = monitor_objs['poprate'].values()
    # Find the PopulationRateMonitor with the longest update period, since we can't run any
    # less than that per iteration of the simulation.
    pop_rate_step = max([mon._bin for mon in popratemons]) * timestep
    # If all the monitors are updated more often than every 100 ms, then we will run the
    # simulation for at least 100 ms per iteration.
    run_per_iter = max(min_run_per_iter, pop_rate_step)
    # Figure out how many of the last PopulationRateMonitor.rate values we should average
    # over to compute the rate in the last run_per_iter time.
    rate_steps = max(int(run_per_iter / pop_rate_step),1)

    timeremaining = params['runtime']
    maxrate = 0*Hz
    while timeremaining > half_timestep and maxrate < maxrate_allowed:
        run_this_iter = min(timeremaining,run_per_iter)
        net.run(run_this_iter)
        timeremaining -= run_this_iter
        rates = [np.mean(mon.rate[-rate_steps:]) for mon in popratemons]
        maxrate = max(rates) * Hz
#        logger.info('Running- {0:.2f}s remain. {1:.2f} Hz poprate'.format(self.timeremaining,
#                                                                         maxrate))
    if timeremaining > half_timestep:
        warn = 'Simulation stopped early: {0:.2f}s remained, {1:.2f} Hz'.format(timeremaining/second,
                                                                                maxrate/Hz)
        print(warn)

    rawdata = monitors_to_rawdata(monitor_objs)
    brian_clear(True,all=True)
    return rawdata

def postproc(params, rawdata):
    '''
    This is where any computations on the rawdata should be done. Results
    that are to be stored should be returned in a dictionary. Any raw data you
    want saved to the HDF5 file should be returned in the same dictionary.
    
    By default we just return the rawdata, which means all recorded data
    will be saved.
    '''
    return rawdata

def make(params):
    '''
    The default network constructor. Takes a set of dictionaries that specify the network
    and returns a dictionary of Brian NeuronGroups and Synapses.
    '''
    from brian.clock import Clock
    input_models = ['PoissonInput','PoissonGroup']

    neuron_groups = {}
    clock = Clock(params['dt'], makedefaultclock=True)
    
    models = params['models']
    populations = params['populations']
    subpopulations = params['subpopulations']
    synapses = params['synapses']
    all_connections = params['connections']
    solver = params['solver']
    solver = None if solver == 'auto' else solver
    experimental_synapses = params['experimental_synapses']
    
    make_connection = make_connection_experimental if experimental_synapses else make_connection_old

    input_populations = {pname:pop for pname,pop in populations.iteritems() 
                                        if pop['model'] in input_models}
    input_connections = {cname:con for cname,con in all_connections.iteritems()
                                          if con['popname_pre'] in input_populations}
    populations = {pname:populations[pname] 
                        for pname in list(set(populations).difference(input_populations))}
    # Now filter the input_connections and the STDP connections
    connections = {cname:all_connections[cname] 
                        for cname in list(set(all_connections).difference(input_connections))
                            if all_connections[cname]['synapse'] != 'STDP'}

    for popname, pop_info in populations.iteritems():
        neuron_groups[popname] = make_population(popname, pop_info, models, 
                                                 all_connections, synapses, clock, solver,
                                                 subpopulations)
        
    for subpopname, subpop_info in sort_subpopulations(subpopulations, neuron_groups):
        neuron_groups[subpopname] = make_subpopulation(neuron_groups[subpop_info['super']], 
                                                                subpopname, subpop_info) 

    #input_connections.update(connections)
    for ipopname, ipop_info in input_populations.iteritems():
        neuron_groups[ipopname] = make_poisson_input(ipopname, ipop_info, synapses, 
                                                     all_connections, neuron_groups,
                                                     clock,solver)

    for conname, con_info in connections.iteritems():
        neuron_groups[conname] = make_connection(neuron_groups[con_info['popname_pre']], 
                                                 neuron_groups[con_info['popname_post']], 
                                                 con_info,
                                                 synapses[con_info['synapse']], 
                                                 clock, solver)
    
    return neuron_groups

def sort_subpopulations(subpopulations, neuron_groups):
    sorted_subpops = []
    for subpopname, subpop_info in subpopulations.iteritems():
        sps = subpop_info['super']
        if sps in neuron_groups:
            sorted_subpops.insert(0,(subpopname,subpop_info))
        else:
            found = False
            for i, subpop in enumerate(sorted_subpops):
                if sps == subpop[0]:
                    found = True
                    sorted_subpops.insert(i+1,(subpopname,subpop_info))
            if not found:
                sorted_subpops.append((subpopname,subpop_info))
    return sorted_subpops

reserved_params = ['reset','threshold','refractory']

def make_population(popname, pop_info, models, connections, synapses, clock, solver, subpops):
    '''
    Constructs a single population by making a Brian NeuronGroup. No support for adaptive resets
    or other fanciness. Just makes the Equations, adds synaptic dynamics if appropriate
    sets the initial conditions and returns the NeuronGroup.
    '''
    from brian.equations import Equations
    from brian.neurongroup import NeuronGroup
    from brian.stdunits import ms
    compileable = ['Euler', 'exponential_Euler', None]

    model = models[pop_info['model']]
    popparams = pop_info['params']

    refractory  = popparams['refractory'] if 'refractory' in popparams else 0*ms
    reset = make_reset(pop_info, model, refractory)
    threshold   = popparams['threshold'] if 'threshold' in popparams else None
    synaptic_curr_name = model['synaptic_curr_name']
    
    eqs_params = {k:v for k,v in popparams.iteritems() if k not in reserved_params}
    eqs = model['equations']
    eqs = Equations(eqs, **eqs_params)
    
    eqs += synapses_on_model(connections, synapses, popname, synaptic_curr_name, subpops)
    ng = NeuronGroup(pop_info['size'], eqs,
                     threshold = threshold,
                     refractory = refractory,
                     reset = reset,
                     method = solver,
                     clock = clock,
                     freeze = True, #TODO: Consider using freeze
                     compile = solver in compileable, #TODO: Consider using compile
                     )
    for var, svs in pop_info['svs'].iteritems():
        set_brian_var(ng, var, svs, pop_info['params'])

    return ng

def make_poisson_input(ipopname, ipop_info, synapses, input_connections, 
                       neuron_groups, clock, solver):
    from brian.directcontrol import PoissonInput, PoissonGroup

    for ic_name,ic in input_connections.iteritems():
        if ic['popname_pre'] == ipopname:
            break
    if ic['popname_pre'] != ipopname:
        print("Unconnected Poisson input")
        pi = None
    else:
        popparams = ipop_info['params']
        rate = popparams['rate']
        N = ipop_info['size']
    
        syn = synapses[ic['synapse']]
        target_name = ic['popname_post']
        target = neuron_groups[target_name]
    
        if ipop_info['model'] == 'PoissonGroup':
            pi = PoissonGroup(N=N, rates=rate, clock=clock)
            conn = make_connection_old(pi, target, ic, syn, clock, solver)
            neuron_groups[ic_name] = conn
        else:
            state = syn['equations']['eqs_model']
            weight = popparams['weight']
            pi = PoissonInput(target=target, N=N, rate=rate, weight=weight, state=state)
    return pi

def make_poisson_group():
    pass

def make_subpopulation(superpop, subpopname, subpop_info):
    size = subpop_info['size']
    params = subpop_info['params']        
    subpop = superpop.subgroup(size)

    for pname, pvalue in [(pname,pvalue) for pname,pvalue in params.iteritems() 
                                            if pname not in reserved_params]:
        try:
            subpop.__getattr__(pname)
            subpop.__setattr__(pname, pvalue)
        except AttributeError:
            pass

    for var, svs in subpop_info['svs'].iteritems():
        set_brian_var(subpop, var, svs, subpop_info['params'])
        
    return subpop

def make_reset(pop_info, model, refractory):
    from brian.reset import NoReset, StringReset, Refractoriness, SimpleCustomRefractoriness
    popparams = pop_info['params']
    if 'reset_str' in model and model['reset_str'] != '':
        '''
        Some annoying hackishness here. Because Brian doesn't allow passing of a namespace
        into the StringReset or SimpleCustomRefractoriness constructors and Python doesn't
        allow modification of the locals() dictionary in 2.7+ we have to find a way to get
        the model parameters into the local namespace before the call to the constructor.
        This is why we construct a resetlocal dictionary and pass it into eval. Yes it's a
        hack, but unless you can find a better way to do it, that's how it's going to be.
        '''
        resetlocals = {'StringReset':StringReset, 'reset_str':model['reset_str']}
        resetlocals.update(popparams)
        reset = eval("StringReset(reset_str)",globals(),resetlocals)
        if 'refractory' in popparams:
            reset = SimpleCustomRefractoriness(reset, refractory)
    elif 'reset' in popparams:
        reset  = popparams['reset']
        if 'refractory' in popparams:
            reset = Refractoriness(reset, refractory)
    else:
        reset = NoReset()
    return reset
    
def synapses_on_model(connections, synapses, popname, synaptic_curr_name, subpops):
    '''
    Finds any connections for which this population is postsynaptic. If any are found
    then we check the synaptic definition to see if the dynamics can be evaluated on the
    neuron model and if so makes new Equations objects for each synapses.
    '''
    from brian.equations import Equations
    
    allpops = [subpopname for subpopname,info in subpops.iteritems() if info['super'] == popname]
    allpops.append(popname)

    eqs = Equations('')
    synapses = {connection['synapse']: {'eqs':synapses[connection['synapse']]['equations'],
                                        'output_var':synapses[connection['synapse']]['output_var'],
                                        'params':connection['params']}
                        for connection in connections.itervalues()
                            if connection['popname_post'] in allpops
                                and connection['synapse'] != 'STDP'}
    allparams = {}
    for syn_info in synapses.itervalues():
        allparams.update(syn_info['params'])
        if syn_info['eqs']['eqs_neuron'] != '':
            eqs += Equations(syn_info['eqs']['eqs_neuron'], **syn_info['params'])

    if synaptic_curr_name != '':
        split_syn = synaptic_curr_name.split(':')
        Isyn_name = split_syn[0]
        Isyn_units = split_syn[1] if len(split_syn) > 1 else 'amp'
            
        Isyn = Isyn_name + ' = '
        if synapses:
            Isyn += '+'.join(s['output_var'] for s in synapses.itervalues())
        else:
            Isyn += '0*'+Isyn_units
        eqs += Equations(' : '.join((Isyn,Isyn_units)), **allparams)

    return eqs

def make_connection_old(source, target, con_info, synapse, clock, solver):
    from brian.connections import Connection
    from scipy.sparse import issparse#, csr_matrix
    import numpy as np

    eqs = synapse['equations']
    connectivity = con_info['connectivity']
    delay = con_info['delays']

    delay_issparse = issparse(delay)

    if delay_issparse:
        max_delay = delay.data.max() if delay.data.size else 0.
    else:
        max_delay = np.max(delay)

    scale = np.NaN
    if 'scale' in con_info['params']:
        # This is here to that we can vary the connection weights by a constant
        # factor of 'scale' for each point in paramspace.
        scale = con_info['params']['scale']

#    TODO: Consider using connect_with_sparse which yields 3x speedup and half the memory usage.
#     one problem is that the delay matrix must have exactly the same nonzero elements as the
#     weight matrix, otherwise the network won't run.
    weight_issparse = issparse(connectivity)
    if weight_issparse and not (delay_issparse or isinstance(delay,np.ndarray)):
        # We only handle homogeneous delays right now.
#        if not delay_issparse:
#            delay =  csr_matrix(delay)
        if not np.isnan(scale):
            connectivity *= scale
        conn = Connection(source,target,eqs['eqs_model'],delay=delay)
        conn.connect_from_sparse(connectivity, column_access=True)
    elif np.isscalar(connectivity):
        weight = con_info['params']['weight']
        if not np.isnan(scale):
            weight *= scale
        fixed = con_info['params']['fixed'] if 'fixed' in con_info['params'] else False
        conn = Connection(source,target,eqs['eqs_model'], max_delay=max_delay)
        if connectivity >= 1.0:
            conn.connect_full(source, target, weight)
        else:
            conn.connect_random(source, target, connectivity, weight, fixed)
    else:
        if not np.isnan(scale):
            connectivity *= scale
        conn = Connection(source,target,eqs['eqs_model'],
                          weight=connectivity,
                          delay=delay,
                          max_delay=max_delay)
        
    return conn

def make_stdp(stdp_params,conn):
    from brian.equations import Equations
    from brian.stdp import STDP

    tau_stdp = stdp_params['tau_stdp']
    wmax = stdp_params['wmax']
    wmin = stdp_params['wmin']
    eta = stdp_params['eta']
    P = stdp_params['P']
    D = stdp_params['D']
    pre = stdp_params['pre']
    post =stdp_params['post']
    rho0 = stdp_params['rho0']
    alpha = rho0*tau_stdp*2

    eqs = stdp_params['eqs']

    params = {'tau_stdp':tau_stdp,
              'P':P,'D':D,'eta':eta,'alpha':alpha # these not really needed here
              }
    eqs_stdp = Equations(eqs, **params)
    stdp = STDP(conn, eqs=eqs_stdp,
                   pre=pre,post=post,
                   wmin=wmin,wmax=wmax)
    return stdp

def make_connection_experimental(source, target, con_info, synapse, clock, solver):
    '''
    Similar to make_population, this function makes a new Synapses class for each
    defined population-to-population connection, creates the neuron-to-neuron connections
    as defined by the user (connectivity), the delays and finally initialises any state variables
    such as the synaptic weights.
    '''
    from brian.experimental.synapses import Synapses
    from brian.experimental.synapses.synaptic_equations import SynapticEquations
    from scipy.sparse import issparse

    eqs = synapse['equations']
    eqs_model = eqs['eqs_model']
    eqs_pre = eqs['eqs_pre'] if eqs['eqs_pre'] != '' else None
    eqs_post = eqs['eqs_post'] if eqs['eqs_post'] != '' else None
    eqs_neuron = eqs['eqs_neuron']
    
    '''
    If you define a multiline string for the action to take on a presynaptic spike event, then
    we split that string and pass it as a list of separate expressions. This way you may define
    a different delay for each action. See the comment below, near the assignment of delays for
    more information. 
    '''
    if eqs_pre and '\n' in eqs_pre:
        eqs_pre = [eq.strip() for eq in eqs_pre.split('\n')]
        eqs_pre = [eq for eq in eqs_pre if eq!='']

    model = SynapticEquations(eqs_model, **con_info['params'])
    syn = Synapses(source=source,
                   target=target,
                   model=model,
                   pre=eqs_pre,
                   post=eqs_post,
                   clock=clock,
                   method=solver,
                   freeze = True, #TODO: Consider using freeze
                   #compile = True, #Synapses do not support compile yet
                   code_namespace=con_info['params'])
    '''
    Now we have to link any variables that are used in the neuron model, but defined in
    the synaptic model.
    This is necessary for the case where we have the dynamics of a variable (say ge) 
    computed in the Synapses, such as: dge/dt = -ge/tau
    and that variable is used directly in the post-synaptic
    model, ie: dv/dt = (ge*(v-vsyn) - v) / C
    Whereby the post-synaptic ge is actually the sum of all the individual ge 
    variables in the Synapses.
    The linking is done by assigning the Synapse state variable to the NeuronGroup
    state variable, which would normally look like this: 
    neurons.ge = synapses.ge
    Unfortunately this has to be done generically in our code, so it looks like the 
    following loop and assumes that the variables to be linked have exactly the same name. 
    '''
    for lhs in SynapticEquations(eqs_neuron)._string:
        if lhs in model._string:
            target.__setattr__(lhs, syn.__getattr__(lhs))
    '''
    Given synapses we set connectivity and delays between two populations.
    The simple case is that both are determined by an executable expression or float. 
    Otherwise we get arrays which define connections individually. Due to how
    the Synapse class works right now, we must pass the indexed array as a string so
    that the assignment is vectorized.
    '''
    connectivity = con_info['connectivity']
    if issparse(connectivity):
        connectivity = connectivity.toarray()
    syn[:,:] = connectivity if isinstance(connectivity,(str,float)) else 'connectivity[i,j]'

    delay = con_info['delays']
    if issparse(delay):
        delay = delay.toarray()
    
    '''
    If you define more than one action to be taken on a presynaptic spike event, the Synapses 
    class will return a list of SynapticDelayVariable objects. In this case we have 
    to handle each one separately, thus the loop over the list. We assume that if the user
    has passed an array of predefined delays, that the third axis corresponds to the
    delay after which each subsequent action is taken. 
    '''
    syn_delay = syn.delay
    if isinstance(syn_delay,list):
        for _k,sd in enumerate(syn_delay):
            sd[:,:] = delay if isinstance(delay,(str,float)) else 'delay[i,j,_k]'
    else:
        syn_delay[:,:] = delay if isinstance(delay,(str,float)) else 'delay[i,j]'

    for varname, svs in con_info['svs'].iteritems():
        if isinstance(svs,(str,float)):
            set_brian_var(syn, varname, svs)
        else:
            var = syn.__getattr__(varname)
            var[:,:] = 'svs[i,j]'

    return syn

