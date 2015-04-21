from functools import wraps

def defaultmonitors(preproc):
    @wraps(preproc)
    def decorated_preproc(params, neuron_groups, monitors):
        '''
        Simply constructs Brian Monitors as specified by the user.
        '''
        from brian.monitor import SpikeMonitor, PopulationRateMonitor, StateMonitor
        from brian.clock import Clock
        from .utils import WeightMonitor

        neuron_groups, usermons = preproc(params, neuron_groups, monitors)

        monitor_objs = {'spikes':{},'poprate':{},
                        'statevar':{}, 'weights':{},
                        'misc':{} # This is for the user to store arbitrary 'monitors'
                        }
        for montypename,mons in usermons.iteritems():
            monitor_objs[montypename].update(mons)

        if 'spikes' in monitors:
            for pop_name in monitors['spikes']:
                monitor_objs['spikes'][pop_name] = SpikeMonitor(neuron_groups[pop_name])

        if 'poprate' in monitors:
            for pop_name, bin_size in monitors['poprate'].iteritems():
                monitor_objs['poprate'][pop_name] = PopulationRateMonitor(neuron_groups[pop_name],
                                                                  bin_size.quantity)
        if 'statevar' in monitors:
            for clock_dt, varname_to_pops in monitors['statevar'].iteritems():
                clk = Clock(dt=clock_dt.quantity)
                for varname, all_pops in varname_to_pops.iteritems():
                    for popname in all_pops:
                        if popname not in monitor_objs['statevar']:
                            monitor_objs['statevar'][popname] = {}
                        monitor_objs['statevar'][popname][varname] = StateMonitor(neuron_groups[popname], 
                                                                          varname, clk, True)
        if 'weights' in monitors:
            #for con_name, (clock_dt, record, bins) in monitors['weights'].iteritems():
            for (clock_dt, record), con_names  in monitors['weights'].iteritems():
                clk = Clock(clock_dt.quantity)
                for con_name, bins in con_names:
                    con = neuron_groups[con_name]
                    monitor_objs['weights'][con_name] = WeightMonitor(con, record, 
                                                                      bins, clock=clk)
    
        return neuron_groups, monitor_objs
    return decorated_preproc

# This decorator is no longer used, since we do not write the results 
# in the worker process anymore.

#def savedefaultmonitors(saverates,savespikes,savestates):
#    def decoratormaker(postproc):
#        @wraps(postproc)
#        def decorated_postproc(params, rawdata, resultsfile=None):
#            from snep.library.tables.results import SubprocessResultsTables
#            local_results = not resultsfile
#            if local_results:
#                resultsfile = SubprocessResultsTables(params['results_file'],
#                                                      params['results_group'])
#                resultsfile.open_file()
#                resultsfile.initialize()
#    
#            if saverates and 'poprate' in rawdata:
#                for pop_name, (times, rates) in rawdata['poprate'].iteritems():
#                    resultsfile.add_population_rates(pop_name, times, rates)
#            if savespikes and 'spikes' in rawdata:
#                for pop_name, spikes in rawdata['spikes'].iteritems():
#                    resultsfile.add_spiketimes(pop_name, spikes)
#            if savestates and 'statevar' in rawdata:
#                for pop_name, all_vars in rawdata['statevar'].iteritems():
#                    for varname, (times, values) in all_vars.iteritems():
#                        resultsfile.add_state_variables(pop_name, varname, times, values)
#        
#            postproc_res = postproc(params, rawdata, resultsfile)
#            if local_results:
#                resultsfile.close_file()
#            return postproc_res
#        return decorated_postproc
#    return decoratormaker
