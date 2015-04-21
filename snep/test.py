def main():
    '''
    For now this file can be used to test more advanced features or anything not appropriate 
    for the basic example project in my_experiment.py
    '''
    from snep.experiment import Experiment
    from snep.utils import Parameter, ParameterArray

    exp = Experiment('./my_experiment')
    et = exp.tables
    et.set_simulation('exponential_Euler',Parameter(0.5,'ms'),Parameter(1.,'second'))

    lif_params = dict(
        reset       =Parameter(-60.,'mV'),
        threshold   =Parameter(-30.,'mV'),
        tau_m       =Parameter(100.,'ms'),
        El          =Parameter(-50.,'mV'),
        )
    et.add_model('lif', ''' dv/dt = (ge+gi-(v-El))/tau_m : volt ''', lif_params)

    exc_size = 3200
    inh_size = 800
    et.add_population('exc', 'lif', exc_size, {'El':Parameter(-49.,'mV')})
    et.add_population('inh', 'lif', inh_size)
    ics = 'reset + rand({0}) * (threshold - reset)'
    et.add_population_initial_conditions('exc', {'v':ics.format(exc_size)})
    et.add_population_initial_conditions('inh', {'v':ics.format(inh_size)})

    ampa_params = dict(tau_e=Parameter(5.,'ms'))
    ampa_eqs = dict(eqs_pre='ge+=w', 
                    eqs_neuron='''dge/dt = -ge/tau_e : volt''',
                    eqs_model='w : volt')
    et.add_synapse('ampa', ampa_eqs, ampa_params)

    gaba_params = dict(tau_i=Parameter(10.,'ms'))
    gaba_eqs = dict(eqs_pre='gi+=w', 
                    eqs_neuron='''dgi/dt = -gi/tau_i : volt''',
                    eqs_model='w : volt')
    et.add_synapse('gaba', gaba_eqs, gaba_params)

    delays = '10*rand()*ms'
    connectivity = 0.1 # can also be 'i!=j' or array of bools

    et.add_connection('exc_exc', 'exc', 'exc', 'ampa', connectivity, delays)
    et.add_connection('exc_inh', 'exc', 'inh', 'ampa', connectivity, delays)
    et.add_connection('inh_exc', 'inh', 'exc', 'gaba', connectivity, delays)
    et.add_connection('inh_inh', 'inh', 'inh', 'gaba', connectivity, delays)

    et.add_connection_initial_conditions('exc_exc',{'w':'5 * rand() * mV'})
    et.add_connection_initial_conditions('exc_inh',{'w':'5 * rand() * mV'})
    et.add_connection_initial_conditions('inh_exc',{'w':'-5 * rand() * mV'})
    et.add_connection_initial_conditions('inh_inh',{'w':'-5 * rand() * mV'})

    et.add_model_parameter_range('lif', {'C':ParameterArray([0.10, 0.15, 0.20],'nF')})
    et.add_population_parameter_range('exc', {'El':ParameterArray([-55.,-45.],'mV')})
    et.add_synapse_parameter_range('ampa', {'d':ParameterArray([1.,2.,3.],'ms')})
    et.add_connection_parameter_range('exc_exc', {'f':ParameterArray([2.,3.],'nS')})

    et.link_parameter_ranges('model', 'lif', 'C', 'synapse', 'ampa', 'd')
    et.link_parameter_ranges('population', 'exc', 'El', 'connection', 'exc_exc', 'f')

    monitors = {'spikes':{'exc', 'inh'},
                'poprate':{'exc':Parameter(100.,'ms')},
                'statevar':{'exc':{'v':Parameter(1., 'ms')
                                   }}}
    exp.add_monitors(monitors)

    exp.process()

if __name__ == '__main__':
    main()
