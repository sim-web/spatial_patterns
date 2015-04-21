try:
    from snep.configuration import config
    if config['network_type'] != 'brian':
        import brian_no_units

    from brian.units import *
    from brian.stdunits import *
    from brian.network import NetworkOperation
    from brian.monitor import Monitor
    from brian.neurongroup import NeuronGroup
    from brian.connections import Connection
    from brian.timedarray import set_group_var_by_array, TimedArray

    def unit_eval(value,units):
        return value if isinstance(value, str) else value * eval(units)

except ImportError:
    import warnings
    class NetworkOperation: pass
    class Monitor: pass
    class NeuronGroup: pass
    class Connection: pass
    class TimedArray: pass
    def set_group_var_by_array(group, var, arr, times=None, clock=None, start=None, dt=None): pass

    def unit_eval(value,units):
        return value
    warnings.warn('Brian was not found, not all units will be applied correctly.')
