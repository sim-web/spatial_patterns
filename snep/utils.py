from snep.library.brianimports import *
import numpy as np
from numpy.random import rand, randn
from scipy.sparse import csr_matrix, lil_matrix, issparse
import copy
''' 
The wild import from snep.library.brian is there to bring in all the brian.units
and brian.stdunits, plus other brian class/function definitions, when available.
The rand, randn imports are necessary for similar reasons.
Those definitions are needed in the global namespace, throughout this file.
In other words: don't modify or remove them.
'''
def p2q(param):
    ''''Parameter-to-Brian.Quantity converter'''
    if issparse(param.value):
        # Unfortunately it seems the Brian Units class doesn't know
        # how to multiply with sparse matrices.
        dense = unit_eval(param.value.todense(),param.units)
        return csr_matrix(dense) if isinstance(param.value,csr_matrix) else lil_matrix(dense)
    else:
        return unit_eval(param.value,param.units)

class Parameter(object):
    '''
    This is how we represent quantities in most of the snep framework. There are limitations with
    Brian Quantity objects which make them hard to work with when trying to load and store them into
    the hdf5 files. Consequently we keep our parameters (and anything with units) in this class
    as long as possible, only converting them to Brian Quantity objects when it makes sense.
    '''
    quantity = property(fget=lambda self: p2q(self))
    def __init__(self, value, units='1', name=None):
        self.value = value
        self.units = units
        self.name = name
        
    def __repr__(self):
        units = '' if self.units == '1' else ','+self.units
        if self.name:
            name = ',name='+self.name
            value = str(type(self.value))
        else:
            name = ''
            value = self.value
        ret = 'Parameter({value}{units}{name})'.format(**locals())
        return ret
    
    def coordvalue(self):
        value = str(self) if self.name else self.value
        return value

    def __str__(self):
        if self.name:
            ret = self.name
        elif isinstance(self.value,str):
            ret = self.value
        else:
            value_str = str(self.value)
            ret = value_str if self.units == '1' else ' '.join((value_str,self.units))
        return ret

    def __cmp__(self, other):
        return not self.__eq__(other)

    def __eq__(self,other):
        return type(self) is type(other) and self.__dict__ == other.__dict__
    
    def __ne__(self,other):
        return not self.__eq__(other)

class ParameterArray(Parameter):
    '''
    Same as Parameter but handles arrays of values and can be iterated on, returning single
    Parameter objects with the appropriate units.
    '''
    def __init__(self, value, units='1', name=None):
        if not (isinstance(value,np.ndarray) or issparse(value) or isinstance(value,str)):
            value = np.array(value)
        Parameter.__init__(self, value, units, name)

    def __iter__(self):
        for p in self.value:
            yield Parameter(p,self.units)

    def __str__(self):
        if self.name:
            valuestr = self.name
        else:
            valuestr = str(self.value.shape)+str(self.value)
            if self.units != '1':
                valuestr = '->'.join((self.units,valuestr))
        return valuestr

    def __eq__(self,other):
        ret = False
        if type(self) is type(other):
            names = self.name==other.name
            units = self.units==other.units
            sparse = issparse(self.value) and issparse(other.value)
            dense = not issparse(self.value) and not issparse(other.value)
            if sparse:
                values = np.abs(self.value-other.value).nnz == 0 
            elif dense:
                if self.value.shape==other.value.shape and sum(self.value.shape) > 1:
                    values = (self.value==other.value).all()
                else:
                    values = self.value==other.value
            else:
                values = False
            
            ret = names and units and values
        return ret

class ParametersNamed(object):
    '''
    Similar to ParameterArray, except that it can contain a list of homogeneous
    objects, each of which has a name. This allows us to have parameter ranges
    that are sets of arrays (which can be a mixture of sparse and dense arrays).
    
    Maybe this should inherit from dict, but it's not clear if that will
    interact with the existing code nicely.
    '''
    def __init__(self, names_values, units='1'):
        self.units = units
        new_list = []
        for k,v in names_values:
            if isinstance(v, ParametersNamed):
                raise Exception('ParametersNamed only stores arrays!')
            elif isinstance(v, (Parameter,ParameterArray)):
                assert(units==v.units)
                v = v.value#self.names_values[k] = v.value
            new_list.append((k,v))
        self.names_values = new_list
        

    def __iter__(self):
        for name,value in self.names_values:
            yield ParameterArray(value,self.units,name)
    
    def iteritems(self):
        for name,value in self.names_values:
            yield name,ParameterArray(value,self.units,name)#yield name, Parameter(value,self.units)

    def __str__(self):
        valuestr = ', '.join(x[0] for x in self.names_values)
        if self.units != '1':
            valuestr = '->'.join((self.units,valuestr))
        return valuestr
    
    def __eq__(self,other):
        return type(self) is type(other) and all(a==b for a,b in zip(self,other))

    def __ne__(self,other):
        return not self.__eq__(other)

Z = lambda f: (lambda x: f(lambda *args: x(x)(*args)))(lambda x: f(lambda *args: x(x)(*args)))
def flatten_dict_of_dicts_the_one_line_version(params):
    ''' Uses a Z-combinator to allow for recursion on an anonymous function'''
    return dict(Z(lambda f: lambda p,x: reduce(lambda l,(k,v): l+f(p+(k,),v), x.iteritems(), [])\
                        if isinstance(x,dict) else [(p,x)])((),params))

def flatten_dict_of_dicts(p):
    '''Turns dict of dictionaries into flat dict mapping path tuples to leaf values'''
    rec = lambda p,x: reduce(lambda l,(k,v): l+rec(p+(k,),v), x.iteritems(), [])\
                    if isinstance(x,dict) else [(p,x)]
    return dict(rec((),p))
    
def flatten(x):
    '''Turns dict of dictionaries into flat list of leaves'''
    return flatten_dict_of_dicts(x).values()

def flatten_params_to_point(params):
    '''Turns nested param dictionary into a flat dictionary mapping paths to values'''
    return flatten_dict_of_dicts(params)

def update_params_at_point(params, paramspace_pt, brian=False):
    '''
    Given params, which is potentially nested dictionaries of parameters,
    and paramspace_pt which is a dictionary of path-tuples to parameters, we
    need to overwrite the appropriate values in the params dictionary.
    params = {'foo':0.0, 'bar':{'baz':'abc'}}
    paramspace_pt = {('foo'):2.0,  ('bar','baz'):'xyz'}  
    '''
    for name, value in paramspace_pt.iteritems():
        p = params
        for n in name[:-1]: p = p[n]
        p[name[-1]] = value.quantity if brian else value
        
class ParameterSpace(object):
    '''
    This class takes a dictionary of parameter ranges which specify a parameter space
    and produces a list of all the points in the space. It also allows two parameters to
    be linked, so that no product is taken over those two parameters (the restrictlist).
    
    It's basically a wrapper around a set of functions written by Konstantin. 
    For a description of any function other than _transitive_set_reduce, talk to him since
    he wrote those. 
    '''
    def __init__(self, thedict, restrictlist):
        self.thedict = thedict
        self.restrictlist = self._transitive_set_reduce(restrictlist)

    def _transitive_set_reduce(self, all_sets):
        '''
        This function sanitises the user defined list of linked
        variables so that _preparedict doesn't fail.
        If any two sets have a common element, then those sets are added due to the transitive
        property of "linked variables". 
        '''
        if not all_sets:
            return []
        head, tail = set(all_sets[0]), [frozenset(t) for t in self._transitive_set_reduce(all_sets[1:])]
        disjoint = [t for t in tail if not head.intersection(t)] # all sets not intersecting with head
        head.update(*set(tail).difference(disjoint)) # add all other sets to head
        return [tuple(s) for s in [head]+disjoint]

    def _product(self, *args, **kwds):
        # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
        # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
        pools = map(tuple, args) 
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)
    
    def _preparedict(self, thedict,restrictlist):
        '''
        Generates a dictionary with combined (keys as tuples, items as lists of lists)
        '''
        otherdict = thedict.copy()
        dummydict = {}
        for elem in restrictlist:
            thelist = []
            for subelem in elem:
                thelist.append(otherdict.pop(subelem))
            dummydict[elem] = self.transposed(thelist)
        otherdict.update(dummydict)
        return otherdict
    
    def transposed(self, lists):
        if not lists: return []
        return map(lambda *row: list(row), *lists)

    def __iter__(self):
        a = self._preparedict(self.thedict,self.restrictlist)
        for valuesvec in self._product(*a.values()):
            b = {keys[0]:keys[1] for keys in zip([thekey for thekey in a.keys()],valuesvec)}
            c = {}
            for kkey in b:
                if isinstance(b[kkey],list):
                    for i,elem in enumerate(b[kkey]):
                        c[kkey[i]] = elem
                else:
                    c[kkey] = b[kkey]
            yield c

def set_brian_var(group, varname, value, localnamespace={}):
    '''
    This is used for two things:
    1) Setting state variables on a NeuronGroup, where if the user specified a string, we
    need to evaluate it with a namespace defined by the neuron parameters before passing the
    resulting array to the group. NeuronGroups don't accept executable string expressions.
    2) Setting state variables on Synapses, which will accept executable string expressions, but
    we need to have the wildcard brian.units and brian.stdunits defined in the global namespace.
    '''
    import re
    if isinstance(value, tuple):
        value, dt = value
        set_group_var_by_array(group, varname, TimedArray(value, dt=dt))
    else:
        if isinstance(group,NeuronGroup) and isinstance(value,str):
            localnamespace['n_neurons'] = len(group)
            value = re.sub(r'\b' + 'rand\(\)', 'rand(n_neurons)', value)
            value = re.sub(r'\b' + 'randn\(\)', 'randn(n_neurons)', value)
            value = eval(value, globals(), localnamespace)
        group.__setattr__(varname, value)

def monitors_to_rawdata(monitors):
    '''
    Extracts the internal data from the Brian Monitors so that we don't have to pass Brian
    objects back out of the run function.
    '''
    rawdata = {'spikes':{},'poprate':{},'statevar':{},'weights':{},'misc':{}}
    for pop_name, mon in monitors['spikes'].iteritems():
        rawdata['spikes'][pop_name] = mon.spiketimes# mon.spikes
    for pop_name, mon in monitors['poprate'].iteritems():
        rawdata['poprate'][pop_name] = (mon.times, mon.rate)
    for pop_name, varmons in monitors['statevar'].iteritems():
        for varname, varmon in varmons.iteritems():
            if pop_name not in rawdata['statevar']:
                rawdata['statevar'][pop_name] = {}
            rawdata['statevar'][pop_name][varname] = (varmon.times, varmon.values)
    for con_name, wmon in monitors['weights'].iteritems():
        w_values = WeightMonitor.convert_values(wmon.values)
        rawdata['weights'][con_name] = (wmon.bins, wmon.times, w_values)
    # In monitors, 'misc' is used to store whatever custom monitor the user needs to
    # get from the preproc stage to the postproc stage.
    for misc_name, mmon in monitors['misc'].iteritems():
        rawdata['misc'][misc_name] = mmon.rawdata() 
    return rawdata

def user_select_experiment_dir(root_dir):
    import os
    from operator import itemgetter
    all_exps = []
    for subdir in os.listdir(root_dir):
        sdp = os.path.join(root_dir,subdir)
        if os.path.isdir(sdp):
            full_file_path = os.path.join(sdp, 'experiment.h5')
            if os.path.exists(full_file_path):
                all_exps.append((full_file_path, subdir))
                
    all_exps = sorted(all_exps, key=itemgetter(1))

    if len(all_exps) > 1:
        print('Select an experiment: ')
        for i, (_ffp, subdir) in enumerate(all_exps):
            print('{0}: {1}'.format(i, subdir))
        inp = -1
        while inp not in range(len(all_exps)):
            try:
                inp = int(raw_input())     
            except ValueError:
                print('Invalid selection')
    else:
        inp = 0
    path, subdir = all_exps[inp]
    return path, subdir

def make_tables_from_path(path):
    '''
    If given path is a file, then that is opened as the ExperimentTables source.
    Otherwise, if it is a directory, then we find all sub-directories that contain
    experiment.h5 files and ask the user to select one.
    '''
    import os
    from snep.tables.experiment import ExperimentTables
    if os.path.isdir(path):
        path, _subdir = user_select_experiment_dir(path)
        print('Opening experiment: ' + path)
    return ExperimentTables(path)

def filter_network_objs_for_run(allnetworkobjects):
    '''
    Given a dictionary of all objects to be passed to the brian.Network
    constructor, we remove anything that is not a NeuronGroup, Connection
    or NetworkOperation. Any NeuronGroup subgroups are also removed since
    their parents are added.
    '''
    canrun = (Connection,NetworkOperation)
    ano = flatten(allnetworkobjects)
    nosubgroups = [obj for obj in ano if obj is not None 
                                          and (isinstance(obj,canrun)
                                               or (isinstance(obj,NeuronGroup)
                                                   and hasattr(obj, '_owner')
                                                   and obj._owner == obj))]
    return nosubgroups

def make_square_figure(nsubplots):
    '''
    Given a desired number of subplots, this returns the number
    of rows and columns that yields the closest thing to a square plot
    '''
    ncols = np.sqrt(nsubplots)
    int_cols = int(ncols)
    if ncols-int_cols < 1e-4:
        nrows = int_cols
    else:
        nrows = int(nsubplots / int_cols)
    ncols = int_cols
    nrows = nrows+1 if ncols * nrows < nsubplots else nrows
    assert(ncols * nrows >= nsubplots)
    return nrows, ncols

def csr_make_ints(indptr,indices):
    # This function ensures the index arrays are integer types
    # because numpy doesn't like it when they're floats.
    indptr = indptr  if indptr.dtype == np.int32 or indptr.dtype == np.int64 \
                        else indptr.astype(np.int32)
    indices= indices if indices.dtype == np.int32 or indices.dtype == np.int64 \
                        else indices.astype(np.int32)
    return indptr, indices

def write_named_param(h5f, aliased_group, name, value):
    sparse = isinstance(value, (lil_matrix,csr_matrix))
    if sparse:
        write_sparse(h5f, aliased_group, name, value)
    else:
        h5f.createArray(aliased_group, name, value)
    return sparse

def write_sparse(h5f, group, arrayname, value):
    lil = isinstance(value, lil_matrix)

    csr = value.tocsr() if lil else value
    csrgroup = h5f.createGroup(group, arrayname)

    indptr, indices = csr_make_ints(csr.indptr, csr.indices)
    h5f.createArray(csrgroup, 'data', csr.data)
    h5f.createArray(csrgroup, 'indptr', indptr)
    h5f.createArray(csrgroup, 'indices', indices)
    h5f.createArray(csrgroup, 'shape', csr.shape)

def read_sparse(group, arrayname):
    csrgroup = group._f_getChild(arrayname)
    data   = csrgroup.data.read()
    indices= csrgroup.indices.read()
    indptr = csrgroup.indptr.read()
    shape  = csrgroup.shape.read()
    # The next few lines make sure the index arrays are integer types
    # because numpy doesn't like it when they're floats.
    indptr = indptr if indptr.dtype == np.int32 \
                            or indptr.dtype == np.int64 \
                        else indptr.astype(np.int32)
    indices= indices if indices.dtype == np.int32 \
                            or indices.dtype == np.int64 \
                        else indices.astype(np.int32)
    csr = csr_matrix((data,indices,indptr),shape=shape)
    return csr

'''
Weight monitor code from Sinyavskiy Oleg, posted on the Brian support group.
https://groups.google.com/d/msg/briansupport/gzkKdciudrE/P6ys-F3sSSEJ
'''
class MatrixMonitorBase(Monitor,NetworkOperation):

    times = property(fget=lambda self:np.array(self._times))
    times_ = times
    values = property(fget=lambda self:self._values)
    values_ = values

    def __init__(self, clock=None, timestep=1, when='end'):
        from brian.clock import guess_clock
        NetworkOperation.__init__(self, None, clock=clock, when=when)
        self.clock = guess_clock(clock)
        self.timestep = timestep
        self.curtimestep = timestep
        self.reinit()

    def __call__(self):
        '''
        This function is called every time step.
        '''
        if self.curtimestep == self.timestep:
            V = self.getValueImpl()
            V = copy.deepcopy(V)
            #print(V)
            self._values.append(V)
            self._times.append(self.clock._t)
        self.curtimestep -= 1
        if self.curtimestep == 0: self.curtimestep = self.timestep

    def reinit(self):
        self._values = []
        self._times = []

    def __getitem__(self, i):
        """Returns the recorded weights values of the state
        """
        return self.values[i]

#    def getvalues(self):
#        newvalues = np.array(self._values)
#        newvalues = newvalues.T #transpose
#        return newvalues

    def getValueImpl(self):
        return NotImplementedError

class WeightMonitor(MatrixMonitorBase):
    bins = property(fget=lambda self:self._bins)
    def __init__(self, C, record=False, bins=None, clock=None, timestep=1, when='end'):
        '''
        C - brian.connection.Connection object
        record - if True the actual values of the weights are kept.
        bins - if not None a histogram is computed on the weights
        '''
        self.C = C
        self._record = record
        self._bins = bins
        super(WeightMonitor,self).__init__(clock, timestep, when)

    def getValueImpl(self):
        W = self.C.W
        weights, hist = None, None
        if self._bins is not None:
            hist, bins = np.histogram(W.alldata,self._bins)
        if self._record:
            weights = csr_matrix((W.alldata,W.allj,W.rowind),W.shape) 
        return (weights,hist)
    
    @staticmethod
    def convert_values(values):
        #from scipy.sparse import vstack
        weights, hist = None, None
        if len(values):
            w, h = values[0]
            if w is not None:
                weights = w.data
                for w, _ in values[1:]:
                    weights = np.vstack((weights,w.data))
                weights = csr_matrix(weights)
                #weights = weights.tocsr()
            if h is not None:
                hist = np.empty((len(values),h.size),dtype=np.int32)
                for i, (_, h) in enumerate(values):
                    hist[i,:] = h
        values = (weights, hist)
        return values

#class WeightMonitor(MatrixMonitorBase):
#    def __init__(self, C, N, clock=None, timestep=1, when='end'):
#        self.C = C
#        self.N = N
#        super(WeightMonitor,self).__init__(clock, timestep, when)
#
#    def getValueImpl(self):
#        return self.C.W.get_col(self.N)

#class STDPMonitor(MatrixMonitorBase):
#    def __init__(self, S, valname, clock=None, timestep=1, when='end'):
#        self.S = S
#        self.valname = valname
#        super(STDPMonitor,self).__init__(clock, timestep, when)
#    def getValueImpl(self):
#        return self.S.__getattr__(self.valname)

class DataMonitor(Monitor):
    def __init__(self,data):
        self._data = data
    def rawdata(self):
        return self._data

class CompareExperiments(object):
    def __init__(self, experiment_A, experiment_B):
        self._a = experiment_A
        self._b = experiment_B
    
    def checkparams(self):
        pa = self._a.get_all_fixed_experiment_params(False)
        pb = self._b.get_all_fixed_experiment_params(False)
        missing_a, missing_b, modified = {},{},{},
        addpath = lambda p,k: p+(k,)
        CompareExperiments._recurse(pa, pb, missing_a, missing_b, modified, (), addpath)
        return {'missing_a':missing_a, 'missing_b':missing_b, 'modified':modified}
    
    def checkranges(self):
        ra = self._a._read_param_ranges()
        rb = self._b._read_param_ranges()
        missing_a, missing_b, modified = {},{},{},
        addpath = lambda p,k: k
        CompareExperiments._recurse(ra, rb, missing_a, missing_b, modified, None, addpath)
        return {'missing_a':missing_a, 'missing_b':missing_b, 'modified':modified}
    
    @staticmethod
    def _recurse(pa, pb, missing_a, missing_b, modified, path, addpath):
        sa,sb = set(pa),set(pb)
        missing_b.update({addpath(path,k):pa[k] for k in sa.difference(sb)})
        missing_a.update({addpath(path,k):pb[k] for k in sb.difference(sa)})
        for k in sa.intersection(sb):
            va,vb = pa[k], pb[k]
            ta,tb = type(va),type(vb)
            if ta is tb and ta is dict:
                CompareExperiments._recurse(va, vb, missing_a, missing_b, modified, addpath(path,k), addpath)
            elif ta is not tb or va != vb:
                modified[addpath(path,k)] = (va, vb)

def compare_experiments(path):
    exps = [make_tables_from_path(path) for _ in range(2)]
    for e in exps: e.open_file(True)
    ce = CompareExperiments(*exps)
    fixed = ce.checkparams()
    ranges = ce.checkranges()
    for e in exps: e.close_file()
    return {'fixed':fixed, 'ranges':ranges}
