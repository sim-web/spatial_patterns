import numpy as np

def compute_state_variable(func, statevars, statevar_names, populations=None):
    allpops = None
    subnetworks = {}
    if not populations:
        populations = {pop:None for pop in statevars.keys()}
    for popname, neuron_idx in populations.iteritems():
        all_vars = statevars[popname]
        statevar_mean = None
        for statevar_name in statevar_names:
            if statevar_name in all_vars:
                t,statevar = all_vars[statevar_name]
                if neuron_idx is not None:
                    statevar = statevar[neuron_idx,:]
                allpops = statevar if allpops is None else np.vstack((allpops,statevar))
                statevar_mean = statevar if statevar_mean is None else np.vstack((statevar_mean,statevar))
        subnetworks[popname] =  func(statevar_mean,axis=0)
    if allpops is not None:
        allpops = func(allpops,axis=0)

    return t, allpops, subnetworks

#def compute_coherence_interval(spiketimes, Tstart, Tend, tau):
#    import time
#    start = time.time()
#    k1 = compute_coherence_interval_old(spiketimes, Tstart, Tend, tau)
#    t_old = time.time()-start
#    #print('old: {0} {1}'.format(res, time.time()-start))
#    start = time.time()
#    k2 = compute_coherence_interval_new(spiketimes, Tstart, Tend, tau)
#    #print('new: {0} {1}'.format(res, time.time()-start))
#    t_new = time.time()-start
#    t_del = (t_new - t_old) / t_old * 100
#    k_del = np.abs(k1-k2)
#    print('t_del {t_del}%, k_del {k_del}'.format(**locals()))
#    return k2

def compute_coherence_interval_old(spiketimes, Tstart, Tend, tau):
    '''
     see Wang96 for the description of the used coherence measure
     this version is an efficient matrix version for efficiency by vectorization
    
     spiketimes is a dictionary which contains the spikes 
         of neuron i in spikes[i] as list of timepoints where neuron 
         i spiked, i.e. spiketimes[i] = [1.0, 1.4] means neuron i spiked
         at t=1000ms and t=1400ms
     Tstart and Tend [ms] specify the time interval where the coherence 
         measure should be computed in
     tau [ms] is the size of the subinterval where the coherence measure 
         is based on (see Wang96)
    '''

    N = len(spiketimes) # number of neurons
    # division of [Tstart,Tend] in K parts of length tau
    K = int(round((Tend - Tstart) / (tau)))
    # matrix to store the vectors [X_i(1),...,X_i(K)], i=1,...,N
    X = np.empty( (K,N) ) 
    #X_column = np.empty(shape=(K,1))
    for i, spikes in spiketimes.iteritems():
        # get i-th neuron's list of spikes
        # We only want spikes inside of Tstart-Tend, so we add an extra bin on the
        # start and end of the interval and then throw those bins away.
        spike_counts = np.histogram(spikes, bins=K, range=(Tstart,Tend))
        X[:,i] = np.minimum(spike_counts[0], 1.)
                
    # we use the numpy matrix instead of array format to have the * operator 
    # acting as matrix multiplication and not component-wise multiplication
    X = np.mat(X) 
    # required helper matrix with ones (to sum up the components of a col of 
    # the matrix X or of one row of X^T)
    one_v = np.ones( (K,1) )
    # column vector full of ones in matrix format
    one_v = np.mat(one_v) 
    z = one_v.transpose()*X
    # the computation is split in numerator and denominator matrices which 
    # is more readable
    kappa_num = X.transpose()*X
    kappa_denum = np.sqrt( z.transpose()*z )
    # replace all zeros in denominator with 1 as we have to divide and if we 
    # replace 0 by 1 then of course the numerator is already 0 but we want it 1
    # so we set also the numerator to 1
    denum_zeros = kappa_denum == 0.0
    #if np.any(denum_zeros):
    kappa_denum[denum_zeros] = 1.
    kappa_num[denum_zeros] = 0.
    
    kappa = 1./(N**2) * np.sum( kappa_num/kappa_denum )
    return kappa

def compute_coherence_interval(spiketimes, Tstart, Tend, tau):
    '''
     see Wang96 for the description of the used coherence measure
     this version is an efficient matrix version for efficiency by vectorization
    
     spiketimes is a dictionary which contains the spikes 
         of neuron i in spikes[i] as list of timepoints where neuron 
         i spiked, i.e. spiketimes[i] = [1.0, 1.4] means neuron i spiked
         at t=1000ms and t=1400ms
     Tstart and Tend [ms] specify the time interval where the coherence 
         measure should be computed in
     tau [ms] is the size of the subinterval where the coherence measure 
         is based on (see Wang96)
    '''
    from scipy import weave
    from scipy.weave.converters import blitz

    # division of [Tstart,Tend] in K parts of length tau
    K_bins = (Tend - Tstart) / (tau)
    K = int(round(K_bins))
    diff = np.abs(K_bins - K)
    assert diff < tau / 10, 'Time range is not divisible by tau' 
    bins = np.arange(K+1)*tau + Tstart
    
    N = len(spiketimes) # number of neurons
    neuron_spikes = spiketimes.values()
    N_spikes = np.array([a.size for a in neuron_spikes])

    # Counter-intuitive result: I tried changing the dtype here to float32
    # assuming it would give me 2x FLOPS. Instead gave me a significant slowdown.
    # See comment here from 'samplebias' for a possible explanation:
    # https://stackoverflow.com/questions/5956783/numpy-float-10x-slower-than-builtin-in-arithmetic-operations
    X = np.zeros((K,N)) 
    code = '''
    for(int i = 0; i < N; ++i)
    {
        py::indexed_ref neuroni_spikes = neuron_spikes[i];
        int N_spikes_i = N_spikes(i); 
        double t_spike = neuroni_spikes[0];
        int j = 0;
        double binL, binR;
        for(int k = 0; k < K; ++k)
        {
            binL = bins(k);
            binR = bins(k+1);
            while(t_spike < binL && j < N_spikes_i-1)
                t_spike = neuroni_spikes[++j];
            if(binL <= t_spike && t_spike < binR)
                X(k,i) = 1;
        }
        if(t_spike == binR)
            X(K-1,i) = 1;
    }
    '''
    ps = ['neuron_spikes', 'N_spikes', 'bins', 'N', 'K', 'X'] 
    weave.inline(code,ps,type_converters=blitz,extra_compile_args=['-O3'])

    z = np.sum(X,axis=0)
    kappa_num = np.dot(X.T,X)

    support_code = '''#include <cmath>'''
    code = '''
    double kappa = 0, zz = 0;

    for(int i = 0; i < N; ++i)
        for(int j = i; j < N; ++j)
            if(0 < (zz = z(i)*z(j)))
                kappa += (i==j?1:2) * kappa_num(i,j) / sqrt(zz);

    kappa /= (N*N);
    return_val = kappa;
    '''
    ps = ['N', 'K', 'z', 'kappa_num']
    kappa = weave.inline(code,ps,support_code=support_code,
                         type_converters=blitz,extra_compile_args=['-O3'])

#    z = z.reshape((N,1))
#    kappa_denum = np.sqrt( np.dot(z,z.T) )
#    denum_zeros = kappa_denum == 0.0
#    kappa_denum[denum_zeros] = 1.
#    kappa_num[denum_zeros] = 0.
#    kappa = 1./(N**2) * np.sum( kappa_num/kappa_denum )

    # This code is kept just cause I wrote it, and it's good evidence that
    # no matter how smart you think you are, you probably can't beat BLAS.
    code = '''
    #define idx(k,n) N*k+n
    double kappa = 0, *z = new double[N](), *X = new double[K*N]();
    
    for(int i = 0; i < N; ++i)
    {
        py::indexed_ref neuroni_spikes = neuron_spikes[i];
        int N_spikes_i = N_spikes(i); 
        double t_spike = neuroni_spikes[0];
        int j = 0;
        double binL, binR;
        for(int k = 0; k < K; ++k)
        {
            binL = bins(k);
            binR = bins(k+1);
            while(t_spike < binL && j < N_spikes_i-1)
                t_spike = neuroni_spikes[++j];
            if(binL <= t_spike && t_spike < binR)
                X[idx(k,i)] = 1;
        }
        if(t_spike == binR)
            X[idx(K-1,i)] = 1;
    }

    for(int n = 0; n < N; ++n)
        for(int k = 0; k < K; ++k)
            z[n] += X[idx(k,n)];

    for(int j = 0; j < N;++j)
    {
        for(int i = j; i < N; ++i)
        {
            double sqrt_zz = sqrt(z[i]*z[j]);
            if(sqrt_zz > 0)
            {
                double dot_ij = 0;
                for(int k = 0; k < K; ++k) dot_ij += X[idx(k,i)] * X[idx(k,j)];
                kappa += (i==j?1:2) * dot_ij / sqrt_zz;
            }
        }
    }
    delete[] z, X;
    kappa /= (N*N);
    return_val = kappa;
    '''

    return kappa

def compute_binned_rms(binsize, times, values):
    '''
    Computes and returns the bin edges and the RMS value of the signal within those bins.
    '''
    timebins = np.arange(0., times[-1]+binsize, binsize)
    sampleperiod = times[1] - times[0]
    samples_per_bin = binsize / sampleperiod
    squared = np.square(values)
    # This is a little trick that sums the values inside of the bins, thanks to the
    # weights parameter of histogram.
    hist, _bin_edges = np.histogram(times, timebins, weights=squared)
    mean_of_bins = hist / np.round(samples_per_bin)
    rms = np.sqrt(mean_of_bins)
    return timebins, rms

def compute_burst_vector(timebins, signal, std_threshold=1.5, abs_threshold=1e-6, min_gap=75e-3):
    signal_std = np.std(signal)
    threshold = max(signal_std*std_threshold, abs_threshold)

    vec = np.zeros(signal.size, dtype=np.int32)
    vec[signal > threshold] = 1

    edgetimes, bursts = squeeze_vector(timebins, vec)
    
    gaps = np.diff(edgetimes)
    shortgaps = np.logical_and(bursts==0, gaps < min_gap)

    leftedges,rightedge = edgetimes[:-1],edgetimes[-1]
    noshortgaps_edges = leftedges[~shortgaps]
    edgetimes = np.empty(noshortgaps_edges.size+1)
    edgetimes[:-1] = noshortgaps_edges
    edgetimes[-1] = rightedge
    bursts = bursts[~shortgaps]

    edgetimes, bursts = squeeze_vector(edgetimes, bursts)

    return edgetimes, bursts, threshold

def squeeze_vector(bins, vec):
    '''
    Util function for compute_burst_vector that removes
    any adjacent 1's or adjacent 0's to form a vector
    that is guaranteed to look like alternating 1s and 0s: 01010101010101
    '''
    was_on = vec[0]
    new_bins = [bins[0]]
    new_vec = [was_on]
    for t,val in zip(bins[1:-1], vec[1:]):
        if was_on != val:
            new_bins.append(t)
            new_vec.append(val)
            was_on = val
    new_bins.append(bins[-1])
    new_bins = np.array(new_bins)
    new_vec = np.array(new_vec)
    return new_bins, new_vec

def compute_duty_cyle(timebins, vector):
    times = np.diff(timebins)
    time_up = times[vector!=0]
    time_down = times[vector==0]
    dutycycle = np.sum(time_up) / np.sum(time_down)
    return dutycycle

def compute_spike_vector_strength(all_spikes, timebins, vector):
    N = len(all_spikes)
    M = vector.size 
    X = np.empty((N,M))
    for i, spikes in all_spikes.iteritems():
        spike_counts, _bins = np.histogram(spikes, bins=timebins)
        X[i,:] = np.minimum(spike_counts, 1.)
    vectorstrengths  = np.dot(X,vector)
    vectorstrengths /= np.sum(vector)
    return vectorstrengths

def compute_kl_divergence(histA,histB):
    A = histA / np.sum(histA,dtype=np.float)
    B = histB / np.sum(histB,dtype=np.float)
    B[B==0.0] = 1e-12
    B = B[A.nonzero()]
    A = A[A.nonzero()]
#    A = A[B.nonzero()]
#    B = B[B.nonzero()]
    return np.sum(A * np.log(A / B))

def cross_corr_coh(spiketimes, Tstart, Tend, tau=2e-3, width=20e-3):
    from brian.stdunits import ms
    '''
    Version from Josef Ladenbauer, with comment:
    Considering the coherence calculation, at the end one shouldn't
    include the autocorrelation values in the matrix diagonal when
    normalizing - at least not for zero lag. 
    =======
    Population averaged pairwise cross correlation of spiketrains 
    (1-0 spike counts over time), see Wang 1996 for details 
    
    Input
    
    spiketimes: dictionary obtained from SpikeMonitor.spiketimes which contains
                the spike time point array of neuron i in spiketimes[i]
    Tstart, Tend: [* ms] specifies the time interval over which the correlation is 
                  measured. Duration of a spike train is (Tstart+width, Tend-width)
    tau: bin size for the 1-0 counts (coincidence window for spikes of different neurons) 
    
    width: maximal time lags (-width, +width), lags in units of tau, width should be
           an integer multiple of tau
           
    Output
    
    kappas[0]: time lags (0 time lag as measure of synchrony)
    kappas[1]: corresponding kappa values (population averages)             
    '''
    
    N = len(spiketimes) # number of neurons
    # division of [Tstart,Tend] in K parts of length tau
    K = int(round((Tend - Tstart) / tau))
    # matrix to store the vectors [X_i(1),...,X_i(K)], i=1,...,N
    X = np.empty( (N,K) ) 
    #X_column = np.empty(shape=(K,1))
    for i, spikes in spiketimes.iteritems():
        # get i-th neuron's list of spikes
        spike_counts = np.histogram(spikes/ms, bins=K, range=(Tstart/ms,Tend/ms))
        X[i,:] = np.minimum(spike_counts[0], 1.)

    # we use the numpy matrix instead of array format to have the * operator 
    # acting as matrix multiplication and not component-wise multiplication
    X = np.mat(X)
    ones_vec = np.ones( (K,1) )  # helper matrix with ones
    ones_vec = np.mat( ones_vec )
    
    k = int(width/tau)  # number of lags
    total_counts_nolag = X[:,k:-k-1] * ones_vec[k:-k-1]
    
    kappas = np.empty( (2,2*k+1) )
    
    for i in range(k+1):
        # negative lags
        total_counts_lag = X[:,k-i:-k-i-1] * ones_vec[k-i:-k-i-1]          
        kappa_mat_num = X[:,k:-k-1] * X[:,k-i:-k-i-1].transpose()
        kappa_mat_denom = np.sqrt(total_counts_nolag * total_counts_lag.transpose())
        
        # replace all zeros in denominator with 1 to avoid division by 0
        denom_zeros = kappa_mat_denom == 0.0
        kappa_mat_denom[denom_zeros] = 1.
        # now, average over upper (or lower) diagonal part of kappa_mat and 
        # exclude all zeros to obtain the population average kappa value
        kappa = np.sum( np.triu(kappa_mat_num/kappa_mat_denom) )
        if kappa > 0:
            kappa = np.float(kappa) / np.sum( np.triu(kappa_mat_num/kappa_mat_denom) > 0.0 )
                            
        kappas[0,k-i] = i * tau
        kappas[1,k-i] = kappa
        
        # positive lags
        if i > 0:
            total_counts_lag = X[:,k+i:-k+i-1] * ones_vec[k+i:-k+i-1]          
            kappa_mat_num = X[:,k:-k-1] * X[:,k+i:-k+i-1].transpose()
            kappa_mat_denom = np.sqrt(total_counts_nolag * total_counts_lag.transpose())
                        
            # replace all zeros in denominator with 1 to avoid division by 0
            denom_zeros = kappa_mat_denom == 0.0
            kappa_mat_denom[denom_zeros] = 1.
            # now, average over upper (or lower) diagonal part of kappa_mat and 
            # exclude all zeros to obtain the population average kappa value
            kappa = 1. / np.sum( np.triu(kappa_mat_num/kappa_mat_denom) > 0.0 ) * \
                    np.sum( np.triu(kappa_mat_num/kappa_mat_denom) )            
            
            kappas[0,k+i] = -i * tau
            kappas[1,k+i] = kappa
           
    return kappas

def CV_ISI(spiketimes, minspikes=5):
    '''
    Coefficient of variation of ISIs
    '''
    if len(spiketimes) < minspikes:  # need at least minspikes spikes to calculate the CV
        return np.NaN
    ISI = np.diff(spiketimes)  # interspike intervals
    return np.std(ISI) / np.mean(ISI)    

def CV_ISI_population(spiketimes, timeL=0., timeR=np.infty):
    from scipy.stats import nanmean
    return nanmean([CV_ISI(spks[np.logical_and(timeL <= spks,spks < timeR)])
                    for spks in spiketimes.itervalues()])

def compute_time_within_bounds(times, values, start, end, upper=np.inf, lower=-np.inf):
    values = values[np.logical_and(start <= times, times < end)]
    within = values[np.logical_and(lower < values, values < upper)]
    return len(within) / float(len(values))

def compute_transition_rates(filter_width, bin_size, 
                             start, stop, initstates, times, values, boundary):
    from brian.units import second
    results = {}
    width_dt = int(filter_width / bin_size) # width in number of bins
    window = np.exp(-np.arange(-2 * width_dt, 2 * width_dt + 1) ** 2 * 1. / (2 * (width_dt) ** 2))
    values = np.convolve(values, window * 1. / np.sum(window), mode='same')
    filter_width /= second #strip units off
    for s,l,r in zip(initstates, start, stop):
        l += filter_width * 1.5
        r -= filter_width * 1.5
        clipped_index = np.logical_and(l <= times, times < r)
        clipped_values = values[clipped_index]
        clipped_times = times[clipped_index]
        over = clipped_values > boundary
        edges, states = squeeze_vector(clipped_times, over)
        if len(edges) > 2:
            # Drop the last because it corresponds to artificial end of phase
            time_in_state = np.diff(edges[:-1])
            assert(np.all(time_in_state > 0))
            valid_states = states[:-1]
            if s and not valid_states.any():
                time_on = np.array([0.0]) # Never spent time in intended state
            else:
                time_on  = time_in_state[ valid_states]
            if not s and valid_states.all():
                time_off = np.array([0.0]) # Never spent time in intended state
            else:
                time_off = time_in_state[~valid_states]
        else:
            # Spent entire time in a single state.
            # If the one state does not match the intended state, record a 0
            # for the intended state
            empty = [0.0] if s != states[0] else []
            time_on  = np.array([r-l] if states[0] else empty)
            time_off = np.array(empty if states[0] else [r-l])
        results[(s,l,r)] = (time_on, time_off, edges, states)
    return results

def compute_value_mean_std(filter_width, bin_size, edges, state, times, values):
    from brian.units import second
    width_dt = int(filter_width / bin_size) # width in number of bins
    window = np.exp(-np.arange(-2 * width_dt, 2 * width_dt + 1) ** 2 * 1. / (2 * (width_dt) ** 2))
    values = np.convolve(values, window * 1. / np.sum(window), mode='same')
    filter_width /= second #strip units off
    std, mean = {True:[],False:[]}, {True:[],False:[]}
    for s,l,r in zip(state, edges[:-1], edges[1:]):
        l += filter_width*2
        r -= filter_width*2
        if r-l > filter_width:
            clipped_values = values[np.logical_and(l <= times, times < r)]
            std[s].append(np.std(clipped_values))
            mean[s].append(np.mean(clipped_values))
    return mean, std

def compute_binned_cv_isi(edges, state, spikes):
    '''
    Computes the CV_ISI for each bin defined by edges. Each bin (between two
    edges) has a state, which can be True or False.
    All the CV values are returned in a dictionary with the state as the key 
    mapping to a list of CV values for that state.
    '''
    from scipy.stats import nanmean
    cv = {True:[],False:[]}
    for s,l,r in zip(state, edges[:-1], edges[1:]):
        all_cv_isi = np.array([CV_ISI(st[np.logical_and(l < st, st < r)]) 
                               for st in spikes.itervalues()])
        if all_cv_isi.size > 0:
            mean_cv = nanmean(all_cv_isi)
            cv[s].append(mean_cv)
    return cv

def compute_firing_rates(spikes,timebins):
    #rates = np.empty(shape=len(spikes))
    bins = np.array([0.,timebins]) if np.isscalar(timebins) else timebins
    diffs = np.diff(bins)
    rates = np.array([np.histogram(st, bins)[0]/diffs for st in spikes.itervalues()])
    return np.squeeze(rates)

def compute_hist_weights(connection,bins,col_lower=0,col_upper=None):
    '''
    connection - brian.connections.Connection object
    bins - defines the bins used in the histogram, see numpy for allowed values
    col_lower, col_upper - bounds for the post-synaptic neuron index
    '''
    W = connection.W
    weights = np.empty(0)
    if col_upper is None:
        col_upper = W.shape[1]
    for i in xrange(W.shape[0]):
        row_i = W[i,:]
        col_filter = np.logical_and(col_lower <= row_i.ind, row_i.ind < col_upper)
        row_i_filtered = row_i[col_filter]
        weights = np.hstack((weights,row_i_filtered))
    hist, bin_edges = np.histogram(weights,bins)
    #moments = compute_weight_moments(weights)
    return hist, bin_edges, weights#, moments

def compute_weight_moments(W):
    from scipy.stats.mstats import moment
    moments = [np.mean(W)]+[moment(W, moment=i) for i in [2,3,4]]
    return moments
    
