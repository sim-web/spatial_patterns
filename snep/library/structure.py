from ..utils import ParameterArray

def compute_connectivity_delays(conns, n_pop, distance_measure, cell_positions, num_target_cells_ijs, 
                          connectivity_profile_ijs, propagation_rate_ijs, monosynaptic):
    '''
    Computes connectivity and delay matrices between two populations.

    n_pop - a dictionary of population sizes.
    distance_measure - a function which takes an array of target cell positions, and a position
                        for the source cell and returns an array of distances.
    cell_positions - a dictionary mapping from population names to arrays of cell positions.
    num_target_cells_ijs - a dictionary from (source,target) tuples to the mean number of target
                            cells for every cell in the source population.
    connectivity_profile_ijs - a dictionary mapping from source population name to functions
                             that take an array of distances and return an array of connection 
                             probabilities.
    propagation_rate_ijs - a dictionary from source population name to the axonal propagation velocity
                in metres per second.
    monosynaptic - a dictionary from (source,target) tuples to a boolean indicating if  
                    connections between source and target cells should always be monosynaptic.
    '''
    import numpy as np
    from random import sample

    connections = {}
    delays = {}
    for (source,target) in conns:
        k_ij = num_target_cells_ijs[(source,target)]
        mono = monosynaptic[(source,target)]
        n_i, n_j = n_pop[source], n_pop[target]

        num_connections_needed_all = np.zeros(n_i,dtype=np.int32) # predefine for cast to int
        num_connections_needed_all[:] = np.random.normal(k_ij,k_ij/20.,size=n_i)
        
        connectivity = np.zeros((n_i,n_j),dtype=np.int32)
        propagationtime = np.zeros((n_i,n_j))
        ii = np.arange(n_i)
        for i in ii:
            num_connections_needed = num_connections_needed_all[i]
            num_already_connected = np.sum(connectivity,1)[i]
            while num_already_connected < num_connections_needed:
                if mono:
                    # Only sample from the unconnected targets
                    unconnected = np.where(connectivity[i,:]==0)[0]
                    try_n_connections = unconnected.size
                    target_idx = np.array(sample(unconnected,try_n_connections))
                else:
                    try_n_connections = num_connections_needed*100
                    target_idx = np.random.randint(0,n_j,try_n_connections)
                target_pos = cell_positions[target][target_idx]
                source_pos = cell_positions[source][i]
                distances = distance_measure(target_pos,source_pos)
                
                p_conn = connectivity_profile_ijs[(source,target)]
                P_x = p_conn(distances)
                #plot(target_idx,P_x)
                
                probs = np.random.uniform(size=try_n_connections)
                new_connections = np.squeeze(np.argwhere(P_x+probs >= 1.))
                connected_idx = target_idx[new_connections]
                #plot_hist(connected_idx)
                missing_conns = num_connections_needed - num_already_connected
                connected_idx = connected_idx[:missing_conns]
                
                bins_idx = np.arange(np.max(connected_idx)+1)
                num_connections_per_neuron = np.bincount(connected_idx)
                connectivity[i,bins_idx] += num_connections_per_neuron

                # Autosynapses must be zeroed here because otherwise we'll terminate early
                if source==target:
                    connectivity[ii,ii] = 0
                num_already_connected = np.sum(connectivity,1)[i]

                proprate_m_per_ms = propagation_rate_ijs[(source,target)]
                propdist_m = np.abs(distances[connected_idx])
                proptime_ms = propdist_m / proprate_m_per_ms
                propagationtime[i,connected_idx] = proptime_ms

        connections[(source,target)] = ParameterArray(connectivity)
        delays[(source,target)] = ParameterArray(propagationtime,'ms')
    return delays, connections

def linear_absolute_positions(name_py, name_in, n_pop, intercellular_dist):
    '''
    This will construct a linear array of cell positions with N pyramidal cells for
    every 1 interneuron. All cells are intercellular_dist apart from each other.
    The indices for the two cell types are returned as arrays in the dictionary idx.
    '''
    import numpy as np
    n_py = n_pop[name_py]
    n_in = n_pop[name_in]
    assert(not n_py % n_in)
    idx = {}
    n_pyr_per_subunit = n_py / n_in

    pos = np.arange(1,n_in+n_py+1) * intercellular_dist
    idx[name_in] = np.arange(n_pyr_per_subunit,n_py+n_in,n_pyr_per_subunit+1)
    py_trick = np.ones(n_py+n_in)
    py_trick[idx[name_in]] = 0
    idx[name_py] = np.where(py_trick)
        
    return pos, idx

