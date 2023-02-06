import bk.compute
import numpy as np
import multiprocessing as mp
import itertools as it


def multiprocessing(func, args, workers):
    with mp.Pool(workers) as ex:
        res = ex.starmap(func, args)
    return list(res)


def jittered_ppc(neuron,phases,jitter_max,n_spikes,n_shuffles,n_workers):
    
    shuffles = multiprocessing(bk.compute.jittered_ppc, 
                                    zip(it.repeat(neuron), 
                                        it.repeat(phases), 
                                        it.repeat(jitter_max), 
                                        it.repeat(n_spikes), 
                                        range(n_shuffles)), 
                                        n_workers)
    return np.array(shuffles)