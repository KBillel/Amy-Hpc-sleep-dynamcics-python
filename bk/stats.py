import numpy as np

import bk.compute


def rayleigh(phases,weights = None):
    r = bk.compute.mean_resultant_length(phases,weights)
    if weights is not None: 
        n = np.sum(weights)
    else:
        n = len(phases)

    
    R = r*n
    z = R**2 / n

    pvalue = np.exp(np.sqrt(1+4*n+4*(n**2 - R**2))-(1+2*n))
    return pvalue