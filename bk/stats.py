import numpy as np
import bk.compute
import bk.multi
import scipy.stats as ss

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


def poisson(baseRate,counts,time):
    eps = np.spacing(1)
    
    lam = baseRate*time
    
    pInc = 1 - ss.poisson.cdf(counts-1,lam)
    pDec = ss.poisson.cdf(counts,lam)
    surprise = np.log((pDec+eps)/(pInc + eps))
    return pInc,pDec,surprise

def ppc(neuron,phases,jitter_max,n_spikes,n_shuffles,n_workers):
    return None

def shuffles_pvalue(shuffle,value):
    return np.min((len(shuffle[shuffle>value])/len(shuffle),len(shuffle[shuffle<value])/len(shuffle)))


def formatting_pvalues(pvalues):

    one_star = pvalues<0.05
    two_star = pvalues<0.01
    three_star = pvalues<0.001

    no_star = pvalues>=0.05
    pvalues = pvalues.astype('object')

    pvalues[one_star] = '*'
    pvalues[two_star] = '**'
    pvalues[three_star] = '***'
    pvalues[no_star] = 'N.S'

    return pvalues