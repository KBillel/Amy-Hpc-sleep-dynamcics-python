import pandas as pd
import numpy as np

def states_to_longstates(states):
    '''
    This function return the long version of the states variables given
    '''
    long = pd.DataFrame()
    for s,i in states.items():
        i['state'] = s
        long = pd.concat((i,long))
        del i['state']
    order = np.argsort(long.start)
    long = long.iloc[order]
    
    return long

def discard_border(state, t):
    state.start = state.start + (t * 1_000_000)
    state.end = state.end - (t * 1_000_000)