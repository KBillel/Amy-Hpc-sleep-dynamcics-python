import pandas
import numpy

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
