import numpy as np
import neuroseries as nts

import scipy.stats


def freezing(speed,treshold):
    """
        BK 8/11/20
        Input 
            speed: speed vector as output by bk.compute.speed (not yet implemented. But it's an nts.frame)
            treshold: arbritary units
    """
    
    fs =  1/scipy.stats.mode(np.diff(pos.index)).mode[0]
    
    freezing = speed<treshold
    if freezing[0] == 1: freezing[0] = 0
    if freezing[-1] == 1: freezing = np.append(freezing,0)
        
    dfreeze = np.diff(freezing.astype(np.int8))
    
    start = np.where(dfreeze == 1)[0]/fs_video + pos.index[0]
    end = np.where(dfreeze == -1)[0]/fs_video + pos.index[0]
    
    freezing_intervals = nts.IntervalSet(start,end,time_units = 's')

    return freezing_intervals