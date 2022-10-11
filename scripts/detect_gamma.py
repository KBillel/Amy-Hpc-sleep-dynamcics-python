import sys
sys.path.append('..')

import itertools

import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import savgol_filter
from scipy.stats import zscore
from tqdm import tqdm

import neuroseries as nts


import time
import matplotlib.pyplot as plt

import seaborn as sns 
import bk.load
import bk.compute
import bk.plot
import bk.signal

import os

plt.rcParams['svg.fonttype'] = 'none'
plt.style.use('ggplot')


def detect_gamma(lfp, threshold=2, max_inter=20, min_duration=10, max_duration=500):
    # This function aim at detecting gamma burst. It will return nts.IntervalSet with beg and end of gamma burst.
    lfp_filt = bk.signal.passband(lfp, 60, 80)  # filter the signal
    env = bk.signal.enveloppe(lfp_filt)  # enveloppe of signal
    z_env = bk.compute.nts_zscore(env)  # zscore enveloppe of signal

    thresholded = z_env.values > threshold
    gamma_intervals = bk.compute.toIntervals(lfp_filt.index.values, thresholded)

    # Cleaning intervals according to parameters
    gamma_intervals = gamma_intervals.merge_close_intervals(max_inter,'ms')
    gamma_intervals = gamma_intervals.iloc[gamma_intervals.duration('ms') < max_duration]
    gamma_intervals = gamma_intervals.drop_short_intervals(min_duration, 'ms').reset_index(drop=True)
    
    gamma_peaks = []
    for gamma in tqdm(gamma_intervals.iloc,total = len(gamma_intervals)):        
        peak = lfp_filt.loc[gamma.start:gamma.end].index.values[np.argmax(lfp_filt.loc[gamma.start:gamma.end])]
        gamma_peaks.append(peak)
    gamma_peaks = nts.Ts(np.array(gamma_peaks))

    return gamma_intervals, gamma_peaks

def main(base_folder, local_path, *args, **kwargs):
    bk.load.current_session_linux(base_folder,local_path)
    states = bk.load.states(new_names = True)
    sides = ['left', 'right']

    gamma = {}
    for state in ['NREM', 'REM']:

        lfp = {}
        for side in sides:
            lfp.update({side: bk.load.lfp_in_intervals(
                bk.load.bla_channels()[side], states[state])})
        lfp.update({'Hpc': bk.load.lfp_in_intervals(
            bk.load.ripple_channel(), states[state])})

        gamma[state] = {}
        for chan in lfp:
            if lfp[chan] is None: continue
            intervals, peaks = detect_gamma(lfp[chan], 2, 30, 30, 200)
            gamma[state][chan] = {'intervals': intervals,
                                  'peaks': peaks}

    os.makedirs('Analysis/Gamma', exist_ok=True)
    try:
        os.remove('gamma_intervals_rem.npy')
        os.remove('gamma_intervals_nrem.npy')
    except:
        print('Could not delete old gamma files')
    
    np.save('Analysis/Gamma/gamma_intervals.npy', gamma, allow_pickle=True)

    return gamma


batch = bk.load.batch(main)
# main('/mnt/electrophy/Gabrielle/GG-Dataset-Light/','Rat08/Rat08-20130708')