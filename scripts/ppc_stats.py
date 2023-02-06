import sys
sys.path.append('..')


import itertools as it
import multiprocessing as mp
from imports import *


def ppc_stats(base_folder, local_path, *args, **kwargs):

    n_spikes = kwargs['n_spikes']
    jitter_max = kwargs['jitter_max']
    n_shuffles = kwargs['n_shuffles']
    n_workers = kwargs['n_workers']

    bk.load.current_session_linux(base_folder, local_path)
    states = bk.load.states(True)
    neurons, metadata = bk.load.spikes()
    lfp = bk.load.lfp_in_intervals(bk.load.ripple_channel(), states['REM'])
    filt_lfp = bk.signal.passband(lfp, 6, 10)
    power, phases = bk.signal.hilbert(filt_lfp)

    ppcs = []
    pvalues = []
    fr_rem = []
    for n in neurons:
        fr_rem.append(bk.compute.fr(n,states['REM']))

        if len(n.restrict(states['REM'])) < 1_500:
            ppcs.append(np.nan)
            pvalues.append(np.nan)
            continue
        n = bk.compute.downsample_spikes(n.restrict(states['REM']), n_spikes)
        ppc = bk.compute.ppc(n, phases)
        ppcs.append(ppc)

        shuffles = bk.multi.jittered_ppc(n, phases, jitter_max, None, n_shuffles, n_workers)
        pvalue = bk.stats.shuffles_pvalue(np.array(shuffles), ppc)
        pvalues.append(pvalue)

    phase_lock = pd.DataFrame({'PPC': ppcs,
                               'pValue': pvalues,
                               'fr_rem':fr_rem})
    phase_lock = pd.concat([metadata, phase_lock], axis=1)

    return phase_lock

if __name__ == '__main__':

    kwargs = {'n_spikes': 1_500,
              'jitter_max': 140,
              'n_shuffles': 1_000,
              'n_workers': 12}

    batch, meta = bk.load.batch(ppc_stats, **kwargs)
    np.save('/mnt/electrophy/Gabrielle/GG-Dataset-Light/All-Rats/ppc_all_1500spikes.npy',(batch,meta))
