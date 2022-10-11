import bk.compute

import scipy
import scipy.signal
import neuroseries as nts
import numpy as np
import basefunction.vBaseFunctions3 as vbf

from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic

def highpass(lfp, low, fs=1250, order=4):
    b, a = scipy.signal.butter(order, low, 'highpass', fs=fs)
    filtered = scipy.signal.filtfilt(b, a, lfp.values)
    return nts.Tsd(np.array(lfp.index), filtered)


def passband(lfp, low, high, fs=1250, order=4):
    b, a = scipy.signal.butter(order, [low, high], 'band', fs=fs)
    filtered = scipy.signal.filtfilt(b, a, lfp.values)
    return nts.Tsd(np.array(lfp.index), filtered)


def hilbert(lfp, deg=False):
    """
    lfp : lfp as an nts.Tsd

    return 
    power : nts.Tsd
    phase : nts.Tsd
    """
    xa = scipy.signal.hilbert(lfp)
    power = nts.Tsd(np.array(lfp.index), np.abs(xa)**2)
    phase = nts.Tsd(np.array(lfp.index), np.angle(xa, deg=deg))
    return power, phase

def instantaneous_frequency(lfp,fs = 1250):
    power,phase = hilbert(lfp)
    inst = np.diff(np.unwrap(phase.values)) * (fs/(np.pi*2))
    return(nts.Tsd(lfp.index.values[:-1],inst))

def enveloppe(lfp):
    xa = scipy.signal.hilbert(lfp)
    env = nts.Tsd(np.array(lfp.index), np.abs(xa))
    return env


def wavelet_spectrogram(lfp, fmin, fmax, nfreq):
    t = lfp.as_units('s').index.values

    f_wv = pow(2, np.linspace(np.log2(fmin), np.log2(fmax), nfreq))
    output = vbf.wvSpect(lfp.values, f_wv)  # [0]

    return t, f_wv, output


def wavelet_spectrogram_intervals(lfp, intervals, q=16, fmin=0.5, fmax=100, num=50):
    t = []
    Sxx = []
    for s, e in intervals.iloc:
        inter = nts.IntervalSet(s, e)
        t_, f, Sxx_ = wavelet_spectrogram(
            lfp.restrict(inter), fmin, fmax, num)
        Sxx_, t_ = scipy.signal.resample(Sxx_, int(len(t_)/q), t_, axis=1)
        t.append(t_)
        Sxx.append(Sxx_)

    Sxx = np.hstack(Sxx)
    t = np.hstack(t)

    return t, f, Sxx


def wavelet_bandpower(lfp, low, high, nfreq=10):
    t, f, Sxx = wavelet_spectrogram(lfp, low, high, nfreq)
    power = nts.Tsd(t, np.nanmean(Sxx, 0), time_units='s')
    return power


def power_bouts(lfp, fmin, fmax, treshold, norm=False, fminNorm=0.5, fmaxNorm=4):
    '''
    This function compute interval when a power in the oscillation is greater then a treshold (zscored)
    '''

    power = wavelet_bandpower(lfp, fmin, fmax)
    if norm:
        powerNorm = wavelet_bandpower(lfp, fminNorm, fmaxNorm)
        power = nts.Tsd(power.index.values, power.values/powerNorm.values)

    power = bk.compute.nts_zscore(power)

    bouts = power.values > treshold
    bouts = bk.compute.toIntervals(power.index.values, bouts)
    return bouts


def flatten_spectrum(freqs,spectrum,freq_range):
    fm = FOOOF()
    fm.fit(freqs,spectrum,freq_range)
    init_ap_fit = init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
    init_flat_spec = fm.power_spectrum - init_ap_fit

    return fm.freqs, init_flat_spec


def nts_whiten(signal):
    import scipy.cluster
    return nts.Tsd(signal.index.values,scipy.cluster.vq.whiten(signal))