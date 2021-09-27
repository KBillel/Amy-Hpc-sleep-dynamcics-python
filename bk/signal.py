import scipy
import scipy.signal
import neuroseries as nts
import numpy as np
import basefunction.vBaseFunctions3 as vbf

def passband(lfp,low,high,fs = 1250,order = 4):
    b,a = scipy.signal.butter(order,[low, high],'band',fs = fs)
    filtered = scipy.signal.filtfilt(b,a,lfp.values)
    return nts.Tsd(np.array(lfp.index),filtered)

def hilbert(lfp,deg = False):
    """
    lfp : lfp as an nts.Tsd
    
    return 
    power : nts.Tsd
    phase : nts.Tsd
    """
    xa = scipy.signal.hilbert(lfp)
    power = nts.Tsd(np.array(lfp.index),np.abs(xa)**2)
    phase = nts.Tsd(np.array(lfp.index),np.angle(xa,deg = deg))
    
    return power,phase

def wavelet_spectrogram(lfp,fmin,fmax,nfreq):
    t = lfp.as_units('s').index.values
    
    f_wv = pow(2,np.linspace(np.log2(fmin),np.log2(fmax),nfreq))
    output = vbf.wvSpect(lfp.values,f_wv)[0]
    
    return t,f_wv,output