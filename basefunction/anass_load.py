import sys, os, platform
import re
import platform
import numpy as np
import pandas as pd
# import neuroseries as nts
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from scipy.ndimage.measurements import center_of_mass
import pickle
import neuroseries as nts

from pylab import *


import bk.load as bk

def namestr(obj, namespace=globals()):
    return [name for name in namespace if namespace[name] is obj]
def closest(lst, K): 
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))], (min(range(len(lst)), key = lambda i: abs(lst[i]-K)))
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_raw_data_directory(raw_data_directory = "\\\AGNODICE\IcyBox"):
    return raw_data_directory

def get_session_path(session_name):
    rat = session_name[0:5] #"Rat08"

    rat_path = os.path.join(get_raw_data_directory(),rat)
    session_path = os.path.join(rat_path,session_name)
    print(session_path)
    return session_path
def pattern_colored(pattern):
    mean = pattern.mean()
    std = pattern.std()
    pattern_high_weight = np.abs(pattern)>=mean+2*std
    pattern_low_weight = np.abs(pattern)<mean+2*std
    return pattern_high_weight, pattern_low_weight

def get_spikes_in_bins(neurons, state):
    """take nd array and state[wake, rem, nrem]"""
    all_state_spike_times, all_state_spikes_in_bins = [],[]
    binsi = []
    for n in neurons:    
        idx_state, time_bin_state = [],[]
        for i in state:
            idx = np.where(np.logical_and(n>=i[0], n<=i[1]))
            idx_state.append(idx)
            t = np.arange(i[0],i[1],0.025)
            time_bin_state.append(t)
        idx_state = np.hstack(idx_state[:])
        time_bin_state = np.hstack(time_bin_state[:])
        all_state_spike_times.append(n[idx_state])
        spikes_in_bin, bins = np.histogram(n[idx_state], time_bin_state)# t)
        all_state_spikes_in_bins.append(spikes_in_bin)
        binsi.append(bins)
    all_state_spikes_in_bins = np.asarray(all_state_spikes_in_bins)
    binsi = np.asarray(binsi)
    return all_state_spikes_in_bins, binsi



def get_session_num(session_name):
    all_rats_folder = os.path.join(get_raw_data_directory(),'All-Rats')
    final_type = sio.loadmat(all_rats_folder + "/AllRats-FinalType.mat")
    final_type = final_type["finalType"]

    regex_pattern = "^[Rat0-9]+(?:-[0-9]+)?$"
    regex = re.compile(regex_pattern)
    list_rats = ['Rat08','Rat09','Rat10','Rat11']
    files = sorted(os.listdir(get_raw_data_directory()))
    sessions, sessions_num = [], []
    sessions_dict = {}

    for rat in list_rats:
        session_num = 1
        rat_folder = os.path.join(get_raw_data_directory(), rat)
        rat_folder_content = os.listdir(rat_folder)
        for session in sorted(rat_folder_content):

            if regex.search(session):
                sessions_dict[session]= session_num
                session_num += 1

    return sessions_dict[session_name]

def get_session_rat(session_name):
    #Written BK 29/07/2020
    print('ToBeImplemented')



def get_positions(session_name):
    """return a dataframe object """
    session_path = get_session_path(session_name)
    pre_run, run, post_run, _, __, ___ = get_all_states(session_name)

    pos_clean = sio.loadmat(session_path + "/posClean.mat")
    pos_clean = pos_clean['posClean']
    pos_clean = np.transpose(pos_clean)

    pos_pre_run = load_epochs(pos_clean, pre_run[0][0], pre_run[0][1])
    pos_run = load_epochs(pos_clean, run[0][0], run[0][1])
    pos_post_run = load_epochs(pos_clean, post_run[0][0], post_run[0][1])
    
    x_pre, y_pre, t_pre = pos_pre_run[1,:], pos_pre_run[2,:], pos_pre_run[0,:]
    x_run, y_run, t_run = pos_run[1,:], pos_run[2,:], pos_run[0,:]
    x_post, y_post, t_post = pos_post_run[1,:], pos_post_run[2,:], pos_post_run[0,:]
#     args = (x_pre, y_pre, t_pre,x_run, y_run, t_run ,x_post, y_post, t_post)
#     all_positions = np.concatenate(args)
#     all_positions = np.column_stack(args)
#     print(np.shape(all_positions))
    positions_dict = {
        'x_pre' : x_pre, 
        'y_pre' : y_pre,
        't_pre' : t_pre,
        'x_run' : x_run,
        'y_run' : y_run, 
        't_run' : t_run,
        'x_post': x_post, 
        'y_post': y_post, 
        't_post': t_post
    }
    
    positions = pd.DataFrame.from_dict(positions_dict, orient='index')       
    return positions.T

def get_all_states(session_name):
    
    session_path = get_session_path(session_name)
    files = os.listdir(session_path)
    runintervals = sio.loadmat(session_path + "/runintervals.mat")
    states =  sio.loadmat(session_path + "/States.mat")

    runintervals = runintervals['runintervals']
    pre_run = runintervals[0]
    run = runintervals[1]
    post_run = runintervals[2]
    rem = states['Rem']
    wake = states['wake']
    nrem = states['sws']
    return pre_run.reshape((1,2)), run.reshape((1,2)), post_run.reshape((1,2)), rem, nrem, wake

def get_structure_shanks(session_name,structure='BLA'):
    
    """structure should be: 'BLA', 'Hpc', 'GP', 'DEn', 'CPu', 'CeCM'..."""
    structures_folder = os.path.join(get_raw_data_directory(),'All-Rats/Structures')

    session_num = get_session_num(session_name)
    rat_num = int(session_name[3:5])
    structure_ = sio.loadmat(structures_folder + "/"+structure+".mat")
    shanks_structure_all_sessions = structure_[structure]
    log = np.where(np.logical_and(shanks_structure_all_sessions[:,0]==rat_num, shanks_structure_all_sessions[:,1]==session_num))
    shanks_ = shanks_structure_all_sessions[log,2]
    return shanks_

def calculate_speed(session_name, s=200):
    """s=FFalse in no smoothing
    return speed_smooth_pre, speed_smooth_run, speed_smooth_post if smooth true
    return speed_pre, speed_run, speed_post if smooth false
    """
    pre_run, run, post_run, _, __, ___ = get_all_states(session_name)
    positions = get_positions(session_name)
    
    dist = positions.diff().fillna(0.)
    dist['Dist_run'] = np.sqrt(dist.x_run**2 + dist.y_run**2)
    dist['Speed_run'] = dist.Dist_run / (dist.t_run)
    speed_run = dist['Speed_run']
    speed_smooth_run = smooth(speed_run,window_len=s, window='flat')

    dist['Dist_pre'] = np.sqrt(dist.x_pre**2 + dist.y_pre**2)
    dist['Speed_pre'] = dist.Dist_pre / (dist.t_pre)
    speed_pre = dist['Speed_pre']
    speed_smooth_pre = smooth(speed_pre,window_len=s,window='hamming')
    
    dist['Dist_post'] = np.sqrt(dist.x_post**2 + dist.y_post**2)
    dist['Speed_post'] = dist.Dist_post / (dist.t_post)
    speed_post = dist['Speed_post']
    speed_smooth_post = smooth(speed_post,window_len=s,window='hamming')
    
    if s:
        return speed_smooth_pre, speed_smooth_run, speed_smooth_post
    else:
        return speed_pre, speed_run, speed_post

def get_all_directions(session_name):
    """get safe, ap laps"""
    
    lap_type_file = sio.loadmat(get_session_path(session_name) + "\\"+session_name+"-LapType.mat")
    safe_laps = lap_type_file["safelaps"]
    ap_laps = lap_type_file["aplaps"]
    lap_type = {
    'safe_laps_all':safe_laps[0][0]['all'],
    'safe_laps_pre':safe_laps[0][0]['prerun'],
    'safe_laps_run':safe_laps[0][0]['run'],
    'safe_laps_post':safe_laps[0][0]['postrun'],
    
    'ap_laps_all':ap_laps[0][0]['all'],
    'ap_laps_pre':ap_laps[0][0]['prerun'],
    'ap_laps_run':ap_laps[0][0]['run'],
    'ap_laps_post':ap_laps[0][0]['postrun']
    }

    return lap_type


def get_neurons(session_name, stru='Hpc'):
    """for now it's working only for pyr"""
    final_type = sio.loadmat(all_rats_folder + "/AllRats-FinalType.mat")
    final_type = final_type["finalType"]
    rat_num = int(session_name[4])
    session_num = get_session_num(session_name)
    this_session = np.where(np.logical_and(final_type[:,0]==rat_num,final_type[:,1]== session_num))
    this_platform = platform.system()
    if this_platform == 'Linux':
        this_session = np.where(np.logical_and(final_type[:,0]==rat_num,final_type[:,1]== 9))

    this_session = final_type[this_session]

    shanks_stru = get_structure_shanks(session_name,structure=stru)
    shanks_all = [n for n in shanks_stru[0,:]]
    all_stru= []
    for n in shanks_all:
        all_stru.append([np.where([this_session[:,2] == n][0])])

    all_stru_clean = []
    i=0
    while i<len(shanks_all):
        all_stru_clean.append(all_stru[i][0])
        i+=1
    all_stru_clean = np.hstack(all_stru_clean)
    np.shape(all_stru)
    all_pyr = np.where([this_session[:,4]== 1])
    pyr_stru = np.intersect1d(all_stru_clean, all_pyr)
    shanks, spike_times = loadSpikeData1(get_session_path(session_name))
    stru_spike_times_tmp = [spike_times.get(key) for key in pyr_stru]
    stru_spike_times = []
    for a in stru_spike_times_tmp:
        stru_spike_times.append(np.asarray(a.axes[0]))
    return stru_spike_times





def load_epochs(raw_data, start, stop):
    timestep = raw_data[0]
    idx_start, idx_stop = None, None
    for i,k in enumerate(timestep):
        if idx_start ==None:
            if k >= start:
                idx_start = i
        if idx_stop == None:
            if stop <= k:
                idx_stop = i
    epoch = raw_data[:,idx_start:idx_stop]
    return epoch
    


def loadSpikeData1(path, index=None, fs = 20000):

    '''
    adapted from: Guillaume Viejo github
    '''
    if not os.path.exists(path):
        print("The path "+path+" doesn't exist; Exiting ...")
        sys.exit()    
    
    new_path = os.path.join(path, 'Analysis/')
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    
    files = os.listdir(new_path)
    if "spike_times.p" in files:
        neurons = pickle.load(open(new_path+"spike_times.p", "rb"))
        shank = pickle.load(open(new_path+"shanks.p", "rb"))
        
    else: 
        files = os.listdir(path)
        clu_files     = np.sort([f for f in files if '.clu.' in f and f[0] != '.'])
        res_files     = np.sort([f for f in files if '.res.' in f and f[0] != '.'])
        clu1         = np.sort([int(f.split(".")[-1]) for f in clu_files])
        clu2         = np.sort([int(f.split(".")[-1]) for f in res_files])
        if len(clu_files) != len(res_files) or not (clu1 == clu2).any():
            print("Not the same number of clu and res files in "+path+"; Exiting ...")
            sys.exit()

        count = 0    
        neurons = {}
        shank = []

        for i in range(len(clu_files)):
            clu = np.genfromtxt(os.path.join(path,clu_files[i]),dtype=np.int32)[1:]
            if np.max(clu)>1:
                res = np.genfromtxt(os.path.join(path,res_files[i]))
                tmp = np.unique(clu).astype(int)
                idx_clu = tmp[tmp>1]
                idx_neu = np.arange(count, count+len(idx_clu))
                for j, n in zip(idx_clu, idx_neu):
                    neurons[n] = pd.Series(index = np.unique(res[clu==j])/fs, data = np.uint8(n), dtype = np.uint8)
                    shank.append(int(clu_files[i].split(".")[-1]))
                count += len(idx_clu)

        # Saving pickles
        pickle.dump(neurons, open(new_path+"spike_times.p", "wb"))
        pickle.dump(shank, open(new_path+"shanks.p", "wb" ) )

    return shank, neurons

def load_bunch_Of_LFP(path,  start, stop, n_channels=90, channel=64, frequency=1250.0, precision='int16', nts = False):
    """adapted from: Guillaume Viejo github"""
    bytes_size = 2
    start_index = int(start*frequency*n_channels*bytes_size)
    stop_index = int(stop*frequency*n_channels*bytes_size)
    fp = np.memmap(path, np.int16, 'r', start_index, shape = (stop_index - start_index)//bytes_size)
    data = np.array(fp).reshape(len(fp)//n_channels, n_channels)
    timestep = np.arange(0, len(data))/frequency
    return timestep, data[:,channel]





def get_lfp(session_name,state,channel=5,start=False,stop=False):
    session_path = get_session_path(session_name)
    state_str = namestr(state, namespace=globals())

    if start & stop:
        timestep, lfp = load_bunch_Of_LFP(lfp_path,  start, stop, n_channels=166, channel=channel, frequency=1250.0, precision='int16')
    
    else:                        
        new_path = os.path.join(session_path, 'Analysis/')
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        
        files = os.listdir(new_path)
        if "timestep_"+state_str[0]+"_ch_"+str(13)+".p" in files:
            
            timestep = pickle.load(open(new_path+"timestep_"+state_str[0]+"_ch_"+str(channel)+".p", "rb"))
            lfp = pickle.load(open(new_path+"lfp_"+state_str[0]+"_ch_"+str(channel)+".p", "rb"))
        else:
            lfp_path = os.path.join(session_path, session_name+'.lfp')
            timestep, lfp = [],[]
            
            for start, stop in state:
                tp, l = load_bunch_Of_LFP(lfp_path, start, stop, n_channels=166, channel=channel, frequency=1250.0, precision='int16')
                timestep.append(tp)
                lfp.append(l)
            timestep = np.hstack(timestep)
            lfp = np.hstack(lfp)
                    # Saving pickles
            pickle.dump(timestep, open(new_path+"timestep_"+state_str[0]+"_ch_"+str(channel)+".p", "wb"))
            pickle.dump(lfp, open(new_path+"lfp_"+state_str[0]+"_ch_"+str(channel)+".p", "wb" ) )
    return timestep, lfp

def smooth(x,window_len=11,window='hanning'):
    """ from SciPy Cookbook
    smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError#, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError##, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError#, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

structures_folder = os.path.join(get_raw_data_directory(),'All-Rats/Structures')