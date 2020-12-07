import numpy as np
import neuroseries as nts

import os
import scipy.stats

def freezing_intervals(speed,threshold, mode='single_speed',clean = False, t_merge = 0.5,t_drop = 1,save = False):
    
    """
        BK 8/11/20
        Input 
            speed: speed vector as output by bk.compute.speed (not yet implemented. But it's an nts.frame)
            treshold: arbritary units
    """
    
    
    if mode.lower() =='single_speed':
        fs =  1/scipy.stats.mode(np.diff(speed.as_units('s').index)).mode[0]
        freezing = speed.values<threshold
        
        if freezing[0] == 1: freezing[0] = 0
        if freezing[-1] == 1: freezing = np.append(freezing,0)

        dfreeze = np.diff(freezing.astype(np.int8))
        start = np.where(dfreeze == 1)[0]/fs + speed.as_units('s').index[0]
        end = np.where(dfreeze == -1)[0]/fs + speed.as_units('s').index[0]
    elif mode.lower() == 'multiple_speed':
        fs =  1/scipy.stats.mode(np.diff(speed.as_units('s').index)).mode[0]
        freezing = np.array((np.sum(speed.as_units('s'),axis = 1))/speed.shape[1] < threshold)
        
        if freezing[0] == 1: freezing[0] = 0
        if freezing[-1] == 1: freezing = np.append(freezing,0)

        dfreeze = np.diff(freezing.astype(np.int8))
        start = np.where(dfreeze == 1)[0]/fs + speed.as_units('s').index[0]
        end = np.where(dfreeze == -1)[0]/fs + speed.as_units('s').index[0]
    elif mode.lower() == 'pca':
        print('not implanted')
    else:
        print('Mode not recognized')
        return False
    freezing_intervals = nts.IntervalSet(start,end,time_units = 's')
    if clean:
        freezing_intervals = freezing_intervals.merge_close_intervals(t_merge,time_units = 's').drop_short_intervals(t_drop,time_units = 's')
        
    
    if save:
        np.save('freezing_intervals',freezing_intervals,allow_pickle = True)
    
    return freezing_intervals

def freezing_video(video_path,output_file,tf,freezing_intervals):
    
    """
        video_path : path to the video to be displaying
        outputfile : path to the video to written
        tf : vector of time containing timing of each frame
        freezing intervals : Intervals when the animal is freezing (as nts.Interval_Set)
    """
    import cv2

    if os.path.exists(output_file):
        print(output_file,'already exist, please delete manually')
        return
    
    tf = nts.Ts(tf,time_units='s')
    freezing_frames = np.where(freezing_intervals.in_interval(tf)>=0)[0]
    fs =  1/scipy.stats.mode(np.diff(tf.as_units('s').index)).mode[0]
    cap  = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    nf = 0
    out = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc('M','J','P','G'), fs, (frame_width,frame_height))
    while True:
        
        ret,frame = cap.read()
        if ret == True:
            if nf in freezing_frames: frame = cv2.circle(frame,(25,25),10,(0,0,255),20)

            cv2.imshow(video_path,frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
            nf +=1
        else:
            break
    cap.release()
    out.release()
    
    return True

def tone_intervals(digital_tone, Fs = 20000, t_merge = 1, t_drop = 1):
    """
        Input : digitalin channel of tone
        Return, nts.IntervalSet for tones, (and Fq of tones)
    """
    edges = TTL_edges(digital_tone,Fs)
    
    tone_intervals = nts.IntervalSet(edges.start,edges.end).merge_close_intervals(t_merge,time_units = 's').drop_short_intervals(t_drop,time_units = 's')
    
    return tone_intervals
    

def TTL_edges(TTL,Fs = 20000):
    if isinstance(TTL[0],(np.bool_,bool)):
        TTL = list(map(int,TTL))
    
    if TTL[0] == 1: TTL[0] = 0
    if TTL[-1] == 1: TTL.append(0)
        
    diff_TTL = np.diff(TTL)
    
    t_start = np.where(diff_TTL == 1)[0]
    t_end = np.where(diff_TTL == -1)[0]
    
    edges = nts.IntervalSet(t_start/Fs,t_end/Fs,time_units = 's')
    return edges
    
def TTL_to_intervals(TTL,Fs = 20000):
    if isinstance(TTL[0],(np.bool_,bool)):
        TTL = list(map(int,TTL))
    
    
    diff_TTL = np.diff(TTL)
    
    t_start = np.where(diff_TTL == 1)[0]
    t_end = np.where(diff_TTL == -1)[0]
    t_TTL = np.array([np.mean(interval) for interval in zip(t_start,t_end)])
    
    
    return (t_start/Fs,t_end/Fs)


def TTL_to_times(TTL,Fs = 20000):
    
    if isinstance(TTL[0],(np.bool_,bool)):
        TTL = list(map(int,TTL))
    
    diff_TTL = np.diff(TTL)
    
    t_start = np.where(diff_TTL == 1)[0]
    t_end = np.where(diff_TTL == -1)[0]
    t_TTL = np.array([np.mean(interval) for interval in zip(t_start,t_end)])
    
    return t_TTL/Fs

def speed(pos,value_gaussian_filter, columns_to_drop=None):
    
    body = []
    for i in pos:
        body.append(i[0])
    body = np.unique(body)
    
    all_speed = np.empty((len(pos)-1,5))
    i = 0
    for b in body:
        x_speed = np.diff(pos.as_units('s')[b]['x'])/np.diff(pos.as_units('s').index)
        y_speed = np.diff(pos.as_units('s')[b]['y'])/np.diff(pos.as_units('s').index)
    
        v = np.sqrt(x_speed**2 + y_speed**2)
        all_speed[:,i] = v
        i +=1
    all_speed = scipy.ndimage.gaussian_filter1d(all_speed,value_gaussian_filter,axis=0)
    all_speed = nts.TsdFrame(t = pos.index.values[:-1],d = all_speed,columns = body)
    if columns_to_drop != None: all_speed = all_speed.drop(columns=columns_to_drop)
    
    return all_speed

def binSpikes(neurons,binSize = 0.025,start = 0,stop = 0):
    if stop == 0:
        stop = np.max([neuron.as_units('s').index[-1] for neuron in neurons])
    bins = np.arange(start,stop,binSize)
    binned = []
    for neuron in neurons:
        hist,b = np.histogram(neuron.as_units('s').index,bins = bins)
        binned.append(hist)
    return np.array(binned),b