import os
import sys
import subprocess

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import time


import xml.etree.ElementTree as ET
import cv2

from tqdm import tqdm


def load_digitalin(path,nchannels=16,Fs = 20000):
    import pandas as pd
    
    digital_word = np.fromfile(path,'uint16')
    sample = len(digital_word)
    time = np.arange(0,sample)
    time = time/Fs

    
    for i in range(nchannels):
        if i == 0: data = (digital_word & 2**i)>0
        else: data = np.vstack((data,(digital_word & 2**i)>0))

    return data

def CountTTL(TTL):
    #Return the number TTL and the index where the last one starts
    
    TTL = list(map(int,TTL))
    diff_TTL = np.diff(TTL)
    
    t_start = np.where(diff_TTL == 1)

    return(len(t_start[0]),t_start[0][-1])

def CountFrames(path):
    #Return the number of frames inside a video
    cap = cv2.VideoCapture(path)
    return(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

def TTLtoTimes(TTL,Fs = 20000):
    
    if isinstance(TTL[0],(np.bool_,bool)):
        TTL = list(map(int,TTL))
    
    diff_TTL = np.diff(TTL)

    t_start = np.where(diff_TTL == 1)[0]
    t_end = np.where(diff_TTL == -1)[0]
    t_TTL = np.array([np.mean(interval) for interval in zip(t_start,t_end)])
    
    return t_TTL/Fs

def xml(session):
    tree = ET.parse(session+'.xml')
    root = tree.getroot()
    
    xmlInfo = {}
    for elem in root:
        for subelem in elem:
            try: 
                xmlInfo.update({subelem.tag:int(subelem.text)})
            except:
                pass
    return xmlInfo

def concat_lfp_dat(session,path_dat):
    
    print('Starting dat concatenation')
    
    to_cat = " + ".join(path_dat)
    p = subprocess.Popen('copy /B ' + to_cat + " " + session+'.dat&',shell = True)
    
    #Check if concatenated file is not corrupted : 
    originalFileSize = []
    for p in path_dat:
        originalFileSize.append(os.path.getsize(p))
    finalSize = np.sum(originalFileSize)
    f = False
    while not f:
        f = os.path.exists(session+'.dat')
        
    pbar = tqdm(total = finalSize,desc = 'Concat : ')
    concatSize = 0
    while concatSize < finalSize:
        time.sleep(10)
        s = os.path.getsize(session+'.dat')
        
        if concatSize == s:
            print('Concatenation in stuck try again')
            p.kill()
            break
        
        pbar.update(s-concatSize)
        concatSize = s
 
    originalFileSize = []
    for p in path_dat:
        originalFileSize.append(os.path.getsize(p))
    finalSize = np.sum(originalFileSize)
    concatSize = os.path.getsize(session+'.dat')
    
    
    if finalSize == concatSize: 
        print('Concat Succefull')
        print('Creating cat event file')
        
        t_rec = []
        for p in path_dat:
            t = os.path.getsize(p)/(fs*nChannels*bytePerSample)
            print(t)
            t_rec.append(t)
        concat_event = np.cumsum(t_rec)*1000 # 1000 is for conversion in ms
        
        with open(session+'.cat.evt','w') as f:
            for c in concat_event:
                f.write(str(c)+ ' cat\n')
        return 1
    else:
        print('Error, please try again. Check freespace on your drives. You need at least',np.sum(originalFileSize)/1e9,'GB')
        return 0
    
    
def concat_digitalin(session,path_digitalin,path_videos,nchannels = 16):
    
    byteSize = 2
    print('Starting digitalin concatenation')
    
    originalFileSize = []
    for p in path_digitalin:
        originalFileSize.append(os.path.getsize(p))
    finalSize = np.sum(originalFileSize)
    
    
    data = np.empty((16,0),dtype = np.bool)
    
    pbar = tqdm(total = int(finalSize/byteSize))
    # Reading all digitalin files in sub-session and comparate number of TTL with number of frames : 
    for p_digitalin,p_video in zip(path_digitalin,path_videos):
        data_ = load_digitalin(p_digitalin,nchannels)
        nTTL,lTTL = CountTTL(data_[0,:])
        nFrames = CountFrames(p_video)

        if nTTL == nFrames: # Same number of frames and TTL is easy case, we don't do anaything
            print('NTTL',nTTL)
            print('Number of Frames',nFrames)
            print('Do not need to correct last TTL')
        elif nTTL-1 == nFrames: # If NTTL and Frames are only one appart we need to delete the last TTL
            print('NTTL',nTTL)
            print('Number of Frames',nFrames)
            print('Last TTL was at ',lTTL)
            print('Correcting TTLs :')
            data_[0,lTTL:] = False
            print('Counting new number of TTL : ' , CountTTL(data_[0,:])[0])
        else:
            print('Some file might be corrupted, need to implement this')

        # Anyway we concatenate digitalin in the RAM
        print('Concatenating digitalin in RAM')
        data = np.hstack((data,data_))
        time.sleep(0.1)
        pbar.update(np.size(data_))
        time.sleep(0.1)


    del data_ #delete temp var for better managing of RAM

    #Recreate the write format of digital in (16channels inside a 16bits uint16)
    digitalwrite = np.empty(data.shape[1],dtype = np.uint16)
    for i in range(nchannels):
        digitalwrite += np.uint16((data[i,:] * 2**i))

    print('Writting digitalin.dat')
    f = open('digitalin.dat','wb')
    f.write(digitalwrite)
    f.close()
    
    
    #Check if concatenated file is not corrupted : 

    concatSize = os.path.getsize('digitalin.dat')
    
    if finalSize == concatSize: 
        print('Concat Succefull')
        
        
        timestamps = TTLtoTimes(data[0,:])
        np.save(session+'-frametime',timestamps)
        del timestamps
        del data
        return 1   
    else:
        print('Error, please try again')
        return 0
    

def downsampleDatFileBK(path, n_channels, fs):
    
    #Code by Guillaume Viejo 
    #Adapted by Billel Khouader oct 2020
    
    """
    downsample .dat file to .lfp 1/16 (20000 -> 1250 Hz)
    
    Since .dat file can be very big, the strategy is to load one channel at the time,
    downsample it, and free the memory.
    
    BK : Modified in order to load dat chunk by chunk in write it as LFP file without 
    loading nChannels time the original dat file
    
    Args:
        path: string
        n_channel: int
        fs: int
    Return: 
        none
    """    
    if not os.path.exists(path):
        print("The path "+path+" doesn't exist; Exiting ...")
        sys.exit()
    listdir     = os.listdir(path)
    datfile     = [f for f in listdir if (f.startswith('Rat') & f.endswith('.dat'))]
    if not len(datfile):
        print("Folder contains no xml files; Exiting ...")
        sys.exit()
    new_path = os.path.join(path, datfile[0])

    f             = open(new_path, 'rb')
    startoffile   = f.seek(0, 0)
    endoffile     = f.seek(0, 2)
    bytes_size    = 2
    n_samples     = int((endoffile-startoffile)/n_channels/bytes_size)
    duration      = n_samples/fs
    f.close()

    chunksize = 100_000
    
    lfp = np.zeros((int(n_samples/16),n_channels),dtype = np.int16)
    
    # Loading
    
    count = 0
    f_lfp = open(session+'.lfp', 'wb')
    
    pbar = tqdm(total = n_samples,desc = 'Extracting LFP file from DAT file (sample)')
    while count < n_samples:
#         print(count)
        f             = open(new_path, 'rb')
        seekstart     = count*n_channels*bytes_size
        f.seek(seekstart)
        block         = np.fromfile(f, np.int16, n_channels*np.minimum(chunksize, n_samples-count))
        f.close()
        block         = block.reshape(np.minimum(chunksize, n_samples-count), n_channels)
        count         += chunksize
        
    # Downsampling        
#         lfp     = scipy.signal.resample_poly(block, 1, 16)
        lfp     = scipy.signal.decimate(block,16,axis = 0)
#         lfp     = scipy.signal.decimate(lfp,4,axis = 0)
#         lfp     = scipy.signal.resample(block, int(len(block)/16))

        f_lfp.write(lfp.astype('int16'))
        pbar.update(chunksize)

        
        del block
        del lfp
    f_lfp.close()  
    return

if __name__ == '__main__':
    path = sys.argv[1]
    
    if not os.path.exists(path):
        print(path, ' does not exists please check directory','Exiting...')
        sys.exit()
        
    session = path.rsplit('\\')[-1]
    os.chdir(path)

    # Looking for Dat (LFPs) files 
    path_dat = [[os.path.join(path,p,f) 
                 for f in os.listdir(os.path.join(path,p)) if f.startswith("amplifier_analogin_auxiliary") & f.endswith('.dat')][0] 
                for p in os.listdir(path) if os.path.isdir(p)]
    print(len(path_dat),' dat files were found')
    for p in path_dat: print("    ",p)
    print('It will be concatenated in this order')


    # Looking for digitalin.dat (TTLs and all digital signal)
    path_digitalin = [[os.path.join(path,p,f) 
                 for f in os.listdir(os.path.join(path,p)) if f.startswith("digitalin") & f.endswith('.dat')][0] 
                for p in os.listdir(path) if os.path.isdir(p)]
    print(len(path_digitalin),' digitalin files were found')
    for p in path_digitalin: print("    ",p)


    #Looking for videos (usefull for counting frames)
    path_videos = [[os.path.join(path,p,f) 
                 for f in os.listdir(os.path.join(path,p)) if f.startswith("Basler") & f.endswith('.mp4')][0] 
                for p in os.listdir(path) if os.path.isdir(p)]
    print(len(path_videos)," Videos were founds : ")
    for p in path_videos: print("    ",p)

    if not (len(path_dat)==len(path_digitalin)==len(path_videos)): 
        print('Not same number of dat, digitalin or videos, please check integrity of each subsession')
        raise ValueError
        sys.exit()
    
        
    xmlInfo = xml(session)

    fs = xmlInfo['samplingRate']
    nChannels = xmlInfo['nChannels']
    bytePerSample = xmlInfo['nBits']/8


    files = os.listdir()
    if session+'.dat' in files:
        overwrite = input(session+'.dat' + ' already exist do you want to overwrite ? (Y/N)')
        if overwrite.lower() == "y": writeDat = True
        else: 
            writeDat = False
            print('Concatenation of ',session+'.dat aborted')
    else: writeDat = True
        
    if 'digitalin.dat' in files:
        overwrite = input('digitalin.dat' + ' already exist do you want to overwrite ? (Y/N)')
        if overwrite.lower() == "y": writeDigital = True
        else: 
            writeDigital = False
            print('Concatenation of digitalin aborted')
    else: writeDigital = True

    
    if session+'.lfp' in files:
        overwrite = input(session+'.lfp' + ' already exist do you want to overwrite ? (Y/N)')
        if overwrite.lower() == "y": extractLFP = True
        else: 
            extractLFP = False
            print('Concatenation of digitalin aborted')
    else: extractLFP = True
    
    t = time.time()
    if writeDat: concat_lfp_dat(session,path_dat)
    print('Dat concatenation done in',time.time()-t,'s')
    
    t = time.time()
    if writeDigital: concat_digitalin(session,path_digitalin,path_videos)
    print('Digital in concatenation done in',time.time()-t,'s')
    
    t = time.time()
    if extractLFP: downsampleDatFileBK(path,nChannels,fs)
    print('LFP extraction done in',time.time()-t,'s')

        
                
    print('Session\'s length base on LFP dat file',os.path.getsize(session+'.dat')/(fs*nChannels*bytePerSample),'s')
    print('Session\'s length base on digitalin.dat file',os.path.getsize('digitalin.dat')/(fs*bytePerSample),'s')
    print('Session\'s length base on downsampled file',os.path.getsize(session+'.lfp')/(1250*nChannels*bytePerSample),'s')

    