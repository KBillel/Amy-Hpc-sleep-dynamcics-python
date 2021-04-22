import os
import sys
import subprocess

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import time

from datetime import datetime

import xml.etree.ElementTree as ET
import cv2
import re

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

def concat_lfp_dat(session,path_dat,subsessions):
    
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
        
        t_rec = [0]
        for p in path_dat:
            t = os.path.getsize(p)/(fs*nChannels*bytePerSample)
            print(t)
            t_rec.append(t)
        concat_event = np.cumsum(t_rec)*1000 # 1000 is for conversion in ms
        
        with open(session+'.cat.evt','w') as f:
            print(concat_event)
            print(subsessions)
            for i,s in enumerate(subsessions):
                print(i)
                f.write(str(concat_event[i])+ f' {subsessions[i]} Start \n')
                f.write(str(concat_event[i+1])+ f' {subsessions[i]} End \n')
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
    
    #Logs
    with open(f'{session}-preprocess.log','a') as log_file:
            log_file.write(f'\n\nNUMBER OF TTL AND TTLs CORRECTIONS\n\n')
            log_file.close()
            
    for p_digitalin,p_video in zip(path_digitalin,path_videos):
        data_ = load_digitalin(p_digitalin,nchannels)
        nTTL,lTTL = CountTTL(data_[0,:])
        nFrames = CountFrames(p_video)
        time.sleep(0.1)
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
            print('NTTL',nTTL)
            print('Number of Frames',nFrames)
            print('Last TTL was at ',lTTL)
            print('Some file might be corrupted, need to implement this')
        oldnTTL = nTTL
        nTTL = CountTTL(data_[0,:])[0]
        
       #Logs 
        with open(f'{session}-preprocess.log','a') as log_file:
            log_file.write(f'{p_digitalin} - {p_video}\n')
            log_file.write(f'NTTL in digitalin file :                        {oldnTTL}\n')
            log_file.write(f'nFrames in video file :                         {nFrames}\n')
            log_file.write(f'Number of TTL after TTL corrections :           {nTTL}\n')
            log_file.write(f'Delta Frames in the end                         {nTTL-nFrames}\n\n')
            log_file.close()
        
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
    
    print(f'Creating {session}-preprocess.log file')
    #Logs
    with open(f'{session}-preprocess.log','a') as log_file:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f'{now} : Starting preprocessing of {session}\n')
            log_file.close()
    
    path_md = f'{session}.md'
    print(path_md)
    if os.path.exists(path_md):
        with open(path_md,'r') as f:
            md_file = f.readlines()
        subsessions = [re.findall('(?<=# )(.*)',l) for l in md_file]
        subsessions = [s[0] for s in subsessions if s]
        print(f'{path_md} was found')
        print(len(subsessions),' subsessions were found')
        for s in subsessions: print("    ",s)
            
            
        with open(f'{session}-preprocess.log','a') as log_file:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f'\n{now} : MD files {path_md}\n')
            for s in subsessions: log_file.write(f'{s}\n')
            log_file.close()
    else:
        print('Could not find md file, are you taking notes ?')
    
    # Looking for Dat (LFPs) files 
    path_dat = [[os.path.join(path,p,f) 
                 for f in os.listdir(os.path.join(path,p)) if f.startswith("amplifier_analogin_auxiliary") & f.endswith('.dat')][0] 
                for p in os.listdir(path) if os.path.isdir(p)]
    print(len(path_dat),' dat files were found')
    for p in path_dat: print("    ",p)
        
    #Logs
    with open(f'{session}-preprocess.log','a') as log_file:
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        log_file.write(f'\n{now} : DAT FILES \n')
        for p in path_dat: log_file.write(f'{p}\n')
        log_file.close()
    
    
    
    print('Files be concatenated in this order.')


    # Looking for digitalin.dat (TTLs and all digital signal)
    path_digitalin = [[os.path.join(path,p,f) 
                 for f in os.listdir(os.path.join(path,p)) if f.startswith("digitalin") & f.endswith('.dat')][0] 
                for p in os.listdir(path) if os.path.isdir(p)]
    print(len(path_digitalin),' digitalin files were found')
    for p in path_digitalin: print("    ",p)
        
    with open(f'{session}-preprocess.log','a') as log_file:
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        log_file.write(f'\n{now} : DIGITALIN FILES\n')
        for p in path_digitalin: log_file.write(f'{p}\n')
        log_file.close()


    #Looking for videos (usefull for counting frames)
    path_videos = [[os.path.join(path,p,f) 
                 for f in os.listdir(os.path.join(path,p)) if f.startswith("Basler") & f.endswith('.mp4')][0] 
                for p in os.listdir(path) if os.path.isdir(p)]
    print(len(path_videos)," Videos were founds : ")
    for p in path_videos: print("    ",p)
    #Logs    
    with open(f'{session}-preprocess.log','a') as log_file:
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        log_file.write(f'\n{now} : VIDEOS FILES\n')
        for p in path_videos: log_file.write(f'{p}\n')
        log_file.close()

    if not (len(path_dat)==len(path_digitalin)==len(path_videos)): 
        print('Not same number of dat, digitalin or videos, please check integrity of each subsession')
    #Logs
        with open(f'{session}-preprocess.log','a') as log_file:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f'\n{now} : WARNING\n')
            for p in path_videos: log_file.write(f'Not same number of dat, digitalin or videos, please check integrity of each subsession')
            log_file.close()
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
        
        
    ############
    #DAT CONCAT#
    ############    
    t = time.time()
    if writeDat:
        with open(f'{session}-preprocess.log','a') as log_file:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f'\n{now} : Starting Dat concatenation\n')
            log_file.close()
        concat_lfp_dat(session,path_dat,subsessions)
    else:
        with open(f'{session}-preprocess.log','a') as log_file:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f'\n{now} : Dat concatenation was skipped\n')
            log_file.close()
    print('Dat concatenation done in',time.time()-t,'s')
    
    with open(f'{session}-preprocess.log','a') as log_file:
        log_file.write(f'Dat concatenation done in{time.time()-t} s\n')
        log_file.close()

    ##################
    #DigitalIN CONCAT#
    ##################    
    t = time.time()
    if writeDigital:
        with open(f'{session}-preprocess.log','a') as log_file:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f'\n{now} : Starting digitalin concatenation\n')
            log_file.close()
        concat_digitalin(session,path_digitalin,path_videos)
    else:
        with open(f'{session}-preprocess.log','a') as log_file:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f'\n{now} : Digitalin concatenation was skipped\n')
            log_file.close()
    print('Digital in concatenation done in',time.time()-t,'s')
    
    with open(f'{session}-preprocess.log','a') as log_file:
        log_file.write(f'Digitalin concatenation done in {time.time()-t} s\n')
        log_file.close()

        
    #############
    #LFP Extract#
    #############         
    t = time.time()
    if extractLFP: 
        with open(f'{session}-preprocess.log','a') as log_file:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f'\n{now} : Starting LFP Extraction\n')
            log_file.close()
        downsampleDatFileBK(path,nChannels,fs)
    else:
        with open(f'{session}-preprocess.log','a') as log_file:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f'\n{now} : LFP Extraction Skipped\n')
            log_file.close()
            
    print('LFP extraction done in',time.time()-t,'s')
    with open(f'{session}-preprocess.log','a') as log_file:
        log_file.write(f'LFP extraction done in {time.time()-t} s\n')
        log_file.close()
    
    time.sleep(0.1)
    
    
    len_dat = os.path.getsize(session+'.dat')/(fs*nChannels*bytePerSample)
    len_digitalin = os.path.getsize('digitalin.dat')/(fs*bytePerSample)
    len_lfp = os.path.getsize(session+'.lfp')/(1250*nChannels*bytePerSample)
    
    print('Session\'s length base on LFP dat file',len_dat,'s')
    print('Session\'s length base on digitalin.dat file',len_digitalin,'s')
    print('Session\'s length base on downsampled file',len_lfp,'s')
    
    
    with open(f'{session}-preprocess.log','a') as log_file:
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        log_file.write(f'\n\n{now} : CHECKING INTEGRITY OF THE DATA AFTER COPY\n')
        log_file.write(f'Session s length base on LFP dat file {len_dat} s\n')
        log_file.write(f'Session s length base on digitalin.dat file {len_digitalin} s\n')
        log_file.write(f'Session s length base on downsampled file {len_lfp} s\n')
        log_file.close()

    
        
        
    
    