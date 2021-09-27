# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:15:33 2021

@author: saudahmad.khan
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from datetime import datetime

allSessions=pd.read_csv("//ADALOVELACE/icybox/All-Rats/Billel/sessionAda.csv")

def getLFP(pathnumber=0,nChannel=166): #Loading the LFP
    if allSessions.Rat[pathnumber]=="Rat10":
        nChannel=172
    else:
        nChannel=166
    return np.memmap(""+allSessions.Path[pathnumber]+"\\"+allSessions.Rat[pathnumber]+"-"+str(allSessions.Day[pathnumber])+".lfp",dtype=np.int16).reshape((-1,nChannel))

def getRipChannel(pathnumber=0,verbose=False): #returns the channel number with good ripple events
    try:
        return int(open(""+allSessions.Path[pathnumber]+"\\"+allSessions.Rat[pathnumber]+"-"+str(allSessions.Day[pathnumber])+".rip.evt", 'r').readline().split()[-1])
    except:
        if(verbose):    
            print("no file found, default channel=13")
        return 13
    
def getSectoredChannel(lfp,ch,freq=1250): #returns any channel from the lfp (hpc,amy,acc etc)
    return lfp[0:freq*(lfp[:,ch].size//freq):,ch].reshape((-1,freq))
    
def getRegionsToCut(temp): #returns regions with bad lfp inside given channel
    return np.nonzero(np.amin(temp[:,0:5],axis=1)==np.amax(temp[:,0:5],axis=1))

def getSleepStates(pathnumber=0): #returns sleep states for every second of the lfp
    states = sio.loadmat(""+allSessions.Path[pathnumber]+'\States.mat')
    
    wake=np.insert(np.array(states['wake'],dtype='object'),2,'wake', axis=1)
    nrem=np.insert(np.array(states['sws'],dtype='object'), 2, 'nrem', axis=1)
    rem=np.insert(np.array(states['Rem'],dtype='object'), 2, 'rem', axis=1)
    drowsy=np.insert(np.array(states['drowsy'],dtype='object'), 2, 'drowsy', axis=1)
    whole_session=np.concatenate((wake, nrem, rem, drowsy))
    whole_session=pd.DataFrame(whole_session, columns=['start', 'stop', 'state'])
    whole_session_sorted=whole_session.sort_values('start', ignore_index=True)
    whole_session_sorted.start[0]=0
    whole_session_sorted.stop=whole_session_sorted.stop+1
    whole_session_sorted.stop[whole_session_sorted.stop.size-1]=getLFP(pathnumber)[:,0].size//1250

    expandedState=np.zeros(whole_session_sorted.stop[whole_session_sorted.stop.size-1],dtype='object')

    for x in range(0,whole_session_sorted.stop.size):
        expandedState[whole_session_sorted.start[x]:whole_session_sorted.stop[x]]=whole_session_sorted.state[x]
    return expandedState


def writeCompressedNPZ(pathnumber):
    if allSessions.Rat[pathnumber]=="Rat10":
        np.savez_compressed(""+allSessions.Rat[pathnumber]+"-"+str(allSessions.Day[pathnumber])+"[compressed].npz",hpc=getSectoredChannel(getLFP(pathno),getRipChannel(pathno)),accelx=getSectoredChannel(getLFP(pathno),166),accely=getSectoredChannel(getLFP(pathno),167),accelz=getSectoredChannel(getLFP(pathno),168),states=getSleepStates(pathno))
    else:
        np.savez_compressed(""+allSessions.Rat[pathnumber]+"-"+str(allSessions.Day[pathnumber])+"[compressed].npz",hpc=getSectoredChannel(getLFP(pathno),getRipChannel(pathno)),accelx=getSectoredChannel(getLFP(pathno),161),accely=getSectoredChannel(getLFP(pathno),162),accelz=getSectoredChannel(getLFP(pathno),163),states=getSleepStates(pathno))

print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
for pathno in range(48,60):
    if allSessions.Rat[pathno]=="Rat09": #since this script is for hpc channel, Rat09 doesn't have any hpc recording and will be skipped.
        continue
    writeCompressedNPZ(pathno)
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))    
