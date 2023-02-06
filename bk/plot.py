import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import neuroseries as nts
import bk.compute
import scipy.stats

def rasterPlot(neurons,window = None,col = 'black',width= 0.5,height = 1,offsets = 1):
    if window is None:
        last_spike = np.empty(len(neurons))
        for i,n in enumerate(neurons):
            last_spike[i] = n.as_units('s').index.values[-1]
        
        window = np.array([[0,np.max(last_spike)]])
            
    if type(window) is list: window = np.array([window])
    window = nts.IntervalSet(window[:,0],window[:,1],time_units = 's')
    neurons_np = []
    
    if isinstance(neurons,nts.time_series.Tsd):
        neurons = [neurons]
    for neuron in neurons:
        neurons_np.append(neuron.restrict(window).as_units('s').index)

    neurons_np = np.array(neurons_np,dtype = 'object')
    
    
    plt.eventplot(neurons_np,color = col,linewidth = width,linelengths=height,lineoffsets=offsets)
    plt.ylabel('Neurons')
    plt.xlabel('Time(s)')
    
def intervals(intervals,col = 'orange',alpha = 0.5,time_units = 's',ymin = 0,ymax = 1,ax = None):
    if ax is None:
        fig,ax = plt.subplots(1,1)
    if type(intervals) != nts.interval_set.IntervalSet:
        print(type(intervals))
        intervals = nts.IntervalSet(intervals['start'],intervals['end'])
    
    for interval in intervals.as_units(time_units).values:
        ax.axvspan(interval[0],interval[1], facecolor=col, alpha=alpha,ymin = ymin, ymax = ymax)

def spectrogram(t,f,spec,log = False,ax = None,vmin = None,vmax = None):
    if ax == None: 
        fig,ax = plt.subplots(1,1)    

    if log: spec = np.log(spec)
    ax.pcolormesh(t,f,spec)

    
def cumsum_curves(x,nbins,col = 'orange',ax = None, log = False):
    x,y = bk.compute.cumsum_ditribution(x,nbins)

    if ax is None:
        fig,ax = plt.subplots(1,1)

    if log:
        ax.semilogx(x,y,col)
    else:
        ax.plot(x,y,col)


def confidence_intervals(x,y,style = 'orange',ax = None):
    if ax is None:
        fig,ax = plt.subplots(1,1)
    
    conf = 1.96*scipy.stats.sem(y,0,nan_policy='omit')
    m = np.nanmean(y,0)
    ax.plot(x,m,style)
    ax.fill_between(x,m+conf,m-conf,color = style,alpha = 0.2)

def curve_and_shades(x,y,method = 'std',style = 'orange',ax = None):
    if ax is None:
        fig,ax = plt.subplots(1,1)
    if method.lower() == 'std':
        shade = np.nanstd(y,0)
    elif method.lower() == 'sem':
        shade = scipy.stats.sem(y,0,nan_policy='omit')
     
    m = np.nanmean(y,0)
    ax.plot(x,m,style)
    ax.fill_between(x,m+shade,m-shade,color = style,alpha = 0.2)
