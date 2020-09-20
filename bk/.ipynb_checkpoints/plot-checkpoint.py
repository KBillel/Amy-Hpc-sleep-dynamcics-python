import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

def rasterPlot(neurons,window,col = 'black'):
    window = nts.IntervalSet(window[:,0],window[:,1],time_units = 's')
    neurons_np = []
    
    if isinstance(neurons,nts.time_series.Tsd):
        neurons = [neurons]
    print(type(neurons))
    for neuron in neurons:
        neurons_np.append(neuron.restrict(window).as_units('s').index)

    neurons_np = np.array(neurons_np,dtype = 'object')
    
    
    plt.eventplot(neurons_np,color = col)
    plt.ylabel('Neurons')
    plt.xlabel('Time(s)')
    
def intervals(intervals,col,time_units = 's'):
    for interval in intervals.as_units(time_units).values:
        plt.axvspan(interval[0],interval[1], facecolor=col, alpha=0.5)