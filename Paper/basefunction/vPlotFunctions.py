import matplotlib.pyplot as plt
import numpy as np

def plotraster(res,clu,pclu=None,cluh=[],prange=[0,np.inf]):
    
        clus = np.unique(clu)
        if not(pclu is None):
                clus = pclu
        if prange[1] == np.inf:
                prange[1] = np.max(res)
                
        smps2plot = np.arange(prange[0],prange[1])
        
        for clui,clu_ in enumerate(pclu):

                spks = res[clu==clu_]
                spks = spks[np.in1d(spks,smps2plot)]
                
                color = 'k'
                if clu_ in cluh:
                        color = 'r'
                        
                plt.plot(spks,np.ones(len(spks))*clui,'|',color=color,mew=3)
                
        plt.yticks(np.arange(len(pclu)),pclu)
        plt.grid()
        plt.xlim(prange[0],prange[1])

def config():
        plt.rcParams['svg.fonttype'] = 'none'
    
def contourf(haxis,vaxis,data,levels=20,cmap='Reds'):

        config()
        cnt = plt.contourf(haxis,vaxis,data,levels,cmap=cmap) 
        for c in cnt.collections:
                        c.set_edgecolor("face") 


def AdjustBox(grid=True,gridcolor='k',spines=False,adjustXlim=True):
	plt.gca().spines['top'].set_visible(spines)
	plt.gca().spines['right'].set_visible(spines)
	tickout()
	plt.gca().yaxis.set_ticks_position('left')
	plt.gca().xaxis.set_ticks_position('bottom')
	if len(plt.gca().get_lines())>0:
		try:
			plt.xlim(GetMinMaxData('x')*1.02)
		except:
			0
	if grid:
		plt.grid(grid,color=gridcolor,linestyle='--',alpha=.5)

def AdjustBoxCountour(grid=True,gridColor='w'):
	tickout()
	plt.gca().yaxis.set_ticks_position('left')
	plt.gca().xaxis.set_ticks_position('bottom')
	if grid:
		plt.grid(color=gridColor,alpha=.75,linestyle='--')

def tickout():
	plt.gca().tick_params(direction='out')

def GetMinMaxData(axis='y'):

	n = len(plt.gca().get_lines())
	if axis is 'y':
		auxmax = np.max([np.max(plt.gca().get_lines()[i].get_ydata()) for i in range(n)])
		auxmin = np.min([np.min(plt.gca().get_lines()[i].get_ydata()) for i in range(n)])
	elif axis is 'x':
		auxmax = np.max([np.max(plt.gca().get_lines()[i].get_xdata()) for i in range(n)])
		auxmin = np.min([np.min(plt.gca().get_lines()[i].get_xdata()) for i in range(n)])
	else:
		print('... invalid axis')
		return None

	return auxmin,auxmax

def tickfontsize(labelsize=14,axis='both'):
	plt.gca().tick_params(axis=axis, which='major', labelsize=labelsize)

def imagesc(HAxis,VAxis,data,cmap='jet',origin='lower'):	

	if np.size(HAxis)==0:
		HAxis=np.arange(.5,np.size(data,1)+1.5)
	else:
		HAxis = np.array(HAxis)
	if np.size(VAxis)==0:
		VAxis=np.arange(.5,np.size(data,0)+1.5)
	else:
		VAxis = np.array(VAxis)

	extent = (HAxis.min(), HAxis.max(), VAxis.min(), VAxis.max())
	handle = plt.imshow(data,extent=extent,aspect='auto',interpolation='nearest',origin=origin,cmap=cmap)
	
	return handle
