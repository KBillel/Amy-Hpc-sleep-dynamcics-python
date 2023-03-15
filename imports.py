import sys
import itertools

import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import savgol_filter
from scipy.stats import zscore
import scikit_posthocs as sp
from tqdm import tqdm

import neuroseries as nts


import time

import matplotlib.pyplot as plt
import seaborn as sns 


from statannotations.Annotator import Annotator


import bk.load
import bk.compute
import bk.plot
import bk.signal
import bk.stats
import bk.multi
import bk.misc

import os


plt.style.use('ggplot')

# plt.rcParams['svg.fonttype'] = 'none'
# plt.rcParams['figure.facecolor'] = '1c2128'
# plt.rcParams['xtick.color'] = 'white'
# plt.rcParams['ytick.color'] = 'white'
# plt.rcParams['text.color'] = 'white'
# plt.rcParams['legend.facecolor'] = '1c2128'
