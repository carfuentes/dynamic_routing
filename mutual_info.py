#imports
import entropy_estimators as ee
import scipy.spatial as ss
from scipy.special import digamma,gamma
from math import log,pi
import numpy.random as nr
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import random as rand
import scipy
from math import sqrt

import pandas as pd
import scipy as sc





#functions
#k-neighbours
def entropy(x,k=3,base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator
         x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x)-1 #"Set k smaller than num. samples - 1"
    d = len(x[0])
    N = len(x)
    intens = 1e-10 #small noise to break degeneracy, see doc.
    x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
    tree = ss.cKDTree(x)
    nn = [tree.query(point,k+1,p=float('inf'))[0][k] for point in x]
    const = digamma(N)-digamma(k) + d*log(2)
    return (const + d*np.mean(list(map(log,nn)),dtype=np.float64))/log(base)


def calc_MI_npeet(x,y):
    return ee.mi(x.reshape((x.shape[0],1)), y.reshape((y.shape[0],1)), base=2)

#binning

def ent(data):
    p_data= pd.Series(data).value_counts()/len(data) # calculates the probabilities
    entropy=sc.stats.entropy(p_data)  # input probabilities to get the entropy 
    return entropy

def entropy_binning(c_xy):
    c_normalized = c_xy/np.sum(c_xy)
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    h = -sum(c_normalized * np.log(c_normalized))  
    return h

def calc_MI_binning(x, y):
    bins=sqrt(x.shape[0]/5)
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    #c_xx=np.histogram2d(x,x,bins)[0]
    return mi/ent(x)


#memory-capacity
def memory_capacity_n(Yt, Xt,startLen,miLen,i_max,n_max):
    MI_i={}
    for i in range(i_max+1):
        MI_i[i]=calc_MI_binning(Xt[startLen+n_max-i:startLen+n_max+miLen-i],Yt[0,n_max:miLen+n_max]) 
    return MI_i

