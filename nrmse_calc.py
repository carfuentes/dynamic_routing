#imports
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error



#functions
def nrmse(y,x):
    return sqrt(mean_squared_error(x,y))/np.std(x)


def nrmse_n(Yt, Xt,i_max,errorLen,startLen,n_max):
    NRMSE_i={}
    for i in range(i_max+1):
        NRMSE_i[i]=nrmse(Yt[0,n_max:errorLen+n_max],Xt[startLen+n_max-i:startLen+n_max+errorLen-i]) 
    
    return NRMSE_i
