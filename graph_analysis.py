#imports
import numpy as np
from matplotlib.pyplot import *
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brute

#functions
def autocorrelation(x):
    result = np.correlate(x, x, mode='full')
    result=result[int(result.size/2):]
    return result/result[0]

def func_sum_exp_power_law(x,a,b,e,f,d):
    return a * np.exp(-1/b * x) + x**e * f+d

def func_sum_exp(x, a, b, c,d,e):
    return a * np.exp(-1/b * x)+ c * np.exp(-1/d * x) + e

def func_power_law(x,e,f,d):
    return x**e * f+d

def exp_func(x, a, b,c):
    return a * np.exp(-1/b * x)+c

def exponential_fitting(x, func,p0=None,start_point=0,MI=False):
    if MI: #x is MI
        xdata=np.array(list(x.keys())[start_point:])
        ydata=np.array(list(x.values())[start_point:])
    else: #x is autocorr
        a=x[:np.argmax(x<0)]
        ydata=a
        xdata=np.arange(a.shape[0])
    popt, pcov = curve_fit(func, xdata, ydata,p0)
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plot(xdata, ydata, '-o', label='data')
    plot(xdata, func(xdata, *popt), 'r-', label="fit")
    #plot(xdata, exp_func(xdata, popt[0],tau,popt[2]), 'b-', label="expected fit")
    xlabel('n',fontsize=20)
    ylabel('MImax',fontsize=20)
    legend(loc=1, prop={'size': 20})
    return popt,xdata,ydata

def spline_plotting(x,y,nrmse):
    if not nrmse:
        spline = UnivariateSpline(x, y-np.max(y)/2, s=0)
        r1, r2 = spline.roots() # find the roots
    else:
        spline = UnivariateSpline(x, y-np.max(abs(y))/2, s=0)
        r1, r2 = spline.roots() # find the roots
    FWHM=abs(r1-r2)
    print("r1=%f and r2=%f"%(r1,r2))
    print("FWHM",FWHM)
    plot(x, y)
    plot(r1,0,marker='o')
    plot(r2,0,marker='o')
    axvspan(r1, r2, facecolor='g', alpha=0.5)
    show()
    return FWHM
   