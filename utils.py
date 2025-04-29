import numpy as np
from scipy.interpolate import UnivariateSpline, make_interp_spline

def _gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
#def _gaussian(x, a, b, c):# Really a lorentzian, just checking
#    return a * (gam/2)**2 / ( (gam/2)**2 + ( x - b )**2)

def _multi_gaussian(x, params):
    template = np.zeros_like(x)
    for param in params:
        template += _gaussian(x, *param)
    return template

def _monomial(x,a,n):
    return a*x**n

def _polynomial(x,params):
    template = np.zeros_like(x)
    for param in params:
        template += _monomial(x, *param)
    return template

def _binary_search(nknots,x,y,dy,maxiter=50):
    """
    Binary search for the smoothness parameter "s" to get a spline with nknots.
    """
    m = len(x)
    std = np.mean(np.sqrt(dy))
    low = np.log10((m - np.sqrt(2*m)) * std**2)-100
    hi = np.log10((m + np.sqrt(2*m)) * std**2)+100
    print([hi,low])
    #hi,low = 100, -10
    for _ in range(maxiter):
        s = 10**((hi+low)/2)
        #print(s)
        ss = UnivariateSpline(x,y,k=3,s=s)
        if len(ss.get_knots()) < nknots:
            hi = np.log10(s)
        elif len(ss.get_knots()) > nknots:
            low = np.log10(s)
        else:
            return s,ss
    return s,ss

def get_spline(knots,edges,wl):
    xvals = [wl[0]] + list(knots[:,0]) + [wl[-1]]
    yvals = [edges[0,0]] + list(knots[:,1]) + [edges[0,1]]
    interp_model = make_interp_spline(xvals,yvals,k=3, axis=-1)
    return interp_model

