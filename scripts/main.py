import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.timeseries import LombScargle

__all__ = ['RV_Detection']

class RV_Detection(object):
    """
    Implements RV detection methods described in
    Toulis & Bean (2021).

    Parameters
    ----------
    time : np.ndarray
       Array of time values.
    vel : np.ndarray
       Array of velocities.
    err : np.ndarray
       Array of standard errors.

    Attributes
    ----------
    df : pandas.core.frame.DataFrame
       Table of input parameters.
    """
    
    def __init__(self, time, vel, err):
        df = pd.DataFrame(columns=['Time', 'Vel', 'Err'])
        df['Time'] = time
        df['Vel'] = vel
        df['Err'] = err
        self.df = df
        self.peak_period = None


     def lomb_scargle(self, minfreq=1.0/50., maxfreq=1.0/0.5):
        """
        Creates Lomb-Scargle periodogram for RV data within a given frequency 
        range.
        
        Parameters
        ----------
        minfreq : float, optional
            Minimum frequency to compute over. Default = 1/50.
        maxfreq : float, optional
            Maximum frequency to compute over. Default = 1/0.5.
            
        Attributes
        ----------
        peak_period : float
            Peak period in periodogram.
        LS_results : np.ndarray
            Results array from astropy.timeseries.LombScarlge.autopower().
        self.all_P : np.ndarray
            Array of length 10000 spanning the minimum and maximum input
            frequencies.
        """
        results = LombScargle(self.df['Time'], self.df['Vel'],
                              dy=self.df['Err']).autopower(minimum_frequency=minfreq,
                                                           maximum_frequency=maxfreq,
                                                           samples_per_peak=50.0)
        argmax = np.argmax(results[1])
        self.peak_period = 1.0/results[0][argmax]
        self.LS_results = results
        self.all_P = 10**(np.linspace(np.log10(maxfreq), 
                                      np.log10(minfreq), 10000))
    

     def get_X(self, x):
        """
        Converts RV data frame into a matrix of covariates.
        """
        X0 = np.stack((np.ones(len(x)), x), axis=1)
        p = len(self.df.columns) - 3 # Number of covariate signals
        # Didn't implement if p > 0
        return X0


     def fit_weighted(self, add_P=np.array([])):
        """
        Fits the RV data.
        
        Parameters
        ----------
        add_P : np.ndarray, optional
            Array of periods which petermines which periodic terms to include.
        """
        ts = self.df['Time']
        y = self.df['Vel']
        se = self.df['Err']
        X0 = self.get_X(ts)
        
        # Adds periodic terms if passed in
        if len(add_P) > 0:
            new_X0 = np.zeros((len(ts), len(np.array(add_P))+X0.shape[1]))
            new_X0[:,0] = X0[:,0]
            new_X0[:,1] = X0[:,1]
            
            c = 2
            for j in range(len(np.array(add_P))):
                new_X0[:,c] = np.cos(2 * np.pi * ts)/add_P[j]
                new_X0[:,c+1] = np.sin(2 * np.pi * ts)/add_P[j]
                c += 2
        
        def loss_function(params, ret_logL=True):
            """
            Quadratic loss function.
            """
            nonlocal se
            s = np.exp(params[0]) # sigma parameter
            v = s**2 + se**2

            fit = np.polyfit(ts, y, deg=1, w=1.0/v)
            model = np.poly1d(fit)
            res = model(ts) - y
            shape = X0.shape[0] - X0.shape[1]
            pdf = norm.pdf(res, scale=np.sqrt(v))

            logL = np.nansum( np.log10(pdf) )
        
            if ret_logL == False:
                yhat = y - res
                return(np.sqrt(v), fit, yhat, -logL,
                       np.correlate(y, yhat)**2, shape,
                       params)
            
            else:
                return -logL # log likelihood
        

        best = minimize(fun=loss_function,
                        x0=[np.log10(np.nanmean(se))],
                        method='L-BFGS-B',
                        tol=1e-9,
                        bounds=[(-10,10)])

        return loss_function(best.x, ret_logL=False)

    
    def null_periodogram(self, theta0, null_samples=100, verbose=False):
        """
        Samples the periodogram under the null hypothesis that
        $\theta$* = $\theta$0.
        """
        n = len(self.df)
        fit = self.fit_weighted(add_P=theta0)
        ## COME BACK HERE AFTER FIT_WEIGHTED DONE
        
        
    def test_period_theta0(self, theta0, all_p=None, null_samples=100, ls0=None):
        if all_p == None:
            all_p=self.all_P
            
        if self.peak_period == None:
            self.lomb_scargle()
            
        id0 = np.argmin(np.abs(all_p - theta0))
        Pf_null = self.null_periodogram(theta0=theta0, null_samples=null_samples)
