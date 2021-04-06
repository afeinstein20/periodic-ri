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


    def lomb_scargle(self, y=None, minfreq=1.0/50., maxfreq=1.0/0.5, ret_results=False):
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
         if y is None:
             y = self.df['Vel']+0.0

         results = LombScargle(self.df['Time'], y,
                               dy=self.df['Err']).autopower(minimum_frequency=minfreq,
                                                            maximum_frequency=maxfreq,
                                                            samples_per_peak=50.0)
         argmax = np.argmax(results[1])
         all_P = 10**(np.linspace(np.log10(maxfreq),
                                  np.log10(minfreq), 10000))
         
         if ret_results == False:
             self.peak_period = 1.0/results[0][argmax]
             self.LS_results = results
             self.all_P = all_P + 0.0

         else:
             return {'periods':1.0/results[0],
                     'power':results[1],
                     'peak_period':1.0/results[0][argmax],
                     'all_P': all_P}
    

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
            new_X0 = np.zeros((len(ts), len(np.array(add_P))*2+X0.shape[1]))
            new_X0[:,0] = X0[:,0]
            new_X0[:,1] = X0[:,1]
            
            c = 2
            for j in range(len(add_P)):
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
                ret_dict = {'sigma':np.sqrt(v),
                            'bhat':fit,
                            'yhat':yhat,
                            'logL':-logL,
                            'R2':np.correlate(y, yhat)**2,
                            'df':shape,
                            'params':params}
                return ret_dict
            
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

        Returns
        -------
        all_power : np.ndarray
           Array of Lomb Scargle power values per each null sample.
        all_P : np.ndarray
           Array of periods sampled.
        theta0 : float
        """

        n = len(self.df)

        # 1. Fit y(t) ~ X + Per(theta0, t) + eta(t)
        fit = self.fit_weighted(add_P=np.array([theta0]))
        Yhat0 = fit['yhat']

        # Residuals under H0
        e0 = self.df['Vel'] - Yhat0
        signs = np.append(np.full(int(n/2), -1), np.full(int(n/2), 1))
        if len(signs) < n:
            signs = np.append(signs, np.full(int(n-len(signs)), 1))


        # Only assume symmetric errors
        for i in range(null_samples):
            e = np.random.choice(signs, size=len(signs), replace=False) * e0
            y_new = Yhat0 + e

            # 2. Lomb-Scargle periodogram
            out = self.lomb_scargle(y=y_new, ret_results=True)
            if i == 0:
                 all_power = np.zeros((null_samples, len(out['power'])))
            all_power[i] = out['power']
        
        return {'power':all_power, 
                'all_P': out['all_P'], 
                'theta0': theta0}
        
    def test_period_theta0(self, theta0, all_p=None, null_samples=100, ls0=None):
        """
        Tests H0: theta* = theta0

        Attributes
        ----------
        sobs : float
           Test statistic between theta0 and theta*.
        S_null : np.ndarray
           Null distributiion test statistics.
        null_pval : float
           Fraction of S_null values greater than or equal to sobs.
        """
        if ls0 is None:
            ls0 = self.lomb_scargle(ret_results=True)

        ind0 = np.argmin(np.abs(ls0['all_P'] - theta0))

        Pf_null = self.null_periodogram(theta0=theta0, null_samples=null_samples)

        # Test statistic
        def tstat(pf):
            nonlocal ind0

            best = np.sort(pf)[-2:]
            return np.max(pf) - pf[ind0]

        # 4a. Observed test statistic
        sobs = tstat(ls0['power'])

        # 4b. Null distribution
        S_null = np.zeros(len(Pf_null['power']))
        for i in range(len(S_null)):
            S_null[i] = tstat(Pf_null['power'][i])

        self.S_null = S_null
        self.sobs   = sobs
        self.null_pval = len(np.where(S_null >= sobs)[0])/len(S_null)
        
        
        
