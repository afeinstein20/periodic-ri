import os, time
import warnings
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
        self.LS_results = None


    def lomb_scargle(self, y=None, minperiod=0.5, maxperiod=50.0, ret_results=False):
        """
        Creates Lomb-Scargle periodogram for RV data within a given frequency 
        range.
        
        Parameters
        ----------
        minperiod : float, optional
           Minimum period to compute over. Default = 0.5 days.
        maxperiod : float, optional
           Maximum period to compute over. Default = 50 days.
        y : np.ndarray, optional
           Array of velocities to run the Lomb-Scargle periodogram
           over. Default is velocity array used to initialize this class
           (self.df['Vel']).
        ret_results : bool, optional
           Option to return results instead of setting results as a class
           attribute. Default = False.

        Attributes
        ----------
        LS_results : np.ndarray
           Results array from astropy.timeseries.LombScarlge.autopower().
        peak_period : float
           Peak period in periodogram.

        Returns
        -------
        results : np.ndarray
           Results array from astropy.timeseries.LombScarlge.autopower().
        peak_period : float
           Peak period in periodogram.
        
        """
        start = time.time()
        if y is None:
            y = self.df['Vel']+0.0
            
        results = LombScargle(self.df['Time'], y,
                              dy=self.df['Err']).autopower(minimum_frequency=1.0/maxperiod,
                                                           maximum_frequency=1.0/minperiod,
                                                           samples_per_peak=50.0)
        argmax = np.argmax(results[1])

        end = time.time()

        if ret_results == False:
            self.LS_results = results
            self.peak_period = 1.0/results[0][argmax]
            self.LS_time = end-start

        else:
            return results, 1.0/results[0][argmax], end-start
        

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

    
    def null_periodogram(self, theta0, null_samples=100, verbose=False, ret_results=False):
        """
        Samples the periodogram under the null hypothesis that
        $\theta$* = $\theta$0.

        Parameters
        ----------
        theta0 : float
           Initial guess of periodic signal in the RV data.
        null_samples : int, optional
           Number of samples to test over. Default = 100 samples.
        verbose : bool, optional
        ret_results : bool, optional
           Returns the results instead of setting class attribute. Default = False.

        Attributes
        -------
        all_null_power : np.ndarray
           Array of Lomb Scargle power values per each null sample.
        """
        np.random.seed(123)

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
            out,_,_ = self.lomb_scargle(y=y_new, 
                                        minperiod=1.0/self.maxfreq,
                                        maxperiod=1.0/self.minfreq,
                                        ret_results=True)
                
            # Creates S_null array
            if i == 0:
                 all_power = np.zeros((null_samples, len(out[1])))
            all_power[i] = out[1]
        
        if ret_results == False:
            self.all_null_power = all_power
        else:
            return all_power
        

    def tstat(self, pf, ind0):
        """ 
        Calculates the test statistic.

        Parameters
        ----------
        pf : np.ndarray
           Lomb-Scargle periodogram power array.
        ind0 : int
           Index of period closest to theta0.
        """
        best = np.sort(pf)[-2:]
        return np.max(pf) - pf[ind0]


    def test_period_theta0(self, theta0, minperiod=0.5, maxperiod=50.0,
                           null_samples=100):
        """
        Tests H0: theta* = theta0

        Parameters
        ----------
        theta0 : float
           Initial guess of periodic signal in the RV data.
        minperiod : float, optional
           Minimum period to search over. Default = 0.5 days.
        maxperiod : float, optional
           Maximum period to search over. Default = 50 days.
        null_samples : int, optional
           Number of null samples to create. Default = 100 samples.

        Attributes
        ----------
        minfreq : float
        maxfreq : float
        all_p : np.ndarray
           Periodic signals searched over.
        sobs : float
           Test statistic between theta0 and theta*.
        S_null : np.ndarray
           Null distributiion test statistics.
        null_pval : float
           Fraction of S_null values greater than or equal to sobs.
        """
        self.minfreq = 1.0/maxperiod
        self.maxfreq = 1.0/minperiod

        self.all_p = 10**(np.linspace(np.log10(1.0/maxperiod), 
                                      np.log10(1.0/minperiod), 
                                      10000))

        if self.LS_results is None:
            self.lomb_scargle(minperiod=minperiod, maxperiod=maxperiod)


        ind0 = np.argmin(np.abs(self.LS_results[0] - theta0))

        self.null_periodogram(theta0=theta0, null_samples=null_samples)

        # 4a. Observed test statistic
        sobs = self.tstat(self.LS_results[1], ind0)

        # 4b. Null distribution
        S_null = np.zeros(len(self.all_null_power))
        for i in range(len(S_null)):
            S_null[i] = self.tstat(self.all_null_power[i], ind0)

        self.S_null = S_null
        self.sobs   = sobs
        self.null_pval = len(np.where(S_null >= sobs)[0])/len(S_null)
        
        
    def get_candidate_periods(self, threshold=0.2):
        """
        Gets candidate perioods to check. Periods must have a power
        value from the Lomb-Scargle periodogram greater than the 
        defined threshold.
        
        Parameters
        ----------
        threshold : float, optional
           Power threshold to exceed. This is a value between [0, 1]. 
           Default = 0.2.

        Returns
        -------
        period_peaks : np.ndarray
           Array of periods that exceed the defined power threshold and
           are the peak of that period region.
        peaks : np.ndarray
           Array of indices that correspond to the peak period/power 
           in the Lomb-Scargle periodogram.
        """
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(self.LS_results[1], height=threshold)
        return self.LS_results[0][peaks], peaks


    def build_confidence_set(self, alpha=0.01, null_samples=100, threshold=0.2,
                             minperiod=0.5, maxperiod=50.0, time_budget=1.0):
        """
        Main methood that builds a confidence set $\theta_{1-\alpha}$ in
        Equations (7) and (12) of Toulis & Bean (2021).

        Parameters
        ----------
        alpha : float, optional
           Confidence level. Default = 0.01.
        null_samples : int, optional
           Number of randomization samples. Default = 100 samples.
        threshold : float, optional
           Value the Lomb-Scargle periodogram power must exceed to be
           considered a decent periodic signal. Default = 0.2.
        minperiod : float, optional
           Minimum period to search over. Default = 0.5 days.
        maxperiod : float, optional
           Maximum period to search over. Default = 50 days.
        time_budget : int, optional
           How many minutes to spend in computation. Default = 1 minute.
        """

        # Runs LS for original data input
        if self.LS_results is None:
            self.lomb_scargle(minperiod=minperiod, maxperiod=maxperiod)
        
        # 1. Get candidate thetas
        #    How many thetas to consider given the time budget
        # 1a. Calculate average Lomb-Scargle periodogram rate
        # Honestly not sure we need this... Might just be adding computing time 
        """
        ls_times=np.zeros(10)
        for i in range(10):
            _, _, t = self.lomb_scargle(minperiod=minperiod, maxperiod=maxperiod,
                                        ret_results=True)
            ls_times[i] = t
        rate = np.nanmean(ls_times)

        # Finds the best threshold to search over given the time budget
        all_thresholds = np.linspace(0.1,0.8,50)
        threshold_lens = np.zeros(len(all_thresholds))
        for i in range(len(all_thresholds)):
            x, _ = self.get_candidate_periods(threshold=all_thresholds[i])
            threshold_lens[i] = len(x)
        best_threshold = np.argmin( np.abs(threshold_lens*null_samples*rate/60 - time_budget) ) # time in minutes
        """

        thetas, theta_inds = self.get_candidate_periods(threshold=threshold)
        
        if null_samples < 100:
            warnings.warn("Number of null samples should be larger than 100.")
            
        for i, theta0 in enumerate(thetas):
            # H0: theta*=theta0
            id0 = np.argmin( np.abs(self.LS_results[0] - theta0) )
            
            # For each theta0, sample from the null.
            Pf_null = self.null_periodogram(theta0, null_samples=null_samples, 
                                            ret_results=True)
            
            # Test statistic
            
            
