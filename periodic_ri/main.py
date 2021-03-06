import os, time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from lightkurve import LightCurve
from scipy.optimize import minimize
from astropy.timeseries import LombScargle

from .gls import Gls

__all__ = ['PeriodicRI']

class PeriodicRI(object):
    """
    Implements periodic signal detection methods 
    described in Toulis & Bean (2021).

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


    def lomb_scargle(self, y=None, minperiod=0.5, maxperiod=50.0, 
                     norm='astropy', ret_results=False):
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
        norm : str, optional
           Sets the normalization for the periodogram. Default is 'astropy', 
           which uses the astropy.timeseries.LombScargle function. Other 
           options include those implemented in GLS: ("ZK", "Scargle", "HorneBaliunas", "Cumming", "wrms", "chisq").
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
            
        if norm == 'astropy':
            results = LombScargle(self.df['Time'], y,
                                  dy=self.df['Err'],
                                  normalization='log').autopower(minimum_frequency=1.0/maxperiod,
                                                                 maximum_frequency=1.0/minperiod,
                                                                 samples_per_peak=50.0,
                                                                 normalization='log')
        else:
            gls = Gls(lc=[self.df['Time'],
                          y,
                          self.df['Err']],
                      fbeg=1.0/maxperiod,
                      fend=1.0/minperiod,
                      norm=norm)
            results = np.array([gls.freq, gls.power])

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
            pdf = stats.norm.pdf(res, scale=np.sqrt(v))

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

    
    def null_periodogram(self, theta0, minperiod=0.5, maxperiod=50.0, norm='astropy',
                         null_samples=100, verbose=False, ret_results=False):
        """
        Samples the periodogram under the null hypothesis that
        $\theta$* = $\theta$0.

        Parameters
        ----------
        theta0 : float
           Initial guess of periodic signal in the RV data.
        minperiod : float, optional
           Minimum period to search over. Default = 0.5 days.
        maxperiod : float, optional
           Maximum period to search over. Default = 50 days.
        norm : str, optional
           Sets the normalization for the periodogram. Default is 'astropy',
           which uses the astropy.timeseries.LombScargle function. Other   
           options include those implemented in GLS: ("ZK", "Scargle", "HorneBaliunas", "Cumming", "wrms", "chisq"). 
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
        shape = (null_samples, n)

        # 1. Fit y(t) ~ X + Per(theta0, t) + eta(t)
        fit = self.fit_weighted(add_P=np.array([theta0]))
        Yhat0 = fit['yhat']

        # Residuals under H0
        e0 = np.full(shape, self.df['Vel'] - Yhat0)

        signs = np.append(np.full(int(n/2), -1), np.full(int(n/2), 1))
        if len(signs) < n:
            signs = np.append(signs, np.full(int(n-len(signs)), 1))

        e = np.random.choice(signs, size=shape) * e0
        y_new = np.full(shape, Yhat0) + e

        # Only assume symmetric errors
        for i in range(null_samples):

            # 2. Lomb-Scargle periodogram
            out,_,_ = self.lomb_scargle(y=y_new[i],
                                        minperiod=minperiod,
                                        maxperiod=maxperiod,
                                        ret_results=True,
                                        norm=norm)
                
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
        if len(pf.shape) == 2:
            return np.nanmax(pf, axis=1) - pf[:,ind0] 
        else:
            return np.nanmax(pf) - pf[ind0]


    def test_period_theta0(self, theta0, minperiod=0.5, maxperiod=50.0,
                           norm='astropy', null_samples=100):
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
        norm : str, optional 
           Sets the normalization for the periodogram. Default is 'astropy',
           which uses the astropy.timeseries.LombScargle function. Other   
           options include those implemented in GLS: ("ZK", "Scargle", "HorneBaliunas", "Cumming", "wrms", "chisq").      
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

        self.all_p = 10**(np.linspace(np.log10(1.0/maxperiod), 
                                      np.log10(1.0/minperiod), 
                                      10000))

        self.lomb_scargle(minperiod=minperiod, maxperiod=maxperiod, norm=norm)


        ind0 = np.argmin(np.abs(self.LS_results[0] - theta0))

        self.null_periodogram(theta0=theta0, minperiod=minperiod,
                              maxperiod=maxperiod, norm=norm,
                              null_samples=null_samples)

        # 4a. Observed test statistic
        sobs = self.tstat(self.LS_results[1], ind0)

        # 4b. Null distribution
        S_null = self.tstat(self.all_null_power, ind0)

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


    def build_confidence_set(self, alpha=0.01, null_samples=100, norm='astropy',
                             min_thresh=0.1, minperiod=0.5, maxperiod=50.0, time_budget=1.0):
        """
        Main methood that builds a confidence set $\theta_{1-\alpha}$ in
        Equations (7) and (12) of Toulis & Bean (2021). Defines the minimum
        power threshold to search over via input time constraints.

        Parameters
        ----------
        alpha : float, optional
           Confidence level. Default = 0.01.
        null_samples : int, optional
           Number of randomization samples. Default = 100 samples.
        minperiod : float, optional
           Minimum period to search over. Default = 0.5 days.
        maxperiod : float, optional
           Maximum period to search over. Default = 50 days.
        min_thresh : float, optional
           Minimum power threshold to search periodogram over.
           Default = 0.1.
        norm : str, optional 
           Sets the normalization for the periodogram. Default is 'astropy',
           which uses the astropy.timeseries.LombScargle function. Other 
           options include those implemented in GLS: ("ZK", "Scargle", "HorneBaliunas", "Cumming", "wrms", "chisq").      
        time_budget : int, optional
           How many minutes to spend in computation. Default = 1 minute.

        Attributes
        ----------
        pvals_m : np.ndarray
           Creates an array of period and associated p-values.
        cset : np.ndarray
           Confidence set of periods for the intialized data.
        """
        from tqdm import tqdm

        # Runs LS for original data input
        self.lomb_scargle(minperiod=minperiod, maxperiod=maxperiod, norm=norm)

        # 1. Get candidate thetas
        #    How many thetas to consider given the time budget
        # 1a. Calculate average Lomb-Scargle periodogram rate
        
        ls_times=np.zeros(10)
        for i in range(10):
            _, _, t = self.lomb_scargle(minperiod=minperiod, maxperiod=maxperiod,
                                          ret_results=True)
            ls_times[i] = t
        rate = np.nanmean(ls_times)

        # Finds the best threshold to search over given the time budget
        all_thresholds = np.linspace(min_thresh,0.8,50)
        threshold_lens = np.zeros(len(all_thresholds))
        for i in range(len(all_thresholds)):
            x, _ = self.get_candidate_periods(threshold=all_thresholds[i])
            threshold_lens[i] = len(x)
        best_threshold = np.argmin( np.abs(threshold_lens*null_samples*rate/60 - time_budget) ) # time in minutes

        # Decides which periods to sample over (given time constraints)
        thetas, theta_inds = self.get_candidate_periods(threshold=all_thresholds[best_threshold])

        if null_samples < 100:
            warnings.warn("Number of null samples should be larger than 100.")
            
        pvals_m = np.zeros((len(thetas), 4))

        for i, theta0 in enumerate(tqdm(thetas)):
            # H0: theta*=theta0
            id0 = np.argmin( np.abs(self.LS_results[0] - theta0) )
            
            # For each theta0, sample from the null.
            Pf_null = self.null_periodogram(theta0, null_samples=null_samples, 
                                            ret_results=True, minperiod=minperiod,
                                            maxperiod=maxperiod, norm=norm)
            
            # 4a. Observed test statistic
            sobs = self.tstat(self.LS_results[1], id0)

            # 4b. Null distribution
            S_null = self.tstat(Pf_null, id0)
            
            pvals_m[i] = np.array([1.0/theta0, 
                                   len(np.where(S_null <= sobs)[0]) / len(S_null),
                                   len(np.where(S_null >= sobs)[0]) / len(S_null),
                                   len(np.where(S_null == sobs)[0]) / len(S_null) 
                                   ])

        # Excludes intervals with values < assigned alpha
        inds = np.where(pvals_m[:,2] > alpha)
        self.pvals_m = pvals_m[inds,:]
        self.cset = pvals_m[:,0][inds]


    def rv_plot(self, ax=None):
        """
        Plots the radial velocity data as a matplotlib.pyplot.errorbar figure.

        Returns
        -------
        matplotlib.figure.Figure
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(14,4))
            ax.set_xlabel('Time', fontsize=16)
            ax.set_ylabel('RVs [km s$^{-1}$]', fontsize=16)

        ax.errorbar(self.df['Time'], self.df['Vel'], yerr=self.df['Err'],
                     marker='o', linestyle='', color='k')

        return ax

    def ls_plot(self, ax=None):
        """
        Plots the Lomb-Scargle periodogram and highlights peaks
        in the data.

        Returns
        -------
        matplotlib.figure.Figure
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(14,4))
            ax.set_xlabel('Period [days]', fontsize=16)
            ax.set_ylabel('Power', fontsize=16)


        ax.plot(1.0/self.LS_results[0], self.LS_results[1], 'k')
        
        _, peak_inds = self.get_candidate_periods()
        ax.plot(1.0/self.LS_results[0][peak_inds],
                 self.LS_results[1][peak_inds], 'darkorange',
                 marker='*', ms=10, linestyle='')

        return ax


    def fold_rvs(self, period=None, plot=False, ax=None):
        """
        Uses lightkurve.LightCurve.fold() to fold the RVs
        on a given period. Adds a column called "Phased_time"
        to self.df.

        Parameters
        ----------
        period : float, optional
           The period on which to fold the data. Default is
           the peak period from the Lomb-Scargle periodogram.
        plot : bool, optional
           Plots the folded RV data and errors. Default is False.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if period is None:
            period = self.peak_period

        lk = LightCurve(time=self.df['Time'],
                        flux=self.df['Vel'],
                        flux_err=self.df['Err'])

        folded = lk.fold(period=period)
        
        if 'Phased_time' not in self.df.columns:
            self.df.insert(3, "Phased_time", folded.time.value)
        else:
            self.df['Phased_time'] = folded.time.value

        if plot:

            if ax is None:
                fig, ax = plt.subplots(figsize=(14,4))
                ax.set_xlabel('Phase', fontsize=16)
                ax.set_ylabel('RVs [km s$^{-1}$]', fontsize=16)
            
            ax.errorbar(self.df['Phased_time'],
                         self.df['Vel'],
                         yerr=self.df['Err'],
                         marker='o', linestyle='',
                         color='k')
            
            return ax
        
