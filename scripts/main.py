import os
import numpy as np
import matplotlib.pyplot as plt
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
    """
    
    def __init__(self, time, vel, err):

    
