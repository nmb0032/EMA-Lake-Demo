import math

#default imports for ema workbench
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import brentq

def get_antropogenic_release(xt, c1, c2, r1, r2, w1):
    """

    :param xt: float polution in lake
    :param c1: float center rbf 1
    :param c2: float center rbf 2
    :param r1: float ratius rbf 1
    :param r2: float ratius rbf 2
    :param w1: float weight of rbf 1
    :return: float
    """
    rule = w1*(abs(xt-c1)/r1)**3+(1-w1)*(abs(xt-c2)/r2)**3
    at1 = max(rule, 0.01)
    at = min(at1,.1)

    return at

def lake_model(b=0.42, q=2.0, mean=0.02,
               stdev=0.001, delta=0.98, alpha=0.4,
               nsamples=100, myears=100, c1=0.25,
               c2=0.25, r1=0.5, r2=0.5,
               w1=0.5, seed=None)
    """
    
    :param b: float decay rate for P in Lake
    :param q: float recycling exponent
    :param mean: float mean of natural inflows 
    :param stdev: float standard deviation of natural inflows
    :param delta: float future utility discount rate
    :param alpha: float utility from pollution
    :param nsamples: int, optional
    :param myears: int, optional
    :param c1: float
    :param c2: float
    :param r1: float
    :param r2: float
    :param w1: float
    :param seed: int, optional for random number gen
    :return: tuple
    """
    np.random.seed(seed)
    Pcrit = brentq(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)

    X = np.zeros((myears,))
    average_daily_P = np.zeros((myears,))
    reliability = 0.0
    inertia = 0
    utility = 0

    for _ in range(nsamples):

