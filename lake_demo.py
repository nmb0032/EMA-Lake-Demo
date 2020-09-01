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
               w1=0.5, seed=None):
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
        X[0] = 0.0
        decision = 0.1
        decisions = np.zeros(myears,)

        natural_inflows = np.random.lognormal(
            math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
            math.sqrt(math.log(1.0+stdev**2/mean**2)),
            size=myears)
        for t in range(1,myears):
            #decision rule is used
            decision = get_antropogenic_release(X[t-1], c1, c2, r1, r2, w1)
            decisions[t] = decision

            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decision +\
                natural_inflows[t-1]
            average_daily_P[t] += X[t]/nsamples
        reliability += np.sum(X < Pcrit)/(nsamples*myears)
        inertia += np.sum(np.absolute(np.diff(decisions)
                                      < 0.02)) / (nsamples*myears)
        utility += np.sum(alpha*decisions*np.power(delta,
                                                   np.arange(myears)))/nsamples
        max_P = np.max(average_daily_P)
        return max_P, utility, inertia, reliability

from ema_workbench import (RealParameter, ScalarOutcome, Constant, Model)

model = Model('lakeproblem', function=lake_model)
model.uncertainties = [RealParameter('b',0.1, 0.45),
                       RealParameter('q',2.0,4.5),
                       RealParameter('mean',0.01,0.05),
                       RealParameter('stdev', 0.001, 0.005),
                       RealParameter('delta',0.93,0.99)]

model.levers = [RealParameter("c1", -2,2),
                RealParameter("c2", -2,2),
                RealParameter("r1", 0, 2),
                RealParameter("r2", 0, 2),
                RealParameter("w1", 0, 1)]

model.outcomes = [ScalarOutcome('max_P'),
                  ScalarOutcome('utility'),
                  ScalarOutcome('inertia'),
                  ScalarOutcome('reliability')]
#overides some of defaults used by model
model.constants = [Constant('alpha', 0.41),
                   Constant('nsamples', 150),
                   Constant('myears', 100)]

from ema_workbench import (SequentialEvaluator, ema_logging, perform_experiments)

ema_logging.log_to_stderr(ema_logging.INFO)

with SequentialEvaluator(model) as evaluator:
    results = evaluator.perform_experiments(scenarios=1000, policies=10)

experiments, outcomes = results
print(experiments.shape)
print(list(outcomes.keys()))

policies = experiments['policy']
for i, policy in enumerate(np.unique(policies)):
    experiments.loc[policies==policy, 'policy'] = str(i)

data = pd.DataFrame(outcomes)
data['policy'] = policies

sns.pairplot(data, hue='policy', vars=list(outcomes.keys()))
plt.show()