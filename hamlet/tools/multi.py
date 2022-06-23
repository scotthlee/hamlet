"""
Multiprocessing-enabled versions of functions from tools.py
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.stats import chi2, norm, binom
from copy import deepcopy
from multiprocessing import Pool

from . import inference as ti
from . import metrics as tm


def jackknife_metrics(targets, 
                      guesses,
                      cutpoint=0.5, 
                      p_adj=None,
                      average='weighted',
                      processes=None):
    # Replicates of the dataset with one row missing from each
    rows = np.array(list(range(targets.shape[0])))
    j_rows = [np.delete(rows, row) for row in rows]

    # using a pool to get the metrics across each
    inputs = [(targets[idx], guesses[idx], cutpoint, p_adj, average)
              for idx in j_rows]
    
    with Pool(processes=processes) as p:
        stat_list = p.starmap(tm.clf_metrics, inputs)
    
    # Combining the jackknife metrics and getting their means
    scores = pd.concat(stat_list, axis=0)
    means = scores.mean()
    
    return scores, means


# Calculates bootstrap confidence intervals for an estimator
class boot_cis:
    def __init__(
        self, y, y_,
        sample_by=None,
        n=100,
        a=0.05,
        method="bca",
        cutpoint=0.5,
        interpolation="nearest",
        average='weighted',
        weighted=True,
        mcnemar=False,
        seed=10221983,
        processes=None,
        p_adj=None,
        by=None,
        boot_mean=False):
        # Converting everything to NumPy arrays, just in case
        stype = type(pd.Series([0]))
        if type(y) == stype:
            y = y.values
        if type(y_) == stype:
            y_ = y_.values
        
        # Getting the point estimates
        stat = tm.clf_metrics(y, y_,
                              p_adj=p_adj,
                              cutpoint=cutpoint,
                              average=average,
                              mcnemar=mcnemar).transpose()
        
        # Pulling out the column names to pass to the bootstrap dataframes
        colnames = list(stat.index.values)
        
        # Making an empty holder for the output
        scores = pd.DataFrame(np.zeros(shape=(n, stat.shape[0])),
                              columns=colnames)
        
        # Setting the seed
        if seed is None:
            seed = np.random.randint(0, 1e6, 1)
        np.random.seed(seed)
        seeds = np.random.randint(0, 1e6, n)
        
        # Generating the bootstrap samples and metrics
        with Pool(processes=processes) as p:
            if p_adj is not None:
                p_vars = binom.rvs(y.shape[0], p_adj, size=n) / y.shape[0]
            else:
                p_vars = [None] * n
            boot_input = [(y, y, p_vars[i], seed) 
                          for i, seed in enumerate(seeds)]
            boot_samples = p.starmap(ti.boot_sample, boot_input)
            inputs = [(y[boot], y_[boot], cutpoint, None) 
                      for i, boot in enumerate(boot_samples)]
            res = p.starmap(tm.clf_metrics, inputs)
            scores = pd.concat(res, axis=0)
            p.close()
            p.join()
        
        # Optionally using the boot means as the main estimates
        if boot_mean:
            stat = scores.mean().to_frame()
        
        # Calculating the confidence intervals
        lower = (a / 2) * 100
        upper = 100 - lower
        quantiles = (lower, upper)
        
        # Making sure a valid method was chosen
        methods = ["pct", "diff", 'emp', "bca"]
        assert method in methods, "Method must be pct, diff, emp, or bca."

        # Calculating the CIs with method #1: the percentiles of the
        # bootstrapped statistics
        if method == "pct":
            cis = np.nanpercentile(scores,
                                   q=quantiles,
                                   interpolation=interpolation,
                                   axis=0)
            cis = pd.DataFrame(cis.transpose(),
                               columns=["lower", "upper"],
                               index=colnames)
        
        # Or with method #2, the canonical "empirical" bootstrap
        elif method == 'emp':
            stat_vals = stat.transpose().values.ravel()
            diffs = scores.values - scores.values.mean(axis=0)
            vars = np.sum(diffs**2, axis=0) / (n - 1)
            zl = norm.ppf(a / 2)
            zu = norm.ppf(1 - (a / 2))
            cis = pd.DataFrame([stat_vals + (zl * np.sqrt(vars)), 
                                stat_vals + (zu * np.sqrt(vars))]).transpose()
            cis.columns = ['lower', 'upper']
            cis.set_index(stat.index, inplace=True)
        
        # Or with method #3: the percentiles of the difference between the
        # obesrved statistics and the bootstrapped statistics
        elif method == "diff":
            stat_vals = stat.transpose().values.ravel()
            diffs = stat_vals - scores
            percents = np.nanpercentile(diffs,
                                        q=quantiles,
                                        interpolation=interpolation,
                                        axis=0)
            lower_bound = pd.Series(stat_vals + percents[0])
            upper_bound = pd.Series(stat_vals + percents[1])
            cis = pd.concat([lower_bound, upper_bound], axis=1)
            cis.set_index(stat.index, inplace=True)
        
        # Or with method #4: the bias-corrected and accelerated bootstrap
        elif method == "bca":
            # Calculating the bias-correction factor
            stat_vals = stat.transpose().values.ravel()
            n_less = np.sum(scores < stat_vals, axis=0)
            p_less = n_less / n
            z0 = norm.ppf(p_less)

            # Fixing infs in z0
            z0[np.where(np.isinf(z0))[0]] = 0.0

            # Estiamating the acceleration factor
            j = jackknife_metrics(y, y_,
                                  cutpoint,
                                  p_adj,
                                  average,
                                  processes)
            diffs = j[1] - j[0]
            numer = np.sum(np.power(diffs, 3))
            denom = 6 * np.power(np.sum(np.power(diffs, 2)), 3 / 2)

            # Getting rid of 0s in the denominator
            zeros = np.where(denom == 0)[0]
            for z in zeros:
                denom[z] += 1e-6
            
            # Finishing up the acceleration parameter
            acc = numer / denom
            self.jack = j
            
            # Calculating the bounds for the confidence intervals
            zl = norm.ppf(a / 2)
            zu = norm.ppf(1 - (a / 2))
            lterm = (z0 + zl) / (1 - acc * (z0 + zl))
            uterm = (z0 + zu) / (1 - acc * (z0 + zu))
            lower_q = norm.cdf(z0 + lterm) * 100
            upper_q = norm.cdf(z0 + uterm) * 100
            self.lower_q = lower_q
            self.upper_q = upper_q

            # Returning the CIs based on the adjusted quintiles
            cis = [
                np.nanpercentile(scores.iloc[:, i],
                                 q=(lower_q[i], upper_q[i]),
                                 interpolation=interpolation,
                                 axis=0) 
                for i in range(len(lower_q))
            ]
            cis = pd.DataFrame(cis, 
                               columns=["lower", "upper"], 
                               index=colnames)
        
        # Putting the stats with the lower and upper estimates
        cis = pd.concat([stat, cis], axis=1)
        cis.columns = ["stat", "lower", "upper"]
        
        # Passing the results back up to the class
        self.cis = cis
        self.scores = scores
        self.p_vars = p_vars
        
        return


def boot_roc(targets, scores, sample_by=None, n=1000, seed=10221983):
    # Generating the seeds
    np.random.seed(seed)
    seeds = np.random.randint(1, 1e7, n)

    # Getting the indices for the bootstrap samples
    p = Pool()
    boot_input = [(targets, sample_by, None, seed) for seed in seeds]
    boots = p.starmap(ti.boot_sample, boot_input)

    # Getting the ROC curves
    roc_input = [(targets[boot], scores[boot]) for boot in boots]
    rocs = p.starmap(roc_curve, roc_input)

    return rocs
