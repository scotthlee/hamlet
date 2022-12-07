"""Generic support functions, mostly for preprocessing"""

import numpy as np
import pandas as pd
import scipy as sp
import os

from multiprocessing import Pool
from matplotlib import pyplot as plt


def check_fname(fname, ids):
    """Checks a list of IDs for a single file name."""
    return fname in ids


def check_fnames(fnames, ids):
    """Checks a list of IDs for a list of file names."""
    with Pool() as p:
        input = [(f, ids) for f in fnames]
        res = p.starmap(check_fname, input)
        p.close()
        p.join()

    return np.array(res)


def trim_zeroes(fname):
    """Removes extra 0s from panel file names."""
    cut = np.where([s == '_' for s in fname])[0][1] + 1
    ending = fname[cut:]
    if len(ending) == 6:
        return fname
    else:
        drop = len(ending) - 6
        ending = ending[drop:]
        return fname[:cut] + ending


def is_file(fn):
    """Checks whether a string is a filename."""
    if '.' in fn:
        extension = fn[fn.index('.'):]
        return True, extension
    else:
        return False, _


def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


# Converts a boot_cis['cis'] object to a single row
def merge_cis(df, stats, round=4):
    df = deepcopy(df)
    for stat in stats:
        lower = stat + '.lower'
        upper = stat + '.upper'
        new = stat + '.ci'
        l = df[lower].values.round(round)
        u = df[upper].values.round(round)
        strs = [
            pd.Series('(' + str(l[i]) + ', ' + str(u[i]) + ')')
            for i in range(df.shape[0])
        ]
        df[new] = pd.concat(strs, axis=0)
        df = df.drop([lower, upper], axis=1)
    return df


def write_stats(stats,
                outcome,
                stats_dir='output/analysis/'):
    stats_filename = outcome + '_stats.csv'
    if stats_filename in os.listdir(stats_dir):
        stats_df = pd.read_csv(stats_dir + stats_filename)
        stats_df = pd.concat([stats_df, stats], axis=0)
        stats_df.to_csv(stats_dir + stats_filename, index=False)
    else:
        stats.to_csv(stats_dir + stats_filename, index=False)
    return


def write_preds(preds,
                outcome,
                mod_name,
                probs=None,
                test_idx=None,
                cohort_prefix='',
                output_dir='output/',
                stats_folder='analysis/'):
    stats_dir = output_dir + stats_folder
    preds_filename = outcome + '_preds.csv'
    if preds_filename in os.listdir(stats_dir):
        preds_df = pd.read_csv(stats_dir + preds_filename)
    else:
        assert test_idx is not None
        preds_df = pd.read_csv(output_dir + cohort_prefix + 'cohort.csv')
        preds_df = preds_df.iloc[test_idx, :]

    preds_df[mod_name + '_pred'] = preds
    if probs is not None:
        if len(probs.shape) > 1:
            probs = np.max(probs, axis=1)
        preds_df[mod_name + '_prob'] = probs
    preds_df.to_csv(stats_dir + preds_filename, index=False)
    return


def merge_cis(c, round=4, mod_name=''):
    str_cis = c.round(round).astype(str)
    str_paste = pd.DataFrame(str_cis.stat + ' (' + str_cis.lower +
                                 ', ' + str_cis.upper + ')',
                                 columns=[mod_name]).transpose()
    return str_paste


def merge_ci_list(l, mod_names=None, round=4):
    if type(l[0] != type(pd.DataFrame())):
        l = [c.cis for c in l]
    if mod_names is not None:
        merged_cis = [merge_cis(l[i], round, mod_names[i])
                      for i in range(len(l))]
    else:
        merged_cis = [merge_cis(c, round=round) for c in l]

    return pd.concat(merged_cis, axis=0)


def crosstab(df, var, levels=None, col='N'):
    if levels is None:
        levels = np.unique(df[var])
    counts = [np.sum([x == l for x in df[var]]) for l in levels]
    out = pd.DataFrame(counts, columns=[col], index=levels)
    return out


def vartab(df, var,
           varname=None,
           levels=None,
           col='N',
           percent=True,
           round=0,
           use_empty=False):
    if varname is None:
        varname = var
    out = crosstab(df, var, col=col, levels=levels)
    if percent:
        if (levels is not None) and (len(levels) == 1):
            percents = (out.N / df[var].shape[0]) * 100
        else:
            percents = (out.N / out.N.sum()) * 100
        out['%'] = np.round(percents, round)
        if round == 0:
            out['%'] = out['%'].astype(int)
    if use_empty:
        empty = pd.DataFrame([''], columns=[col], index=[''])
        out = pd.concat([out, empty], axis=0)
    var_index = [varname] + [''] * (out.shape[0] - 1)
    out = out.set_index([var_index, out.index.values.astype(str)])
    return out
