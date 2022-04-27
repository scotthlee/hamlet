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


def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    
    return total_size
