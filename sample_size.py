import pandas as pd
import numpy as np
import scipy as sp

import hamlet.tools.inference as ti


def sample(sens, spec, p, N, n_samples=1000):
    prevs = sp.stats.binom(N, p).rvs(n_samples) / N
    samps = []
    for prev in prevs:
        pn = 1 - prev
        tp, fn = sens * prev, (1 - sens) * prev
        tn, fp = spec * pn, (1 - spec) * pn
        probs = np.array([tn, fp, fn, tp])
        rv = sp.stats.multinomial(N, probs)
        samps.append(rv.rvs())
    return np.array(samps).reshape(n_samples, 4)


# Setting the basic parameters
alpha = .05
sens = .80
spec = .80
sizes = np.arange(100, 15000, 100)
ab_p = .12
abtb_p = .07

# Generating the samples
prevs = [ab_p, abtb_p]
prev_samps = []
for prev in prevs:
    samps = []
    for size in sizes:
        samp = sample(sens, spec, prev, size)
        samp_sens = samp[:, 3] / samp[:, [2, 3]].sum(1)
        samp_spec = samp[:, 0] / samp[:, [0, 1]].sum(1)
        samp_count = np.diff(samp[:, [1, 2]]) / samp[:, [2, 3]].sum(1)
        samps.append([samp_sens, samp_spec, samp_count])
    prev_samps.append(samps)

# Getting the quantiles
prev_quants = []
for ps in prev_samps:
    sens_quants = np.array([np.quantile(s[0], [.025, .975]) for s in ps])
    spec_quants = np.array([np.quantile(s[1], [.025, .975]) for s in ps])
    count_quants = np.array([np.quantile(s[2], [.025, .975]) for s in ps])
    prev_quants.append([sens_quants, spec_quants, count_quants])
    
# Getting the MOEs
size_doub = np.vstack([sizes, sizes]).T
prev_moes = []
for pq in prev_quants:
    sens_moes = np.diff(pq[0]) / 2
    spec_moes = np.diff(pq[1]) / 2
    count_moes = np.diff(pq[2]) / 2
    prev_moes.append([sens_moes, spec_moes, count_moes])

# Writing the MOEs to csv
moe_cols = ['sensitivity', 'specificity', 'count_diff']
out_dir = '/Users/scottlee/OneDrive - CDC/Documents/projects/hamlet/analysis/'
ab_moes = pd.DataFrame(np.concatenate(prev_moes[0], 1),
                       columns=moe_cols)
ab_moes['n'] = sizes
ab_moes['n_pos'] = sizes * ab_p
ab_moes.to_csv(out_dir + 'ab_sample_size.csv', index=False)

abtb_moes = pd.DataFrame(np.concatenate(prev_moes[1], 1),
                         columns=moe_cols)
abtb_moes['n'] = sizes
abtb_moes['n_pos'] = sizes * abtb_p
abtb_moes.to_csv(out_dir + 'abtb_sample_size.csv', index=False)

