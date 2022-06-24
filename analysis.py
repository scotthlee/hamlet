import numpy as np
import pandas as pd

from importlib import reload

from hamlet.tools import multi as tm
from hamlet.tools import inference as ti


# Loading the original data for getting cutpoints based on prevalence
samp = pd.read_csv('D:/data/hamlet/samp.csv')
find_cols = [
    'infiltrate', 'reticular', 'cavity',
    'nodule', 'pleural_effusion', 'hilar_adenopathy',
    'linear_opacity', 'discrete_nodule', 'volume_loss',
    'pleural_reaction', 'other'
]
find_prob_cols = [s + '_prob' for s in find_cols]
N = samp.shape[0]
ab_p = np.round(samp.abnormal.sum() / N, 2)
abtb_p = np.round(samp.abnormal_tb.sum() / N, 2)
find_p  = np.round(samp[find_cols].sum() / N, 4)

# Loading the validation and test predictions
file_dir = 'output/'
ab_val = pd.read_csv(file_dir + 'abnormal/stats/val_probs.csv')
ab_test = pd.read_csv(file_dir + 'abnormal/stats/test_probs.csv')

abtb_val = pd.read_csv(file_dir + 'abnormal_tb/stats/val_probs.csv')
abtb_test = pd.read_csv(file_dir + 'abnormal_tb/stats/test_probs.csv')

find_val = pd.read_csv(file_dir + 'findings/stats/val_preds.csv')
find_test = pd.read_csv(file_dir + 'findings/stats/test_preds.csv')

# Getting the cutpoints
ab_cuts = ti.get_cutpoint(ab_val.abnormal, 
                          ab_val.abnormal_prob, 
                          p_adj=ab_p)
abtb_cuts = ti.get_cutpoint(abtb_val.abnormal,
                            abtb_val.abnormal_prob,
                            p_adj=abtb_p)
find_cuts = ti.get_cutpoints(find_val[find_cols].values,
                             find_val[[s + '_prob' for s in find_cols]].values,
                             column_names=find_cols,
                             p_adj=find_p)
all_cuts = {'abnormal': ab_cuts, 
            'abnormal_tb': abtb_cuts, 
            'findings': find_cuts}
pickle.dump(all_cuts, open(file_dir + 'cuts.pkl', 'wb'))

# Getting the confidence intervals
ab_j_cis = tm.boot_cis(ab_test.abnormal,
                     ab_test.abnormal_prob,
                     cutpoint=ab_cuts['j'],
                     p_adj=ab_p)
ab_ct_cis = tm.boot_cis(ab_test.abnormal,
                        ab_test.abnormal_prob,
                        cutpoint=ab_cuts['count_adj'],
                        p_adj=ab_p)
abtb_j_cis = tm.boot_cis(abtb_test.abnormal,
                         abtb_test.abnormal_prob,
                         cutpoint=abtb_cuts['j'],
                         p_adj=abtb_p)
abtb_ct_cis = tm.boot_cis(abtb_test.abnormal,
                          abtb_test.abnormal_prob,
                          cutpoint=abtb_cuts['count_adj'],
                          p_adj=abtb_p)
find_j_cis = [tm.boot_cis(find_test[c],
                          find_test[c + '_prob'],
                          p_adj=find_p.values[i],
                          cutpoint=find_cuts[c]['j'])
              for i, c in enumerate(find_cols)]
find_ct_cis = [tm.boot_cis(find_test[c],
                           find_test[c + '_prob'],
                           p_adj=find_p.values[i],
                           cutpoint=find_cuts[c]['count_adj'])
               for i, c in enumerate(find_cols)]

all_cis = [
    ab_j_cis, ab_ct_cis, abtb_j_cis,
    abtb_ct_cis, find_j_cis
]
pickle.dump(all_cis, open(file_dir + 'cis.pkl', 'wb'))