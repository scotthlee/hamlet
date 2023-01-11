import numpy as np
import pandas as pd
import pickle

from importlib import reload
from sklearn.metrics import roc_auc_score as auroc

from hamlet.tools import multi as tm
from hamlet.tools import inference as ti
from hamlet.tools import metrics


# Setting the directory and columns of interest
data_dir = '/Users/scottlee/OneDrive - CDC/Documents/projects/hamlet/'
samp = pd.read_csv(data_dir + 'samp.csv')
find_cols = [
    'infiltrate', 'reticular', 'cavity',
    'nodule', 'pleural_effusion', 'hilar_adenopathy',
    'linear_opacity', 'discrete_nodule', 'volume_loss',
    'pleural_reaction', 'other'
]
find_prob_cols = [s + '_prob' for s in find_cols]

# Loading the validation and test files
val = samp[samp.split == 'val'].reset_index(drop=True).sort_values('id')
val_probs = pd.read_csv(data_dir + 'val_predictions.csv')
val = pd.merge(val, val_probs, on=['id'])

test = samp[samp.split == 'test'].reset_index(drop=True).sort_values('id')
test_probs = pd.read_csv(data_dir + 'test_predictions.csv')
test = pd.merge(test, test_probs, on=['id'])

# Loading the external datasets;
nih = pd.read_csv(data_dir + 'output/other/nih.csv')
shen = pd.read_csv(data_dir + 'output/other/shen.csv')
mcu = pd.read_csv(data_dir + 'output/other/mcu.csv')
viet = pd.read_csv(data_dir + 'output/other/viet.csv')

ext_dfs = [nih, shen, mcu, viet]
ext_names = ['nih', 'shenzhen', 'mcu', 'vietnam']

# Getting the baeline prevalence for the different outcomes, excluding
# images gathered specifically for the study (i.e., only using images
# gathered under the screening program's normal operating conditions)
all_df = pd.read_csv(data_dir + 'all.csv')
all_df = all_df[[s in ['immigrant', 'refugee'] for s in all_df.source]]
N = all_df.shape[0]
ab_p = np.round(all_df.abnormal.sum() / N, 2)
abtb_p = np.round(all_df.abnormal_tb.sum() / N, 2)
find_p  = np.round(all_df[find_cols].sum() / N, 4)
all_df = []

# Getting the cutpoints for HaMLET
ab_cuts = ti.get_cutpoint(val.abnormal,
                          val.abnormal_prob,
                          p_adj=ab_p)
abtb_cuts = ti.get_cutpoint(val.abnormal_tb,
                            val.abnormal_tb_prob,
                            p_adj=abtb_p)
find_cuts = ti.get_cutpoints(val[find_cols].values,
                             val[find_prob_cols].values,
                             column_names=find_cols,
                             p_adj=find_p)
all_cuts = {'abnormal': ab_cuts,
            'abnormal_tb': abtb_cuts,
            'findings': find_cuts}

# Getting the confidence intervals for our data
abtb_j_cis = tm.boot_cis(test.abnormal_tb,
                         test.abnormal_tb_prob,
                         cutpoint=abtb_cuts['j'],
                         p_adj=abtb_p)
abtb_ct_cis = tm.boot_cis(test.abnormal_tb,
                          test.abnormal_tb_prob,
                          cutpoint=abtb_cuts['count_adj'],
                          p_adj=abtb_p)
find_j_cis = [tm.boot_cis(test[c].fillna(0),
                          test[c + '_prob'],
                          p_adj=find_p.values[i],
                          cutpoint=find_cuts[c]['j'])
              for i, c in enumerate(find_cols)]
find_ct_cis = [tm.boot_cis(test[c].fillna(0),
                           test[c + '_prob'],
                           p_adj=find_p.values[i],
                           cutpoint=find_cuts[c]['count_adj'])
               for i, c in enumerate(find_cols)]
ham_cis = [abtb_j_cis, abtb_ct_cis, find_j_cis, find_ct_cis]
pickle.dump(ham_cis, open(data_dir + 'ham_cis.pkl', 'wb'))

# Cutoffs for the external datasets
nih_cuts = ti.get_cutpoint(nih.abnormal,
                           nih.abnormal_prob)
viet_ab_cuts = ti.get_cutpoint(viet.abnormal,
                               viet.abnormal_prob)
viet_abtb_cuts = ti.get_cutpoint(viet.abnormal_tb,
                                 viet.abnormal_tb_prob)
shen_cuts = ti.get_cutpoint(shen.abnormal,
                            shen.abnormal_prob)
mcu_cuts = ti.get_cutpoint(mcu.abnormal,
                           mcu.abnormal_prob)

# And their intervals
nih_cis = tm.boot_cis(nih.abnormal,
                      nih.abnormal_prob,
                      cutpoint=nih_cuts['j'])
viet_ab_cis = tm.boot_cis(viet.abnormal,
                          viet.abnormal_prob,
                          cutpoint=viet_ab_cuts['j'])
viet_abtb_cis = tm.boot_cis(viet.abnormal_tb,
                            viet.abnormal_tb_prob,
                            cutpoint=viet_abtb_cuts['j'])
shen_cis = tm.boot_cis(shen.abnormal,
                       shen.abnormal_tb_prob,
                       cutpoint=shen_cuts['j'])
mcu_cis = tm.boot_cis(mcu.abnormal,
                      mcu.abnormal_tb_prob,
                      cutpoint=mcu_cuts['j'])
ext_cis = [nih_cis, viet_ab_cis, viet_abtb_cis,
           shen_cis, mcu_cis]
pickle.dump(ext_cis, open(data_dir + 'ext_cis.pkl', 'wb'))
