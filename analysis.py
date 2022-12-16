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

# Getting the cutpoints
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

# Getting the confidence intervals
ab_j_cis = tm.boot_cis(test.abnormal,
                       test.abnormal_prob,
                       cutpoint=ab_cuts['j'],
                       p_adj=ab_p)
ab_ct_cis = tm.boot_cis(test.abnormal,
                        test.abnormal_prob,
                        cutpoint=ab_cuts['count_adj'],
                        p_adj=ab_p)
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
all_cis = [
    ab_j_cis, ab_ct_cis, abtb_j_cis,
    abtb_ct_cis, find_j_cis, find_ct_cis
]
pickle.dump(all_cis, open(data_dir + 'cis.pkl', 'wb'))

# Loading the external datasets;
# TO DO: refactor generate_predictions so it can write to an existing 
# set of labels.
nih = pd.read_csv(data_dir + 'output/other/nih.csv')
shen = pd.read_csv(data_dir + 'output/other/shen.csv')
mcu = pd.read_csv(data_dir + 'output/other/mcu.csv')
viet = pd.read_csv(data_dir + 'output/other/viet.csv')

ext_dfs = [nih, shen, mcu, viet]
ext_names = ['nih', 'shenzhen', 'mcu', 'vietnam']

# Making a table of AUCs for the external datasets
na_str = np.nan
ab_ab = [
    auroc(nih.abnormal, nih.abnormal_prob),
    na_str,
    na_str,
    auroc(viet.abnormal, viet.abnormal_prob)
]
ab_abtb = [
    auroc(nih.abnormal, nih.abnormal_tb_prob),
    na_str,
    na_str,
    auroc(viet.abnormal, viet.abnormal_tb_prob)
]
abtb_ab = [
    na_str,
    auroc(shen.abnormal, shen.abnormal_prob),
    auroc(mcu.abnormal, mcu.abnormal_prob),
    auroc(viet.abnormal_tb, viet.abnormal_prob)
]
abtb_abtb = [
    na_str,
    auroc(shen.abnormal, shen.abnormal_tb_prob),
    auroc(mcu.abnormal, mcu.abnormal_tb_prob),
    auroc(viet.abnormal_tb, viet.abnormal_tb_prob)
]
auc_tab = pd.DataFrame([ab_ab, ab_abtb, abtb_ab, abtb_abtb],
                       index=['ab/ab', 'ab/abtb',
                                'abtb/ab', 'abtb/abtb'],
                       columns=ext_names).transpose()
auc_tab = (auc_tab * 100).round(2)
auc_tab.to_csv(data_dir + 'analysis/tables/ext_aucs.csv')

# Making a table of specs at 70% sens for the external TB datasets
abtb_ab = pd.concat([
    metrics.spec_at_sens(test.abnormal_tb, test.abnormal_prob),
    metrics.spec_at_sens(shen.abnormal, shen.abnormal_prob),
    metrics.spec_at_sens(mcu.abnormal, mcu.abnormal_prob),
    metrics.spec_at_sens(viet.abnormal_tb, viet.abnormal_prob)
]).reset_index(drop=True)
abtb_ab.index = ['hamlet'] + ext_names[1:]
abtb_abtb = pd.concat([
    metrics.spec_at_sens(test.abnormal_tb, test.abnormal_tb_prob),
    metrics.spec_at_sens(shen.abnormal, shen.abnormal_tb_prob),
    metrics.spec_at_sens(mcu.abnormal, mcu.abnormal_tb_prob),
    metrics.spec_at_sens(viet.abnormal_tb, viet.abnormal_tb_prob)
]).reset_index(drop=True)
abtb_abtb.index = ['hamlet'] + ext_names[1:]
sens90 = pd.concat([abtb_ab, abtb_abtb], axis=1)
sens90.to_csv(data_dir + 'analysis/tables/sens90.csv')
