import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

from hamlet.tools.generic import check_fnames, trim_zeroes


SEED = 2022
TEST_N = 8000
ADD_NEW_ONLY = False

# Setting the directorires
base_dir = 'C:/Users/yle4/data/hamlet/'
ds_dir = base_dir + 'misc/labels/'

if ADD_NEW_ONLY:
    new_dir = base_dir + 'source/new/'
    new_files = os.listdir(new_dir)
    all_df = pd.read_csv(base_dir + 'all.csv')
    samp_df = pd.read_csv(base_dir + 'samp.csv')
    new_ids = [f[:-4] for f in new_files]
    ids, args1, args2 = np.intersect1d(all_df.id.values,
                                       new_ids,
                                       return_indices=True)
    good_df = all_df.loc[all_df.id.isin(ids)]
    good_df['split'] = 'train'
    good_df['file'] = [new_files[i] for i in args2]
    good_df['batch'] = samp_df.batch.max() + 1
    samp_df = pd.concat([samp_df, good_df], axis=0)
    samp_df.to_csv(base_dir + 'samp.csv', index=False)
    [os.rename(new_dir + f, base_dir + 'train/img/' + f)
     for f in [new_files[i] for i in args2]]

# Reading the source Excel files
panels = [pd.read_csv(ds_dir + 'panel/' + f, encoding='latin')
          for f in os.listdir(ds_dir + 'panel/')]
for df in panels:
    ids = ['pan_' + s for s in df.ID.values.astype('str')]
    df.ID = ids

immigrant = pd.read_csv(ds_dir + 'immigrant.csv', encoding='latin')
immigrant.ID = pd.Series(['im_' + s for s in immigrant.ID.astype('str')])

refugee = pd.read_csv(ds_dir + 'refugee.csv', encoding='latin')
refugee.ID = ['ref_' + s for s in refugee.ID.astype('str')]

non_iom = panels + [immigrant, refugee]
iom = pd.read_csv(ds_dir + 'iom.csv')
iom['sex'] = 'na'

# Renaming columns
col_dict = {'ID': 'id',
            'panel_sitecode': 'panel_site',
            'abnormal_img': 'abnormal',
            'DS_ChestXrayFinding': 'abnormal',
            'DS_Infiltrate': 'infiltrate',
            'DS_ReticularMarkSuggestFibrosis': 'reticular',
            'DS_CavitaryLesion': 'cavity',
            'DS_Nodule': 'nodule',
            'DS_Pleural': 'pleural_effusion',
            'DS_HilarAdenopathy': 'hilar_adenopathy',
            'DS_MiliaryFindings': 'miliary',
            'DS_DiscreteLinearOpacity': 'linear_opacity',
            'DS_DiscreteFibroticScar': 'linear_opacity',
            'DS_DiscreteNodule': 'discrete_nodule',
            'DS_DiscreteFibroticScarVolumeLoss': 'volume_loss',
            'DS_IrregularThickPleuralReaction': 'pleural_reaction',
            'DS_Other': 'other',
            'DS_SCResults_Smear1Results': 'smear_1',
            'DS_SCResults_Smear2Results': 'smear_2',
            'DS_SCResults_Smear3Results': 'smear_3',
            'DS_SCResults_Culture1Results': 'culture_1',
            'DS_SCResults_Culture2Results': 'culture_2',
            'DS_SCResults_Culture3Results': 'culture_3',
            'DS_TBClass_NoClass': 'no_class',
            'DS_TBClass_ClassA': 'class_a',
            'DS_TBClass_ClassB1Pul': 'class_b1_pulm',
            'DS_TBClass_ClassB1Extrapul': 'class_b1_extrapulm',
            'DS_TBClass_ClassB2LTBI': 'class_b2_ltbi',
            'DS_TBClass_ClassB3Contact': 'class_b3_contact',
            'DS_TBClass_ClassBOther': 'class_b_other'
}

for df in non_iom:
    df.columns = df.columns.str.replace(' ', '')
    df.rename(columns=col_dict, inplace=True)

# Pulling out columns to make a combined dataset
abn_col = ['abnormal']
demo_cols = [
    'id', 'exam_country', 'exam_date', 'birth_country',
    'date_of_birth', 'panel_site', 'sex'
]
find_cols = [
    'infiltrate', 'reticular', 'cavity',
    'nodule', 'pleural_effusion', 'hilar_adenopathy',
    'miliary', 'linear_opacity', 'discrete_nodule',
    'volume_loss', 'pleural_reaction', 'other',
]
test_cols = [
    'smear_1', 'smear_2', 'smear_3',
    'culture_1', 'culture_2', 'culture_3'
]
class_cols = [
    'class_a', 'class_b1_pulm', 'class_b1_extrapulm',
    'class_b2_ltbi', 'class_b3_contact', 'class_b_other'
]
all_cols = demo_cols + abn_col + find_cols + test_cols + class_cols

# Merging the datasets
non_iom = [df[all_cols] for df in non_iom]
non_iom = [df.iloc[:, ~df.columns.duplicated()] for df in non_iom]
non_iom = pd.concat(non_iom, axis=0)
all_df = pd.concat([iom[all_cols], non_iom], axis=0)
all_df['abnormal_tb'] = np.array(all_df[find_cols].sum(axis=1) > 0,
                                 dtype=np.uint8)

# Dropping rows without CXR readings
all_df = all_df.dropna(axis=0, subset=['abnormal'])
all_df = all_df.drop_duplicates(subset='id', keep='first')

# Making a data source variable
all_df['source'] = ''
all_df.source[['im_' in str(id) for id in all_df.id.values]] = 'immigrant'
all_df.source[['ref_' in str(id) for id in all_df.id.values]] = 'refugee'
all_df.source[['iom_' in str(id) for id in all_df.id.values]] = 'iom'
all_df.source[['pan_' in str(id) for id in all_df.id.values]] = 'panel'

# Making the first data flow table
n_tabs = pd.crosstab(all_df.source, 'n')
ab_tabs = pd.crosstab(all_df.source, all_df.abnormal)
ab_tabs['pct_ab'] = ab_tabs[1] / n_tabs.n
abtb_tabs = pd.crosstab(all_df.source, all_df.abnormal_tb)
abtb_tabs['pct_abtb'] = abtb_tabs[1] / n_tabs.n
tabs = pd.concat([n_tabs, ab_tabs, abtb_tabs], axis=1)
tabs.drop([0.0], axis=1, inplace=True)
tabs.columns = ['n', 'ab', 'pct_ab', 'abtb', 'pct_abtb']
tabs.to_csv(base_dir + 'all_ages_tab.csv')

# Dropping data from entrants under 15 years old
all_df['exam_date'] = pd.to_datetime(all_df.exam_date, errors='coerce')
all_df['date_of_birth'] = pd.to_datetime(all_df.date_of_birth, errors='coerce')
all_df.dropna(axis=0, inplace=True, subset=['exam_date', 'date_of_birth'])
ages = all_df.exam_date - all_df.date_of_birth
days = ages.dt.days.values
all_df['age_days'] = days
adults = np.where(days >= 15*365)[0]
kids = np.where(days < 15*365)[0]
all_df.iloc[kids, :].to_csv(base_dir + 'kids.csv', index=False)
all_df = all_df.iloc[adults, :].reset_index(drop=True)

# Making the source tables for adults
n_tabs = pd.crosstab(all_df.source, 'n')
ab_tabs = pd.crosstab(all_df.source, all_df.abnormal)
ab_tabs['pct_ab'] = ab_tabs[1] / n_tabs.n
abtb_tabs = pd.crosstab(all_df.source, all_df.abnormal_tb)
abtb_tabs['pct_abtb'] = abtb_tabs[1] / n_tabs.n
tabs = pd.concat([n_tabs, ab_tabs, abtb_tabs], axis=1)
tabs.drop([0.0], axis=1, inplace=True)
tabs.columns = ['n', 'ab', 'pct_ab', 'abtb', 'pct_abtb']
tabs.to_csv(base_dir + 'adults_tab.csv')

# Saving the dataset to file
all_df.to_csv(base_dir + 'all.csv', index=False)
presplit_dir = base_dir + 'presplit/'
fnames = os.listdir(presplit_dir)
short_fnames = [s[:-4] for s in fnames]
fname_dict = dict(zip(short_fnames, fnames))

# Quick check for images with no record
ids = all_df.id.values.astype('str')
no_record = np.setdiff1d(short_fnames, ids)
[os.rename(presplit_dir + fname_dict[f],
           base_dir + 'source/bad/no_record/' + f[i])
 for f in no_record]

fnames = os.listdir(presplit_dir)
short_fnames = [s[:-4] for s in fnames]

# Building the splits
has_img = np.intersect1d(ar1=all_df.id.values,
                         ar2=short_fnames,
                         return_indices=True)
fnames = [fname_dict[f] for f in has_img[0]]
short_fnames = [f[:-4] for f in fnames]
samp_df = all_df.iloc[has_img[1], :].drop_duplicates(subset='id').reset_index()
ids = samp_df.id.values.astype('str')
samp_df['file'] = [fname_dict[id] for id in ids]

# Making the flow table for adults with valid images
n_tabs = pd.crosstab(samp_df.source, 'n')
ab_tabs = pd.crosstab(samp_df.source, samp_df.abnormal)
ab_tabs['pct_ab'] = ab_tabs[1] / n_tabs.n
abtb_tabs = pd.crosstab(samp_df.source, samp_df.abnormal_tb)
abtb_tabs['pct_abtb'] = abtb_tabs[1] / n_tabs.n
tabs = pd.concat([n_tabs, ab_tabs, abtb_tabs], axis=1)
tabs.drop([0.0], axis=1, inplace=True)
tabs.columns = ['n', 'ab', 'pct_ab', 'abtb', 'pct_abtb']
tabs.to_csv(base_dir + 'valid_tab.csv')

# Specifying the panel sites to use for reference reads
sites = samp_df.panel_site.values.astype('str')
good_sites = [
              'Cho Ray', 'ASVIET1', 'Luke',
              'ASPHIL1', 'Consultorios de Visa', 'AMDOMI1',
              'AMDOMI2', 'Servicios Medicos Consulares', 'AMMEXI1',
              'Clinica Medical Internacional', 'AMMEXI2',
              'Medicos Especializados', 'AMMEXI3',
              'Servicios Medicos de la Frontera'
]
good_any = np.array([s in good_sites for s in sites], dtype=np.uint8)
panel_pos = np.array(['pan_' in s for s in ids], dtype=np.uint8)
iom_read = np.array(['iom_' in s for s in ids], dtype=np.uint8)
pref_reads = np.array(good_any + panel_pos + iom_read > 0, dtype=np.uint8)

# Setting up the reference reads for validation and testing
abtb = samp_df.abnormal_tb.values
ab = samp_df.abnormal.values
abnotb = np.array((ab == 1) & (abtb == 0), dtype=np.uint8)
samp_df['ab_kind'] = 'normal'
samp_df.ab_kind[abnotb == 1] = 'no TB'
samp_df.ab_kind[abtb == 1] = 'TB'

abn = samp_df.abnormal.values
ref_mtb = np.array(samp_df.class_a == 1, dtype=np.uint8)
ref_abn = np.array(((panel_pos == 1) | (iom_read == 1)) & (abn == 1),
                    dtype=np.uint8)
ref_nrm = np.array((good_any == 1) & (abn == 0),
                   dtype=np.uint8)

np.random.seed(SEED)
ref_abn_samp = np.random.choice(np.where(ref_abn == 1)[0],
                                 size=TEST_N,
                                 replace=False)
ref_nrm_samp = np.random.choice(np.where(ref_nrm == 1)[0],
                                 size=TEST_N,
                                 replace=False)

val_samp = np.random.choice(range(TEST_N),
                            size=int(TEST_N / 2),
                            replace=False)
test_samp = np.setdiff1d(range(TEST_N), val_samp)

val_ids = np.concatenate([ref_abn_samp[val_samp],
                          ref_nrm_samp[val_samp]])
test_ids = np.concatenate([ref_abn_samp[test_samp],
                           ref_nrm_samp[test_samp]])
abn_split = np.array(['train'] * samp_df.shape[0])
abn_split[val_ids] = 'val'
abn_split[test_ids] = 'test'

# Writing the CSV back to disk with the split info
samp_df['split'] = abn_split
samp_df.to_csv(base_dir + 'samp.csv', index=False)

# And now moving the validation and test images
split_dict = dict(zip(samp_df.id.values,
                      samp_df.split.values))

for f in samp_df.file.values:
    ds = split_dict[f[:-4]]
    path = base_dir + ds + '/img/'
    os.rename(presplit_dir + f, path + f)
