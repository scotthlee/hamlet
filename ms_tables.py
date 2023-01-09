import pandas as pd
import numpy as np

from hamlet.tools.generic import crosstab, vartab


# Read in the full dataset
data_dir = '~/OneDrive - CDC/Documents/projects/hamlet/'
df = pd.read_csv(data_dir + 'samp.csv')

# Making age group variables
df['age_group'] = pd.cut(df.age_years,
                         bins=[15, 25, 35, 45, 55, 65, 99],
                         labels=['15-24', '25-34', '35-44',
                                 '45-54', '55-64', '>=65'])
df['age_group'] = df.age_group.astype(str)

# Making geographic region and subregion variables
countries = pd.read_csv(data_dir + 'countries.csv')
countries['upper_name'] = countries.name.str.upper()
region_dict = dict(zip(countries.upper_name, countries.region))
sub_dict = dict(zip(countries.upper_name, countries['sub-region']))

region_dict.update({'nan': 'NA'})
sub_dict.update({'nan': 'NA'})

exams = df.exam_country.str.upper().astype(str)
exams.replace('VIETNAM', 'VIET NAM', inplace=True)
exams.replace('RUSSIA', 'RUSSIAN FEDERATION', inplace=True)
exams.replace('BOSNIA AND HERCEGOVINA',
              'BOSNIA AND HERZEGOVINA',
              inplace=True)
exams.replace('UNITED KINGDOM',
              'UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND',
              inplace=True)
exams.replace('MOLDOVA', 'MOLDOVA, REPUBLIC OF', inplace=True)
exams.replace('WEST BANK', 'PALESTINE, STATE OF', inplace=True)
exams.replace('TAIWAN', 'TAIWAN, PROVINCE OF CHINA', inplace=True)
exams.replace('CONGO, DEMOCRATIC REP OF',
               'CONGO, DEMOCRATIC REPUBLIC OF THE',
               inplace=True)
exams.replace('IVORY COAST', "CÔTE D'IVOIRE", inplace=True)
exams.replace('CAPE VERDE', 'CABO VERDE', inplace=True)
exams.replace('SLOVAK REPUBLIC', 'SLOVAKIA', inplace=True)
exams.replace('KOSOVO', 'SERBIA', inplace=True)
exams.replace('TANZANIA, UNITED REP OF',
              'TANZANIA, UNITED REPUBLIC OF',
              inplace=True)
exams.replace('LAOS',
              "LAO PEOPLE'S DEMOCRATIC REPUBLIC",
              inplace=True)
exams.replace('GAMBIA, THE', 'GAMBIA', inplace=True)
exams.replace('BOLIVIA', 'BOLIVIA (PLURINATIONAL STATE OF)', inplace=True)
exams.replace('CONGO, REPUBLIC OF', 'CONGO', inplace=True)
exams = exams.values
df['exam_region'] = [region_dict[c] for c in exams]
df['exam_subregion'] = [sub_dict[c] for c in exams]

# Making summary variables for TB testing
smears = df[['smear_1', 'smear_2', 'smear_3']].sum(1) > 0
df['smear'] = smears.astype(np.uint8)
cultures = df[['culture_1', 'culture_2', 'culture_3']].sum(1) > 0
df['culture'] = cultures.astype(np.uint8)
df['tb_disease'] = df.class_a.fillna(0)

# Adding a variable for the data source
source = np.array([''] * df.shape[0], dtype='U16')
ids = df.id.values.astype(str)
source[np.where(['iom_' in s for s in ids])[0]] = 'IOM Telerad QC'
source[np.where(['ref_' in s for s in ids])[0]] = 'MiMOSA'
source[np.where(['pan_' in s for s in ids])[0]] = 'Panels'
source[np.where(['im_' in s for s in ids])[0]] = 'eMedical'
df['source'] = source

# Dividing the data into splits for making separate tables
train = df[df.split == 'train']
val = df[df.split == 'val']
test = df[df.split == 'test']
splits = [df, train, val, test]

# Making the tables, starting with demographics first
dem_tabs = []
tab_cols = [
    'age_group', 'sex', 'source',
    'exam_region', 'exam_subregion'
]
tab_names = [
    'Age Group', 'Sex', 'Data Source',
    'Exam Region', 'Exam Sub-Region'
]
split_names = ['All', 'Train', 'Validate', 'Test']
var_levels = [np.unique(df[v]) for v in tab_cols]
for j, s in enumerate(splits):
    tabs = []
    for i, c in enumerate(tab_cols):
        tab = vartab(df=s,
                      var=tab_cols[i],
                      varname=tab_names[i],
                      levels=var_levels[i],
                      use_empty=True)
        tabs.append(tab)
    tabs = pd.concat(tabs, axis=0)
    dem_tabs.append(tabs)

dem_tabs = pd.concat(dem_tabs, axis=1)
dem_tabs.to_csv(data_dir + 'analysis/dem_tabs.csv')

# And now the TB-related table
tb_tabs = []
tab_cols = [
    'abnormal', 'abnormal_tb', 'tb_disease',
    'infiltrate', 'reticular', 'cavity',
    'nodule', 'pleural_effusion', 'hilar_adenopathy',
    'miliary', 'linear_opacity', 'discrete_nodule',
    'volume_loss', 'pleural_reaction', 'other'
]
tab_names = [
    'Abnormal',
    'Abnormal (Suggestive of TB)',
    'TB Disease',
    'Infiltrate or Consolidation',
    'Reticular Findings',
    'Cavitary Lesion',
    'Nodule or Mass with Poorly Defined Margins',
    'Pleural Effusion', 'Hilar/mediastinal Adenopathy',
    'Miliary Findings', 'Discrete Linear Opacity',
    'Discerete Nodule(s) Without Calcification',
    'Volume Loss or Retraction', 'Irregular Thick Pleural Reaction',
    'Other'
]
var_levels = [np.unique(df[v]) for v in tab_cols]
for j, s in enumerate(splits):
    tabs = [vartab(df=s,
                   var=tab_cols[i],
                   varname=tab_names[i],
                   levels=[1],
                   round=1)
            for i, c in enumerate(tab_cols)]
    tabs = pd.concat(tabs, axis=0)
    tb_tabs.append(tabs)

tb_tabs = pd.concat(tb_tabs, axis=1)
tb_tabs.to_csv(data_dir + 'analysis/tb_tabs.csv')


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
