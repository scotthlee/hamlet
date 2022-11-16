import pandas as pd
import numpy as np

from hamlet.tools.generic import crosstab, vartab
    

# Read in the full dataset
data_dir = '~/OneDrive - CDC/Documents/projects/hamlet/'
df = pd.read_csv(data_dir + 'samp.csv')

# Making age group variables
df['age_years'] = (df.age_days / 365).round().astype(int)
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
df['tb_disease'] = np.array(smears + cultures > 0, dtype=np.uint8)

# Dividing the data into splits for making separate tables
train = df[df.split == 'train']
val = df[df.split == 'val']
test = df[df.split == 'test']
splits = [df, train, val, test]

split_tabs = []
tab_cols = [
    'abnormal', 'abnormal_tb', 'tb_disease',
    'age_group', 'sex', 'exam_region', 
    'exam_subregion'
]
tab_names = [
    'Abnormal', 'Abnormal TB', 'TB Disease', 
    'Age Group', 'Sex', 'Exam Region', 
    'Exam Sub-Region'
]
split_names = ['All', 'Train', 'Validate', 'Test']
var_levels = [np.unique(df[v]) for v in tab_cols]
for j, s in enumerate(splits):
    tabs = [vartab(df=s, 
                   var=tab_cols[i], 
                   varname=tab_names[i],
                   levels=var_levels[i],
                   use_empty=True)
            for i, c in enumerate(tab_cols)]
    tabs = pd.concat(tabs, axis=0)
    split_tabs.append(tabs)

split_tabs = pd.concat(split_tabs, axis=1)
split_tabs.to_csv(data_dir + 'analysis/table1.csv')
    