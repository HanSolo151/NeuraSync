import pandas as pd
import numpy as np

data_lrrk2 = pd.read_csv("D:\NeuraSync\research paper\Metabolomic_Analysis_of_LRRK2_PD_1_of_5_14Mar2025 (1).csv")

data_lrrk2['MCR'] = data_lrrk2['TESTNAME'].str.extract(r'MZ([\d.]+)').astype(float)
data_lrrk2['RT'] = data_lrrk2['TESTNAME'].str.extract(r'RT([\d.]+)').astype(float)
data_lrrk2['Charge'] = data_lrrk2['TESTNAME'].str.contains('pos').astype(int)

data_lrrk2['SEX'] = data_lrrk2['SEX'].map({'Female': 0, 'Male': 1})
data_lrrk2['COHORT'] = data_lrrk2['COHORT'].map({'Control': 0, 'Prodromal': 1, 'PD': 2})

try:
  data_lrrk2 = data_lrrk2.drop(columns=['TYPE', 'UNITS', 'RUNDATE', 'PROJECTID', 'PI_NAME', 'PI_INSTITUTION', 'PROJECTID', 'TESTNAME'])
except:
  pass

data_lrrk2

