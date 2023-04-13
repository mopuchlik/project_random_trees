# -*- coding: utf-8 -*-
'''
Created on Mon Mar 14 23:32:19 2022

load and transform/impute data  

@Changelog
2023.04.13 first version

@author: MIchal Opuchlik <michal.opuchlik@gmail.com>
'''
#%% libs
from datetime import datetime
import pandas as pd
import os
import numpy as np
import time

#current working directory
os.chdir('D:/Dropbox/programowanie/projekt_microdata')
# os.chdir('/home/michal/Dropbox/programowanie/projekt_microdata')
print('Current working directory: {0}'.format(os.getcwd()))

#%% general data

gen_data = pd.read_csv('general_data.csv')

# some data info 
gen_data.info()
gen_data.dtypes
x = gen_data.describe()
print(x)

gen_data.shape

# drop some unnecessary columns
cols_1 = ['EmployeeCount', 'Over18', 'StandardHours'] 
gen_data.drop(columns=cols_1, inplace=True)

#%%  ## fixing NAs

# # TotalWorkingYears
# assumed that TotalWorkingYears = YearsAtCompany
gen_data['TotalWorkingYears'].isna().sum()

ind = gen_data['TotalWorkingYears'].isna()
ind.sum()

gen_data['TotalWorkingYears'].values[ind] = gen_data['TotalWorkingYears'].values[ind]

#%% # NumCompaniesWorked
# check before
gen_data['NumCompaniesWorked'].isna().sum()

ind_x = gen_data['NumCompaniesWorked'].isna()

# take average rate of job change (in years hence floor)
ind_rate = -ind_x & gen_data['NumCompaniesWorked'] != 0

job_change_rate = gen_data['TotalWorkingYears'][ind_rate] \
    / gen_data['NumCompaniesWorked'][ind_rate]

job_change_rate = job_change_rate.mean()

gen_data['NumCompaniesWorked'].values[ind_x] = \
    (gen_data['TotalWorkingYears'][ind_x] / job_change_rate).apply(np.floor)

#%% ##### employee survey data #################

es_data = pd.read_csv('employee_survey_data.csv')
es_data.columns

# EnvironmentSatisfaction
# JobSatisfaction
# WorkLifeBalance

es_data['EnvironmentSatisfaction'].isna().sum()
es_data['JobSatisfaction'].isna().sum()
es_data['WorkLifeBalance'].isna().sum()

ind_1 = es_data['EnvironmentSatisfaction'].isna()
ind_2 = es_data['JobSatisfaction'].isna()
ind_3 = es_data['WorkLifeBalance'].isna()

# small correlation so set NAs to neutral levels 2=Medium
# TODO one could think to get it through some regressions
es_data['EnvironmentSatisfaction'].values[ind_1] = 2
es_data['JobSatisfaction'].values[ind_2] = 2
es_data['WorkLifeBalance'].values[ind_3] = 2

#%% ##### manager_survey_data #################

ms_data = pd.read_csv('manager_survey_data.csv')
ms_data.columns

ms_data['JobInvolvement'].isna().sum()
ms_data['PerformanceRating'].isna().sum()

#%% ##### MERGE

data = pd.merge(gen_data, es_data, on = 'EmployeeID', how = 'outer')
data = pd.merge(data, ms_data, on = 'EmployeeID', how = 'outer')

# last checks
data.shape
data.isna().sum()

#%% ######## in and out_time ###################

in_time = pd.read_csv('in_time.csv')
out_time = pd.read_csv('out_time.csv')

#%%
# get diff in hours between login and logout
in_work_h = pd.DataFrame()

start_time = time.time()

for i in range(1, len(in_time.columns)):
    in_time_day = pd.to_datetime(in_time.iloc[:, i])
    out_time_day = pd.to_datetime(out_time.iloc[:, i])

    diff = out_time_day - in_time_day
    diff = diff.dt.seconds / 60**2

    in_work_h = pd.concat([in_work_h, diff], axis=1)

end_time = time.time()
time_diff = (end_time - start_time) 
print('Execution time:', time_diff, 'sec')


# %% calculate stats in rows (for each guy)

result_lst = []

start_time = time.time()

for i in range(in_work_h.shape[0]):
    x = in_work_h.iloc[i, :]

    time_max = x.max()
    time_min = x.min()
    time_mean = x.mean()
    time_quant_10 = x.quantile(0.1)
    time_quant_50 = x.quantile(0.5)
    time_quant_90 = x.quantile(0.9)

    dict_tmp = {'time_max': x.max(), 
                'time_min': x.min(), 
                'time_mean': x.mean(), 
                'time_quant_10': x.quantile(0.1),
                'time_quant_50': x.quantile(0.5),
                'time_quant_90': x.quantile(0.9)}
    result_lst.append(dict_tmp)

stats = pd.DataFrame(result_lst) 

end_time = time.time()
time_diff = (end_time - start_time) 
print('Execution time:', time_diff, 'sec')

#%% merge together
stats['EmployeeID'] = data.index + 1
data = data.merge(stats, on='EmployeeID', how='outer')

# %% write data
data.to_csv('final_data.csv', index=False)

