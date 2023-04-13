# -*- coding: utf-8 -*-
'''
Created on Fri Feb 18 06:45:15 2022

implementation of decision tree model (xgboost for a time being)

@Changelog
2023.04.13 first version

@author: mich
'''

# %% libs

from sklearn import tree
import seaborn as sns
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, accuracy_score, mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
import os

# %% load data

# # desktop
cwd = 'D:/Dropbox/programowanie/projekt_microdata/'
data = pd.read_csv(cwd + '/final_data.csv', 
                   sep = ',', 
                   decimal = '.')

# # laptop
# cwd = 'D:/Dropbox/programowanie/projekt_microdata/'
# data = pd.read_csv(cwd + '/final_data.csv',
#                    sep=',',
#                    decimal='.',)

data.head(10)

os.chdir(cwd)
print("Current working directory: {0}".format(os.getcwd()))

#%% change Attrition to 0-1
data['Attrition'].replace(('Yes', 'No'), (1, 0), inplace=True)

# %% categorical dummies for string-based variables in dtypes

data.shape
data.dtypes

vars = data.columns[~data.columns.isin(['Attrition'])]
vars = list(vars)
vars = [
    'Age',
    'DistanceFromHome',
    'Education',
    'WorkLifeBalance',
    'JobInvolvement',
    'PerformanceRating',
    'time_max',
    'time_min',
    'time_mean',
    'time_quant_10',
    'time_quant_50',
    'time_quant_90',
    'EnvironmentSatisfaction',
    'StockOptionLevel',
    'PercentSalaryHike',
]

prepared_data = data.copy()
# check 
prepared_data['Attrition'].isna().sum()
for var in ['BusinessTravel',
            'Department',
            'EducationField',
            'Gender',
            'JobRole',
            'MaritalStatus',]:

    onehot_data = OneHotEncoder(sparse_output=False)
    onehot_data = onehot_data.fit_transform(np.array(data[var]).reshape(-1, 1))
    in_data = pd.DataFrame(onehot_data)
    in_data.columns = [var + str(x) for x in in_data.columns]
    prepared_data = pd.DataFrame.join(prepared_data, in_data)
    vars = vars + list(in_data.columns)

# prepared_data
# prepared_data.shape

# %% split to train and test

X_train, X_test, y_train, y_test = train_test_split(
    prepared_data[vars], prepared_data['Attrition'], test_size=0.2, random_state=1)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

# %% nice pairwise plots
var_to_plot = [
    'Age',
    # 'DistanceFromHome',
    # 'PerformanceRating',
    # 'Education', 'WorkLifeBalance', 'JobInvolvement',
    # 'PerformanceRating',
    # 'time_max','time_min','time_mean',
    # 'time_quant_10', 'time_quant_50',
    'time_quant_90',
    # 'EnvironmentSatisfaction', 'StockOptionLevel',
    'PercentSalaryHike',
]
# whole dataset
data_to_compare = data[var_to_plot]

sns.set(style='ticks', color_codes=True)
g = sns.pairplot(data_to_compare)
plt.show()

# train vs test
g = sns.pairplot(X_train[var_to_plot])
plt.show()
g = sns.pairplot(X_test[var_to_plot])
plt.show()

#%% model xgboost

# params
params = {
    'learning_rate': 0.06,
    'max_depth': 4,
    'n_estimators': 900,
    # 'metric': 'auc',
    'subsample': 0.78,
    'colsample_bytree': 0.94,
    # 'min_samples_leaf': 300,
    # use_label_encoder:False
}

model_xgb = xgb.XGBRFClassifier(**params)
model_xgb = model_xgb.fit(X_train.values, y_train.values)
# %% diagnostics: importance table
imp = calc_importance(model = model_xgb, variables = vars)

plt.figure(figsize=(15, 15))
plt.scatter(x=imp['impo'], y=imp['vars'])
plt.show()

# %% diagnostics: Gini train
train = X_train.copy()
train['pr'] = model_xgb.predict_proba(X_train)[:, 0]

test = X_test.copy()
test['pr'] = model_xgb.predict_proba(X_test)[:, 0]

gini_train = calc_gini(prediction=train['pr'].values, realization=y_train)
gini_test = calc_gini(prediction=test['pr'].values, realization=y_test)

print(gini_train)
print(gini_test)

# %% diagnostics: bootstrapped Gini (with quantiles) for test
gini_boot_train = calc_gini_boot(model=model_xgb,
                                 data=X_train,
                                 realization=y_train, 
                                 iters=100)
print(np.quantile(gini_boot_train, 0.05))
print(gini_train)
print(np.quantile(gini_boot_train, 0.95))

gini_boot_test = calc_gini_boot(model=model_xgb,
                                data=X_test,
                                realization=y_test, 
                                iters=100)
print(np.quantile(gini_boot_test, 0.05))
print(gini_test)
print(np.quantile(gini_boot_test, 0.95))

# %% diagnostics: plot single tree
# TODO generalize for xgboost and scikit

# def plot_single_tree(model, data, which_tree):
#     tree_to_plot = model.estimators_[which_tree - 1]
#     tree.plot_tree(
#         tree_to_plot,
#         feature_names=list(data.columns.values),
#         class_names=[0, 1],
#         filled=True,)

# plot_single_tree(model_xgb, X_train, 1)
# plt.show()

xgb.plot_tree(model_xgb)
plt.show()

fig, ax = plt.subplots(figsize=(50, 50))
xgb.plot_tree(model, num_trees=90, ax=ax)
# plt.show()
plt.savefig("single_tree.pdf")
plt.close()

# %% diagnostics: plot learnign curve

# TODO check what is wrong with the warnings
import warnings
warnings.filterwarnings("ignore")

learn = learning_curves(model_xgb, prepared_data[vars], prepared_data['Attrition'], 500)

plt.plot(np.sqrt(learn['train_errors']), 'r+', linewidth=2, label='train_errors')
plt.plot(np.sqrt(learn['val_errors']), 'b-', linewidth=3, label='val_errors')
plt.legend()
plt.show()

# %%
