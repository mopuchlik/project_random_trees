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
from sklearn.metrics import roc_curve, accuracy_score, mean_squared_error, confusion_matrix

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
import os

# custom diagnostics
import modules.diags as diags

#%% paths 

path_main = 'D:/Dropbox/programowanie/projekt_microdata/projekt_random_trees'
# path_main = '/home/michal/Dropbox/programowanie/projekt_microdata/projekt_random_trees'
# path_main = 'C:/Users/michal/Dropbox/programowanie/projekt_microdata/projekt_random_trees' 

path_input = path_main + '/data_output'

# %% load data

data = pd.read_csv(path_input + '/final_data.csv', 
                   sep = ',', 
                   decimal = '.')


data.head(10)

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

    onehot_data = OneHotEncoder(sparse=False)
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

#%% model xgboost ####################################################

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

#%% model GradientBoosting ####################################################
params = {
    'learning_rate': 0.2,
    'max_depth': 6,
    'n_estimators': 30,
    # 'metric': 'auc',
    'subsample': 0.78,
    # 'colsample_bytree': 0.94,
    # 'min_samples_leaf': 300,
    # use_label_encoder:False
}

model_gbc = GradientBoostingClassifier(**params)
model_gbc = model_gbc.fit(X_train.values, y_train.values)

#%% choose model to analyze

# model_curr = model_xgb
model_curr = model_gbc


# %% diagnostics: importance table
imp = diags.calc_importance(model = model_curr, variables = vars)

plt.figure(figsize=(15, 15))
plt.scatter(x=imp['impo'], y=imp['vars'])
plt.show()

# %% diagnostics: Gini train
train = X_train.copy()
train['pr'] = model_curr.predict_proba(X_train)[:, 0]

test = X_test.copy()
test['pr'] = model_curr.predict_proba(X_test)[:, 0]

gini_train = diags.calc_gini(prediction=train['pr'].values, realization=y_train)
gini_test = diags.calc_gini(prediction=test['pr'].values, realization=y_test)

print(gini_train)
print(gini_test)

# %% diagnostics: bootstrapped Gini (with quantiles) for test
gini_boot_train = diags.calc_gini_boot(model=model_curr,
                                       data=X_train,
                                       realization=y_train, 
                                       iters=100)
print(np.quantile(gini_boot_train, 0.05))
print(gini_train)
print(np.quantile(gini_boot_train, 0.95), '\n')

gini_boot_test = diags.calc_gini_boot(model=model_curr,
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

# plot_single_tree(model_curr, X_train, 1)
# plt.show()

xgb.plot_tree(model_curr)
plt.show()

fig, ax = plt.subplots(figsize=(50, 50))
xgb.plot_tree(model_curr, num_trees=90, ax=ax)
# plt.show()
plt.savefig("single_tree.pdf")
plt.close()

# %% diagnostics: plot learning curve

# TODO check what is wrong with the warnings
import warnings
warnings.filterwarnings("ignore")

learn = diags.learning_curves(model_curr, prepared_data[vars], prepared_data['Attrition'], 50)

plt.plot(np.sqrt(learn['train_errors']), 'r+', linewidth=2, label='train_errors')
plt.plot(np.sqrt(learn['val_errors']), 'b-', linewidth=3, label='val_errors')
plt.legend()
plt.show()

# %% cross-validation

print(cross_val_score(model_curr, X_train, y_train, cv=3, scoring="accuracy", verbose=0))
print(cross_val_score(model_curr, X_train, y_train, cv=3, scoring="recall", verbose=0))
print(cross_val_score(model_curr, X_train, y_train, cv=3, scoring="roc_auc", verbose=0))

# confusion matrix
y_train_pred = cross_val_predict(model_curr, X_train, y_train, cv=3)
confusion_matrix(y_train, y_train_pred) / len(y_train) * 100

# %%
