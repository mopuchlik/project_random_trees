# -*- coding: utf-8 -*-
'''
Created on Fri Feb 18 06:45:15 2022

diagnostic functions for decision trees

@Changelog
2023.04.13 first version

TODO load all diagnostic functions as a module

@author: mich
'''

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split

def calc_importance(model, variables):
    '''
    Calculate importances in a given tree model.

    retrieve sorted importance table based on model (must be fitted) for given variables
    and return a sorted descending list.

    :model model object (tested for xgboost)
    :variables list of variables
    :returns list of importances 
    '''
    importance_table = pd.DataFrame(
        {'impo': model.feature_importances_, 'vars': variables})
    importance_table.sort_values(['impo'], ascending=False, inplace=True)

    return importance_table

def calc_gini(realization, prediction):
    '''
    Calculate Gini coeff based on realization and prediction.

    :realization realized values
    :prediction predicted values
    :returns gini coefficient
    '''
    gini = -1 * (2 * roc_auc_score(realization, prediction) - 1)

    return gini

def calc_gini_boot(model, data, realization, iters):
    '''
    Calculate bootstrapped Gini coeff based on realization and prediction.

    :model model object (tested for xgboost)
    :data input data
    :realization realized values
    :iters number of boostrap iterations
    :returns list of randomized Gini coefficients based on random samples from data
    '''
    result = []
    for x in range(1, iters):
        id_w = np.random.choice(range(0, data.shape[0]), 
                                size=data.shape[0],
                                replace=True).astype('int')

        data_temp = data.iloc[id_w, :]
        prob_temp = model.predict_proba(data_temp)[:, 0].astype('float')
        realization_temp = realization.iloc[id_w].values

        gini_test_tmp = calc_gini(realization=realization_temp,
                                  prediction=prob_temp)

        result.append(gini_test_tmp)

    return result

def learning_curves(model, X, y, step):
    '''
    Calculate learning curve.

    :model model object (tested for xgboost)
    :X input data
    :y realized values
    :step step of increase of dataset
    :returns trains and validation errors as available data window is inceasing by step
    '''

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    print(f'Len: {len(X_train)}')

    train_errors, val_errors = [], []
    for m in range(1, len(X_train), step):  # 3528 podzielnosc bez reszty 42, 49, 36, 24, 12 etc
        if m <= len(X_train):
            model.fit(X_train[:m], y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
            val_errors.append(mean_squared_error(y_val_predict, y_val))
        else: break    
    result = {'train_errors': train_errors, 
              'val_errors': val_errors}
    result = pd.DataFrame(result)

    return result