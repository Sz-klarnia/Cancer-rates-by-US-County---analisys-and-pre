# basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats

# preprocessing, feature and model selection
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,MinMaxScaler,PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif,RFE, RFECV
from sklearn.pipeline import make_pipeline
import sklearn.pipeline as pipeline
from sklearn.model_selection import GridSearchCV,train_test_split,TimeSeriesSplit

#  models and metrics
import sklearn.linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn import metrics

# statistical libraries
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

# warnings
import warnings
import pprint




def find_best_algorytm(X_train,X_test,y_train,y_test,pipelines,param_grids):
    warnings.filterwarnings("ignore")
    best_score = 0
    best_model = np.NaN

    for i in range(len(pipelines)):
        grid = GridSearchCV(pipelines[i],
                           param_grid=param_grids[i],
                           cv=5,
                           refit=True,
                           verbose=1,
                           n_jobs = 3)
        grid.fit(X_train,y_train)
        scr = grid.score(X_test,y_test)
        if scr > best_score:
            best_score = scr
            best_model = grid.best_params_
        
    print(f"Best model is {best_model} with R2 score: {best_score}")
    return best_model

def fill_missing(data,col_name):
    X_filled = data.iloc[:,3:].dropna().drop(col_name,axis=1)
    y_filled = data.iloc[:,3:].dropna()[col_name]
    idx = data[data.isna()[col_name] == True].index
    X_to_fill = data.iloc[:,3:].loc[idx].drop(col_name,axis=1)

    X_train,X_test,y_train,y_test = train_test_split(X_filled,y_filled,test_size=0.2)


    lr = make_pipeline(StandardScaler(),LinearRegression())
    lr.fit(X_train,y_train)
    score = lr.score(X_test,y_test)
    
    if score > 0.6:
        to_fill = lr.predict(X_to_fill)
        series = pd.Series(to_fill, index=idx)
    
        data[col_name].fillna(series,inplace=True)
        return data
    else:
        print("R2 score to low")
        

def categorize(data,col_name,target_col_name,thresholds):
    categorized = {}
    iterations = len(thresholds)+1
    for i in range(len(thresholds)+1):
        if i == 0:
            categorized[f"Under {str(thresholds[i])}"] = data[data[col_name] <= thresholds[i]][target_col_name]
        elif i > 0 and i < len(thresholds):
            categorized[f"Over {str(thresholds[i - 1])}"] = data[(data[col_name] >thresholds[i-1])&(data[col_name]<=thresholds[i])][target_col_name]
        else:
            categorized[f"Over {str(thresholds[i - 1])}"] = data[data[col_name] > thresholds[i-1]][target_col_name]
    return categorized



def differences(data_dict, array1,array2,array3,array4):
    
    # equality of variance test
    levene = stats.levene(array1,array2,array3,array4)
    print(f"Levene test result = {levene}")
    print("----")
    if levene[1] < 0.05:
        print("Unequal variances, performing kruskall wallis test\n")
        result  = stats.kruskal(array1,array2,array3,array4)
        print(f"{result}\n")
        if result[1] < 0.05:
            #post_hoc
            print("Differences between groups, performing post hoc ttests")
            t_results = {}
    
            for category in data_dict.keys():
                for category1 in data_dict.keys():
                     if category == category1:
                        pass
                     else:
                        t_results[(category,category1)] = f"{stats.ttest_ind(data_dict[category], data_dict[category1], equal_var=False)} \n"
            pprint.pprint(t_results)
        else:
            print("No significant difference between groups")
        
    
    else:
        print("Equal variances, performing kruskall wallis test")
        result = stats.f_oneway(array1,array2,array3,array4)
        print(f"{result}\n")
        if result[1] < 0.05:
            #post_hoc
            t_results = {}
            print("Differences between groups, performing post hoc ttests")
            for category in data_dict.keys():
                for category1 in data_dict.keys():
                    if category == category1:
                        pass
                    else:
                        t_results[(category,category1)] = f"{stats.ttest_ind(data_dict[category], data_dict[category1], equal_var=True)}\n"
            pprint.pprint(t_results)
        else:
            print("No significant difference between groups")





