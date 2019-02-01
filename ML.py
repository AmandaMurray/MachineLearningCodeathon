# from __future__ import division, print_function, unicode_literals
import pandas as pd
# Common imports
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from six.moves import urllib
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# from future_encoders import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def load_insurance_data():
    csv_path = "insurance/insurance.csv"
    return pd.read_csv(csv_path)

def main():
    #loading data
    insurance = load_insurance_data()
    insurance.head()

    #understanding the data
    print ("info")
    insurance.info()
    print ("describe")
    print(insurance.describe())

    #splitting the test set and training set
    train_set, test_set = train_test_split(insurance, test_size = 0.2, random_state = 42)
    test_set.head()

    #printing histograms
    insurance.hist(bins=50, figsize=(20,15))
    # plt.show()

    #learning the correlations
    corr_matrix = insurance.corr()
    print(corr_matrix["charges"].sort_values(ascending=False))

    #plotting scatter matrices
    attributes = ["charges", "age", "bmi", "children"]
    matrix = scatter_matrix(insurance[attributes], figsize=(12, 8))
    # plt.show()

    # there are no particular fetures we can extract

    #data cleaning
    sample_incomplete_rows = insurance[insurance.isnull().any(axis=1)].head()
    print(sample_incomplete_rows)
    #there is no missing data

    smoker_cat = insurance['smoker'].values.reshape(-1,1)
    sex_cat = insurance['sex'].values.reshape(-1,1)
    region_cat = insurance['region'].values.reshape(-1,1)

    cat_encoder = OneHotEncoder(sparse = False)

    hot_smoker_cat = cat_encoder.fit_transform(smoker_cat)
    hot_sex_cat = cat_encoder.fit_transform(sex_cat)
    hot_region_cat = cat_encoder.fit_transform(region_cat)
    # print(cat_encoder.categories_)

    # pipeline
    cat_attribs = ['smoker', 'sex', 'region']
    full_pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    insurance_prepared = full_pipeline.fit_transform(train_set)

    #applying a linear regression model
    insurance_labels = train_set["charges"].copy()
    lin_reg = LinearRegression()
    lin_reg.fit(insurance_prepared, insurance_labels)
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

    #evaluating
    some_data = train_set.iloc[:5]
    some_labels = insurance_labels.iloc[:5]
    some_data_prep = full_pipeline.transform(some_data)

    print("Predictions:", lin_reg.predict(some_data_prep))
    print ("Labels: ", list(some_labels))

    #calculating errors
    insurance_predictions = lin_reg.predict(insurance_prepared)
    lin_mse = mean_squared_error(insurance_labels, insurance_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)

    #cross validation
    lin_scores = cross_val_score(lin_reg, insurance_prepared, insurance_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)

    #Try With Random Forrest regression
    forest_reg  = RandomForestRegressor(random_state=42)
    forest_reg.fit(insurance_prepared, insurance_labels)

    #Forest cross-validation
    forest_scores = cross_val_score(forest_reg, insurance_prepared, insurance_labels, scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores)

    #perform a grid-search to select the best hyperparameters
    param_grid = [
    {'n_estimators': [1,3,10,30], 'max_features': [2,3,4,5,6,8]},
    {'bootstrap':[False], 'n_estimators': [1,3,10,30], 'max_features':[2,3,4,5,6]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring= "neg_mean_squared_error", return_train_score=True)
    grid_search.fit(insurance_prepared, insurance_labels)

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    #identify the best hyperparameters
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

    #evaluating on the test set
    final_model = grid_search.best_estimator_

    x_test = test_set.drop("charges", axis=1)
    y_test = test_set["charges"].copy()

    x_test_prepared = full_pipeline.transform(x_test)
    final_predictions = final_model.predict(x_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)

main()
