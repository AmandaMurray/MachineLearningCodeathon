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

def load_happy_data():
    csv_path = "world-happiness-report/2016.csv"
    return pd.read_csv(csv_path)

def main():
    #loading data
    happy = load_happy_data()
    happy.head()

    #understanding the data
    print ("info")
    happy.info()
    print ("describe")
    print(happy.describe())

    #splitting the test set and training set
    train_set, test_set = train_test_split(happy, test_size = 0.2, random_state = 42)
    test_set.head()

    #printing histograms
    happy.hist(bins=50, figsize=(20,15))
    # plt.show()

    #learning the correlations
    corr_matrix = happy.corr()
    print(corr_matrix["Happiness Score"].sort_values(ascending=False))

    #plotting scatter matrices
    attributes = ["Happiness Score", "Economy (GDP per Capita)", "Health (Life Expectancy)", "Family", "Freedom", "Generosity", "Trust (Government Corruption)", "Dystopia Residual"]
    matrix = scatter_matrix(happy[attributes], figsize=(12, 8))
    # plt.show()

main()
