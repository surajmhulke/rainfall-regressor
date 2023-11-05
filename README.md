# rainfall-regressor
# Predicting Rainfall in Coastal Karnataka

## Table of Contents
- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Importing Dataset](#importing-dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development and Evaluation](#model-development-and-evaluation)
- [Conclusion](#conclusion)

## Introduction

This project focuses on predicting rainfall in Coastal Karnataka using neural network models. We leverage weather data, including various meteorological parameters, alongside carbon emission and population data. This information helps us create a predictive model for monthly rainfall.

## Importing Libraries

First, we import essential libraries for data manipulation, visualization, and model development. Key libraries include Keras, Scikit-Learn, Matplotlib, Pandas, NumPy, and more.

```python
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import warnings
from xgboost import XGBRegressor
```
Importing Dataset

We load two critical datasets for this project:

    Carbon Emission and Population Data: This dataset includes information on the year, total population, average carbon emissions per capita, and average temperature change.

    Weather Data: Weather data specifically focuses on rainfall in Coastal Karnataka. It provides monthly precipitation values, covering various years.

python

# Load carbon emission data
carb = pd.read_csv("/kaggle/input/carbon-emissions-and-population-from-1961-to-2015/carbon.csv")
# Load weather data
weather_data = pd.read_csv("/kaggle/input/rainfall-in-india/rainfall in india 1901-2015.csv")

Exploratory Data Analysis (EDA)

We explore the datasets to gain insights into the data distribution, summary statistics, and data quality. EDA helps us understand our data better and identify patterns.

python

# Data exploration
data.describe()

Feature Engineering

Feature engineering is a crucial step where we preprocess the data, combine datasets, and handle missing values. In this project, we select specific weather columns and stack them for easier model input.

python

# Stack data columns "Months" into rows
_data = weather_data[[
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'
]]

Model Development and Evaluation

For the neural network model, we use Keras to create a feedforward neural network. Model training involves compiling the model, defining the architecture, and training on the training dataset.

python

# Neural Network Model
NN_model = Sequential()
# Compile the network
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

Conclusion

This project showcases the process of predicting monthly rainfall in Coastal Karnataka using neural network models. We load and preprocess data, develop a neural network model, and train it to make predictions. You can now predict this month's and the next month's rainfall using this model.
