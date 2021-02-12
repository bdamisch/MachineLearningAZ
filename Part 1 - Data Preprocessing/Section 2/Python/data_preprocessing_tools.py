# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print('split into x and y')
print(X)
print(y)
print('\n')

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
#remember that the upper bound is excluded in python, hence why it is 3 instead of 2
print('handled missing data')
print(X)
print('\n')

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print('one-hot encoded ind var, turns countries into 1 hot vectors')
print(X)
print('\n')

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print('one-hot encoded dep var')
print(y)
print('\n')

# Splitting the dataset into the Training set and Test set
# the test set is supposed to be a brand new set, don't do training with it
# feature scaling gets mean and standard deviation of the feature to perform scaling
# if feature scaling is applied before split, then it'll get the mean + std dev
# of all the values including the ones in the test set
# shouldn't have that info, it would be info leakage from test set

# feature scaling occurs after split to prevent info leakage from test set 
from sklearn.model_selection import train_test_split
# test_size -> 80% of observation in training set, 20% in test set
# splitting randomly 80% into training set and 20% in test set, random_state is the seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print('split into training and test sets')
#matrix of features of training set
print(X_train)
#matrix of features of test set
print(X_test)
#dep var of training set
print(y_train)
#dep var of test set
print(y_test)
print('\n')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
print('feature scaling')
sc = StandardScaler()
# goal of standardization/feature scaling is to have all the values of the feature
# in the same range, since the vectors are already in ranges of -/+3, there's nothing
# that we should do.
# transform the age and salary columns --> 3:
# fit == computes the mean and std of all features (age and salary)
# transform == gets all features in the same scale
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
# don't take the standard and mean for all the features for the test set
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print('x train')
print(X_train)
print('x test')
print(X_test)





