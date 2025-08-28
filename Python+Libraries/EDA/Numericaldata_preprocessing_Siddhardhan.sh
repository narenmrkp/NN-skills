import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# loading the data from csv file to a pandas dataframe
diabetes_data = pd.read_csv('/content/diabetes.csv')
# first 5 rows of the dataframe
diabetes_data.head()
# number of rows & columns
diabetes_data.shape
diabetes_data.describe()
X = diabetes_data.drop(columns='Outcome', axis =1)
Y = diabetes_data['Outcome']
print(X)
print(Y)
# 0 --> Non - Diabetic, 1 --> Diabetic
scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)
print(standardized_data)
X = standardized_data
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
