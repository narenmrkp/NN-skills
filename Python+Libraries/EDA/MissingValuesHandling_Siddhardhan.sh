# Methods to Handle Missing Values:   1. Imputation      2. Dropping
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# loading the dataset to a Pandas DataFrame
dataset = pd.read_csv('/content/Placement_Dataset.csv')
dataset.head()
dataset.shape
dataset.isnull().sum()
# analyse the distribution of data in the salary
fig, ax = plt.subplots(figsize=(8,8))
sns.distplot(dataset.salary)

# Replace the missing values with Median value
dataset['salary'].fillna(dataset['salary'].median(),inplace=True)
dataset.isnull().sum()
# filling missing values with Mean value:
# dataset['salary'].fillna(dataset['salary'].mean(),inplace=True)
# filling missing values with Mean value:
# dataset['salary'].fillna(dataset['salary'].mode(),inplace=True)

# Dropping Method
salary_dataset = pd.read_csv('/content/Placement_Dataset.csv')
salary_dataset.shape
salary_dataset.isnull().sum()
# drop the missing values
salary_dataset = salary_dataset.dropna(how='any')
salary_dataset.isnull().sum()
salary_dataset.shape
