import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# total bill vs tip dataset
tips = sns.load_dataset('tips')
tips.head()
# setting a theme for the plots
sns.set_theme()
# visualize the data
sns.relplot(data=tips, x ='total_bill',y='tip',col='time',hue='smoker',style='smoker',size='size')
# load the iris dataset
iris = sns.load_dataset('iris')
iris.head()
sns.scatterplot(x='sepal_length',y='petal_length',hue='species',data=iris)
sns.scatterplot(x='sepal_length',y='petal_width',hue='species',data=iris)

# loading the titanic dataset
titanic = sns.load_dataset('titanic')
titanic.head()
titanic.shape
sns.countplot(x='class',data=titanic)
sns.countplot(x='survived',data=titanic)
sns.barplot(x='sex',y='survived',hue='class',data=titanic)

# house price dataset
from sklearn.datasets import load_boston
house_boston = load_boston()
house = pd.DataFrame(house_boston.data, columns=house_boston.feature_names)
house['PRICE'] = house_boston.target
print(house_boston)
house.head()
sns.distplot(house['PRICE'])
correlation = house.corr()
# constructing a Heat Map
plt.figure(figsize=(10,10))
sns.heatmap(, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

