    # Basics (Pandas)
import pandas as pd
import numpy as np
# Create a dummy DataFrame
data = {
    'ID': range(1, 11),
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Helen', 'Ivy', 'Jack'],
    'Age': np.random.randint(20, 40, size=10),
    'Salary': np.random.randint(30000, 80000, size=10),
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'Marketing', 'HR', 'IT', 'Finance']
}
df = pd.DataFrame(data)
df
# View top 5 rows
df.head()
# View last 5 rows 
df.tail()
# View basic info
print("\nInfo:")
df.info()
# Summary statistics
print("\nDescribe:\n")
df.describe()
# Column selection
print("\nAges:\n", df['Age'])
# Row filtering
print("\nEmployees with Age > 30:\n", df[df['Age'] > 30])
# Sorting
print("\nSorted by Salary (descending):\n", df.sort_values(by='Salary', ascending=False))
# Grouping
print("\nAverage Salary by Department:\n", df.groupby('Department')['Salary'].mean())
# Rename column
df.rename( columns={'Salary': 'Annual_Salary'}, inplace=True)
print("\nRenamed 'Salary' to 'Annual_Salary':\n")
df
# Drop column
df.drop('Department', axis=1, inplace=True)
print("\nAfter Dropping 'Bonus':\n")
df
# Null check
print("\nAny missing values?\n", df.isnull().sum())
# Check Duplicated
df.duplicated().sum()
---------------------------------------------------------------------------------
Video-2 (Pandas):
# empty sereis
import pandas as pd 
import numpy as np
pd.Series()
# creating series from list 
age = [10,20, 30, 40, 50]
age_ser = pd.Series(age)
age_ser
# creating sereis from dictionary 
name_dict = {
    'a': 'Noor',
    'b': 'Jhon',
    'c': 'Sara'
}
pd.Series(name_dict)
# creating sereis from scaler values
pd.Series(0, index=[1,2,3,4])
# creating series with numpy functions 
print("With linspace :\n ", pd.Series(np.linspace(0,100,5)))
print("\n with arange :\n", pd.Series(np.arange(10)))
print("\n with array :\n",pd.Series(np.array([0,2,3,4])))
    # Working with DataFrames
letters = ['a','b','c','d','e']
data = pd.DataFrame(data=letters, columns=['letters'])
data
mydict = {
    'name':['noor','raza','shazia'],
    'Age':[10,20,30],
    'gender':['male','male','female']
}
df = pd.DataFrame(mydict)
df
# selecting columns
df['name']
# filtering 
df[df['Age'] > 20]
-------------------------------------------------------------------------------------------------------------
Video-3 (Pandas)
    # Working with Databases
!pip install mysql-connector-python
import mysql.connector 
import pandas as pd 
# make connection
con = mysql.connector.connect(host='localhost',user='root',password='',database='dummy_db')
df = pd.read_sql_query('SELECT * FROM mytabel', con)
    # Working with Json
df = pd.read_json("lec 17, dataset2.json")
    # Working with APIs
import pandas as pd
import requests
response = requests.get('https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page=1')
response
temp_df = pd.DataFrame(response.json()['results'])[['id','title','overview','release_date','popularity','vote_average','vote_count']]
temp_df.head()
    # CSV Files with Parameters
df = pd.read_csv("lec 17, dataset.csv", nrows=100, usecols=['Name','Sex','Age'])
df
# encoding (utf-8, latin1)
df = pd.read_csv('lec 17, dataset3.csv', encoding='utf-8')
df
# read data in chunks 
dfs = pd.read_csv('lec 17, dataset.csv',chunksize=50)
for chunk in dfs:
    print(chunk.shape)
-----------------------------------------------------------------------------------------------------------
EDA Analysis:
import pandas as pd
df = pd.read_csv('lec 17, dataset.csv')
df.head()
# 1 check shape of your dataset 
df.shape
# 2 explore missing values in dataset 
df.isnull().sum()
# 3 fill missing values with fillna method 
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Cabin'].fillna(df['Cabin'].mode(),inplace=True)
# 4. check duplicated values 
df.duplicated().sum()
# 5. explore basic statistics
df.describe()
# 6. dataset info
df.info()
    # Line Plot
df.plot(y='Age', title='Age Distribution')
    # bar Plot
df['Embarked'].value_counts().plot(kind='bar',title='Passenger by city',color='green',figsize=(10,7))
    # Bar (Horizontal Bar Plot)
df['Parch'].value_counts().plot(kind='barh',title='with family', color='red',figsize=(8,5))
    # Histogram
df['Age'].plot(kind='hist', bins=20, title='Histogram of Age', color='skyblue')
    # Box Plot
df[['Age','Fare']].plot(kind='box',figsize=(10,8))
    # Pie Plot
df['Survived'].value_counts().plot(kind='pie',autopct='%1.1f%%',title='Survived Rate',startangle=90,figsize=(8,6))
    # Area Plot
df[['Age','Fare']].head(20).plot(kind='area',alpha=0.5,title='Area Plot')
    # KDE Density Plot
df['Age'].plot(kind='kde', title='Age Density Plot')
    # Scatter Plot
df.plot(kind='scatter', x='Age', y='Fare', title='Fare vs Age')
    # Plot with Groupby Method
df.groupby('Pclass')['Survived'].mean().plot(kind='bar', title='Survival Rate by Class')
    # Plot with Condition
df[df['Sex'] == 'male']['Age'].plot(kind='kde', label='Male', legend=True)
df[df['Sex'] == 'female']['Age'].plot(kind='kde', label='Female', legend=True, title='Age Density by Gender')
-------------------------------------------------------------------------------------------------------------------------------------
DataSelection/Filtering:
import pandas as pd
df = pd.read_csv('lec 17, dataset.csv')
df.head()
# get any sinlge column
sex_values = df['Sex'] --> sex_values
# fetch more than on column (subset of data)
sub_df = df[['Sex','Age','Fare']] --> sub_df
# get male passenger data (age, sex, fare,survival)
df[df['Sex'] == 'female'][['Age','Sex','Fare','Survived']]
# get older pessanger data ((age, sex, fare,survival))
df[df['Age'] >=60][['Age','Sex','Fare','Survived']]
# get pessanger (only male) (age: between 25 and 30) and (Fare:between: 50 and 150)
df[(df['Sex']=='male') & (df['Age'] >=25) & (df['Age'] <=30) & (df['Fare'] >=50) & (df['Fare'] <=150)]
# get first row 
df.loc[0]
# get first row but only age value
df.loc[0,'Age']
# get first 4 rows, only age, sex, fare
df.loc[:3, ['Age','Sex','Fare']]
# select data based on label range
df.loc[50:80, ['Age','Fare']]
# frist 3 rows and column 
df.iloc[:3,:3]
# 10 to 15 rows and sepcific 3 (Name Sex Age) columns
df.iloc[10:16,[3,4,5]]
df.iloc[[100,150,200],[1,4,5,11]]       # 100th, 150th, 200th rows only with columns of 1st,4th,5th,11th
df.iloc[-3:,[-5,-1]]                    # Last 3 rows, specific columns of -5, -1 positions
df.at[2,'Name']                         # 2nd row and specific column name
df.iat[2,3]                             # 2nd row and column position(3rd)
# filter pessangers with 25, 30 and 40 ages
df[df['Age'].isin([25, 30, 40])][['Age', 'Name', 'Sex']]
# multiple fitering 
# filter pessangers with 25, 30 ages and only male and only those passenger that was traveling towards Q city
df[(df['Age'].isin([25,30])) & (df['Sex'].isin(['male']) & (df['Embarked'].isin(['S'])))][['Age','Sex','Name','Embarked']]
# find out passenger Maguire, Mr. John Edward
df[df['Name'].str.contains('Maguire, Mr. John Edward')]
# Fetch female that was traveling towards C city
df[(df['Sex'].str.contains('female')) & (df['Embarked'].str.contains('C'))][['Name','Sex','Embarked']]
# get only 5 to 10 years passanger
df[df['Age'].between(5,10)][['Name','Age','Sex']]
# get only those female that has two families 
df[(df['Sex'].str.contains('female') & (df['Parch'].between(1,2)))][['Age','Sex','Parch']]
# Passenger who has between 25,30 age, went towards C,only Female,
df[(df['Age'].between(25,30)) & (df['Embarked'].str.contains('C')) & (df['Sex'].isin(['female']))][['Age','Sex','Embarked']]
    # query Method
# Passenger who has between 25,30 age, went towards C,only Female,
df.query("Age.between(25,30) and Embarked.str.contains('C') and Sex.isin(['female'])")[['Sex','Age','Embarked']]
    # loc[] and Lambda
# get only passengers with 2 siblings and spouse
df.loc[lambda x: x['SibSp']==2][['Age','Sex','SibSp']]
# with isin (only ages of 1,2,3)
df.loc[lambda x: x['Age'].isin([1,2,3])][['Name','Age']]
# with between (only bw 50 and 60)
df.loc[lambda x: x['Age'].between(50,60)][['Age','Sex']]
# with contain
df.loc[lambda x: x['Name'].str.contains('Peacock, Miss. Treasteall')]
---------------------------------------------------------------------------------------------------------------
GroupBy Method
1. How many passengers survived in each class?
df.groupby("Pclass")['Survived'].sum()
2. What is the average age of passengers by gender?
df.groupby('Sex')['Age'].mean()
3. How many passengers are there per embarkation port?
df.groupby('Embarked')['PassengerId'].count()
4. What is the average fare paid by class and gender?
df.groupby(['Pclass','Sex'])['Fare'].mean()
5. How many siblings/spouses aboard per class?
df.groupby('Pclass')['SibSp'].sum()
6. What is the maximum fare in each passenger class?
df.groupby('Pclass')['Fare'].max()
7 What is the mean and std of age by survival?
df.groupby('Survived')['Age'].agg(['mean', 'std'])
8. Find the oldest and youngest passengers in each passenger class?
df.groupby('Pclass')['Age'].agg(['max','min'])
9. Calculate total fare and average age for each embarked port?
df.groupby('Embarked').agg({'Fare':'sum', 'Age':'mean'})
10. Find the Most Common Embarked Port for each Passenger Class?
df.groupby("Pclass")['Embarked'].apply(lambda x: x.mode().iloc[0])
11. Calculate the Age Range (Max - Min) for each Title (Mr, Mrs, etc.):
df['title'] = df['Name'].str.extract('([A-Za-z]+)\.')
df.groupby('title')['Age'].agg(lambda x: x.max() - x.min())
12. Group Passengers by Age Range and Calculate the Survival Rate:
# Group passengers by age range and calculate survival rate
age_bins = [0, 18, 30, 50, 100]
age_labels = ['0-18', '19-30', '31-50', '51+']
df['AgeRange'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
df.groupby('AgeRange')['Survived'].mean() * 100
13. Find the Passengers with the Highest Fare in each Age Group:
# Create age bins and labels
age_bins = [0, 18, 30, 50, 100]
age_labels = ['0-18', '19-30', '31-50', '51+'] 
df['agegroup'] = pd.cut(df['Age'],bins=age_bins,labels=age_labels)
df['agegroup']
df.loc[df.groupby('agegroup', observed=True)['Fare'].idxmax().dropna()][['Name', 'Fare']]
14. Find the Most Common Ticket Number for each Passenger Name :
# Find the most common ticket number for each passenger name prefix
df.groupby('Name')['Ticket'].apply(lambda x: x.mode().iloc[0])
