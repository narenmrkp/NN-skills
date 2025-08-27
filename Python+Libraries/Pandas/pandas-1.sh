# Pandas-Nerchuko:
------------------
Series: 1-D Array, DataFrame: 2-D Array
!pip install pandas --> import numpy as np --> import pandas as pd
    # Converting all into series
my_list = [10,20,30,40] --> arr = np.array(my_list)
pd.Series(data=my_list)
labels = ['a','b','c','d'] --> pd.Series(data = my_list, index=labels)
pd.Series(arr,labels)
d = {'a':10,'b':20,'c':30,'d':40} --> pd.Series(data=d)
pd.Series(data = labels)
ser1 = pd.Series([1,2,3,4], index=['nandu','nani','venky','jaswanth']) --> ser1['nandu'] --> ser1.values
ser2 = pd.Series([1,2,4,6], index=['nani','chakri','nandu','venky']) --> ser2['nandu'] --> ser2.values
ser1 + ser2     # chakri:NaN, jaswanth:NaN, nandu:5, nani:3, venky:9
dict1 = {'state':['Andhra Pradesh','Telangana'],'year':[2020,2021]} --> df = pd.DataFrame(dict1) --> display(df)
df = pd.DataFrame(dict1, index=['row1','row2'])
df2 = pd.DataFrame(np.random.randn(5,5)) --> display(df2)
np.random.seed(10)
rows= ['row1','row2','row3','row4','row5']
col = ['col1','col2','col3','col4','col5']
df2 = pd.DataFrame(np.random.randn(5,5), index = rows, columns=col ) --> display(df2)
df2['col1'] --> df2[['col1','col2']]
df2['new'] = df2['col1'] + df2['col2']
df2.drop('new',axis=1)      # it removes column 'new' from this only, not from df2 dataframe, so we need to use inplace=True
df2.drop('new', axis=1, inplace=True)       # it removes column of 'new'
df2.drop('row1', axis=0, inplace=True)      # it removes row of 'row1'
df2.loc['row2'] --> df2.loc[:,'col2'] --> df2.iloc[1] --> df2.iloc[:,1:] --> df2.loc[:,['col3','col4']]
df = pd.DataFrame({'A':[1,2,np.nan],'B':[6,np.nan,np.nan],'C':[1,2,3]}) --> display(df)
df.isna().sum() --> df.dropna() --> df.dropna(axis=1)
df['A'].fillna(value = df['A'].mean())  # fillup by mean/median/mode
data = {'Company:['GOOG','GOOG','MSFT','MSFT','FB','FB'],'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],'Sales':[200,120,340,124,243,350]}
df = pd.DataFrame(data)
by_comp = df.groupby('Company') --> by_comp.mean()
by_comp.std() --> by_comp.min() --> by_comp.max() --> by_comp.count()
bycomp.describe()
df1 = pd.Dataframe({'A':['A0','A1','A2','A3'],'B':['B0','B1','B2','B3'],
                    'C':['C0','C1','C2','C3'],'D':['D0','D1','D2','D3']},index=[0,1,2,3])
df2 = pd.Dataframe({'A':['A4','A5','A6','A7'],'B':['B4','B5','B6','B7'],
                    'C':['C4','C5','C6','C7'],'D':['D4','D5','D6','D7']},index=[4,5,6,7])
df3 = pd.Dataframe({'A':['A8','A9','A10','A11'],'B':['B8','B9','B10','B11'],
                    'C':['C8','C9','C10','C11'],'D':['D8','D9','D10','D11']},index=[8,9,10,11])
pd.concat([df1,df2,df3])
pd.concat([df1,df2,df3],axis=1)
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[44,55,66,55],'col3':['a','b','c','d']})
df.head() --> df.tail()
df['col2'].unique() --> df['col2'].nunique()
df['col2'].value_counts()
def square(x):
    return x*x
df['col1'] = df['col1'].apply(square) --> df['col3'].apply(len) --> df['col1'].sum() --> del df['col3']
df.columns --> df.index
