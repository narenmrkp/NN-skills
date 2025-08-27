Akshit Madan: Youtuber (Build with Akshit)
--------------------------------------- Start of Pandas -------------------------------------------------------------
import pandas as pd --> pd.__version__
lst = [1,2,3,4,5] --> print(lst)
series = pd.Series(lst) --> print(series) --> print(type(series))
empty = pd.Series([]) --> empty     # here by default dtype is float-64
a = pd.Series(['p','q','r','s','t'], index = [10,11,12,13,14]) --> a
a = pd.Series(['p','q','r','s','t'], index = [10,11,12,13,14], name = 'alphabets') --> a
scalar_series = pd.Series(0.5) --> scalar_series    # output: 0.5
scalar_series = pd.Series(0.5, index = [1,2,3]) --> scalar_series   # output: 0.5 0.5 0.5 with indexes 1 2 3 vertically print
dict_series = pd.Series({'p':1, 'q':2, 'r':3, 's':4, 't':5}) --> dict_series --> dict_series[0:3]
max(dict_series)
dict_series1 = pd.Series({'p':[1,5,6], 'q':[2,6,7], 'r':[3,9,0], 's':[4,4,5], 't':[5,1,2]}) --> dict_series1
df = pd.DataFrame() --> print(df) --> display(df)
lst = [1,2,3,4,5] --> df = pd.DataFrame(lst) --> df
lst1 = [[1,2,3,4,5],[11,12,13,14,15]] --> df = pd.DataFrame(lst1) --> display(df)
a = [{'a':5,'b':7,'c':9,'d':2}, {'a':4,'b':8,'c':19,'d':12}] --> df = pd.DataFrame(a)
b = {'Roll_No':pd.Series([1,2,3,4,5]), 'Maths':pd.Series([67,89,23,90,56]), 'Physics':pd.Series([12,98,44,90,78])}
df = pd.DataFrame(b) --> display(df)
df = pd.read_csv(r'C:\Users\narian\Desktop\datasets\Salary_Data.csv') --> display(df) --> type(df)
df.columns --> df.shape --> df.size --> df.head() --> df.head(2) --> df.tail() --> df.tail(8)
df.describe() --> df.info()
df2 = pd.read_csv(r'C:\Users\narian\Desktop\datasets\Restaurant.csv') --> display(df2) --> df2.head()
df2.shape --> df2.info() --> df2.describe()
df = pd.read_csv(r'C:\Users\narian\Desktop\datasets\sample.csv') --> df.head()
df.isnull().sum() --> df.isnull().sum().sum() --> df2 = df.dropna()
df3 = df.dropna(axis=1)     # it deletes all Null values of all columns
df.dropna(how = 'any')      # if any row value is Null then remove that row
df.dropna(how = 'all')      # if entire row values is Null then remove that row
df.dropna(inplace = True)   # it replaces original df
df.fillna(0) --> df.fillna(30)
df.fillna( {'Physics':'Absent', 'Chemistry':0, 'Maths':'NA'})
df.fillna(method='ffill')   # it replaces with previous row value
df.fillna(method='ffill', axis = 1)   # here Null value replaces with previous Column value
df.fillna(method='bfill')   # here Null value replaces with Next row value
df.fillna(method='bfill', inplace=True)   # here Null value replaces with Next row value, but df gets changed permanently, original df gets affected.
df['Physics'].fillna( value=df['Physics'].mean())   # here Null value replaces with Mean value
df.replace(to_replace=26, value=30) --> df.replace(34,10000)
df.replace(to_replace=[50,51,52,53,54,55,56,57,58,59], value='A')
df.replace(to_replace=[50,51,52,53], value=['A','B','C','D'])
df['Physics'].replace(to_replace=[50,51,52,53], value=['A','B','C','D'],inplace=True)
df.replace('[A-Za-z]',0, regex=True)
df.replace(to_replace=15,method='ffill') --> df.replace(to_replace=15,method='bfill')
df = pd.read_csv('C:/Users/narian/Desktop/datasets/sample2.csv', index_col=['Roll No']) --> df.head()
df.loc[1] --> df.loc[5] --> df.loc[5,6,7,8] --> df.loc[5,'Physics'] --> df.loc[5:15,'Chemistry']
df.loc[df['Physics']<50] --> df.loc[df['Physics']>80] --> df.loc[df['Physics']>80, ['Maths']]
df.iloc[0] --> df.iloc[[0,1,2]] --> df.iloc[:,0] --> df.iloc[:,1] --> df.iloc[0:5,1] --> df.iloc[0:5,1:4]
branch_group = df.groupby(by='Branch') --> branch_group --> branch_group.groups
df.groupby(by = ['Branch','Section']) --> branch_group.groups
for group, data_frame in branch_group:
    print(group)
    print(data_frame)
df1 = pd.DataFrame({'Roll No':[1,2,3,4,5],'Physics':[34,67,34,89,12]}) --> df1
df2 = pd.DataFrame({'Roll No':[1,2,3,4,5],'Chemistry':[78,33,39,81,90]}) --> df2
pd.merge(df1,df2, on='Roll No') --> pd.merge(df1,df2)
df3 = pd.DataFrame({'Roll No':[1,2,3,6,7],'Physics':[34,67,34,89,12]}) --> df3
df4 = pd.DataFrame({'Roll No':[1,2,3,8,9],'Chemistry':[34,67,34,89,12]}) --> df4
pd.merge(df3,df4) --> pd.merge(df3,df4,how='left') --> pd.merge(df3,df4,how='Right') --> pd.merge(df3,df4,how='outer')
------------------------------------ End of Pandas -------------------------------------------------------------------------------------
------------------------------------ Start of Numpy ------------------------------------------------------------------------------
################################################################
import numpy as np
arr1 = np.array([10,20,30,40,50]) --> print(arr1) --> print(type(arr1))
arr2 = np.array([10,20,30],[40,50,60],[70,80,90]) --> print(arr2) --> print(type(arr2))
print(arr1[0:2]) --> print(arr1[-1:]) --> print(arr1[:3])
print(arr2[0:2,0:2,0:2]) --> print(arr2[-1:,-1:,-1:]) --> print(arr2[:3,:3,:3])
arr3 = np.array( [[10,20,30,40],[50,60,70,80]])
print(arr3[0,1:3])  # output: [20,30]
print(arr3[1,1:3])  # output: [60,70]
print(arr2[2,1:2])  # output: [80,90]
print(np.shape(arr3))   # output: (2,4) Rows-2, Columns-4
###############################################################################
Akshit Madan: Youtuber (Build with Akshit):
--------------------------------------------
import numpy as np
lst = [1,2,3,4,5] --> print(lst)    # python
a = np.array([1,2,3,4,5]) --> print(a) --> type(a)
b = np.array([1,2,3,4,5],[6,7,8,9,10]) --> print(b) --> type(b)
c = np.array([1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]) --> print(c)
print(a.size) --> print(b.size) --> print(c.size)
print(a.shape) --> print(b.shape) --> print(c.shape)
print(a.dtype) --> print(b.dtype) --> print(c.dtype)
c.transpose()       # it changes Rows into Columns and Columns into Rows
np.empty((4,4),dtype=int)   # it creates randomly 4 rows x 4 Columns
x = np.ones(6)  # output: array([1., 1., 1., 1., 1., 1.])
y = np.ones((3,5)) or z = np.ones((3,5), dtype = int)  # it creates 3 Rows x 5 Columns all are 1's only
y = np.zeros(3,6) or z = np.zeros((3,6), dtype = int)   # it creates 3 Rows x 6 Columns all are 0's only
z = np.ones((3,5), dtype = str)
z = np.ones((3,5), dtype = bool)    # all are 'True'
z = np.zeros((3,5), dtype = bool)   # all are 'False'
a = np.arange(1,20) --> print(a)    # [1 2 3 4 5......19]
a = np.arange(1,20,2) --> print(a)  # [1 3 5 7 9....19]
a = np.arange(2,20,2) --> print(a)  # [2 4 6 8 10....18]
a = a.reshape((3,3))    # it converts above 1-D array into 3-D array
a = a.flatten()         # it again converts n-D array into 1-D array
a = a.ravel()           # it also converts n-D array into 1-D array
Ravel: It is only ref of original array, if we modify here, original also will get modified, Its Faster & does not occupy memory, library-level function
Flatten: Returns copy of original array, Even if we modify this, original not get changed, It is slower & occupies memory, It is a method of n-D array object.

a = np.arange(1,51) --> a = a.reshape(10,5) --> a[0] --> a[2] --> a[0,0] --> a[3,4]
a[2:5]  # it prints from 2nd to 4th rows
a[0:10] or a[:10] # it prints all rows from 0 to 9
a[:, 2]     # it prints all elements of 2nd index column [columns: 0,1,2,3....]
a[2:5, 4]   # it prints rows(2nd index row to 4th index row) elements of 4th index column array([15,20,25])
a[:,:]      # it prints all rows and all columns
a[:, 2:5]   # it prints all rows elements from 2nd index column to 4th index column
a[:, 2:5].dtype

---------------------------------- End of Numpy --------------------------------------------------------------------------------------
---------------------------------- Start of Matplot ----------------------------------------------------------------------------------
Build with Akshit:
--------------------
    # Scatter plots
import matplotlib.pyplot as plt --> import numpy as np --> import pandas as pd
plt.style.use('dark_background')
rollno = [1,2,3,4,5,6,7,8,9,10]
marks = [10,20,30,40,50,60,70,80,90,100]
plt.scatter(rollno, marks) --> plt.show()
plt.scatter(rollno, marks, color='green') --> plt.show()
plt.scatter(rollno, marks, color='green', marker='*') --> plt.show()
plt.figure(figsize=(12,8)) --> plt.scatter(rollno, marks, color='blue', marker='*') --> plt.show()
plt.figure(figsize=(8,8)) --> plt.plot(rollno, marks, 'bo', markersize=20) --> plt.show()       # bo --> blue,circle, gv--> green,v-shape
temp_pune = [25,34,21,45,28,6,43,18,7,2]
humd_pune = [28,25,29,20,26,50,19,29,52,55]
temp_bangalore = [34,35,36,37,28,27,26,25,31,20]
humd_bangalore = [40,38,36,35,42,44,41,40,34,45]
plt.figure(figsize=(8,8)) --> plt.plot(temp_pune,humd_pune,'ro',markersize=15) --> plt.show()
plt.figure(figsize=(8,8)) --> plt.xticks(np.arange(0,60,5)) --> plt.yticks(np.arange(10,60,5))
plt.plot(temp_pune,humd_pune,'ro',markersize=15) --> plt.xlabel('Temperature')
plt.plot(temp_bangalore,humd_bangalore,'bo',markersize=15) --> plt.xlabel('Temperature') --> plt.ylabel('Humidity') --> plt.show()

df = pd.read_csv('/content/IRIS.csv') --> df.head()
plt.scatter(df['sepal_length'],df['petal_length']) --> plt.show()
plt.plot(df['sepal_width'],df['petal_width'], 'go') --> plt.show()
plt.figure(figsize=(8,8)) --> plt.xticks(np.arange(1,10,0.5)) --> plt.yticks(np.arange(1,10,0.5))
plt.plot(df['sepal_length'],df['petal_length'], 'ro', alpha=0.5, markersize=8) --> plt.xlabel('Sepal Length') --> plt.ylabel('Petal Length') --> plt.show()
    # Line Plots
import matplotlib.pyplot as plt --> import numpy as np --> import pandas as pd
plt.style.use('dark_background')
rollno = [1,2,3,4,5,6,7,8,9,10]
marks = [10,20,30,40,50,60,70,80,90,100]
plt.plot(rollno, marks, 'r-') --> plt.show()
plt.plot(rollno, marks, linestyle='-') --> plt.show()
plt.plot(rollno, marks, linestyle='--', color='#728569') --> plt.show()
plt.plot(rollno, marks, linestyle=':', color='orange') --> plt.show()
plt.plot(rollno, marks, linestyle='-.', color='orange') --> plt.show()
plt.plot(rollno, marks, linestyle=':', linewidth=10) --> plt.show()
study_hours = [2,3,4,4,,5,6,7,7,8,9,9,10,11,11,12]
marks = [6,10,15,20,34,44,55,60,55,67,70,80,90,99,100]
plt.figure(figsize=(8,8)) --> plt.xticks(np.arange(0,15,1)) --> plt.yticks(np.arange(0,100,5))
plt.plot(study_hours,marks,'r-') --> plt.xlabel('Study Hours') --> plt.ylabel('Marks') --> plt.show()
plt.plot(study_hours,marks,'r-') --> plt.plot(study_hours,marks,'bo') --> plt.xlabel('Study Hours') --> plt.ylabel('Marks') --> plt.show()
    # Bar Plots
import matplotlib.pyplot as plt --> import numpy as np --> import pandas as pd
plt.style.use('dark_background')
subjects = ['Maths','English','Science','Social Studies','Computer']
marks = [89,90,45,78,99]
plt.bar(subjects,marks, color='green') --> plt.show()
colors = ['red','blue','green','orange','purple'] --> plt.bar(subjects, marks, color=colors,width=0.6,edgecolor='white',linewidth=4,linestyle='--') --> plt.show()
plt.barh(subjects, marks, color=colors) --> plt.show()
subjects = ['Maths','English','Science','Social Studies','Computer']
marks1 = [89,90,45,78,99] --> marks2 = [78,56,34,90,12]
plt.figure(figsize=(8,8))
plt.bar(subjects,marks1) --> plt.bar(subjects,marks2) --> plt.xlabel('Subjects') --> plt.ylabel('Marks') --> plt.show()
subjects_len = np.arange(len(subjects))
width = 0.4
plt.figure(figsize=(8,8))
plt.bar(subjects_len, marks1, width=width) --> plt.bar(subjects_len + width, marks2, width=width) --> plt.xlabel('Subjects') --> plt.ylabel('Marks') --> plt.show()
plt.bar(subjects_len, marks1, width=width, color=colors) --> plt.bar(subjects_len + width, marks2, width=width, color=colors, alpha=0.5) --> plt.xlabel('Subjects') --> plt.ylabel('Marks') --> plt.show()
df = pd.read_csv('content/supermarket.csv') --> df.head()
payment_df = pd.DataFrame(df['Payment'].value_counts()) --> display(payment_df)
colors = ['red','blue','green']
plt.bar(payment_df.index, payment_df['Payment'], color=colors) --> plt.show()
    # Hist Plots
import matplotlib.pyplot as plt --> import numpy as np --> import pandas as pd
plt.style.use('dark_background')
marks_50_students = np.random.randint(0,100,(50)) --> marks_50_students
plt.hist(marks_50_students) --> plt.show()
bins = np.arange(0,100,5)
plt.figure(figsize=(6,6)) --> plt.hist(marks_50_students, bins=bins, color='orange') --> plt.xticks(np.arange(0,100,5)) --> plt.show()
plt.figure(figsize=(6,6)) --> plt.hist(marks_50_students, bins=bins, color='orange', orientation='horizontal') --> plt.yticks(np.arange(0,100,5)) --> plt.show()
plt.figure(figsize=(6,6)) --> plt.hist(marks_50_students, bins=bins, color='orange', rwidth=0.6) --> plt.xticks(np.arange(0,100,5)) --> plt.show()
plt.figure(figsize=(6,6)) --> plt.hist(marks_50_students, bins=bins, color='orange', histtype='step') --> plt.xticks(np.arange(0,100,5)) --> plt.show()
marks_50_students1 = np.random.randint(0,100,(50)) --> marks_50_students2 = np.random.randint(0,100,(50))
bins = np.arange(0,100,5)
plt.figure(figsize=(6,6))
plt.hist([ marks_50_students1, marks_50_students2], bins=bins, color=['orange','blue'])
plt.xticks(np.arange(0,100,5)) --> plt.xlabel('Marks') --> plt.ylabel('Frequency') --> plt.title('Marks of Students of 2 Classes') --> plt.show()
    # Pie Plots
import matplotlib.pyplot as plt --> import numpy as np --> import pandas as pd
plt.style.use('dark_background')
subjects = ['Physics','Chemistry','Maths','English','Computers']
marks = [89,90,45,23,95]
plt.pie(marks, labels=classes) --. plt.show()
colors = ['red','blue','green','#9803fc','#03c2fc'] --> plt.pie(marks, labels=classes, colors=colors) --. plt.show()
explode_values=[0.1,0.2,0,0,0] --> textprops={'fontsize':14, 'color':'k'} --> wedgeprops={'linewidth':3,'linestyle':'--','edgecolor':'white'}
plt.pie(marks, labels=classes, colors=colors, autopct='%0.2f%%', explode=explode_values, shadow=True) --> plt.show()
plt.pie(marks, labels=classes, colors=colors, autopct='%0.2f%%', explode=explode_values, radius=1.6, textprops = textprops, wedgeprops=wedgeprops)
plt.title('Subjects and Avg Marks') --> plt.legend() --> plt.show()
df = pd.read_csv('/content/supermarket.csv') --> df.head()
payment_df = pd.DataFrame(df['Payment'].value_counts()) --> payment_df
plt.pie(payment_df['Payment'], labels=payment_df.index,colors=['red','blue','green'], autopct='%0.2f%%') --> plt.show()
    # Subplots
















----------------------------------------- End of Matplot --------------------------------------------------------------------------------------------------------------------
----------------------------------------- Start of Seaborn -------------------------------------------------------------------------------------------------
Build with Akshit:
--------------------
import seabron as sns --> import numpy as np --> import pandas as pd --> import matplotlib.pyplot as plt
roll_no = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
marks = [23,45,67,89,56,34,21,45,67,32,67,76,33,21,45]
sample_df = pd.DataFrame({"RollNo": roll_no, "Marks": marks})
sample_df.head()
    # Line Plot
sns.lineplot(x='RollNo', y='Marks', data=sample_df) --> plt.title('Student Marks')
seaborn_df = sns.load_dataset('planets') --> seaborn_df.head()
df = pd.read_scv('/content/hr_data.csv') --> df.head()
sns.lineplot(x='number_project', y='average_monthly_hours', data=df)
sns.lineplot(x='promotion_last_5years', y='left', data=df)
plt.figure(figsize=(12,6)) --> sns.lineplot(x='department',y='left',data=df)
plt.figure(figsize=(12,6)) --> sns.lineplot(x='number_project',y='average_monthly_hours',data=df, hue='left',style='department',palette='flare')
    # Dist Plot
sns.distplot(df['time_spend_company'])
sns.distplot(df['left'])
sns.distplot(df['average_monthly_hours'])
df.describe()
bins=[2,3,4,5,6,7,8,9,10]
sns.distplot(df['time_spend_company'],bins=bins,color='green')    # optional: kde=False/True, hist=Flase/True, rug=True,vertical=True
plt.xticks(bins)
sns.distplot(df['time_spend_company'],bins=bins, rug=True,hist_kws={'color':'red','edgecolor':'blue','linewidth':3,'alpha':0.5})
sns.distplot(df['time_spend_company'],bins=bins, rug=True,kde_kws={'color':'orange','linewidth':3})
    # Scatter Plot
titanic_df = sns.load_dataset('titanic') --> titanic_df.head()
sns.scatterplot(x='age', y='fare', data = titanic_df, hue = 'alive')
plt.figure(figsize=(12,6))
sns.scatterplot(x='age', y='fare', data = titanic_df, hue='alive', style='class', palette='gist_rainbow',alpha=0.5) --> plt.title('Titanic data Analysis')
sns.lineplot(x='age', y='fare', data = titanic_df, color='green')
    # Bar Plot
titanic_df.head()
sns.barplot(x='class', y='fare', data = 'titanic_df', hue='sex', palette='inferno')
sns.barplot(y='class', x='fare', data = 'titanic_df', hue='sex', palette='inferno', orient='h')     # for this x axis should be Numerical values
sns.barplot(x='class', y='fare', data = 'titanic_df', hue='sex', palette='inferno',ci=100,errcolor='#7289da',errwidth=3)
sns.barplot(x='class', y='fare', data = 'titanic_df', hue='sex', palette='inferno', saturation=0.5)
    # Heatmaps
flight_df = sns.load_dataset('flights') --> flight_df.head()
flight_df = flight_df.pivot("month","year","passengers") --> flight_df.head()
plt.figure(figsize=(12,6)) --> ax = sns.heatmap(flight_df) --> ax
plt.figure(figsize=(14,8)) --> ax = sns.heatmap(flight_df, annot=True, fmt='d') --> ax
plt.figure(figsize=(14,8)) --> ax = sns.heatmap(flight_df, annot=True, fmt='d', linecolor='k',linewidths='3') --> ax
plt.figure(figsize=(14,8)) --> ax = sns.heatmap(flight_df, annot=True, fmt='d', linecolor='k',linewidths='3', cmap ='Blues') --> ax     # cbar =False
grid_kws = {"height_ratios":(.4,.05),"hspace": .4}
f,(ax,cbar_ax) = plt.subplots(2,gridspec_kw=grid_kws)
ax = sns.heatmap(flight_df, cbar_kws = {"orientation":"horizontal"},ax=ax,cbar_ax=cbar_ax,)
titanic_df = sns.load_dataset('titanic') --> plt.figure(figsize=(12,8)) --> sns.heatmap(titanic_df.corr())
------------------------------------ End of Seaborn (Akshit) --------------------------------------------------------------------------------------------

pip install seaborn --> conda install seaborn --> pip install matplotlib
import matplotlib.pyplot as plt --> import seaborn as sns --> import pandas as pd
    # Line Plot thru matplot lab
var1 = [1,2,3,4,5,6,7] --> var2 = [2,3,4,1,5,6,7]
plt.plot(var1,var2) --> plt.show()
    # line plot thru seaborn
x1 = pd.DataFrame({"var1":var1, "var2":var2})
sns.lineplot(x=var1,y=var2) --> plt.show()      # without dataframe
sns.lineplot(x="var1", y="var2", data=x1) --> plt.show()    # with dataframe
    # Line Plot thru seaborn using csv file dataset
y1 = sns.load_dataset("penguins")       # penguins is csv file, without loading, we can use this from direct github repo "mwaskom/seaborn-data"
y1 = sns.load_dataset("penguins").head(6)       # first 6 rows of data
sns.lineplot(x="bill_length_mm", y="flipper_length_mm", data=y1,hue="sex",size=50) --> plt.show()    # here hue used to display legend on Top corner of graph, sex is one of the column of penguin csv file
sns.lineplot(x="bill_length_mm", y="flipper_length_mm", data=y1,hue="sex",style="sex") --> plt.show()   # here style using for lines, dotted lines...etc in chart/graph
sns.lineplot(x="bill_length_mm", y="flipper_length_mm", data=y1,hue="sex",style="sex",palette="Accent",markers=["o",">"]) --> plt.show()   # here many types of pallette there for different colours, markers are pointers in lines in chart/graph
sns.lineplot(x="bill_length_mm", y="flipper_length_mm", data=y1,hue="sex",style="sex",palette="Accent",dashes=False, legend=False) --> plt.show()   # when use False, it removes dotted line style, removes legend also in chart
plt.grid() --> plt.title("Line Chart") --> plt.show()
    # Bar Plot thru seaborn
sns.barplot(x=y1.island,y=y1.bill_length_mm) --> plt.show()
order1 = ["Dream","Torgersen","Biscoe"]     # here we define custom ordering for x values of chart
sns.barplot(x="island",y="bill_length_mm", data=y1, hue="sex",order=order1, hue_order=["Female","Male"], ci=100,n_boot=2) --> plt.show()     # ci is interval of y values in chart
sns.barplot(x="island",y="bill_length_mm", data=y1, hue="sex",order=order1, hue_order=["Female","Male"], orient="h") --> plt.show()    # For vertical to horizontal, x and y values should be numerical only
sns.barplot(x="island",y="bill_length_mm", data=y1, hue="sex",order=order1, hue_order=["Female","Male"], orient="v",saturation=0.3,errcolor="b",errwidth=5,capsize=0.2, dodge=False) --> plt.show()    
sns.barplot(x="bill_depth_mm",y="bill_length_mm",data=var,orient="h") --> plt.show()
sns.barplot(x="island",y="bill_length_mm",data=var,color="g") --> plt.show()
sns.set(style="darkgrid")
sns.barplot(x="island",y="bill_length_mm", data=y1, hue="sex",order=order1, hue_order=["Female","Male"], orient="v",saturation=100,errcolor='b',alpha=0.7) --> plt.show()
----------------------------------------- End of Seaborn -----------------------------------------------------------------------------------------------------------------------------------
