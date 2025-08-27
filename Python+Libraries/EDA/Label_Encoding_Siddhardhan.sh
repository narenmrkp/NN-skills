# Label Encoding: converting the labels into numeric form
# importing the Dependencies
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# loading the data from csv file to pandas dataFrame
cancer_data = pd.read_csv('/content/data.csv')
# first 5 rows of the dataframe
cancer_data.head()
# finding the count of different labels
cancer_data['diagnosis'].value_counts()
# load the Label Encoder function
label_encode = LabelEncoder()
labels = label_encode.fit_transform(cancer_data.diagnosis)
# appending the labels to the DataFrame
cancer_data['target'] = labels
cancer_data.head()
cancer_data['target'].value_counts()

# Label Encoding of iris data
# loading the data from csv file to pandas dataFrame
iris_data = pd.read_csv('/content/iris_data.csv')
iris_data.head()
iris_data['Species'].value_counts()
# loding the label encoder
label_encoder_1 = LabelEncoder()
iris_labels = label_encoder_1.fit_transform(iris_data.Species)
iris_data['target'] = iris_labels
iris_data.head()
iris_data['target'].value_counts()
