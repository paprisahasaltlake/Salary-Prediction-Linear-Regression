#Problem Statement

"""Predicting the salary for a job position is crucial for a business' HR & talent function for optimizing compensation strategy and talent retention in a highly competetive labour market. The aim of this project is to build a salary prediction model for existing and future job seekers by examining an existing dataset of job postings.

The analysis is aimed at explaining every step of the process from defining the problem, discovering dataset, developing model and deploying into production. The model applies data transformation and machine learning on features such as work experience, Job Type, Majors, Industry Type, Degree and Miles from metropolis. The final aim is to predict salary for a job posting based on these available features.

The dataset includes available features or labelled columns for analysis as follows:

Job ID/jobId : Given Job ID for the role
Company ID : Company ID for the respective Job ID advertised
Degree : Applicant's qualification/degree
Major : Degree Specialization
Industry : Job ID's categorized industry such as Oil, Auto, Health, Finance etc.
Experience (Years) : Requried Experience for the role
Miles from Metropolis : Distance of the job location in miles from the nearest metropolitan city
Salary : In x1000 dollars of the respective Job ID
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import sklearn as sk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

#loading data into a pandas dataframe

from google.colab import drive
drive.mount('/content/drive')

url1 = '/content/drive/My Drive/Demo/train_dataset.csv'
train_dataset_df = pd.read_csv(url1)

url2 = '/content/drive/My Drive/Demo/test_dataset.csv'
test_dataset_df = pd.read_csv(url2)

url3 = '/content/drive/My Drive/Demo/train_salaries.csv'
train_salaries_df = pd.read_csv(url3)

#Renaming columns for ease of exploration

train_dataset_df.rename(columns={'jobId':'Job ID', 'companyId':'Company ID', 
                                  'jobType':'Job Type','degree':'Degree','major':'Major',
                                  'industry':'Industry','yearsExperience':'Experience (Years)',
                                 'milesFromMetropolis':'Miles from Metropolis'}, inplace = True)

test_dataset_df.rename(columns={'jobId':'Job ID', 'companyId':'Company ID', 
                                  'jobType':'Job Type','degree':'Degree','major':'Major',
                                  'industry':'Industry','yearsExperience':'Experience (Years)',
                                 'milesFromMetropolis':'Miles from Metropolis'}, inplace = True)

train_salaries_df.rename(columns={'jobId':'Job ID','salary':'Salary'}, inplace = True)

train_dataset_df.info()

print(train_dataset_df.shape)
print(test_dataset_df.shape)
print(train_salaries_df.shape)

"""Both train and test datasets have the same lengths and data types"""

train_dataset_df.head()

test_dataset_df.info()

train_salaries_df.info()

train_salaries_df.head()

"""As discussed earlier, we've seen the feature training and target sets contain the same Job IDs except that the latter includes out target variable 'Salary'. We would like to see the data sorted by its Job ID all in one place instead of having to open two datasets."""

#Merging the training features and salaries (target) dataset along Job ID column 
train_merged = pd.merge(train_dataset_df, train_salaries_df, on ='Job ID', how = 'inner')
train_merged.head()

#Checking if any Salary attributes have '0' value
len(train_merged[train_merged['Salary']==0])

#Displaying the rows that have '0' as their salary values
train_merged[train_merged['Salary']==0]

#Removing rows with Salary value 0

train_merged = train_merged[train_merged.Salary!= 0]

print(train_merged.shape)
train_merged.reset_index(drop = True).head()

#Checking for duplicate values

train_merged.duplicated().any()

#Visualizing Salary
f, ax = plt.subplots(1,2,figsize=(18,6))
sns.distplot(train_merged['Salary'], ax=ax[0], bins=40, kde=True, norm_hist=True)
ax[0].axvline(np.mean(train_merged['Salary']), color='black')
ax[0].axvline(np.median(train_merged['Salary']), color='darkblue', linestyle='--')
ax[0].set_title('Salary Histogram',fontsize = 14)
sns.boxplot(train_merged['Salary'], ax=ax[1], color='gold')
ax[1].set_title('Salary Boxplot', fontsize = 14)
f.suptitle('Distribution of Salary', fontsize = 18)

print("The Black line in Dist. Plot shows the mean at: ",round(train_merged['Salary'].mean()))
print("The Dotted line in the Dist. Plot shows the median at: ", round(train_merged['Salary'].median()))

#Getting rid of redundant variables from the dataset

train_merged = train_merged.drop('Job ID', axis = 1)

test_dataset_df = test_dataset_df.drop('Job ID', axis = 1)
test_dataset_df = test_dataset_df.drop('Company ID', axis = 1)

train_merged.head()

#One hot encode categorical data in the dataset

train_merged = pd.get_dummies(train_merged)
train_merged.head()

#Dividing data into attributes and labels

x = train_merged.drop('Salary', axis = 1)
#To avoid model overfitting, which happens if the target parameter 'Salary' is 
#not removed from training data 

y = train_merged['Salary']

#Performing the split (Training - 80%, Testing - 20%)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=40)

print("Nmber of training samples:",x_train.shape[0])
print("Nmber of test samples :", x_test.shape[0])

#Creating Linear Regressor Object
lr = LinearRegression()

#Fitting model using x and y attributes
lr.fit(x_train, y_train)

#Making Prediction such that y_hat gives an array of Target Value (Salary)
y_hat = lr.predict(x_test)
print("The first 5 predictied salaries: ", y_hat[0:5])

#Having established a baseline model, we can predict Salaries in the test set

y_hat=lr.predict(x_test) #Predicting the training data
y_hat

#Evaluation with MSE
mse = mean_squared_error(y_test, y_hat)
print("Mean Squared Error is: ", mse)

print('The Accuracy of the model is: ', lr.score(x,y))

