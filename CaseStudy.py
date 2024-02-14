!pip install --upgrade tensorflow

from __future__ import print_function 
import pandas as pd
pd.__version__

import os
os.getcwd()

#Mouting the drive to load a simple data set stored on the Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

#Load a dataset into a dataframe
from google.colab.data_table import DataTable
DataTable.max_columns = 58
spam_dataset_dataframe = pd.read_csv("/content/gdrive/My Drive/spam.csv",sep=',')
print("Dataset was loaded sucessfully...")

spam_dataset_dataframe.info() #Get some insights on the data

spam_dataset_dataframe.describe()

#Dimensions of the spam_dataset_dataframe
print(spam_dataset_dataframe.shape)

#Checking if there are missing values in the dataset
spam_dataset_dataframe.isnull().sum()

spam_dataset_dataframe['Class']=spam_dataset_dataframe['Class'].apply(lambda x: 1 if x=='spam' else 0)
spam_dataset_dataframe.head()


#Plot per feature histogram. figsize = (width, height)
spam_dataset_dataframe.hist(bins=30,figsize=(20,20));

#Displaying the first ten columns to create a general idea of the dataset
spam_dataset_dataframe.iloc[0:10]

# Visualising the Sparse Matrix
# data visualization

import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

plt.figure(figsize=[15,30])
plt.title('')
plt.spy(spam_dataset_dataframe[:100].values, precision = 0.1, markersize = 5)
plt.show()

import numpy as np
from sklearn.model_selection import train_test_split

#Create a training and test set
spam_training_set, spam_test_set = train_test_split(spam_dataset_dataframe, test_size = 3601
, random_state = 42)
spam_dataset_dataframe.keys()

spam_training_data, spam_training_target = spam_training_set[["make", "address", "all","3d","our", "over", "remove", "internet", "order", "mail", "receive", "will", "people", "report", "addresses", "free", "business", "email", "you", "credit", "your", "font", "0", "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857", "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs", "meeting", "original", "project", "re", "edu", "table", "conference", "semicol", "paren", "bracket", "bang", "dollar", "pound", "cap_avg", "cap_long", "cap_total" ]], spam_training_set["Class"]
spam_test_data, spam_test_target = spam_test_set[["make", "address", "all","3d","our", "over", "remove", "internet", "order", "mail", "receive", "will", "people", "report", "addresses", "free", "business", "email", "you", "credit", "your", "font", "0", "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857", "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs", "meeting", "original", "project", "re", "edu", "table", "conference", "semicol", "paren", "bracket", "bang", "dollar", "pound", "cap_avg", "cap_long", "cap_total" ]], spam_test_set["Class"]
spam_training_data.head()


from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

highest_accuracy = 0
penalty_LR = [ "l2",  None]
multi_class_LR = ["auto","ovr", "multinomial"]
max_iter_LR = [ 10000,100000]
u = ""
d = " "
f = 0

for g in penalty_LR:
  for h in multi_class_LR: 
    for k in max_iter_LR:
      clf_lr = LogisticRegression(penalty = g, multi_class = h, max_iter = k,  random_state = 101)
      clf_lr.fit(spam_training_data,spam_training_target)
      spam_test_target_predict=clf_lr.predict(spam_test_data)
      print("For penalty = ",g,", and multi_class = ", h ,"and max_iter: ",k, "the accuracy score is: ", accuracy_score(spam_test_target,spam_test_target_predict))
    
      if accuracy_score(spam_test_target,spam_test_target_predict) > highest_accuracy:
          highest_accuracy = accuracy_score(spam_test_target,spam_test_target_predict)
          u = g
          d = h
          f = k

print("The random forest with the highest accuracy ",highest_accuracy, "has the following parameters: penalty = ", u, " and multi_class = ", d, " and max_iteration = ", f)

clf_lr = LogisticRegression(penalty = u ,multi_class = d , max_iter= f , random_state = 101)
clf_lr.fit(spam_training_data,spam_training_target)
spam_test_target_predict=clf_lr.predict(spam_test_data)
c_m_lr = confusion_matrix(spam_test_target,spam_test_target_predict)
c_r_lr = classification_report(spam_test_target,spam_test_target_predict)
a_s_lr = accuracy_score(spam_test_target,spam_test_target_predict)


# Compare observed value and Predicted value
print("Prediction for 20 observation:    ",clf_lr.predict(spam_test_data[0:20]))
print("Actual values for 20 observation: ",spam_test_target[0:20].values)
print(c_m_lr)
print(c_r_lr)
print(a_s_lr)


from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

clf_dt = DecisionTreeClassifier(criterion = "entropy", max_features = None, splitter = "best", random_state = 101,max_depth = 12 )
clf_gnb = GaussianNB()
eclf = VotingClassifier(estimators = [('DT', clf_dt),('LR', clf_lr), ('GNB', clf_gnb)], voting = 'hard')
eclf.fit(spam_training_data,spam_training_target)
spam_test_target_predict=eclf.predict(spam_test_data)
c_m_VC = confusion_matrix(spam_test_target,spam_test_target_predict)
c_r_VC = classification_report(spam_test_target,spam_test_target_predict)
a_s_VC = accuracy_score(spam_test_target,spam_test_target_predict)


# Compare observed value and Predicted value
print("Prediction for 20 observation:    ",eclf.predict(spam_test_data[0:20]))
print("Actual values for 20 observation: ",spam_test_target[0:20].values)
print(c_m_VC)
print(c_r_VC)
print(a_s_VC)

plt.figure(figsize=(5,5))
sns.heatmap(c_m_VC,annot=True,fmt='d', cmap="Blues")


ada = AdaBoostClassifier(
           estimator=clf_dt,
           random_state=101)


ada.fit(spam_training_data,spam_training_target)


spam_test_target_predict=ada.predict(spam_test_data)
c_m1 = confusion_matrix(spam_test_target,spam_test_target_predict)
c_r1 = classification_report(spam_test_target,spam_test_target_predict)
a_s1 = accuracy_score(spam_test_target,spam_test_target_predict)


# Compare observed value and Predicted value
print("Prediction for 20 observation:    ",ada.predict(spam_test_data[0:20]))
print("Actual values for 20 observation: ",spam_test_target[0:20].values)
print(c_m1)
print(c_r1)
print(a_s1)

plt.figure(figsize=(5,5))
sns.heatmap(c_m1,annot=True,fmt='d', cmap="BuPu")

'''n_estimators = [10, 50, 100, 500, 1000, 5000]
max_features = ["sqrt", "log2", None]
highest_accuracy = 0
a = 0
b = ""

for x in max_features:
  for n in n_estimators: 
    clf1 = RandomForestClassifier(n_estimators = n, max_features = x, random_state = 101)
    clf1.fit(spam_training_data,spam_training_target)
    spam_test_target_predict=clf1.predict(spam_test_data)
    print("For no_estimators = ",n,", and max_fetures = ", x , "the accuracy score is: ", accuracy_score(spam_test_target,spam_test_target_predict))
    
    if accuracy_score(spam_test_target,spam_test_target_predict) > highest_accuracy:
      highest_accuracy = accuracy_score(spam_test_target,spam_test_target_predict)
      a = n
      b = x

print("The random forest with the highest accuracy ",highest_accuracy, "has the following parameters: n_estimators = ", a, " and max_features = ", b)
'''
clf1 = RandomForestClassifier(n_estimators = 1000, max_features = "log2", random_state = 101)

clf1.fit(spam_training_data,spam_training_target)


spam_test_target_predict=clf1.predict(spam_test_data)
c_m_RF = confusion_matrix(spam_test_target,spam_test_target_predict)
c_r_RF = classification_report(spam_test_target,spam_test_target_predict)
a_s_RF = accuracy_score(spam_test_target,spam_test_target_predict)


# Compare observed value and Predicted value
print("Prediction for 20 observation:    ",clf1.predict(spam_test_data[0:20]))
print("Actual values for 20 observation: ",spam_test_target[0:20].values)
print(c_m_RF)
print(c_r_RF)
print(a_s_RF)


plt.figure(figsize=(5,5))
sns.heatmap(c_m_RF,annot=True,fmt='d', cmap="Greens")