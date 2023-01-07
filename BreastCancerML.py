#!/usr/bin/env python
# coding: utf-8

# # Gathering Data:

# In[1]:


import numpy as np
import pandas as pd

pd.options.display.max_columns = 100


# In[2]:


data = pd.read_csv("data.csv")


# In[3]:


len(data.index),len(data.columns)


# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


# Exploring Data Analysis

data.info()


# In[8]:


data.isna().sum()


# In[9]:


#We found no null values
data = data.dropna(axis= 'columns')


# In[10]:


data.describe(include='O')


# In[11]:


data.diagnosis.value_counts()


# In[12]:


data.head(2)


# In[13]:


diagnosis_unique= data.diagnosis.unique()
diagnosis_unique
#To Iddentity unique values in diagnosis column


# In[14]:


#Data Visualization:
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


# In[15]:


plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.hist(data.diagnosis)
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")

plt.subplot(1,2,2)

sns.countplot('diagnosis',data=data);


# In[16]:


px.histogram(data,x='diagnosis')


# In[17]:


cols=["diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean"]

sns.pairplot(data[cols],hue="diagnosis")
plt.show()


# In[18]:


size = len(data['texture_mean'])

area= np.pi*(10*np.random.rand(size))**2
colors= np.random.rand(size)

plt.xlabel("texture mean")
plt.ylabel("radius mean")
plt.scatter(data['texture_mean'],data['radius_mean'],s= area,c=colors,alpha=0.5);


# In[19]:


#Data Encoding:
from sklearn.preprocessing import LabelEncoder

label_encoder_Y = LabelEncoder()
data.diagnosis = label_encoder_Y.fit_transform(data.diagnosis)


# In[20]:


data.head(3)


# In[21]:


print(data.diagnosis.value_counts())
print("\n",data.diagnosis.value_counts().sum())


# In[22]:


#Finding correlation btw other features
cols1 = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols1))
data[cols1].corr()


# In[23]:


plt.figure(figsize=(12,9))
plt.title("Correlation Graph")

cmap = sns.diverging_palette(1000,120,as_cmap=True)
sns.heatmap(data[cols1].corr(),annot=True,fmt='.1%',linewidths=0.5,cmap=cmap);


# In[24]:


plt.figure(figsize=(15,10))
fig =px.imshow(data[cols1].corr());
fig.show()


# In[25]:


#Model Implementation:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Machine Learning Model:

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# In[26]:


#Accuracy, Error and Validations Check:

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,classification_report
from sklearn.model_selection import KFold,cross_validate,cross_val_score
from sklearn.svm import SVC
from sklearn import metrics



# In[27]:


data.columns


# In[28]:


#Dependent and Independent Variables
Indp =['radius_mean','perimeter_mean','area_mean','symmetry_mean','compactness_mean','concave points_mean']

yhat=['diagnosis']


X= data[Indp]


# In[29]:


y= data.diagnosis


# In[30]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=10)


# In[31]:


#Feature Scling:

sc=StandardScaler()

X_train= sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)


# In[32]:


#Model Function: Returns, Score,Accuracy, Prediction

def model_building(model,X_train,X_test,y_train,y_test):

    model.fit(X_train,y_train)
    score= model.score(X_train,y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(pred,y_test)
    
    return score,pred,accuracy


# In[33]:


#model List

model_list ={'Logistic_Regression' : LogisticRegression(),
             'Decision_Tree': DecisionTreeClassifier(criterion='entropy',random_state=5),
             'RandomForest': RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=5),
             'SVC': SVC()
            }
print(list(model_list.keys()))
print(list(model_list.values()))


# In[34]:


#Model Implementation:

#Plot for Confusion Matrix:

def cm_graph(cm):
    sns.heatmap(cm,annot=True,fmt='d') 
    plt.show()


# In[35]:


df_prediction=[]
confusion_marix=[]
df_prediction_names=['model_name','score','accuracy_score','accuracy_percentage']

for mo_name,model in zip (list(model_list.keys()),list(model_list.values())):
    (score,pred,accuracy)= model_building(model,X_train,X_test,y_train,y_test)
    print("\n\n Classification Report of",str(mo_name))
    
    print(classification_report(y_test,pred))
    df_prediction.append([mo_name,score,accuracy,"{0:.2%}".format(accuracy)])
    
    confusion_marix.append(confusion_matrix(y_test,pred))
    
    df_pred= pd.DataFrame(df_prediction,columns=df_prediction_names)


# In[36]:


len(confusion_marix)


# In[37]:


plt.figure(figsize=(10, 2))
plt.title("Confusion Metric Graph")


for index, cm in enumerate(confusion_marix):
      
          
        cm_graph(cm) # Call the Confusion Metrics Graph
        plt.tight_layout(pad=True)
       


# In[38]:


df_pred


# In[39]:


df_pred.sort_values('score',ascending=False)


# In[40]:


#HyperTuning: K-Fold--
def cross_val_scoring(model):
    model.fit(data[Indp],data[yhat])
    predictions= model.predict(data[Indp])
    accuracy=accuracy_score(predictions,data[yhat])
    
    print("\n Full Data Accuracy:",round(accuracy,2))
    print("Cross Validation Of:",str(name),"\n")
    
    k_fold=KFold(n_splits=5)
    
    for train_index,test_index in k_fold.split(data):
        X_train= data[Indp].iloc[train_index,:]
        y_train= data[yhat].iloc[train_index]
        X_test = data[Indp].iloc[test_index,:]
        y_test=data[yhat].iloc[test_index]
    
        model.fit(X_train,y_train)
    
        err=[]
        err.append(model.score(X_train,y_train))
    
        print("\n Score:",round(np.mean(err),2))
          


# In[41]:


for name,model in zip(list(model_list.keys()),list(model_list.values())):
    cross_val_scoring(model)


# In[42]:


#gridSearchAlgo:

from sklearn.model_selection import GridSearchCV

#DecisionTree

model=DecisionTreeClassifier()

para={'max_features':['auto','sqrt','log2'],
           'min_samples_split':[2,3,4,5,6,7,8,9,10],
           'min_samples_leaf':[2,3,4,5,6,7,8,9,10]}
gsc=GridSearchCV(model,para,cv=10)

gsc.fit(X_train,y_train)

print("\n Best Score:")
print(gsc.best_score_)

print("\n best estimator:",gsc.best_estimator_)

print("\n best parameter:",gsc.best_params_)


# In[43]:


#KNeighbours:

model = KNeighborsClassifier()


param_grid = {
    'n_neighbors': list(range(1, 30)),
    'leaf_size': list(range(1,30)),
    'weights': [ 'distance', 'uniform' ]
}

gsc = GridSearchCV(model, param_grid, cv=10)

gsc.fit(X_train, y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estinator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_)


# In[44]:


model = SVC()

param_grid = [
              {'C': [1, 10, 100, 1000], 
               'kernel': ['linear']
              },
              {'C': [1, 10, 100, 1000], 
               'gamma': [0.001, 0.0001], 
               'kernel': ['rbf']
              }
]



gsc = GridSearchCV(model, param_grid, cv=10)

gsc.fit(X_train, y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estinator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_)


# In[45]:


model = RandomForestClassifier()


random_grid = {'bootstrap': [True, False],
               'max_depth': [40, 50, None], 
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2], 
               'min_samples_split': [2, 5],
               'n_estimators': [200, 400]}

gsc = GridSearchCV(model, random_grid, cv=10) 

gsc.fit(X_train, y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estinator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_)


# 
