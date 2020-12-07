#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


# In[4]:


name=['sepal-length','sepal-width','pedal-length','pedal-width','class']
df=pd.read_csv(url,names=name)


# In[5]:


df


# In[6]:


print(df.groupby('class').size())


# In[7]:


sb.pairplot(data=df,palette='hus1')


# In[8]:


df.hist()
nd=df.drop()


# In[9]:


sb.boxplot(data=df)


# In[10]:


df.columns


# In[11]:


X=df[['sepal-length', 'sepal-width', 'pedal-length', 'pedal-width']]
Y=df['class']


# In[13]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
# X


# In[14]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=7)


# In[15]:


scoring='accuracy'


# In[16]:


models=[]
models.append(('LR',LogisticRegression()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('SVM',SVC()))
print(models)


# In[17]:


lr=LogisticRegression()
svc=SVC()
knn=KNeighborsClassifier()
lr.fit(X_train,Y_train)
svc.fit(X_train,Y_train)
knn.fit(X_train,Y_train)


# In[20]:


plr=lr.predict(X_test)
pknn=knn.predict(X_test)
pscv=svc.predict(X_test)


# In[21]:


from sklearn.metrics import confusion_matrix,classification_report


# In[23]:


print(confusion_matrix(plr,Y_test),classification_report(plr,Y_test))


# In[24]:


print(confusion_matrix(pknn,Y_test),classification_report(pknn,Y_test))


# In[25]:


print(confusion_matrix(pscv,Y_test),classification_report(pscv,Y_test))


#   ** As we see that the precison and accuracy are better in KNN classifier as compared to others** 





