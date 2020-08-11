#!/usr/bin/env python
# coding: utf-8

# Task_2:- To Explore Supervised Machine Learning

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("http://bit.ly/w-data")
print("DATA IMPORTTED")

df.head()


# In[3]:


df.dtypes


# In[4]:


df.plot(x='Hours', y='Scores', style='o')  
plt.title('TIME vs SCORE')  
plt.xlabel('Studied Time in Hours')  
plt.ylabel('Exam Score in Percentage')  
plt.show()


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X = df.iloc[:,:-1].values
Y= df.iloc[:,-1].values
# df[['Hours']].values  
#y = df[['Scores']].values


# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)


# In[8]:


X_train.reshape(1,-1)
Y_train.reshape(1,-1)


# In[9]:


from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()
regression_model.fit(X_train, Y_train)


# In[10]:


line = regression_model.coef_*X+regression_model.intercept_
_
plt.scatter(X, Y)
plt.title('Regression Line')
plt.xlabel('Studied Time in Hours')  
plt.ylabel('Exam Score in Percentage')
plt.plot(X, line)
plt.show()


# In[11]:


X_test #.reshape(-2,1) 


# In[12]:


Y_pred = regression_model.predict(X_test)
Y_pred 


# In[13]:


Y_test


# In[14]:


df_1 = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
df_1


# In[15]:


def Time():
    print('note:hours of study bt 0-12')
    h = float(input('Enter the hours of study:'))
    return h


# In[16]:


Hours = Time()


# In[17]:


pred = regression_model.predict([[Hours]])
print("No of Hours = {}".format(Hours))
print("Predicted Score = {}".format(pred[0]))


# MODEL EVALUATION:-

# In[19]:


from sklearn import metrics  
print('MEAN ABSOLUTE ERROR:',metrics.mean_absolute_error(Y_test, Y_pred))


# In[ ]:




