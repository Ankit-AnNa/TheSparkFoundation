#!/usr/bin/env python
# coding: utf-8

# Task_4- To Explore Decision Tree Algorithm

# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris


# In[2]:


iris = datasets.load_iris()


# In[3]:


df = pd.DataFrame(iris.data)
df


# In[4]:


df_new = df.rename(columns={0: 'sepal length (cm)',1:'sepal width (cm)',2:'petal length (cm)',3:'petal width (cm)'})
df_new.head()


# In[5]:


x=iris.target
x


# In[6]:


from sklearn import tree
Dtree = tree.DecisionTreeClassifier()
Dtree = Dtree.fit(df_new, x)


# In[7]:


import graphviz
dot_data = tree.export_graphviz(Dtree, out_file=None,feature_names=iris.feature_names,class_names=iris.target_names,filled=True, rounded=True,special_characters=True)  
graph = graphviz.Source(dot_data)


# So that's a Decision_Tree:-

# In[8]:


graph


# In[ ]:




