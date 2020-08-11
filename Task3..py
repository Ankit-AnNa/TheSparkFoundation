#!/usr/bin/env python
# coding: utf-8

# TASK_3-To Explore Unsupervised Machine Learning:-

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets


# In[2]:


iris = datasets.load_iris()
iris 


# In[3]:


df = pd.DataFrame(iris.data)
df


# In[4]:


df_new = df.rename(columns={0: 'sepal length (cm)',1:'sepal width (cm)',2:'petal length (cm)',3:'petal width (cm)'})
print('Final Import and Improved Data')
df_new.head()


# In[5]:


x = df_new.iloc[:,].values


# In[6]:


from sklearn.cluster import KMeans
cluster = []

kmeans_kwargs = {
"init": "random",
"n_init": 10,
"max_iter": 300,
"random_state": 42,
 }
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(x)
    cluster.append(kmeans.inertia_)


# In[7]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), cluster)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("CLUSTER")
plt.show()


# In[8]:


kmeans = KMeans(init= "random",n_init= 10,max_iter= 300,random_state= 42)
y_kmeans = kmeans.fit_predict(x)


# In[9]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1],s = 100, c = 'blue', label = 'Iris-Setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1],s = 100, c = 'indigo', label = 'Iris-Versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],s = 100, c = 'yellow', label = 'Iris-Virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:




