#!/usr/bin/env python
# coding: utf-8

# Task_5- To explore Business Analytics

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("SampleSuperstore.csv")


# In[3]:


print("Import Super-store Data--")
df


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.columns


# ->Finding Error or null value in data and solve it-

# In[7]:


pd.DataFrame( df.isnull().sum(), columns= ['Number of missing values'])


# According to upper table we see they have not an error or null value in data

# In[8]:


df.describe()


# In[9]:


df.corr()


# In[10]:


import seaborn as sns
sns.heatmap(df.corr(), annot=True) 


# As we see Correlation in 'Sales & Profit' and 'Sales & Quantity' is Majore- 

# -> So Now we Analysis Data's-

# In[11]:


sns.scatterplot(df['Profit'], df['Sales'], hue=df['Region'], palette='Set2') 


# In[12]:


df['Region'].value_counts()


# As Above Data tells the Demand by Region

# In[13]:


df['Quantity'].value_counts()


# As Above Data tells the Maximum Demanding Quantity 

# ->Now let's Make Grafe between Sales-Quantity and Quantity-Profit-

# In[14]:


sns.scatterplot(df['Quantity'], df['Sales'], hue=df['Region'], palette='Set2')


# In[15]:


sns.scatterplot( df['Profit'],df['Quantity'], hue=df['Region'], palette='Set2')


# As We see above that No Profit in company in Quantity 1 to 8.

# In[16]:


sns.boxplot('Discount','Profit', data=df)


# As We see above that No Profit in company when Discount is above 40%.

# In[17]:


t=pd.DataFrame(np.percentile(df['Profit'],[25,40,60,80,90,99.5],axis=0)).transpose()
t.columns=['Prce_20','Prce_40','Prce_60','Prce_80','Prce_90','Prce_99.5']


# In[18]:


t


# In[19]:


GP1=df.groupby(['Region']).Profit.mean().reset_index()
GP1


# In[20]:


GP1.sort_values("Profit", axis = 0, ascending = False, 
                 inplace = True, na_position ='first') 
GP_1=GP1.reset_index()
GP_1.drop(['index'], axis = 1)


# In[21]:


import matplotlib.pyplot as plt
plt.barh(GP_1['Region'],GP_1['Profit'])
plt.xlabel('PROFIT')
plt.ylabel('FROM REGION')
plt.show()


# As we see above Less Profit in Central Region so need to improve it 

# In[22]:


GP2=df.groupby(['Quantity']).Profit.mean().reset_index()
GP2


# In[23]:


GP2.sort_values("Profit", axis = 0, ascending = False, 
                 inplace = True, na_position ='first') 
GP_2=GP2.reset_index()
GP_2.drop(['index'], axis = 1)


# In[24]:


plt.barh(GP_2['Quantity'],GP_2['Profit'])
plt.xlabel('Profit Gain from Product')
plt.ylabel('Quantity of Product')
plt.show()


# As We See Quantity 1 to 6 as not given good profit to company

# In[25]:


GP3=df.groupby(['State']).Profit.mean().reset_index()
GP3


# In[26]:


#arreng in asending oder
GP3.sort_values("Profit", axis = 0, ascending = False, 
                 inplace = True, na_position ='first') 
GP_3=GP3.reset_index()
GP_3.drop(['index'], axis = 1)


# As above Table We see 39 to 48 number's city,In city company face the maximum loss

# In[27]:


sns.pairplot(df, kind="reg")  
plt.show()


# In[28]:


GP=df.groupby(['Country','State','City','Postal Code','Segment','Region','Category','Sub-Category'])
GP.first() 


# Above table help to understand all data in simple way via groupby function
