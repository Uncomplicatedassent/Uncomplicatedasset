#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[8]:


df=pd.read_csv("insurance.csv")
df


# In[9]:


df.info()


# In[10]:


df.head(20)


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


plt.xlabel("bmi")
plt.ylabel("charges")
plt.scatter(df.bmi,df.charges,color = 'blue',marker = '*')


# In[16]:


from sklearn import linear_model
reg=linear_model.LinearRegression()


# In[17]:


reg.fit(df[['bmi']],df.charges)


# In[19]:


reg.predict([[42]])


# In[ ]:




