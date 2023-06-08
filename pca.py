#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()


# In[3]:


print(cancer['DESCR'])
print(cancer['target_names'])


# In[4]:


df = pd.DataFrame(cancer['data'],columns = cancer['feature_names'])
df


# In[5]:


df.head()


# In[6]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)


# In[7]:


scaled_data = scaler.transform(df)


# In[8]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(scaled_data)


# In[9]:


X_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(X_pca.shape)


# In[11]:


plt.figure(figsize = (8,6))
plt.scatter(X_pca[:,0],X_pca[:,1],c = cancer['target'],cmap = 'plasma')
plt.xlabel("First principal component")
plt.ylabel("Second principal component")


# In[13]:


pca.components_


# In[15]:


import seaborn as sns
df_comp = pd.DataFrame(pca.components_,columns = cancer['feature_names'])
sns.heatmap(df_comp, cmap = 'plasma',)


# In[ ]:




