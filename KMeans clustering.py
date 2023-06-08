#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()


# In[6]:


import numpy as np
from sklearn.datasets import make_blobs


# In[7]:


X, Y_true = make_blobs(n_samples = 300, centers = 4, cluster_std = 0.60, random_state = 0)
plt.scatter(X[:,0], X[:,1], s=50)


# In[8]:


from sklearn.cluster import KMeans
KMeans = KMeans(n_clusters = 4)
KMeans.fit(X)


# In[31]:


Y_KMeans = KMeans.predict(X)
Y_KMeans


# In[32]:


from sklearn.metrics import pairwise_distances_argmin


# In[33]:


def find_clusters(X, n_clusters, rseed =2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        labels = pairwise_distances_argmin(X, centers)
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels


# In[38]:


Y_KMeans = KMeans.predict(X)
Y_KMeans
centers,labels = find_clusters(X,4)
plt.scatter(X[:,0], X[:,1], c=Y_KMeans, s=50, cmap = 'viridis')
plt.scatter(centers[:,0],centers[:,1], c='black',s=200,alpha =0.5)
plt.show()


# In[ ]:




