#!/usr/bin/env python
# coding: utf-8

# <h1> Task #2 </h1>
# <h1> Ravi Teja </h1>
# <h2> linear regression</h2>

# In[1]:


#import all libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# <h1> step-1: collecting data </h1>

# In[2]:


raw_data = pd.read_csv("http://bit.ly/w-data")
raw_data.head(10)


# <h1> step-2 data preprocessing</h1>
# 

# In[3]:


raw_data.isnull().sum().sum()


# In[5]:


raw_data.dtypes


# <h1> step-3 data visualization</h1>

# In[7]:


raw_data.plot(x = 'Hours', y = 'Scores', kind = 'scatter')
plt.title('Hours vs Scores')
plt.xlabel('Hours studied')
plt.ylabel('Marks scored')
plt.show()


# <h1> step-4 preparing the data</h1>

# In[8]:


y = raw_data['Scores']
X = raw_data.drop('Scores', axis = 1)


# In[9]:


#splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,random_state = 0)


# <h1> step-5 training algorithm</h1>
# <p> now we have splited the data for test and traing hence we use it for train the algorithm</p>

# In[10]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[12]:


#plotting the line 
line = lr.coef_ * X + lr.intercept_

#plotting the tested data
plt.scatter(X, y)
plt.plot(X, line)
plt.xlabel('hours')
plt.ylabel('scores')
plt.show()


# <h1> step-6 model predictions </h1>

# <p> What will be the predicted scores if the given hours value is 9.25 in a day?

# In[21]:


pred_test = lr.predict(X_test)


# In[23]:


metrics.mean_absolute_error(y_test, pred_test)


# In[25]:


lr.predict([[9.25]])

