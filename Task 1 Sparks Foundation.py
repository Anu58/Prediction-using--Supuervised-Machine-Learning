#!/usr/bin/env python
# coding: utf-8

# # Name : Anushka Bhatia

# # Data Science and Business Analytics 

# # Predict the percentage of a student based on the number of study hours

# # Task 1: Prediction using Supervised ML

# including the required Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # READING THE DATA FROM DATASET

# In[2]:


url =  "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")


# In[3]:


data


# # TO CHECK THE MISSING VALUES

# In[4]:


data.isnull().sum()


# # TO GENERATE DESCRIPTIVE STATISTICS

# In[5]:


data.describe()


# # DATA VISUALIZATION

# In[6]:


x = data['Hours']
y = data['Scores']


# In[7]:


plt.scatter(x,y)
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied',fontsize=20)
plt.ylabel('Percentage Scores',fontsize=20)
plt.show()


# # LINEAR REGRESSION MODEL

# Preparing the Data

# In[8]:


x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=0)


# # TRAINING THE MODEL

# The values taken from the training dataset are trained using linear regression model algorithm.

# In[10]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
print("Training successful")


# # PLOTTING THE REGRESSION LINE

# Now,printing the coefficient and intercept values required to plot the regression line.

# In[11]:


reg.coef_


# In[12]:


reg.intercept_


# In[13]:


line = reg.coef_*x+reg.intercept_
plt.scatter(x,y,color="red")
plt.title("Hours vs Scores(Percentage)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores(Percentage)")
plt.plot(x,line,color="green")
plt.show()


# # PREDICTING THE SCORES

# In[14]:


#testing the data in hours
print(x_test)

#predicting the scores
y_pred = reg.predict(x_test)


# Comparing Predicted data with Actual Data|

# In[15]:


data = pd.DataFrame({'Actual' : y_test, 'Predicted' :y_pred})
data
        


# Predicting the score of the student for given time can be directly calculated by predict function

# In[16]:


#predict values by own data
hours = [9.25]
own_pred = reg.predict([hours])
print("No. of Hours = ",format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # EVALUATING THE MODEL

#  They are used for everything how well distinct algorithms perform well on datasets.Here,I have evaluated the model using mean absolute error,mean absolute error ,mean squared error and root meam squared error.

# In[18]:


from sklearn import metrics
print("Mean Absolute error:", metrics.mean_absolute_error(y_test,y_pred))
print("Mean Squared error:",metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Squared error:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:




