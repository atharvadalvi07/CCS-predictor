#!/usr/bin/env python
# coding: utf-8

# ## Engr 1330 - Computational Thinking and Data Science Spring 2022
# # Concrete Strength Predictor Final Project
# **Atharva Dalvi**

# ## Concrete Strength Predictor Final Project - Background
# 
# The Compressive Strength of Concrete determines the quality of Concrete. 
# The strength is determined by a standard crushing test on a concrete cylinder, that requires engineers to build small concrete cylinders with different combinations of raw materials and test these cylinders for strength variations with a change in each raw material. 
# The recommended wait time for testing the cylinder is 28 days to ensure correct results, although there are formulas for making estimates from shorter cure times.
# The formal 28-day approach consumes a lot of time and labor to prepare different prototypes and test them; the method itself is error prone and mistakes can cause the wait time to drastically increase.
# 
# One way of reducing the wait time and reducing the number of combinations to try is to make use of digital simulations, where we can provide information to the computer about what we know and the computer tries different combinations to predict the compressive strength.
# This approach can reduce the number of combinations we can try physically and reduce the total amount of time for experimentation. 
# But, to design such software we have to know the relations between all the raw materials and how one material affects the strength. 
# It is possible to derive mathematical equations and run simulations based on these equations, but we cannot expect the relations to be same in real-world. 
# Also, these tests have been performed for many numbers of times now and we have enough real-world data that can be used for predictive modelling.
# 
# 
# ## Objective(s):
# - Literature scan on concrete design, and utility of a predictive approach
# - Analyse an existing concrete compressive strength database and build a data model to predict the compressive strength of a concrete mixture.
# - Build an interface to allow users to enter concrete mixtures and return an estimated strength and an assessment of the uncertainty in the estimate
# - Build an interface to allow users to add observations to the underlying database, and automatically update the Data Model to incorporate the new observations
# 
# ## Tasks: 
# 
# **Literature Research:**
# - Describe the challenge of concrete mixture design and the importance of compressive strength.
# - Summarize the value of a data model in the context of the conventional approach to strength prediction
# 
# **Database Acquisition**
# - Get the database from the repository: https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls
# - Supply links to any additional databases discovered during the literature research
# - A copy of the database is located at: http://54.243.252.9/engr-1330-psuedo-course/CECE-1330-PsuedoCourse/6-Projects/P-ConcreteStrength/concreteData.xls If you cannot access the original database you can use this copy
# 
# **Exploratory Data Analysis**
# - Describe (in words) the database.
# - Reformat as needed (column headings perhaps) the database for subsequent analysis.
# - Select possible data model structures (multi-feature linear model, power-law, ...) (see the equations below)
# - Select possible data model "fitting" tools (ordinary least squares,lasso regression, decision trees, random forests, ...)
# 
# **Model Building**
# - Build data models
# - Assess data model quality (decide which model is best) 
# - Build the input data interface for using the "best" model
# - Using your best model determine projected concrete strength for 5 possible mixtures in the table below:
# 
# |Cement|BlastFurnaceSlag|FlyAsh |CoarseAggregate|FineAggregate|Water|Superplasticizer|Age|
# |:---|:---|:---|:---|:---|:---|:---|:---|
# |175.0|13.0|172.0|1000.0|856.0|156.0|4.0|3.0|
# |320.0|0.0|0.0|970.0|850.0|192.0|0.0|7.0|
# |320.0|0.0|126.0|860.0|856.0|209.0|5.70|28.0|
# |320.0|73.0|54.0|972.0|773.0|181.0|6.0|45.0|     
# |530.0|359.0|200.0|1145.0|992.0|247.0|32.0|365.0|                
#        
# 

# In[1]:


#Importing all the libraries 
import numpy as np
import pandas as pd
import statistics 
import math
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# # Exploratory Data Analysis

# In[2]:


#making a dataframe of the csv file
df = pd.read_csv("1330 concrete.csv")
df


# In[3]:


#Getting the information (i.e, the number of rows and the data types present in each column) and the basic statistical measures 

df.info()

df.describe()


# In[4]:


df.rename(columns = {"Cement (component 1)(kg in a m^3 mixture)":"cement",
                            "Blast Furnace Slag (component 2)(kg in a m^3 mixture)":"blast furnace",
                            "Fly Ash (component 3)(kg in a m^3 mixture)":"flyash",
                            "Superplasticizer (component 5)(kg in a m^3 mixture)":"superplasticizer",
                            "Fine Aggregate (component 7)(kg in a m^3 mixture)":"fine aggregate"}, inplace = True)
df


# In[5]:


#making a pairplot of the database
sns.pairplot(df)


# In[6]:


#Computing the correlation coefficient between all the columns in the dataset and displaying them as a heat map

corr = df.corr()
sns.heatmap(corr, annot=True, cmap='Blues')
plt.title("Feature Correlation Heatmap")


# so the three main variables having significant correlation is 'cement', 'superplasticizer' and 'age'

# # Multiple Linear Regression

# In[7]:


x = ['cement' , 'superplasticizer' , 'Age (day)']
X = df[x]
Y = df['ccs']


# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


# In[9]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

print("the coeff are",lm.coef_)
print("the intercept is ",lm.intercept_)


# In[10]:


#Testing the model on the training set
y_train_pred = lm.predict(X_train)

plt.scatter(y_train,y_train_pred,color='r')
plt.plot([y_train.min(), y_train.max()], [y_train_pred.min(), y_train_pred.max()], color = 'black', lw=2)
plt.xlabel("Y_train")
plt.ylabel("Y_train_pred")
plt.title("Predictions vs. actual values in the training set")


# In[11]:


#Testing the model on the test set

y_test_pred = lm.predict(X_test)

plt.scatter(y_test,y_test_pred,color='r')
plt.plot([y_test.min(), y_test.max()], [y_test_pred.min(), y_test_pred.max()], color = 'black', lw=2)
plt.xlabel("Y_test")
plt.ylabel("Y_test_pred")
plt.title("Predictions vs. actual values in the test set")


# In[12]:


#Computing the MSE and the RMSE values for the predictions made on the training set

from sklearn import metrics

RMSE_train = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
print("the root mean square error of the predictions made on training set is",RMSE_train)

#Computing the MSE and the RMSE values for the predictions made on the test set

RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
print("the root mean square error of the predictions made on test set is",RMSE_test)


# In[13]:


import sklearn.metrics
y_pred = lm.predict(X)
sklearn.metrics.r2_score(Y, y_pred)


# # Exponential Data Model

# In[14]:


Y = np.log(Y)
X = sm.add_constant(X)
mod = sm.OLS(Y, X)
mod = mod.fit()
print(mod.summary())


# # Power-Law Model

# In[15]:


# Plot regression against actual data
plt.figure(figsize=(12, 6))
plt.plot(df['cement'], df['ccs'], marker='o', linewidth=0)          # scatter plot showing actual data
plt.xlabel('cement')
plt.ylabel('ccs')
#plt.legend(['Observed Values','Data Model'])
plt.grid()
plt.title("Cement vs Cement compressive strength")


# In[16]:


df['lnX']=df['cement'].apply(math.log)
df['lnY']=df['ccs'].apply(math.log)
df.head()


# In[17]:


plt.plot(df['lnX'], df['lnY'], marker='o', linewidth=0)           # scatter plot showing actual data
plt.xlabel('cement')
plt.ylabel('ccs')
#plt.legend(['Observed Values','Data Model'])
plt.grid()
plt.title("Cement vs Cement compressive strength")


# In[18]:


# Initialise and fit linear regression model using `statsmodels`
model1 = smf.ols('lnY ~ lnX', data=df) # model object constructor syntax
model1 = model1.fit()
# Predict values
y_pred11 = model1.predict()

beta0 = model1.params[0] # the fitted intercept
beta1 = model1.params[1]
sse = model1.ssr
rsq = model1.rsquared


# In[19]:


titleline = "Cement vs Cement compressive strength \n Data model y = " + str(round(beta0,3)) + " + " + str(round(beta1,3)) + "x" # put the model into the title
titleline = titleline + '\n SSE = ' + str(round(sse,4)) + '\n R^2 = ' + str(round(rsq,4)) 

# Plot regression against actual data
plt.figure(figsize=(12, 6))
plt.plot(df['lnX'], df['lnY'], 'o')           # scatter plot showing actual data
plt.plot(df['lnX'], y_pred11, 'r', linewidth=1)   # regression line
plt.xlabel('Ln cement')
plt.ylabel('Ln ccs')
plt.legend(['Observed Values','Data Model'])
plt.grid()
plt.title(titleline)

plt.show();


# **From the above three models, the best R^2 value is of multiple linear regression. So we take multiple linear regression as our final model**

# # Interface to allow users to enter concrete mixtures and return an estimated strength and an assessment of the uncertainty in the estimate

# In[20]:


# number of elements
n = int(input("Enter number of set of data : "))
print("please enter 2 diffrent numbers with a space in between") 

# Below line read inputs from user using map() function
a = list(map(float,input("\nenter the Cement (kg in a m^3 mixture) value : ").strip().split()))[:n]
b = list(map(float,input("\nenter the Blast Furnace Slag (kg in a m^3 mixture) value : ").strip().split()))[:n]
c = list(map(float,input("\nenter the Fly Ash (kg in a m^3 mixture) value : ").strip().split()))[:n]
d = list(map(float,input("\nenter the Water (kg in a m^3 mixture) value : ").strip().split()))[:n]
e = list(map(float,input("\nenter the Superplasticizer (kg in a m^3 mixture) value : ").strip().split()))[:n]
f = list(map(float,input("\nenter the Coarse Aggregate (kg in a m^3 mixture) value : ").strip().split()))[:n]
g = list(map(float,input("\nenter the Fine Aggregate (kg in a m^3 mixture) value : ").strip().split()))[:n]
h = list(map(float,input("\nenter the Age (day) value : ").strip().split()))[:n]

df2 = {'cement': a, 'superplasticizer': e, 'age': h}
userinput = pd.DataFrame(df2)
print("----------------------------------------------------")
userpred = pd.DataFrame({'cement compressive strength':lm.predict(userinput)})
print("the value of cement concrete strength is")
userpred


# # Interface to allow users to add observations to the underlying database, and automatically update the Data Model to incorporate the new observations

# In[21]:


copydf = pd.read_csv("1330 concrete.csv")
#number of elements
n = int(input("Enter number of set of data : "))
print("please enter 2 diffrent numbers with a space in between") 

# Below line read inputs from user using map() function
a = list(map(float,input("\nenter the Cement (kg in a m^3 mixture) value : ").strip().split()))[:n]
b = list(map(float,input("\nenter the Blast Furnace Slag (kg in a m^3 mixture) value : ").strip().split()))[:n]
c = list(map(float,input("\nenter the Fly Ash (kg in a m^3 mixture) value : ").strip().split()))[:n]
d = list(map(float,input("\nenter the Water (kg in a m^3 mixture) value : ").strip().split()))[:n]
e = list(map(float,input("\nenter the Superplasticizer (kg in a m^3 mixture) value : ").strip().split()))[:n]
f = list(map(float,input("\nenter the Coarse Aggregate (kg in a m^3 mixture) value : ").strip().split()))[:n]
g = list(map(float,input("\nenter the Fine Aggregate (kg in a m^3 mixture) value : ").strip().split()))[:n]
h = list(map(float,input("\nenter the Age (day) value : ").strip().split()))[:n]
j = list(map(float,input("\nenter the Concrete compressive strength(MPa, megapascals) value : ").strip().split()))[:n]

for i in range(n):
    copydf.loc[len(copydf)+1] = [a[i], b[i], c[i], d[i], e[i], f[i], g[i], h[i], j[i]]
copydf


# In[22]:


newindep = copydf[['Cement (component 1)(kg in a m^3 mixture)' , 'Superplasticizer (component 5)(kg in a m^3 mixture)' , 'Age (day)']]
newdep = copydf['ccs']
lm1 = LinearRegression()
lm1.fit(newindep,newdep)

copydf_pred = lm1.predict(newindep)

plt.scatter(newdep, copydf_pred,color='r')
plt.plot([newdep.min(), newdep.max()], [copydf_pred.min(), copydf_pred.max()], color = 'black', lw=2)
plt.xlabel("ccs")
plt.ylabel("ccs pred")
plt.title("Predictions vs. actual values in the training set")


# # Predicting the concrete strength for the given values

# In[23]:


predictordata = pd.DataFrame({'cement' : [175.0,320.0,320.0,320.0,530.0],
                              'blast furnace slag' : [13.0,0.0,0.0,73.0,359.0],
                              'FlyAsh' : [172.0,0.0,126.0,54.0,200.0],
                              'CoarseAggregate' : [1000.0,970.0,860.0,972.0,1145.0],
                              'FineAggregate' : [856.0,850.0,856.0,773.0,992.0],
                              'Water' : [156.0,192.0,209.0,181.0,247.0],
                              'superplasticizer' : [4.0,0.0,5.7,6.0,32.0],
                              'Age (day)' : [3.0,7.0,28.0,45.0,365.0]})
predictordata


# In[24]:


x1 = ['cement' , 'superplasticizer' , 'Age (day)']
X1 = predictordata[x]
ccspred = lm.predict(X1)
predictordata['ccs'] = ccspred   
predictordata

