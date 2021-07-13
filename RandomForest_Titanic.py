#!/usr/bin/env python
# coding: utf-8

# ## Exercise 4: RandomForest Classifier (5 Points)
# 
# In this exercise, you are going to work with the famous [titanic](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv) dataset.
# The RMS titanic was a British passenger liner that sank on its maiden voyage due to a collision with an iceberg on 15 April 1912. More than 1,500 people died in this tragedy.
# 
# The dataset contains information of 891 passengers out of the estimated 2,224 passengers and crew aboard.
# In the following, you are going to analyze the dataset and build a `RandomForest` classifier to predict the survival of a passenger.
# 
# **Note:** Since there is no definite _correct_ answer for analyzing, cleaning, and preparing data as well as for building and training machine learning models, we will not provide you with unittests for this exercise. Therefore, it is even more important to comment your code such that we can understand your ideas and intentions.

# ### a) Reading the data (0.5 Points)
# 
# Read the provided dataset file `titanic.dsv` using the Python library `pandas`.
# 
# The dataset contains the following attributes:
# * `survived`: If the passenger survived the accident (1=yes, 0=no)
# * `pclass`: Ticket class (1=expensive, 2=normal, 3=cheap)
# * `sex`: Sex of the passenger (male or female)
# * `age`: Age of the passenger
# * `sibsp`: Number of siblings/spouses on board
# * `parch`: Number of parents/children on board
# * `fare`: Fare
# * `embarked`: Town in which the passenger went on board (C=Cherbourg, S=Southampton, Q=Queenstown)
# * `class`:Ticket class (First=expensive, Second=normal, Third=cheap)
# * `who`: Sex of the passenger (man or woman)
# * `adult_male`: If the passenger is an adult male
# * `deck`: The deck on which the cabin of the passenger was located
# * `embark_town`: Town in which the passenger went on board
# * `alive`: If the passenger survived
# * `alone`: If the passenger was traveling alone

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().system('head -n 5 titanic.dsv')
titanic = pd.read_csv('titanic.dsv', sep = "|" )
display(titanic.head())


# ### b) Data analysis and data cleaning (1.5 Points)
# 
# Analyze the titanic dataset using `pandas` and `seaborn` and clean the data accordingly. Your solution should explicitly state which cleaning steps you apply and why you apply them.

# In[ ]:


titanic.shape


# At the first instant, it is noticable that the columns "who", "alive" and "class" are not needed because we already have their qualitative correspondings, and one of the same properties embarked or embark_town can also be removed.
# 

# In[ ]:


titanic.drop(["who","alive","embarked","class"],axis=1,inplace=True)


# In[ ]:


titanic.dtypes


# In[ ]:


titanic['deck'].value_counts()


# There are pretty high difference between useful deck classes and the dataset.
# 

# In[ ]:


list_of_Nan = [ i for i in titanic['deck'] if type(i) is not str]
print(len(list_of_Nan))


# It means that we have many irrelevant values in deck column, it will be better to drop the whole deck column which we can never utilize.

# In[ ]:


titanic.drop(['deck'],axis=1,inplace=True)


# In[ ]:


titanic.describe()


# We should have had 891 values for each column, but a large amount of age data are missing. What we need to do is dropping those rows from our dataset.
# 

# In[ ]:


titanic.dropna(inplace=True)
titanic.head(10)


# In[ ]:


titanic.describe()


# In[ ]:


titanic["age"].value_counts()


# Age column has many decimal numbers and we're rounding all the numbers by transforming directly to integer.

# In[ ]:


titanic["age"] = [int(i) for i in titanic['age']]
len(titanic)


# If someone has relative, this one isn't alone. A new column called "relatives" can sit in for all the sibsp, parch and alone. The total relative numbers bigger than 3 will be gathered in one group.

# In[ ]:


titanic.drop(["alone"],axis=1,inplace=True)
titanic.head()


# The siebling-spouse and the parent-children pairs can be collected in one group called relatives.

# In[ ]:


titanic['relatives']= titanic['sibsp'] + titanic['parch']
titanic.drop(["sibsp","parch"],axis=1,inplace=True)
    


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
fig, axes = plt.subplots(1,4, figsize=(20,8))
sns.boxplot(ax=axes[0], y=titanic["pclass"])
sns.boxplot(ax=axes[1], y= titanic["age"])
sns.swarmplot(ax=axes[1], y= titanic["age"],size=1.5,color="0.25")
sns.boxplot(ax=axes[2], y= titanic["relatives"])
sns.boxplot(ax=axes[3], y= titanic["fare"])
#sns.violinplot(ax=axes[3], y= titanic["fare"],size=1.5,color="0.25")


# In[ ]:


display(titanic[titanic["age"] > 65])


# This visualization was inefficient and that brings us to apply some adjustment to our data. Firstly, the outlier datas can be collected in the closest group so that they don't lose their effect totally. As for the age datas, the values over 65 will be removed since there are some outliers exceptionally like a survived man in the age of 80 which can manipulate the general behaviour of the dataset. 

# In[ ]:


for i in [4,5,6,7] :
    titanic["relatives"].replace(i,3,inplace=True)
titanic = titanic[titanic["age"] < 65]


# In[ ]:


display(titanic[titanic["fare"] > 220 ])


# The ratio of the survivors from the richest class is quite a lot and these datas are very precious for me. The ones over 500 are transferred to the second expensive class with the fare of 263.000 

# In[ ]:


titanic["fare"].replace(512.3292, 263.000, inplace= True)


# The values between 50 and 263.00 will be scaled to the range 50-100 so as to minimize the standing out of the outliers.

# In[ ]:


def f (x) :
    res = (x-50)/(263-50) * (100-50) + 50
    return res

titanic['fare']=titanic['fare'].apply(lambda x: f(x) if (x > 50.0) else x )
titanic.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
fig, axes = plt.subplots(1,4, figsize=(20,8))
sns.boxplot(ax=axes[0], y=titanic["pclass"],labels = ["pclass"])
sns.boxplot(ax=axes[1], y= titanic["age"])
sns.swarmplot(ax=axes[1], y= titanic["age"],size=1,color="0.25")
sns.boxplot(ax=axes[2], y= titanic["relatives"])
sns.boxplot(ax=axes[3], y= titanic["fare"])
sns.swarmplot(ax=axes[3], y= titanic["fare"],size=1,color="0.25")


# In[ ]:


titanic = titanic.reset_index(drop=True)


# ### c) Data preparation (1 Point)
# 
# Before you are able to train a machine learning model on your current dataset, you have to perform some data preprocessing. Concretely, the attributes `sex` and `embark_town` are given as strings. However, the `RandomForest` classifier you are going to build in the next exercise requires the data to be numeric (np.float32). Therefore, encode your data accordingly.
# 
# **Hint:** You can encode your data manually or have a look at sklearn [LabelEncocer](https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.LabelEncoder.html).

# In[ ]:


titanic["embark_town"].value_counts()


# There are totally 3 different towns which the passengers embarked from. These values will be categorized into groups called 0, 1, 2 .

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
titanic['embark_town'] = le.fit_transform(titanic["embark_town"])
titanic["sex"] = le.fit_transform(titanic["sex"])
titanic["adult_male"] = le.fit_transform(titanic["adult_male"])
titanic.head()


# In[ ]:


titanic = titanic.astype("float32")
titanic.dtypes


# In[ ]:


sns.pairplot(titanic,hue="survived")


# ### c) Training a RandomForest Classifier and assessing the model performance (2 Points)
# 
# Your task is to predict the survival of an individual passenger of the titanic. For this, build a `RandomForest` classifier using the `sklearn` library. Do not use more dann $200$ trees in the forest.
# 
# 
# Afterwards, use $5$-fold cross-validation and the prediction accuracy as metric to assess your model performance.

# After some tries, the columns adult_male, which can also be presented better by the features sex and age, embark_town and relatives have been proven to have quite a tiny effects to classifying the datas. Therefore, we'll keep on modelling with other existing features.

# In[ ]:


X = titanic[['sex',"pclass","age","fare"]]
Y = titanic["survived"]
Y.replace((1.0,0.0),('Yes',"No"),inplace=True)
Y.reset_index(drop=True)


# In[ ]:


Y.value_counts()


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(
                n_estimators = 200,
                criterion = "gini",
                max_depth = 5,
                min_samples_leaf =12,
                random_state=42, class_weight='balanced')

kfold = KFold(n_splits=5)
result = cross_val_score(classifier,X,Y,cv=kfold)
mean = np.mean(result)
result,mean


# In[ ]:


from sklearn.metrics import accuracy_score,classification_report
classifier.fit(X,Y)
Y_pred = classifier.predict(X)
accuracy_score(Y_pred,Y)


# Lastly, we want to find out the most distinguishing features for the survival classification. Manually split the given data into train and test data using a `test_size` of $25\%$, train the RandomForest classifier, and report the feature importance for each input variable.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42 )
clf = classifier.fit(X_train,Y_train)
Y_predicted = classifier.predict(X_test)
accuracy_score(Y_test,Y_predicted)


# In[ ]:


print(classification_report(Y_test,Y_predicted))


# In[ ]:


feature_importance = pd.DataFrame(classifier.feature_importances_,
                                 index = X.columns,
                                 columns = ["importance"]).sort_values("importance", ascending=False)
feature_importance


# To sum up, we lost many datas because of missing age datas, because it is about the life of the people, so sensitive. At the modelling, we decided to give the optimized parameter by applying manually some other values nearby. The model gave us maybe not the best but acceptable results. The difference between the prediction accuracies of our training and test sets isn't high. A surprising result, which the classification report shows us, is that our model works in predicting the non-survived persons much better than survived ones; however, we have %79 accuracy score in average. 
