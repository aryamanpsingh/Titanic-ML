
# coding: utf-8

# In[496]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import numpy as np


# In[598]:


#Import training and testing csv
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# In[564]:


#Find total null values
train_df[50:100]


# In[599]:


#Set null ages to mean age
test_df.Age = test_df.Age.fillna(test_df.Age.mean())
train_df.Age = train_df.Age.fillna(train_df.Age.mean())

#Find mean fare by class - TRAIN
mean_fare_1 = train_df.Fare[train_df.Pclass==1].mean()
mean_fare_2 = train_df.Fare[train_df.Pclass==2].mean()
mean_fare_3 = train_df.Fare[train_df.Pclass==3].mean()

#Set null values to fair by class - TRAIN
train_df.Fare[train_df.Pclass==1] = train_df.Fare[train_df.Pclass==1].fillna(mean_fare_1)
train_df.Fare[train_df.Pclass==2] = train_df.Fare[train_df.Pclass==2].fillna(mean_fare_2)
train_df.Fare[train_df.Pclass==3] = train_df.Fare[train_df.Pclass==3].fillna(mean_fare_3)

#Find mean fare by class - TEST
mean_fare_1 = test_df.Fare[test_df.Pclass==1].mean()
mean_fare_2 = test_df.Fare[test_df.Pclass==2].mean()
mean_fare_3 = test_df.Fare[test_df.Pclass==3].mean()

#Set null values to fair by class - TEST
test_df.Fare[train_df.Pclass==1] = test_df.Fare[train_df.Pclass==1].fillna(mean_fare_1)
test_df.Fare[train_df.Pclass==2] = test_df.Fare[train_df.Pclass==2].fillna(mean_fare_2)
test_df.Fare[train_df.Pclass==3] = test_df.Fare[train_df.Pclass==3].fillna(mean_fare_3)

#Set Sex column to numbers
train_df.Sex[train_df.Sex=='male']=1
train_df.Sex[train_df.Sex=='female']=2
test_df.Sex[test_df.Sex=='male']=1
test_df.Sex[test_df.Sex=='female']=2

#Set Embarked column to numbers
test_df.Embarked[test_df.Embarked=='C']=1
test_df.Embarked[test_df.Embarked=='Q']=2
test_df.Embarked[test_df.Embarked=='S']=3
train_df.Embarked[train_df.Embarked=='C']=1
train_df.Embarked[train_df.Embarked=='Q']=2
train_df.Embarked[train_df.Embarked=='S']=3

test_df.Embarked = test_df.Embarked.fillna(0)
train_df.Embarked = train_df.Embarked.fillna(0)


# In[604]:


#Define features
features = ['Pclass','Sex','Embarked','SibSp','Parch','Age','Fare']
X = train_df[features]
y = train_df.Survived
test_X = test_df[features]


# In[605]:


#Define train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size=0.2)


# In[606]:


#Define Random Forest Classifier - fit with X and y
forest_model = RandomForestClassifier(random_state=1,n_estimators=400,max_features=6)
forest_model.fit(X,y)
forest_model_pred = forest_model.predict(test_X)

#Define Decision Tree Classifier, fit with training test split
tree_model = DecisionTreeClassifier(random_state=1,min_samples_split=0.9,min_samples_leaf=0.1)
tree_model.fit(train_X,train_y)
tree_model_pred = tree_model.predict(val_X)
#Calculating ROC-AUC score
tree_model_error = roc_auc_score(val_y, tree_model_pred)


# In[607]:


#Define Gradient Boosting Classifier - fit with X and y
gb = GradientBoostingClassifier(random_state=1,max_features=5,subsample=0.5,min_samples_leaf=4,min_impurity_decrease=0.2,min_samples_split=16)
gb.fit(X,y)
gb_pred = gb.predict(test_X)
#print(roc_auc_score(val_y,gb_pred))


# In[608]:


output = pd.DataFrame({'PassengerId': test_df.PassengerId,
                       'Survived': gb_pred})
output.to_csv('submission.csv', index=False)

