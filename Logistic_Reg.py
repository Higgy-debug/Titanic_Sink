* IMPORT LIBRARIES:

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import math

titanic_sink = pd.read_csv(r"C:\Users\HP\Downloads\train.csv")

titanic_sink.head(11)

* ANALYZING DATA:


sns.countplot(x="Sex" , data=titanic_sink)
sns.countplot(x="Survived" , hue="Sex" , data=titanic_sink)
sns.countplot(x="Survived" , hue = "Pclass" , data=titanic_sink)
titanic_sink["Age"].plot.hist()

titanic_sink.info()

sns.countplot(x="SibSp" , data=titanic_sink)

* DATA WRANGLING:


titanic_sink.isnull()



titanic_sink.isnull().sum()



sns.heatmap(titanic_sink.isnull() , yticklabels = False)



sns.heatmap(titanic_sink.isnull() , yticklabels = False , cmap="viridis")



sns.boxplot(x="Pclass" , y="Age", data=titanic_sink)



sns.boxplot(x="Sex" , y="Age", data=titanic_sink)



titanic_sink.head(6)



titanic_sink.drop("Cabin" , axis=1 , inplace=True)



titanic_sink.head(5)



titanic_sink.dropna(inplace=True)



titanic_sink.isnull().sum()



titanic_sink.head(3)



pd.get_dummies(titanic_sink['Sex'])



sex=pd.get_dummies(titanic_sink['Sex'])
sex.head(5)



sex=pd.get_dummies(titanic_sink['Sex'] , drop_first=True)
sex.head(5)



hirthi=pd.get_dummies(titanic_sink["Embarked"])



hirthi.head(5)



hirthi=pd.get_dummies(titanic_sink["Embarked"] , drop_first = True)



hirthi.head(5)



appu = pd.get_dummies(titanic_sink['Pclass'] , drop_first = True)
appu.head(4)



titanic_sink=pd.concat([titanic_sink , sex , hirthi , appu] , axis=1)
titanic_sink.head(5)



titanic_sink.drop(['Sex' , 'Embarked' , 'PassengerId' , 'Name' ,'Ticket'] ,axis = 1 ,inplace=True)
titanic_sink()



titanic_sink.drop('Pclass' , axis=1 , inplace=True)



titanic_sink.head()

* TRAINING THE DATA


X = titanic_sink.drop("Survived" , axis=1)
y = titanic_sink["Survived"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state= 1)



from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()

logmodel.fit(X_train , y_train)
predictions=logmodel.predict(X_test)



from sklearn.metrics import classification_report
classification_report(y_test , predictions)

* ACCURACY CHECK:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test , predictions)



from sklearn.metrics import accuracy_score
accuracy_score(y_test , predictions)
