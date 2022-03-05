#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
fruit = pd.read_csv('fruit_new.csv')
fruit.drop('fruit_label', axis='columns', inplace=True)
fruit.head()


# In[2]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label1 = le.fit_transform(fruit['fruit_name'])
label2 =  le.fit_transform(fruit['fruit_subtype'])
label1


# In[3]:


label2


# In[4]:


fruit.drop("fruit_name", axis=1, inplace=True)
fruit.drop("fruit_subtype", axis=1, inplace=True)
fruit["fruit_name"] = label1
fruit["fruit_subtype"] = label2
fruit


# In[5]:


X= fruit.drop(labels= 'fruit_subtype', axis = 1)
y= fruit['fruit_subtype']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train, y_train)
p=clf.predict(X_test)
c=multilabel_confusion_matrix(y_test,p)
c


# In[6]:


print('accuracy: ',accuracy_score(y_test,p))

plt.figure(figsize=(20,12))
tree.plot_tree(clf,rounded=True,filled=True)
plt.show()


# In[7]:


clf=DecisionTreeClassifier(criterion= 'gini',max_depth= 17,min_samples_leaf= 3,min_samples_split= 12,splitter= 'random')
clf.fit(X_train,y_train)
print('accuracy: ',accuracy_score(y_test,p))
plt.figure(figsize=(20,12))
tree.plot_tree(clf,rounded=True,filled=True)
plt.show()


# In[8]:


from sklearn.metrics import classification_report
acc = accuracy_score(y_test, p)
print("Accuracy: ", acc)
print(classification_report(y_test, p))
print("Classification Report: ", acc)


# In[ ]:





# In[ ]:




