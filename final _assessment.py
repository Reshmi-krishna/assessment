#!/usr/bin/env python
# coding: utf-8

# In[160]:


import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sn


# In[161]:


traindata=pd.read_csv("C:/Users/Resh/Downloads/train_ctrUa4K.csv")
traindata.head()


# In[162]:


traindata['Loan_Status'].value_counts(normalize=True)


# In[163]:


traindata.head()


# In[164]:


traindata['Dependents'].replace('3+', 3,inplace=True)


# In[165]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
traindata['Gender']=le.fit_transform(traindata['Gender'])
traindata['Married']=le.fit_transform(traindata['Married'])
traindata['Self_Employed']=le.fit_transform(traindata['Self_Employed'])
traindata['Property_Area']=le.fit_transform(traindata['Property_Area'])
traindata['Loan_Status']=le.fit_transform(traindata['Loan_Status'])
traindata['Education']=le.fit_transform(traindata['Education'])

traindata.head()


# In[166]:


traindata=traindata.drop(['Loan_ID'],axis=1)
traindata


# In[167]:


traindata.isna().sum()


# In[168]:


traindata1=traindata.replace(['3+'], '4')
traindata1
for i in ['LoanAmount','Dependents' ,'Loan_Amount_Term','Credit_History']:
    traindata1[i]=traindata1[i].fillna(traindata1[i].median())


# In[169]:


train=pd.get_dummies(traindata1)
train


# In[170]:


x = train.drop('Loan_Status', axis=1)
y = train.Loan_Status


# In[171]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[172]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


# In[173]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# In[201]:


my_logit_model = LogisticRegression()
model2=my_logit_model.fit(x_train, y_train)
pred=model2.predict(x_test)
print('accuracy is',accuracy_score(y_test,pred))


kfold = KFold(n_splits=10, random_state=7,shuffle=True)
results = cross_val_score(my_logit_model, x_train, y_train, cv=kfold)
print("Logistic Regression Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[175]:


testdata=pd.read_csv("C:/Users/Resh/Downloads/test_lAUu6dG.csv")
testdata


# In[176]:


from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
testdata['Gender']=le1.fit_transform(testdata['Gender'])
testdata['Married']=le1.fit_transform(testdata['Married'])
testdata['Self_Employed']=le1.fit_transform(testdata['Self_Employed'])
testdata['Property_Area']=le1.fit_transform(testdata['Property_Area'])
testdata['Education']=le1.fit_transform(testdata['Education'])
testdata.head()


# In[177]:


testdata1=testdata.replace(['3+'], '3')
testdata1
for i in ['LoanAmount','Dependents' ,'Loan_Amount_Term','Credit_History']:
    testdata1[i]=testdata1[i].fillna(testdata1[i].median())
testdata1
x_test1=testdata1.drop(['Loan_ID'],axis=1)
x_test2=pd.get_dummies(x_test1)
x_test2


# In[182]:


x


# In[178]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# In[179]:


randForest = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
randForest.fit(x_train,y_train)
y_pred_class  = randForest.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

print('Accuracy is',accuracy_score(y_pred_class,predictions))



# In[180]:


randForestNew = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
randForestNew.fit(x,y)


# In[184]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_test2=scaler.fit_transform(x_test2)
x_test2


# In[204]:


y_test_pread_class = randForestNew.predict(x_test2)
y_prediction1=model2.predict(x_test2)


# In[205]:


y_test_pread_class
y_prediction=pd.DataFrame(y_test_pread_class)
y_prediction2=pd.DataFrame(y_prediction1)


# In[189]:


solution=pd.read_csv("C:/Users/Resh/Downloads/sample_submission_49d68Cx.csv")
solution


# In[192]:


solution['Loan_Status']=y_prediction
solution


# In[195]:


solution2 = solution.reset_index(drop=True)
solution2
solution2['Loan_Status'].replace ({1: 'Y', 0: 'N'},inplace=True)
solution3=pd.DataFrame(solution2)
solution3


# In[199]:


solution3.to_csv('C:/Users/Resh/Downloads/assesment3.csv',index=False)


# In[206]:


solution['Loan_Status']=y_prediction2
solution


# In[207]:


solution2 = solution.reset_index(drop=True)
solution2
solution2['Loan_Status'].replace ({1: 'Y', 0: 'N'},inplace=True)
solution3=pd.DataFrame(solution2)
solution3


# In[209]:


solution3.to_csv('C:/Users/Resh/Downloads/assesment4.csv',index=False)
#got an accuracy of 0.73


# In[ ]:





# In[63]:





# In[ ]:





# In[64]:





# In[ ]:





# In[66]:





# In[ ]:




