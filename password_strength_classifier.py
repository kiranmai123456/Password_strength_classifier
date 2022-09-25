#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb


# In[2]:


d=pd.read_csv("data.csv",error_bad_lines=False)
df=pd.DataFrame(d)
df


# In[3]:


df.head()


# In[4]:


df.isna().sum()


# In[5]:


df.dropna(inplace=True)


# In[6]:


df.isna().sum()


# In[7]:


x = np.array(df["password"])
y = np.array(df["strength"])


# In[8]:


sb.countplot(y)


# In[9]:


def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer(tokenizer=word)


# In[11]:


x=vect.fit_transform(x)


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[13]:


x_train.shape , x_test.shape


# In[14]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)


# In[15]:


x_pred=model.predict(x_train)
x_pred


# In[16]:


x_t_pred=model.predict(x_test)
x_t_pred


# In[17]:


model.score(x_train,x_pred)


# In[18]:


model.score(x_test,x_t_pred)


# In[19]:


import getpass
user=getpass.getpass("Enter Password:")
data=vect.transform([user]).toarray()
output=model.predict(data)
print(output)


# In[25]:


import getpass
user=getpass.getpass("Enter Password:")
data=vect.transform([user]).toarray()
output=model.predict(data)
print(output)
if(output==1):
    print("output 1 means your password is in medium level... It's better to set a strong password")
elif(output==0):
    print("output 0 means your password is too weak... It's better to set a strong password")
else:
    print("output 2 means your password is strong")


# In[21]:


from sklearn.metrics import r2_score
r2_score(y_test,x_t_pred)


# In[22]:


r2_score(y_train,x_pred)

