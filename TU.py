#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


# In[2]:


DIR='../input/brain-tumor-classification-mri/Training'
CAT=[]
for cat in os.listdir(DIR):
    CAT.append(cat)


# In[3]:


CAT


# In[7]:


training_data=[]
training_label=[]
for cat in os.listdir(DIR):
    PATH=os.path.join(DIR,cat)
    for img in os.listdir(PATH):
        img_arr=cv2.imread(os.path.join(PATH,img),cv2.IMREAD_GRAYSCALE)
        new_arr=cv2.resize(img_arr,(100,100))
        #plt.imshow(img_arr,cmap='gray')
        training_data.append(new_arr)
        training_label.append(cat)
        #plt.show()


# In[11]:


X=[]
Y=[]
for img in training_data:
    X.append(img)


# In[19]:


Y=training_label


# In[21]:


Y[-1]


# In[22]:


import pandas as pd
Y=pd.DataFrame(Y)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y_lb=le.fit_transform(Y)


# In[23]:


Y_lb


# In[26]:


set(Y_lb)


# In[39]:


Y_lb[826]


# glioma_tumor=0
# meningioma_tumor=1
# No_tumor=2
# pituitary_tumor=3

# In[40]:


X=np.array(X).reshape(-1,100,100,1)


# In[47]:


X=X/255


# In[41]:


Y_lb=np.array(Y_lb)


# In[43]:


from sklearn.model_selection import train_test_split


# In[48]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y_lb,random_state=1)


# In[42]:


from keras.models import Sequential
from keras.layers import Conv2D,Dropout,MaxPooling2D
from keras.layers import Flatten,Activation
from keras.layers import Dense


# In[54]:


model=Sequential()
model.add(Conv2D(128,(3,3),activation='relu',input_shape=(100, 100, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(4,activation='softmax'))

model.compile(optimizer='RMSProp',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history=model.fit(X_train,Y_train,batch_size=10,epochs=10)


# In[56]:


history.history


# In[61]:


plt.plot(history.history['accuracy'],label='Accuracy')
plt.plot(history.history['loss'],label='Loss')
plt.legend(loc="upper right")
plt.show()


# In[106]:


y_pred=model.predict(X_test)
Y_pred=pd.DataFrame(y_pred)
Y_pred['pred']=Y_pred.idxmax(axis=1)
Y_pred['test']=Y_test
Y_pred['diff']=Y_pred['pred']-Y_pred['test']
Y_pred['diff'].value_counts()


# In[108]:


acc=(615/(615+56+22+11+7+7))
acc


# In[110]:


Pr={0:'glioma_tumor',1:'meningioma_tumor',2:'no_tumor',3:'pituitary_tumor'}


# In[111]:


Y_pred['pred']=Y_pred['pred'].map(Pr)


# In[137]:



def enter_new_data(DIR):
    img_arr=cv2.imread(DIR,cv2.IMREAD_GRAYSCALE)
    new_arr=cv2.resize(img_arr,(100,100))
    new_arr=np.array(new_arr).reshape(-1,100,100,1)
    y_pred=model.predict(new_arr)
    Y_pred=pd.DataFrame(y_pred)
    Y_pred['pred']=Y_pred.idxmax(axis=1)
    Y_pred['pred']=Y_pred['pred'].map(Pr)
    pre= Y_pred['pred']
    return pre


# In[138]:


DIR='../input/brain-tumor-classification-mri/Testing/no_tumor/image(10).jpg'
img_arr=cv2.imread(DIR,cv2.IMREAD_GRAYSCALE)
new_arr=cv2.resize(img_arr,(100,100))
plt.imshow(new_arr,cmap='gray')


# In[139]:


enter_new_data(DIR)


# In[77]:


plt.plot(pred)
plt.plot(Y_test)
plt.show()

