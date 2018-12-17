
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf

dataPhase = np.load("trainLenetdataPhase.npy")
dataMag = np.load("trainLenetdataMag.npy")
dataY = np.load("trainLenetdataY.npy")


# In[2]:


from keras.models import Sequential
from keras.layers import Convolution2D
model = Sequential()
model.add(Convolution2D(filters=6,kernel_size=(5,5),strides=1,activation='relu',input_shape=(32,32,2)))


# In[3]:


from keras.layers import MaxPooling2D
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Convolution2D(filters=16,kernel_size=(5,5),strides=1,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


# In[4]:


from keras.layers import Flatten
model.add(Flatten())


# In[5]:


from keras.layers import Dense
model.add(Dense(120,input_shape=(400,),activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(20,activation='softmax'))


# In[6]:


from keras.optimizers import Adam
optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])


# In[7]:



print(dataMag.shape,dataPhase.shape)
dataIn = np.stack((dataMag,dataPhase),axis=3)
dataIn = np.reshape(dataIn,(dataIn.shape[0],dataIn.shape[1],dataIn.shape[2],dataIn.shape[3]))
print(dataIn.shape)


# In[8]:


model.summary()


# In[9]:


model.fit(x=dataIn,y=dataY,epochs=20,batch_size=32)


# In[10]:


model.save('Lenet_model_with_MagAndPhase.h5') 






