
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import load_model


# In[2]:


LenetA = load_model('Lenet_model_with_Mag.h5')
LenetB = load_model('Lenet_model_with_MagAndPhase.h5') 
LenetC = load_model('Lenet_model_with_overlap16_turkey_Mag.h5') 
AlexModel = load_model('AlexNet_with_Mag.h5') 


# In[3]:


AlexvaldataY = np.load("valAlexnetdataY.npy")
LenetdataY = np.load("trainLenetdataY.npy")
print(LenetdataY.shape)
testLenetdataPhase = np.load("testLenetdataPhase.npy")
testLenetdataMag = np.load("testLenetdataMag.npy")
testLenet_tukeydataMag = np.load("testLenet_tukeydataMag.npy")
testAlexnetdataPhase = np.load("testAlexnetdataPhase.npy")
testAlexnetdataMag = np.load("testAlexnetdataMag.npy")


# In[4]:


print(testLenetdataPhase.shape)
print(testLenetdataMag.shape)
print(testLenet_tukeydataMag.shape)
print(testAlexnetdataPhase.shape)
print(testAlexnetdataMag.shape)
dataIn = np.stack((testLenetdataMag,testLenetdataPhase),axis=3)
dataIn = np.reshape(dataIn,(dataIn.shape[0],dataIn.shape[1],dataIn.shape[2],dataIn.shape[3]))


# In[5]:


y1 = LenetA.predict(testLenetdataMag)
y2 = LenetB.predict(dataIn)
y3 = LenetC.predict(testLenet_tukeydataMag)
y4 = AlexModel.predict(testAlexnetdataMag)


# In[6]:


print(y1.shape)
print(y2.shape)
print(y3.shape)
print(y2.shape)


# In[7]:


results=[]
err = 0
for i in range(y4.shape[0]):
    ans = -1
    ans1 = np.argmax(y1[i])
    ans2 = np.argmax(y2[i])
    ans3 = np.argmax(y3[i])
    ans4 = np.argmax(y4[i])
    choice = np.zeros(20)
    choice[ans1] += 1
    choice[ans2] += 1
    choice[ans3] += 1
    choice[ans4] += 1
    best = 0
    best_arg = []
    best_arg_ct = 0
    for i in range(20):
        if(choice[i]>=best):
            best = choice[i]
            best_arg.append(i)
    #print(best_arg[len(best_arg)-1],np.argmax(choice))
    for j in range(len(best_arg)):
        if(best_arg[j]==best):
            best_arg_ct = best_arg_ct + 1
    if(best_arg_ct>=2):
        print("2 vs 2")
        ans = ans4
    else:
        ans = best_arg[len(best_arg)-1]
    
    if(ans==-1):
        print("invalid ans")
        exit(1)
    results.append(ans)


# In[10]:


results = np.asarray(results,dtype='int64')
print(results.shape)
print(results)


# In[11]:


np.save('results.npy',results)

