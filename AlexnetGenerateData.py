import numpy as np
import cv2
import os
import re
import threading
AlexnetdataPhase = []
AlexnetdataMag = []
AlexnetdataY=[]

dir_str = input("input directory(train,val,test):")
def Pspec2_224(Phase_path):
    Pfig = cv2.imread(Phase_path)
    resized_image = cv2.resize(Pfig, (224, 224))
    #cv2.imwrite('output.jpg', resized_image)
    temp= np.array(resized_image)
    temp = np.resize(temp,(224,224,3))
    return temp
def Mspec2_224(Mag_path):
    Mfig = cv2.imread(Mag_path)
    resized_image = cv2.resize(Mfig, (224, 224))
    temp= np.array(resized_image)
    temp = np.resize(temp,(224,224,3))
    return temp
def job(i):
    _dir=dir_str+"Spectrogram"+str(i)
    for files in os.listdir(_dir):
        index = re.search(r'\d+', files).group()
        print(index)
        Phase_path = "./"+dir_str+"Spectrogram"+str(i)+"/Pspectrogram"+index+".png"
        p = Pspec2_224(Phase_path)
        Mag_path = "./"+dir_str+"Spectrogram"+str(i)+"/Mspectrogram"+index+".png"
        m = Mspec2_224(Mag_path)
        with lock:
            AlexnetdataPhase.append(p)
            AlexnetdataMag.append(m)
            AlexnetdataY.append([float(i==j) for j in range(20)])


if(dir_str!="train" and dir_str!="val" and dir_str!="test"):
    print("invalid directory")
    exit(1)
if(dir_str =="train" or dir_str == "val"):
    threads = []
    for i in range(20):
        global lock
        lock = threading.Lock()
        threads.append(threading.Thread(target = job, args = (i,)))
        threads[i].start()
    for i in range(20):
        threads[i].join()
        
    AlexnetdataPhase=np.asarray(AlexnetdataPhase)
    AlexnetdataMag=np.asarray(AlexnetdataMag)
    AlexnetdataY=np.asarray(AlexnetdataY)
    np.save(dir_str+"AlexnetdataPhase.npy",AlexnetdataPhase)
    np.save(dir_str+"AlexnetdataMag.npy",AlexnetdataMag)
    np.save(dir_str+"AlexnetdataY.npy",AlexnetdataY)
elif(dir_str=="test"):
    _dir=dir_str+"Spectrogram"
    for i in range(2387):
        Phase_file_name = _dir+"/"+"hanning_Phase_testdata"+str(i)+".png"
        Mag_file_name = _dir+"/"+"hanning_Mag_testdata"+str(i)+".png"
        m = Mspec2_224(Mag_file_name)
        AlexnetdataMag.append(m)
        p = Pspec2_224(Phase_file_name)
        AlexnetdataPhase.append(p)
        print(i)
    np.save(dir_str+"AlexnetdataPhase.npy",AlexnetdataPhase)
    np.save(dir_str+"AlexnetdataMag.npy",AlexnetdataMag)