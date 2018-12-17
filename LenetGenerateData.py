import numpy as np
import cv2
import os
import re
import threading
LenetdataPhase = []
LenetdataMag = []
LenetdataY=[]
Lenet_tukey_dataMag = []
def Pspec2_32(Phase_path):
    Pfig = cv2.imread(Phase_path)
    #print("./"+dir_str+"Spectrogram"+str(which_label)+"/Pspectrogram"+index+".png")
    gray = cv2.cvtColor(Pfig, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray, (32, 32))
    #cv2.imwrite('output.jpg', resized_image)
    temp= np.array(resized_image)
    temp = np.resize(temp,(32,32,1))
    return temp
def Mspec2_32(Mag_path):
    Mfig = cv2.imread(Mag_path)
    #print("./"+dir_str+"Spectrogram"+str(which_label)+"/Mspectrogram"+index+".png")
    gray = cv2.cvtColor(Mfig, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray, (32, 32))
    temp= np.array(resized_image)
    temp = np.resize(temp,(32,32,1))
    return temp
def job(i):
    _dir=dir_str+"Spectrogram"+str(i)
    for files in os.listdir(_dir):
        index = re.search(r'\d+', files).group()
        print(index)
        Phase_path = "./"+dir_str+"Spectrogram"+str(which_label)+"/Pspectrogram"+index+".png"
        Mag_path = "./"+dir_str+"Spectrogram"+str(which_label)+"/Mspectrogram"+index+".png"
        p = Pspec2_32(Phase_path)
        m = Mspec2_32(Mag_path)
        with lock:
            LenetdataPhase.append(p)
            LenetdataMag.append(m)
            LenetdataY.append([float(i==j) for j in range(20)])

dir_str = input("input directory(train,val,test):")
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
        
    LenetdataPhase=np.asarray(LenetdataPhase)
    LenetdataMag=np.asarray(LenetdataMag)
    LenetdataY=np.asarray(LenetdataY)
    np.save(dir_str+"LenetdataPhase.npy",LenetdataPhase)
    np.save(dir_str+"LenetdataMag.npy",LenetdataMag)
    np.save(dir_str+"LenetdataY.npy",LenetdataY)
elif(dir_str=="test"):
    _dir=dir_str+"Spectrogram"
    for i in range(2387):
        #print(files)
        hanning_Mag_filename = _dir+"/"+"hanning_Mag_testdata"+str(i)+".png"
        hanning_Phase_filename = _dir+"/"+"hanning_Phase_testdata"+str(i)+".png"
        tukey_Mag_filename = _dir+"/"+"overlap16_turkey_Mag_testdata"+str(i)+".png"
        m = Mspec2_32(hanning_Mag_filename)
        LenetdataMag.append(m)
        p = Pspec2_32(hanning_Phase_filename)
        LenetdataPhase.append(p)
        Tm = Mspec2_32(tukey_Mag_filename)
        Lenet_tukey_dataMag.append(Tm)
        print(i)
    np.save(dir_str+"LenetdataPhase.npy",LenetdataPhase)
    np.save(dir_str+"LenetdataMag.npy",LenetdataMag)
    np.save(dir_str+"Lenet_tukeydataMag.npy",Lenet_tukey_dataMag)