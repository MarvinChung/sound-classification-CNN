import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import os
import re
import threading
import cv2
thread_n = 20
def draw_phase_spectrogram(signal,Phase_path):
    #Phase
    Pfreq,Ptime,PS = spectrogram(signal, fs=1.0, window=('hanning'), nperseg=None, noverlap=None, nfft=256, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='phase')
    #Pfreq,Ptime,PS = spectrogram(signal, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=16, nfft=256, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='phase')
    #plt.pcolormesh(Ptime, Pfreq, PS)
    #plt.xlabel('Time [sec]')
    #plt.ylabel('Frequency [Hz]')
    #cur_axes = plt.gca()
    #cur_axes.axes.get_xaxis().set_visible(False)
    #cur_axes.axes.get_yaxis().set_visible(False)
    #plt.savefig("./Spectrogram"+str(which_label)+"/Pspectrogram"+index+".png")
    Pfig = cv2.normalize(PS, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(Phase_path,Pfig)
def draw_mag_spectrogram(signal,Mag_path):
    #Magnitude
    Mfreq,Mtime,MS = spectrogram(signal, fs=1.0, window=('hanning'), nperseg=None, noverlap=None, nfft=256, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
    #Mfreq,Mtime,MS = spectrogram(signal, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=16, nfft=256, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
    #plt.pcolormesh(Mtime, Mfreq, MS)
    #plt.xlabel('Time [sec]')
    #plt.ylabel('Frequency [Hz]')
    #cur_axes = plt.gca()
    #cur_axes.axes.get_xaxis().set_visible(False)
    #cur_axes.axes.get_yaxis().set_visible(False)
    #plt.savefig("./Spectrogram"+str(which_label)+"/Mspectrogram"+index+".png")
    Mfig = cv2.normalize(MS, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(Mag_path,Mfig)
def draw_tukey_noverlap_phase_spectrogram(signal,Phase_path):
    #Phase
    #Pfreq,Ptime,PS = spectrogram(signal, fs=1.0, window=('hanning'), nperseg=None, noverlap=None, nfft=256, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='phase')
    Pfreq,Ptime,PS = spectrogram(signal, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=16, nfft=256, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='phase')
    #plt.pcolormesh(Ptime, Pfreq, PS)
    #plt.xlabel('Time [sec]')
    #plt.ylabel('Frequency [Hz]')
    #cur_axes = plt.gca()
    #cur_axes.axes.get_xaxis().set_visible(False)
    #cur_axes.axes.get_yaxis().set_visible(False)
    #plt.savefig("./Spectrogram"+str(which_label)+"/Pspectrogram"+index+".png")
    Pfig = cv2.normalize(PS, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(Phase_path,Pfig)
def draw_tukey_noverlap_mag_spectrogram(signal,Mag_path):
    #Magnitude
    #Mfreq,Mtime,MS = spectrogram(signal, fs=1.0, window=('hanning'), nperseg=None, noverlap=None, nfft=256, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
    Mfreq,Mtime,MS = spectrogram(signal, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=16, nfft=256, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
    #plt.pcolormesh(Mtime, Mfreq, MS)
    #plt.xlabel('Time [sec]')
    #plt.ylabel('Frequency [Hz]')
    #cur_axes = plt.gca()
    #cur_axes.axes.get_xaxis().set_visible(False)
    #cur_axes.axes.get_yaxis().set_visible(False)
    #plt.savefig("./Spectrogram"+str(which_label)+"/Mspectrogram"+index+".png")
    Mfig = cv2.normalize(MS, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(Mag_path,Mfig)

def read_files(_dir,which_label):
    #index =0
    for files in os.listdir(_dir):
        signal = np.load(_dir+files)
        index = re.search(r'\d+', files).group()
        print(_dir,index)
        Phase_path = "./"+dir_str+"Spectrogram"+str(which_label)+"/Pspectrogram"+index+".png"
        Mag_path = "./"+dir_str+"Spectrogram"+str(which_label)+"/Mspectrogram"+index+".png"
        draw_mag_spectrogram(signal,Mag_path)
        draw_phase_spectrogram(signal,Phase_path)
        #index+=1

tdir = []
dir_str = input("input directory(train,val,test):")
if(dir_str!="train" and dir_str!="val" and dir_str!="test"):
    print("invalid directory")
    exit(1)
if(dir_str =="train" or dir_str == "val"):
    tdir.append("./"+dir_str+"/Tettigonioidea1/")
    tdir.append("./"+dir_str+"/Tettigonioidea2/")
    tdir.append("./"+dir_str+"/drums_Snare/")
    tdir.append("./"+dir_str+"/Grylloidea1/")
    tdir.append("./"+dir_str+"/drums_MidTom/")
    tdir.append("./"+dir_str+"/drums_HiHat/")
    tdir.append("./"+dir_str+"/drums_Kick/")
    tdir.append("./"+dir_str+"/drums_SmallTom/")
    tdir.append("./"+dir_str+"/guitar_chord2/")
    tdir.append("./"+dir_str+"/Frog1/")
    tdir.append("./"+dir_str+"/Frog2/")
    tdir.append("./"+dir_str+"/drums_FloorTom/")
    tdir.append("./"+dir_str+"/guitar_7th_fret/")
    tdir.append("./"+dir_str+"/drums_Rim/")
    tdir.append("./"+dir_str+"/Grylloidea2/")
    tdir.append("./"+dir_str+"/guitar_3rd_fret/")
    tdir.append("./"+dir_str+"/drums_Ride/")
    tdir.append("./"+dir_str+"/guitar_chord1/")
    tdir.append("./"+dir_str+"/guitar_9th_fret/")
    tdir.append("./"+dir_str+"/Frog3/")
    #draw_phase_spectrogram(np.load("./train/guitar_9th_fret/1.npy"),18,1)
    for i in range(20):
        os.makedirs(dir_str+"Spectrogram"+str(i), exist_ok = True)
    threads = []
    for i in range(thread_n):
        global lock
        lock = threading.Lock()
        threads.append(threading.Thread(target = read_files , args = (tdir[i],i,)))
        threads[i].start()

         
    for i in range(thread_n):
      threads[i].join()
    print("Done")
elif(dir_str == "test"):
    samples = np.load("test.npy")
    os.makedirs(dir_str+"Spectrogram", exist_ok = True)
    for i in range(samples.shape[0]):
        tukey_mag_path = "./"+dir_str+"Spectrogram/overlap16_turkey_Mag_testdata"+str(i)+".png"
        hanning_mag_path = "./"+dir_str+"Spectrogram/hanning_Mag_testdata"+str(i)+".png"
        hanning_phase_path = "./"+dir_str+"Spectrogram/hanning_Phase_testdata"+str(i)+".png"
        draw_tukey_noverlap_mag_spectrogram(samples[i],tukey_mag_path)
        draw_mag_spectrogram(samples[i],hanning_mag_path)
        draw_phase_spectrogram(samples[i],hanning_phase_path)
    print("Done")
