Sound Classification with CNN
---
Sound Datas (already convert to .npy format):
![](https://i.imgur.com/YH7SjRk.png)


**Method**: Changing the sound datas to spectrogram 
**Model Inputs**: Spectrograms


**Models:**
1. Lenet
    -1.training data using magnitude
    -2.training data using magnitude and phase
    -3.training with different spectrogram compare to 1,2
3. Alenet
    -1.training data using magnitude
---

**Lenet 1:**
training data using magnitude
model summary:
![](https://i.imgur.com/iIsGqDF.png =90%x90%)


| Input Image | size | window | overlap
| -------- | -------- | -------- | -------- |
| Spectrogram    | 32 × 32 × 1  |hanning| none


| layer | kernel size | stride | activation|
| -------- | -------- | -------- | -------- |
| 1.convolution|5 × 5|1| relu
| 2.maxpooling |2 × 2|2|relu
|3.convolution|5 × 5|1|relu
|4.maxpooling|2 × 2|2|relu
|5.flatten|-|-|-
|6.FC|-|-|relu
|7.FC|-|-|relu
|8.FC|-|-|softmax


| optimizer | loss | metrics |epochs
| -------- | -------- | -------- |-------- |
| adam    | categorical_crossentropy     | accuracy|20
![](https://i.imgur.com/ax02eJh.png =50%x50%)
**Lenet 2:**
training data using magnitude and phase
model summary:same as Lenet1
![](https://i.imgur.com/JHNvt4g.png =50%x50%)
**Lenet3:**
spectrogram using turkey window and noverlap = 16 
training data using magnitude
model summary: same as Lenet1
![](https://i.imgur.com/Nia8fDF.png =70%x70%)

**Alenet**
![](https://i.imgur.com/8j8Qz4y.png =90%x90%)
| Input Image | size | window | overlap | note | 
| -------- | -------- | -------- | -------- | -------- |
| Spectrogram|224 × 224 × 3|hanning|None| the proccess will make the image to 227×227×3

| layer | kernel size | stride | activation|
| -------- | -------- | -------- | -------- |
| 1.convolution|11 × 11|4| relu
| 2.maxpooling |3 × 3|1|relu
| 3.convolution|5 × 5|1| relu
| 4.maxpooling |3 × 3|2|relu
| 5.convolution|3 × 3|1|relu
| 6.convolution|3 × 3|1|relu
| 7.convolution|3 × 3|1|relu
| 8.maxpooling|3 × 3|2|relu
| 9.flatten|-|-|-
|10.FC|-|-|relu
|11.FC|-|-|relu
|12.FC|-|-|relu
|13.FC|-|-|softmax


| optimizer | loss | metrics |epochs
| -------- | -------- | -------- |-------- |
| adam    | categorical_crossentropy     | accuracy|3
![](https://i.imgur.com/1lZW4RE.png =50%x50%)

**Expirment1**:
---
Testing with validation set
Lenet1(Spectrogram with magnitude) VS Lenet2(Spectrogram with magnitude and phase)

A = spectrogram(signal, fs=1.0, window=('hanning'), nperseg=None, noverlap=None, nfft=256, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
B = spectrogram(signal, fs=1.0, window=('hanning'), nperseg=None, noverlap=None, nfft=256, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='phase')

Lenet1 : A
Lenet2 : A+B
![](https://i.imgur.com/ax02eJh.png =50%x50%)

![](https://i.imgur.com/JHNvt4g.png =50%x50%)
Lenet2 is worse
**What I have learned:**
Phase does not really matter.


Expirment 2
---
AlexNet VS Lenet with same spectrogram
![](https://i.imgur.com/1lZW4RE.png =50%x50%)
![](https://i.imgur.com/dgRl36x.png =50%x50%)
Alexnet image:224×224×3
Lenet image:32×32×1
Alexnet with only 3 epochs can generate better result on validation set
Alexnet accuracy on train set:
![](https://i.imgur.com/MZrLMps.png)
Lenet accuracy on train set:
![](https://i.imgur.com/NugXJgE.png)
Although Lenet has higher accuracy on train set, Alexnet behaves better on validation set. 
**Alexnet progess:**
![](https://i.imgur.com/IiAW9qB.png =75%x75%)
**Lenet progess:**
![](https://i.imgur.com/eyvVTsj.png =75%x75%)
However Alexnet converge quicker than Lenet.
**What I have learned:**
Assuming more complicated network and input generate better results.


Expirment3
---
Lenet1 VS Lenet3

Lenet1 spectrogram: spectrogram(signal, fs=1.0, window=('hanning'), nperseg=None, noverlap=None, nfft=256, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
Lenet3 spectrogram: spectrogram(signal, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=16, nfft=256, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='magnitude')
![](https://i.imgur.com/ax02eJh.png =50%x50%)


![](https://i.imgur.com/Nia8fDF.png =50%x50%)
Lenet3 is better.
Lenet3 using tukey window and with noverlap = 16.
Lenet using hanning window and noverlap parameter is none.
**What I have learned:**
Spectrogram with overlapping may generate better result.

How to execute program:
---

1. Ensure you have the training directory train if you want to train your own model.
2. type python3 Spectrograms.py
* The program will ask you which directory(train/val/test) you want to generate spectrograms.
* val is for validation set which is optional, type test if you only want to see the result.
* Generate directories with format: train/val/test+Spectrogram+label(ex: trainSpectrogram0).
* Generate spectrograms(MSpectrogram:magnitude spectrogram, PSpectrogram:phase spectrogram) to the corresponding label directory.
3. type python3 LenetGenerateData.py 
* Generate numpy arrays for training.
* The program will ask you which set(train/val/test) you want to generate numpy arrays.
* val is for validation set which is optional, type test if you only want to see the result.
* LenetdataPhase.npy, LenetdataMag.npy, LenetdataY.npy **with a prefix train/val/test** will be created.
5. type python3 AlexnetGenerateData.py
* Generate numpy arrays for training.
* The program will ask you which set(train/val/test) you want to generate numpy arrays.
* val is for validation set which is optional, type test if you only want to see the result.
* AlexnetdataPhase.npy, AlexnetdataMag.npy, AlexnetdataY.npy **with a prefix train/val/test** will be created.
6. Open jupyter notebook (optional)
* Use Lenet_with_Mag.ipynb to create Lenet1 model
* Use Lenet_with_MagAndPhase.ipynb to create Lenet2 model
* To create Lenet3 or modify spectrogram, you need to change the spectrogram function in Spectrograms.py and repeat the above steps.
* Use Alex.ipynb to create Alex model
7. Use Python3 to do the same thing as step 6 (optional)
- type python3 Lenet_with_Mag.py
- type python3 Lenet_with_MagAndPhase.py
- type python3 Alex.py
8. Open jupyter notebook (optional)
* Use ValidationResult.ipynb to see validation result
9. Type python3 Voting.py
* Generate the resuls.npy base on the test.npy
* Using Lenet1, Lenet2, Lenet3 and Alexnet to vote the result
