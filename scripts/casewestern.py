# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:27:59 2020

@author: Abdeljalil Letrache <abdeljalilletrache@outlook.com>
"""

from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm
import seaborn as sn
from scipy.io import loadmat
from skimage.transform import resize
import pywt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.utils import to_categorical

#%% Matplotlib LaTeX Settings
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('axes', titlesize=12)
plt.rc('text', usetex=True)

#%% General settings
path_data = '/home/abdeljalil/Workspace/casewestern/'

NB = ['Normal_0.mat', 'Normal_1.mat', 'Normal_2.mat', 'Normal_3.mat']

B07 = ['B007_0.mat', 'B007_1.mat', 'B007_2.mat', 'B007_3.mat']
B14 = ['B014_0.mat', 'B014_1.mat', 'B014_2.mat', 'B014_3.mat']
B21 = ['B021_0.mat', 'B021_1.mat', 'B021_2.mat', 'B021_3.mat']

IR07 = ['IR007_0.mat', 'IR007_1.mat', 'IR007_2.mat', 'IR007_3.mat']
IR14 = ['IR014_0.mat', 'IR014_1.mat', 'IR014_2.mat', 'IR014_3.mat']
IR21 = ['IR021_0.mat', 'IR021_1.mat', 'IR021_2.mat', 'IR021_3.mat']

OR07 = ['OR007@6_0.mat', 'OR007@6_1.mat', 'OR007@6_2.mat', 'OR007@6_3.mat']
OR14 = ['OR014@6_0.mat', 'OR014@6_1.mat', 'OR014@6_2.mat', 'OR014@6_3.mat']
OR21 = ['OR021@6_0.mat', 'OR021@6_1.mat', 'OR021@6_2.mat', 'OR021@6_3.mat']

FULL_DATA = [NB, B07, B14, B21, IR07, IR14, IR21, OR07, OR14, OR21]

plotting_data = ['B007_0.mat', 'B014_0.mat', 'B021_0.mat', 'IR007_0.mat', 'IR014_0.mat','IR021_0.mat','OR007@6_0.mat','OR014@6_0.mat','OR021@6_0.mat']

plotting_sample_data_labels = ['BF 0.18', 'BF 0.36', 'BF 0.54', 'IF 0.18', 'IR 0.36', 'IR 0.54', 'OF 0.18', 'OF 0.36', 'OF 0.54']

plotting_labels = ['Normal', 'BF 0.18', 'BF 0.36', 'BF 0.54', 'IF 0.18', 'IR 0.36', 'IR 0.54', 'OF 0.18', 'OF 0.36', 'OF 0.54']


#%% Plotting data generation principal
fig = plt.figure(figsize=(4,1.5))
gs = GridSpec(1, 3, figure=fig)
axis1 = fig.add_subplot(gs[0,:2])
axis2 = fig.add_subplot(gs[0,2])

data = loadmat(os.path.join(path_data, 'Normal_0.mat'))
vib = data[list(data.keys())[3]]
vib = vib[64*64:2*64*64].reshape(64, 64)

axis1.plot(vib.ravel(), c='k', lw=1)
axis1.set_xticks([])
axis1.set_yticks([])
axis1.set_xlabel('Time-domain signal (length 4096)')


axis2.imshow(vib, cmap='gray', aspect=1)
axis2.set_xticks([])
axis2.set_yticks([])
axis2.set_xlabel('64'r'$\times$''64 image')

fig.tight_layout(w_pad=.5)

fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/cw_bearings_data_generation.pdf')

#%% Plotting data generation principal [French]
fig = plt.figure(figsize=(4,1.5))
gs = GridSpec(1, 3, figure=fig)
axis1 = fig.add_subplot(gs[0,:2])
axis2 = fig.add_subplot(gs[0,2])

data = loadmat(os.path.join(path_data, 'Normal_0.mat'))
vib = data[list(data.keys())[3]]
vib = vib[64*64:2*64*64].reshape(64, 64)

axis1.plot(vib.ravel(), c='k', lw=1)
axis1.set_xticks([])
axis1.set_yticks([])
axis1.set_xlabel('Domaine temporel (longeur 4096)')


axis2.imshow(vib, cmap='gray', aspect=1)
axis2.set_xticks([])
axis2.set_yticks([])
axis2.set_xlabel('Image 64'r'$\times$''64')

fig.tight_layout(w_pad=.5)

fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/cw_bearings_data_generation_fr.pdf')

#%% Plotting Faults
fig, axes = plt.subplots(3,3,figsize=(3,3.8))

for i in range(9):
    data = loadmat(os.path.join(path_data, plotting_data[i]))
    vib = data[list(data.keys())[3]]
    vib = vib[64*64:2*64*64].reshape(64, 64)
    axes.ravel()[i].imshow(vib, cmap='gray')
    axes.ravel()[i].set_xticks([])
    axes.ravel()[i].set_yticks([])
    axes.ravel()[i].set_title(plotting_sample_data_labels[i])
fig.tight_layout(h_pad=.2, pad=.5)
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/cw_bearings_faults_samples.pdf')

#%% Generating dataset
train_data_x = zeros((0,64,64))
train_data_y = zeros((0,1))

for bearing_state in enumerate(FULL_DATA):
    for load in bearing_state[1]:
        data = loadmat(os.path.join(path_data, load))
        vibration = data[list(data.keys())[3]]
        
        number_of_samples = (vibration.shape[0]//(64*64))
        usable_length = number_of_samples*64*64
        vibration = vibration[:usable_length]
        vibration = vibration.reshape(-1,64,64)
        
        train_data_x = concatenate((train_data_x, vibration), axis=0)

        labels = ones((number_of_samples,1)) * bearing_state[0]
        train_data_y = concatenate((train_data_y, labels), axis=0)

train_data_y = to_categorical(train_data_y)

#%% Fourier analysis

data = loadmat(os.path.join(path_data, FULL_DATA[1][2]))
vibration = data[list(data.keys())[3]]
vibration_trunc = vibration[:2000]

sampling_frequency = 12*10**3
sampled_points = vibration_trunc.shape[0]

fft_y=fft.fft(vibration_trunc)
fft_x = linspace(0, sampling_frequency, sampled_points)

plt.plot(fft_x,abs(fft_y))


#%% Wavelet analysis
sampling_frequency = 48*10**3
data = loadmat(os.path.join(path_data, FULL_DATA[7][2]))
vibration = data[list(data.keys())[3]]
vibration_trunc = vibration[:5000]

[coef, freq] = pywt.cwt(vibration_trunc, arange(1,129),'morl',1/sampling_frequency)
#%%

plt.contourf(linspace(0,5000,5000), freq, square(abs(coef.reshape(128,5000))), levels=40)

#%% Plotting classes samples count
counts = []

for i in range(10):
    count = (train_data_y==i).sum()
    counts.append(count)

fig, axis = plt.subplots(1,1, figsize=(3.5,2))
axis.bar(arange(len(counts)),counts, fill=False,color='k')
axis.set_yticks(unique(array(counts)))
axis.set_yticklabels(unique(array(counts)))
axis.set_xticks(arange(len(plotting_labels)))
axis.set_xticklabels(plotting_labels,rotation=45)
axis.set_xlabel('Classes')
axis.set_ylabel('Number of samples')

fig.savefig('D:\\Thesis\\Document\\figures\\cw_bearings_faults_count.pdf', bbox_inches = 'tight')

#%% Faults and counts

fig, axes = plt.subplots(3,3,figsize=(3,3.8))

for i in range(9):
    data = loadmat(os.path.join(path_data, plotting_data[i]))
    vib = data[list(data.keys())[3]]
    vib = vib[64*64:2*64*64].reshape(64, 64)
    axes.ravel()[i].imshow(vib, cmap='gray')
    axes.ravel()[i].set_xticks([])
    axes.ravel()[i].set_yticks([])
    axes.ravel()[i].set_title(plotting_sample_data_labels[i])
fig.tight_layout(h_pad=.2, pad=.5)
fig.savefig('D:\\Thesis\\Document\\figures\\cw_bearings_faults_samples_count.pdf')

#%% Creating model
model = Sequential()
model.add( Conv2D(32, (3,3), strides=(1, 1), padding='same', input_shape=(1,64,64), name='Conv1') )
model.add( MaxPool2D((2,2), padding='same', name='MaxPool1') )
model.add( Conv2D(64, (3,3), strides=(1, 1), padding='same', name='Conv2') )
model.add( MaxPool2D((2,2), padding='same', name='MaxPool2') )
model.add( Conv2D(128, (3,3),strides=(1, 1), padding='same', name='Conv3') )
model.add( MaxPool2D((2,2), padding='same', name='MaxPool3') )
model.add( Flatten() )
model.add( Dense(128, activation='relu', name='Dense1') )
model.add( Dropout(rate=0.1)
model.add( Dense(64, activation='relu', name='Dense2') )
model.add( Dropout(rate=0.1)
model.add( Dense(10, activation='softmax', name='Dense3') )

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#%% Splitting train and test data

x_train, x_test, y_train, y_test = train_test_split(train_data_x, train_data_y, test_size=.2, shuffle=True)

#%% Fitting the model
history = model.fit(x_train.reshape(-1,1,64,64), y_train, validation_split=.15, epochs=40, batch_size=32)

#%%Evaluating the model
[acc, score] = model.evaluate(x_test.reshape(-1,1,64,64),y_test)
#%% Plotting training process

fig, axes = plt.subplots(2,1,figsize=(5.5,5), sharex=True)
axes[0].plot(history.history['loss'],c='k', lw=1)
axes[0].plot(history.history['val_loss'],'--',c='k', lw=1)
axes[0].legend(['Training loss','Validation loss'], ncol=2,framealpha=.5)
axes[0].grid(ls=':')
axes[0].set_ylabel('Loss (categorical corssentropy)')

axes[1].plot(array(history.history['acc'])*100,c='k', lw=1)
axes[1].plot(array(history.history['val_acc'])*100,'--',c='k', lw=1)
axes[1].legend(['Training accuracy','Validation accuracy'], ncol=2,framealpha=.5)
axes[1].grid(ls=':')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy (\%)')

fig.tight_layout()
#fig.savefig('D:\\Thesis\\Document\\figures\\cw_bearings_faults_classification_training.pdf')

#%% Plotting confusion matrix
cm = confusion_matrix(y_test.argmax(axis=1), model.predict(x_test.reshape(-1,1,64,64)).argmax(axis=1),normalize='pred')

fig, axis = plt.subplots(1,1,figsize=(5,4))

cm_per = cm*100/cm.max(axis=1)
sn.heatmap(cm*100, annot=True, fmt=".1f", cmap='gray_r', ax=axis, cbar=False)
axis.set_yticklabels(plotting_labels, rotation='horizontal')
axis.set_xticklabels(plotting_labels, rotation=45)
fig.tight_layout()
axis.set_xlabel('Predicted label')
axis.set_ylabel('Actual Label')

for _, spine in axis.spines.items():
    spine.set_visible(True)

#fig.savefig('D:\\Thesis\\Document\\figures\\cw_bearings_faults_classification.pdf', bbox_inches = 'tight')

