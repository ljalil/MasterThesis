#%% Loading Libraries
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import os
import pywt
from skimage.transform import resize
#%%
from keras.models import Sequential
from keras.layers import LSTM, Conv2D, MaxPool2D, Flatten, Dropout, Dense, Masking
from keras.utils import to_categorical
#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score
#%% Matplotlib LaTeX Settings
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('axes', titlesize=12)
plt.rc('text', usetex=True)

#%% General Settings
main_path = '/home/abdeljalil/Workspace/FEMTODATA/'
frequency = 25.5*(10**3)
period = 1/frequency
sample_length = 2560
vibrations_sample_length = 2560

#%% Continuous Wavelet Transform Settings
time = linspace(0, 0.1, 2560)
scales = arange(1,257)
wavelet = 'morl'

#%% Preparing data for plottign fault progress
bearing = 'Bearing1_1'
path = os.path.join(main_path, bearing)
files_list = sorted(os.listdir(path))
number_of_files = len(files_list)
files_to_plot = (linspace(0,number_of_files-2,8)).astype('int32')
coefs_h = []
coefs_v = []
for file in files_to_plot:
    file_path = os.path.join(path, files_list[file])
    data = pd.read_csv(file_path, header=None)
    print(file_path)
    vibration_h = data[4].values
    vibration_v = data[5].values
    [coef_h, freq] = pywt.cwt(vibration_h, scales, wavelet, 1/frequency)
    [coef_v, freq] = pywt.cwt(vibration_v, scales, wavelet, 1/frequency)
    coefs_h.append(abs(coef_h)**2)
    coefs_v.append(abs(coef_v)**2)

resize_shape = (128,128)
resized_coefs_h = []
resized_coefs_v = []
for coef in zip(coefs_h, coefs_v):
    resized_coefs_h.append(resize(coef[0][:128//2,:], resize_shape))
    resized_coefs_v.append(resize(coef[1][:128//2,:], resize_shape))

#%% Plotting scaleograms: horizontal vibrations
fig, axes = plt.subplots(2, 4, figsize=(6,3.5), sharex=True, sharey=True)
for index, axis in enumerate(axes.flatten()):
    cntr = axis.contourf(resized_coefs_h[index], levels=80)
    axis.set_title(r'{}\%'.format(round((index+1)*100/8)))
#    for c in cntr.collections:
 #       c.set_edgecolor("face")

for axis in axes.flatten():
    axis.set_xticks([0,128])
    axis.set_xticklabels([0,0.1])
    axis.set_yticks([0,128])
    axis.set_yticklabels([163/1000, 20800/2000])

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel(r"Time (s)")
plt.ylabel("Frequency (kHz)",labelpad=15)
fig.tight_layout(w_pad=.2, h_pad=0.2)
plt.gcf().subplots_adjust(left=.14,bottom=0.15)
#fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/femto_scaleograms_h.pdf')
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/femto_scaleograms_h.png')


#%% Plotting scaleograms: vertical vibrations
fig, axes = plt.subplots(2, 4, figsize=(6,3.5), sharex=True, sharey=True)
for index, axis in enumerate(axes.flatten()):
    cntr = axis.contourf(resized_coefs_v[index], levels=80)
    axis.set_title(r'{}\%'.format(round((index+1)*100/8)))
#    for c in cntr.collections:
#        c.set_edgecolor("face")

for axis in axes.flatten():
    axis.set_xticks([0,128])
    axis.set_xticklabels([0,0.1])
    axis.set_yticks([0,128])
    axis.set_yticklabels([163/1000, 20800/2000])

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel(r"Time (s)")
plt.ylabel("Frequency (kHz)",labelpad=15)
fig.tight_layout(w_pad=.2, h_pad=0.2)
plt.gcf().subplots_adjust(left=.14,bottom=0.15)
#fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/femto_scaleograms_v.pdf')
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/femto_scaleograms_v.png')


#%% Plotting scaleograms: horizontal vibrations [french]
fig, axes = plt.subplots(2, 4, figsize=(6,3.5), sharex=True, sharey=True)
for index, axis in enumerate(axes.flatten()):
    cntr = axis.contourf(resized_coefs_h[index], levels=80)
    axis.set_title(r'{}\%'.format(round((index+1)*100/8)))
#    for c in cntr.collections:
#        c.set_edgecolor("face")

for axis in axes.flatten():
    axis.set_xticks([0,128])
    axis.set_xticklabels([0,0.1])
    axis.set_yticks([0,128])
    axis.set_yticklabels([163/1000, 20800/2000])

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel(r"Temps (s)")
plt.ylabel("Fréquence (kHz)",labelpad=15)
fig.tight_layout(w_pad=.2, h_pad=0.2)
plt.gcf().subplots_adjust(left=.14,bottom=0.15)
#fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/femto_scaleograms_h_fr.pdf')
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/femto_scaleograms_h_fr.png')


#%% Plotting scaleograms: vertical vibrations [french]
fig, axes = plt.subplots(2, 4, figsize=(6,3.5), sharex=True, sharey=True)
for index, axis in enumerate(axes.flatten()):
    cntr = axis.contourf(resized_coefs_v[index], levels=80)
    axis.set_title(r'{}\%'.format(round((index+1)*100/8)))
#    for c in cntr.collections:
#        c.set_edgecolor("face")

for axis in axes.flatten():
    axis.set_xticks([0,128])
    axis.set_xticklabels([0,0.1])
    axis.set_yticks([0,128])
    axis.set_yticklabels([163/1000, 20800/2000])

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel(r"Temps (s)")
plt.ylabel("Fréquence (kHz)",labelpad=15)
fig.tight_layout(w_pad=.2, h_pad=0.2)
plt.gcf().subplots_adjust(left=.14,bottom=0.15)
#fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/femto_scaleograms_v_fr.pdf')
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/femto_scaleograms_v_fr.png')


#%% Creating dataset from scaleograms

resize_shape = (128,128)
data_x = zeros((0,2,128,128))
data_y = zeros((0,))

first_healthy_samples = 80
last_faulty_samples = 80

bearings_list = sorted(os.listdir(main_path))

for bearing in bearings_list[:7]:
    bearing_path = os.path.join(main_path, bearing)
    bearing_files = sorted(os.listdir(bearing_path))

    for file_healthy in bearing_files[:first_healthy_samples]:
        data = pd.read_csv(os.path.join(bearing_path, file_healthy),header=None)
        vibration_h = data[4].values
        vibration_v = data[5].values
        [coef_h, _] = pywt.cwt(vibration_h, scales, wavelet, 1/frequency)
        [coef_v, _] = pywt.cwt(vibration_v, scales, wavelet, 1/frequency)
        resized_power_coef_h = resize(abs(coef_h)**2, resize_shape)
        resized_power_coef_v = resize(abs(coef_v)**2, resize_shape)
        resized_power_coef_h = resized_power_coef_h.reshape(1,128,128)
        resized_power_coef_v = resized_power_coef_v.reshape(1,128,128)
        stacked_coefs = concatenate((resized_power_coef_h, resized_power_coef_v), axis=0)
        #Append generated coefficients to data_x
        data_x = concatenate((data_x, stacked_coefs.reshape(1, 2, 128, 128)), axis=0)
    data_y = concatenate((data_y, zeros(first_healthy_samples,)), axis=0)
    print(bearing, ' healthy finished')

    for file_faulty in bearing_files[-last_faulty_samples:]:
        data = pd.read_csv(os.path.join(bearing_path, file_faulty),header=None)
        vibration_h = data[4].values
        vibration_v = data[5].values
        [coef_h, _] = pywt.cwt(vibration_h, scales, wavelet, 1/frequency)
        [coef_v, _] = pywt.cwt(vibration_v, scales, wavelet, 1/frequency)
        resized_power_coef_h = resize(abs(coef_h)**2, resize_shape)
        resized_power_coef_v = resize(abs(coef_v)**2, resize_shape)

        resized_power_coef_h = resized_power_coef_h.reshape(1,128,128)
        resized_power_coef_v = resized_power_coef_v.reshape(1,128,128)
        stacked_coefs = concatenate((resized_power_coef_h, resized_power_coef_v), axis=0)
        #Append generated coefficients to data_x
        data_x = concatenate((data_x, stacked_coefs.reshape(1, 2, 128, 128)), axis=0)

    data_y = concatenate((data_y, ones(last_faulty_samples,)), axis=0)
    print(bearing, ' faulty finished')
    print('Data shape: ', data_x.shape)
#%%
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=True)
#%%

model = Sequential()
model.add( Conv2D(16, (3,3), strides=(1,1), padding='same', input_shape=(2,128,128), name='Conv1') )
model.add( MaxPool2D((2,2), padding='same', name='MaxPool1') )
model.add( Conv2D(32, (3,3), strides=(1,1), padding='same', name='Conv2') )
model.add( MaxPool2D((2,2), padding='same', name='MaxPool2') )
model.add( Conv2D(64, (3,3), strides=(1,1), padding='same', name='Conv3') )
model.add( MaxPool2D((2,2), padding='same', name='MaxPool3') )
model.add( Flatten() )
model.add( Dense(512, activation='relu', name='Dense2') )
model.add(Dropout(rate=0.1))
model.add( Dense(64, activation='relu', name='Dense3') )

model.add(Dropout(rate=0.1))
model.add( Dense(2, activation='sigmoid', name='Dense4') )

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


history = model.fit(train_x, to_categorical(train_y), batch_size=64,validation_split=.2, epochs=50)

#%% Calculating Metrics
rounding = 4
score, acc = model.evaluate(test_x, to_categorical(test_y))
pred = model.predict(test_x)
pred = argmax(pred, axis=1)
print('Accuracy: {}%'.format(round(acc*100, rounding)))
print('Loss: {}'.format(round(score, rounding)))
print('Precision: {}'.format(round(precision_score(test_y, pred), rounding)))
print('Recall: {}'.format(round(recall_score(test_y, pred), rounding)))
print('F-1: {}'.format(round(f1_score(test_y, pred), rounding)))

#%% Plotting Training
fig, axes = plt.subplots(2,1,figsize=(5.5,5), sharex=True)
axes[0].plot(history.history['loss'],c='k', lw=1)
axes[0].plot(history.history['val_loss'],'--',c='k', lw=1)
axes[0].legend(['Training','Validation'], ncol=2,framealpha=.5)
axes[0].grid(ls=':')
axes[0].set_ylabel('Loss (binary corssentropy)')

axes[1].plot(array(history.history['acc'])*100,c='k', lw=1)
axes[1].plot(array(history.history['val_acc'])*100,'--',c='k', lw=1)
axes[1].legend(['Training','Validation'], ncol=2,framealpha=.5)
axes[1].grid(ls=':')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy (\%)')

fig.tight_layout()
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/femtocwt_training.pdf')

#%% Plotting Training [French]
fig, axes = plt.subplots(2,1,figsize=(5.5,5), sharex=True)
axes[0].plot(history.history['loss'],c='k', lw=1)
axes[0].plot(history.history['val_loss'],'--',c='k', lw=1)
axes[0].legend(['Entraînement','Validation'], ncol=2,framealpha=.5)
axes[0].grid(ls=':')
axes[0].set_ylabel('Perte (entropie croisée binaire)')

axes[1].plot(array(history.history['acc'])*100,c='k', lw=1)
axes[1].plot(array(history.history['val_acc'])*100,'--',c='k', lw=1)
axes[1].legend(['Training','Validation'], ncol=2,framealpha=.5)
axes[1].grid(ls=':')
axes[1].set_xlabel('Epoques')
axes[1].set_ylabel('Précision (\%)')

fig.tight_layout()
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/femtocwt_training_fr.pdf')


#%% ROC AUC

predictions = model.predict(test_x)
fpr, tpr, thresholds = roc_curve(test_y,predictions.max(axis=1) )
clf_roc_auc = roc_auc_score(test_y, predictions.max(axis=1))
#%% Plot ROC Curve
fig, axis = plt.subplots(1,1, figsize=(4,3))
axis.plot(fpr,tpr,c='k', lw=1)
axis.plot([0, 1], [0, 1], 'k--', lw='.8')
axis.set_xlabel('False Positive Rate')
axis.set_ylabel('True Positive Rate')
axis.grid(ls=':')
axis.axis([0, 1, 0, 1])
axis.legend(['ROC Curve (AUC={})'.format(around(clf_roc_auc,4))],framealpha=.5)
fig.tight_layout()
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/femtocwt_roc_auc.pdf')

#%% Plot ROC Curve [French]
fig, axis = plt.subplots(1,1, figsize=(4,3))
axis.plot(fpr,tpr,c='k', lw=1)
axis.plot([0, 1], [0, 1], 'k--', lw='.8')
axis.set_xlabel('Taux de faux positif')
axis.set_ylabel('Taux de vrai positif')
axis.grid(ls=':')
axis.axis([0, 1, 0, 1])
axis.legend(['Courbe ROC (AUC={})'.format(around(clf_roc_auc,4))],framealpha=.5)
fig.tight_layout()
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/femtocwt_roc_auc_fr.pdf')

#%% Confusion matrix
cm = confusion_matrix(test_y, predict),normalize='pred')
