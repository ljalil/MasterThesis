# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:58:48 2020

@author: Abdeljalil Letrache <abdeljalilletrache@outlook.com>
"""

#%% Loading basic libraries
import os
from numpy import *
import pandas as pd
import pywt
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
from skimage.transform import resize
import importlib
#%% Import machine learning libraries
from keras.models import Sequential
from keras.layers import LSTM, Conv1D, MaxPool1D, Flatten, Dropout, Dense, Masking
from keras.utils import to_categorical

#%% Matplotlib LaTeX Settings
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('axes', titlesize=12)
plt.rc('text', usetex=True)

#%% General settings

def decomposition_length(dwt_settings):
    wav = pywt.Wavelet(dwt_settings['wavelet'])
    #TODO: adjust this to be more general
    dwl =2560 
    for i in range(dwt_settings['level']):
        dwl = int((dwl + wav.dec_len -1)/2)
    return dwl

vibrations_sample_length = 2560
general_settings = {
    'main_path' :'/home/abdeljalil/Workspace/FEMTODATA/', 
    'frequency' : 25.6*(10**3),
    'sampling_period' : 1/25.6*(10**3),
    'sample_length' : 2560
    }

#%% Utilities Functions 
def RUL_modeler(life : int, kind : str, classes = 5):
    if kind == 'linear':
        return flip(arange(life)).reshape(-1,1)

    elif kind == 'nonlinear':
        x = array([0, 1.4*life/2, life])
        y = array([life, 1.2*life/2, 0])
        polynom = poly1d(polyfit(x,y,3))
        return polynom(arange(0,life)).reshape(-1,1)

    elif kind == 'piecewise':
        pass

    elif kind == 'classification':
        return linspace(0, classes, life, endpoint=False).astype('int32').reshape(-1,1)

    return None

def plot_nonlinear_RUL(life : int, xylim : int, savefig = False):
    linear_rul = RUL_modeler(life, kind='linear')
    nonlinear_rul_x = arange(0,life)
    nonlinear_rul_y = RUL_modeler(life, kind='nonlinear')

    fig, axis = plt.subplots(1,1,figsize=(5.5,4))
    axis.plot(linear_rul,'--', c='k', lw=1)
    axis.plot(nonlinear_rul_x, nonlinear_rul_y, '-', c='k', lw=1)
    axis.set_xlabel('Aging')
    axis.set_ylabel('Remaining life')
    axis.legend(['Linear RUL', 'Nonlinear RUL'], loc='lower center', ncol=2, framealpha=.5)
    axis.set_xlim([0,xylim])
    axis.set_ylim([0,xylim])
    axis.grid(ls=':')

    if savefig:
        fig.savefig('D:\\Thesis\\Document\\figures\\nonlinear_rul.pdf')

#%% Discrete Wavelet Transform Functions 

def plot_bearing_dwt(main_bearings_path, bearing_name,wavelet, level, vibration_orientation = 'vertical',  normalizey=True):
    bearing_directory = os.path.join(main_bearings_path, bearing_name)
    files_for_plotting = linspace(1, len( os.listdir( bearing_directory ) )-1, 5).astype('int32')

    fig, axes = plt.subplots(5,1, figsize=(5,5), sharex=True, sharey=normalizey)

    for iterator in enumerate(files_for_plotting):
        file_name = os.listdir(bearing_directory)[iterator[1]]
        vibrations = pd.read_csv(os.path.join(bearing_directory, file_name), header=None)

        if vibration_orientation == 'vertical':
            vibrations = vibrations[4].values
        elif vibration_orientation == 'horizontal':
            vibrations = vibrations[5].values

        coef = pywt.wavedec(vibrations, wavelet, level=level)
        axes[iterator[0]].plot(coef[1], c='k', lw=1)
        axes[iterator[0]].set_title('Fault Progression {}\%'.format(iterator[0]*25))
        axes[iterator[0]].grid(True)

        if normalizey:
            axes[iterator[0]].set_ylim([-60,60])
            axes[iterator[0]].set_yticks([-60,0,60])

    fig.tight_layout(h_pad=.4)

def plot_bearing_trigonometric_features(main_bearings_path, bearing_name, dwt_settings):
    bearing_directory = os.path.join(main_bearings_path, bearing_name)
    dec_length = decomposition_length(dwt_settings)
    decomposition_coefficients = zeros((0, dec_length))
    fig, axes = plt.subplots(nrows = 1, ncols=2, figsize=(6,3))
    for file_name in os.listdir(bearing_directory):
        file_path = os.path.join(bearing_directory, file_name)

        df = pd.read_csv( file_path , header = None)
        vibrations = df[4].values
        coef = pywt.wavedec(vibrations, dwt_settings['wavelet'], dwt_settings['level'])
        single_decomposition_coefficients = coef[0].reshape(1,-1)

        print('single : ',single_decomposition_coefficients.shape)
        decomposition_coefficients = concatenate((decomposition_coefficients, single_decomposition_coefficients), axis=0)

    arcsinh_func = std(arcsinh(decomposition_coefficients), axis=1)
    arctan_func = std(arctan(decomposition_coefficients), axis=1)
    axes[0].scatter(arange(0,arcsinh_func.shape[0]), arcsinh_func, c='k', s=1)
    axes[1].scatter(arange(0,arctan_func.shape[0]), arctan_func, c='k', s=1)

def calculate_feature(coefficients, function : str, cum : bool):
    '''
    Given a set of dwt coefficients, it performs transformation indicated by function argument and returns the cumulative sum
    '''
    if function == 'skewness':
        feature = scipy.stats.skew(coefficients, axis=1)

    elif function == 'kurtosis':
        feature = scipy.stats.kurtosis(coefficients, axis=1)

    elif function == 'arcsinh':
        feature = arcsinh(std(coefficients, axis=1))

    elif function == 'arctan':
        feature = arctan(std(coefficients, axis=1))

    elif function == 'entropy':
        feature = scipy.stats.entropy(abs(coefficients), axis=1)

    elif function == 'rms':
        feature = sqrt( mean( square(coefficients), axis=1 ) )

    elif function == 'ubound':
        feature = (coefficients.max(axis=1) + 1/2 * (coefficients.max(axis=1) - coefficients.min(axis=1))/(coefficients.shape[1]-1))
    if cum:
        feature = cumsum(feature)/sqrt(abs(cumsum(feature)))

    return feature

def construct_dwt_features_dataset(main_bearings_path, bearings_list, dwt_settings, cum, rul_kind, classes = 5):
    '''
    Given a list of bearings, this function uses calculate_feature() to calculate different cumulative features for each bearing and the corresponding RUL
    '''
    RUL_X = zeros((0, 6))
    RUL_Y = zeros((0,1))

    dec_length = decomposition_length(dwt_settings)

    for bearing in bearings_list:
        bearing_path = os.path.join(main_bearings_path, bearing)

        bearing_coefficients = zeros((0, dec_length))

        for file in sorted(os.listdir(bearing_path)):
            file_path = os.path.join(bearing_path, file)
            df = pd.read_csv( file_path , header = None)
            vibrations = df[4].values

            coefficients = pywt.wavedec(vibrations, dwt_settings['wavelet'], level=dwt_settings['level'])[0].reshape(1,-1)
            bearing_coefficients = concatenate((bearing_coefficients, coefficients), axis=0 )

        #f_skewness = calculate_feature(bearing_coefficients, 'skewness').reshape(-1,1)
        f_kurtosis = calculate_feature(bearing_coefficients, 'kurtosis', cum).reshape(-1,1)
        f_arcsinh = calculate_feature(bearing_coefficients, 'arcsinh', cum).reshape(-1,1)
        f_arctan = calculate_feature(bearing_coefficients, 'arctan', cum).reshape(-1,1)
        f_entropy = calculate_feature(bearing_coefficients, 'entropy', cum).reshape(-1,1)
        f_rms = calculate_feature(bearing_coefficients, 'rms', cum).reshape(-1,1)
        f_ubound = calculate_feature(bearing_coefficients, 'ubound', cum).reshape(-1,1)

        features = concatenate(( f_kurtosis, f_arcsinh, f_arctan, f_entropy, f_rms, f_ubound), axis=1)
        RUL_X = concatenate((RUL_X, features), axis=0)

        RUL = RUL_modeler(len(os.listdir(bearing_path)), rul_kind, classes = classes)
        RUL_Y = concatenate((RUL_Y, RUL), axis=0)

    return RUL_X, RUL_Y

def smooth_feature(feature, window, degree):
    return scipy.signal.savgol_filter(feature, window, degree)

#%% Discrete  Wavelet Transform 
dwt_settings = {
    'wavelet' : 'db4',
    'level' : 4
}

#%% Construct Dataset of Cumulative and Non-cumulative Trigonometric & Other Features 
rulx, ruly = construct_dwt_features_dataset(general_settings['main_path'] , ['Bearing1_1'], dwt_settings, cum=False, rul_kind='nonlinear')

rulxcum, rulycum = construct_dwt_features_dataset(general_settings['main_path'] , ['Bearing1_1'], dwt_settings, cum=True, rul_kind='nonlinear')

#%% Plot Trigonometric Features 
arcsinhf = rulx[:,1]
arctanf = rulx[:,2]
arcsinhfs = smooth_feature(arcsinhf, 101,1)
arctanfs = smooth_feature(arctanf, 101, 1)

fig, axes = plt.subplots(1,2, figsize=(5.5,2))
axes[0].plot(arcsinhf, c='k',alpha=.2, lw=1)
axes[0].plot(arcsinhfs, c='royalblue', lw=1)
axes[0].set_title(r'$\sigma(asinh)$')
axes[0].set_ylim([0,4])
axes[1].plot(arctanf, c='k', alpha=.2,lw=1)
axes[1].plot(arctanfs, c='royalblue', lw=1)
axes[1].set_title(r'$\sigma(atan)$')
axes[1].set_ylim([.2,2])
for axis in axes:
    axis.set_xlabel(r'Time ($sec \times 10$)')
    axis.grid(ls=':')
    axis.set_xticks([0,1000,2000,3000])
    axis.set_xlim([0,3000])
    axis.legend(['Noisy','Filtered'],framealpha=.5, ncol=2,fontsize='small',loc='upper center')
fig.tight_layout()
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/trigonometric_features.pdf')


#%% Plot Trigonometric Features [French] 
arcsinhf = rulx[:,1]
arctanf = rulx[:,2]
arcsinhfs = smooth_feature(arcsinhf, 101,1)
arctanfs = smooth_feature(arctanf, 101, 1)

fig, axes = plt.subplots(1,2, figsize=(5.5,2))
axes[0].plot(arcsinhf, c='k',alpha=.2, lw=1)
axes[0].plot(arcsinhfs, c='royalblue', lw=1)
axes[0].set_title(r'$\sigma(asinh)$')
axes[0].set_ylim([0,4])
axes[1].plot(arctanf, c='k', alpha=.2,lw=1)
axes[1].plot(arctanfs, c='royalblue', lw=1)
axes[1].set_title(r'$\sigma(atan)$')
axes[1].set_ylim([.2,2])
for axis in axes:
    axis.set_xlabel(r'Temps ($sec \times 10$)')
    axis.grid(ls=':')
    axis.set_xticks([0,1000,2000,3000])
    axis.set_xlim([0,3000])
    axis.legend(['Bruyante','Lissée'],framealpha=.5, ncol=2,fontsize='small',loc='upper center')
fig.tight_layout()
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/trigonometric_features_fr.pdf')



#%% Plot Trigonometric Features 
features_func = ['kurtosis',r'$\sigma(asinh)$',r'$\sigma(atan)$','entropy','rms','ubound']
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(5,3.5), sharex=True)
for i in range(len(features_func)):
    feature = smooth_feature(rulx[:,i], 101, 1)
    cumfeature = rulxcum[:,i]
    axes.flatten()[i].plot(feature ,c='k', lw=1)
    axes.flatten()[i].set_ylabel(features_func[i])
    ax2 = axes.flatten()[i].twinx()
    ax2.plot(cumfeature,c='royalblue', lw=1)
    ax2.set_ylabel('C-{}'.format(features_func[i]),c='royalblue')
#    axes.flatten()[i].plot(rulxcum[:,i],c='r')
    #axes.flatten()[i].set_title(features_func[i]) 
    axes.flatten()[i].set_xticks([0,2800])
#    axes.flatten()[i].grid(ls=':')
    axes.flatten()[i].set_xticks([0,1000,2000,3000])
    axes.flatten()[i].set_xlim([0,3000])

    axes.flatten()[i].set_ylim([floor(feature.min()), ceil(feature.max())])
    axes.flatten()[i].set_yticks([floor(feature.min()), ceil(feature.max())])
    ax2.set_ylim([floor(feature.min()-5), ceil(feature.max()+5)])
    ax2.set_yticks([floor(cumfeature.min()-5), ceil(cumfeature.max()+5)])

axes.flatten()[3].set_ylim([4.5,4.9])
axes.flatten()[3].set_yticks([4.5,4.9])
fig.tight_layout()
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/trig_classic_cumulative_features.pdf')

#%% Calculate monotonicity
import FitnessMetrics

features_list = ['kurtosis','asinh','atan','entropy','rms','ubound']
for feature in range(6):
    print(features_list[feature],' = ', FitnessMetrics.monotonicity(smooth_feature(rulx[:,feature], 101, 1)))

for feature in range(6):
    print('Cumulative ',features_list[feature],' = ', FitnessMetrics.monotonicity(smooth_feature(rulxcum[:,feature], 101, 1)))

#%% Calculate trendability 
b1_1 = construct_dwt_features_dataset(general_settings['main_path'], ['Bearing1_1'], dwt_settings, cum=False, rul_kind='nonlinear')

b1_2 = construct_dwt_features_dataset(general_settings['main_path'], ['Bearing1_2'], dwt_settings, cum=False, rul_kind='nonlinear')

b1_3 = construct_dwt_features_dataset(general_settings['main_path'], ['Bearing1_3'], dwt_settings, cum=False, rul_kind='nonlinear')

b1_4 = construct_dwt_features_dataset(general_settings['main_path'], ['Bearing1_4'], dwt_settings, cum=False, rul_kind='nonlinear')

b1_5 = construct_dwt_features_dataset(general_settings['main_path'], ['Bearing1_5'], dwt_settings, cum=False, rul_kind='nonlinear')

b1_6 = construct_dwt_features_dataset(general_settings['main_path'], ['Bearing1_5'], dwt_settings, cum=False, rul_kind='nonlinear')
#%%
importlib.reload(FitnessMetrics)
fi = 3
for fi in range(6):
    feat_list = [b1_1[0][:,fi], b1_2[0][:,fi], b1_3[0][:,fi], b1_4[0][:,fi], b1_5[0][:,fi], b1_6[0][:,fi]]
    print(features_list[fi],':',FitnessMetrics.trendability(feat_list))

#%% Construct Neural Network Training Data From Cumulative Features
bearings_list = ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4']
rulxnn = []
rulynn = []

for bearing in bearings_list:
    bearx, beary = construct_dwt_features_dataset(general_settings['main_path'] , [bearing] , dwt_settings, cum=True, rul_kind='nonlinear')
    rulxnn.append(bearx)
    rulynn.append(beary)

#%%
rulxnnfull = zeros((0,6))
rulynnfull = zeros((0,1))
#%%
rulxnnfull = []
rulynnfull = []
for bearx, beary in zip(rulxnn, rulynn):
    missing_lines = int(ceil(bearx.shape[0]/seq_length)*seq_length-bearx.shape[0])
    missing_lines_x = ones((missing_lines, 6))*-10
    missing_lines_y = ones((missing_lines, 1))*-10
    rulxnnfull.append(concatenate((bearx,missing_lines_x), axis=0).reshape(-1,seq_length,num_features))
    rulynnfull.append(concatenate((beary,missing_lines_y), axis=0).reshape(-1, seq_length))

#%%
#rulxnnshaped = rulxnnfull.reshape(-1,seq_length,num_features)
#rulynnshaped = rulynnfull.reshape(-1,seq_length)

#%% Prepare data for LSTM
seq_length = 100
num_features = 6
#%% Create LSTM Neural Network

model = Sequential()
model.add(Masking(mask_value=-10, input_shape=(seq_length, num_features)))
model.add(LSTM(100, return_sequences=True , name='lstm_1'))
model.add(LSTM(100, return_sequences=True, name='lstm_2'))
model.add(LSTM(100,  name='lstm_3'))
model.add(Dense(120, activation='relu')) 
model.add(Dense(100))
model.compile(los-s='mean_squared_error', metrics=['mae'], optimizer='adam')

#%%

history = model.fit(rulxnnshaped, rulynnshaped.reshape(-1,900), epochs=20, batch_size=1)
#%%

model = Sequential()
model.add( Dense(12, activation='relu', input_shape=(6,))  )
model.add( Dense(1)  )
model.compile(loss='mean_squared_error', metrics=['mae'], optimizer='adam')
#%%
history = model.fit(rulxnnfull, rulynnfull.ravel(), shuffle=True,epochs=10, batch_size=64, validation_split=.2)
#%%
pred = model.predict(rulxnnfull)
plt.plot(pred)
plt.plot(rulynnfull)
#%%
test_bearings_list = ['Bearing1_5','Bearing1_6']
test_rulxnn = []
test_rulynn = []

for bearing in test_bearings_list:
    bearx, beary = construct_dwt_features_dataset(general_settings['main_path'] , [bearing] , dwt_settings, cum=True, rul_kind='nonlinear')
    test_rulxnn.append(bearx)
    test_rulynn.append(beary)
#%%
pred = model.predict(test_rulxnn[1])
plt.plot(3500-pred)
plt.plot(test_rulynn[1].ravel())

#%% Calculate Continuous Wavelet Transform
bearing = 'Bearing1_1'
path = os.path.join(general_settings['main_path'], bearing)
files_list = sorted(os.listdir(path))
files = (linspace(0,len(files_list)-1,8)).astype('int32')
coefs = []
scales = arange(1,129)
time = linspace(0,0.1, 2560)
wavelet = 'morl'

for file in files:
    file_path = os.path.join(path, files_list[file])
    data = pd.read_csv(file_path, header=None)

    vibration = data[4].values
    [coef, freq] = pywt.cwt(vibration, scales, wavelet, 1/general_settings['frequency'])
    coefs.append(abs(coef))
#%%
fig, axes = plt.subplots(
