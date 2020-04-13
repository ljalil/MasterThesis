# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:36:53 2020

@author: Abdeljalil
"""


import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

freq = 100
T = 1./freq
N = 300

x = np.linspace(0,N*T, N)
signal = np.sin(2*np.pi*5*x) + 3*np.sin(2*np.pi*15*x) + 2*np.sin(2*np.pi*30*x)+ np.random.randn(N)

fig, axes = plt.subplots(1,2,figsize=(8,3))
axes[0].plot(x, signal, c='k', lw=1.4)
axes[0].set_xticks([0,1,2,3])
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True)

fft_y = np.fft.fft(signal)
fft_y = abs(fft_y[:N//2])
fft_x = np.linspace(0,freq/2,N//2)

axes[1].plot(fft_x, (2/N)*fft_y, c='k', lw=1.4)
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True)