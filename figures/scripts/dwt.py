# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:43:45 2020

@author: Abdeljalil
"""


import numpy as np
import scipy.signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pywt

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('axes', titlesize=12)
plt.rc('text', usetex=True)


t = np.linspace(0, 20, 2000)
#chirp = scipy.signal.chirp(t, f0=6, f1=1, t1=10, method='linear')
x = np.linspace(0, 1, num=2048)
chirp = np.sin(250 * np.pi * x**2)

wavelet = 'sym3'
cA1, cD1 = pywt.dwt(chirp, wavelet)
cA2, cD2 = pywt.dwt(cA1, wavelet)
cA3, cD3 = pywt.dwt(cA2, wavelet)
#plt.plot(coeffs[2])
#plt.plot(chirp)
scales = np.arange(1,65)
#[coeffs, freq] = pywt.cwt(chirp,scales,'morl',sampling_period=1./2048)
#plt.contourf(x,freq,abs(coeffs), levels=50,cmap='viridis')

#%% Plot DWT Coefficients
fig = plt.figure(constrained_layout=True,figsize=(6,3.5))
gs = GridSpec(4, 2, figure=fig)

ax_chirp = fig.add_subplot(gs[0, :])
ax_cA1 = fig.add_subplot(gs[1, 0])
ax_cD1 = fig.add_subplot(gs[1, 1])
ax_cA2 = fig.add_subplot(gs[2, 0])
ax_cD2 = fig.add_subplot(gs[2, 1])
ax_cA3 = fig.add_subplot(gs[3, 0])
ax_cD3 = fig.add_subplot(gs[3, 1])

ax_chirp.plot(chirp,c='k')

ax_cA1.plot(cA1,c='k')
ax_cD1.plot(cD1,c='k')
ax_cA2.plot(cA2,c='k')
ax_cD2.plot(cD2,c='k')
ax_cA3.plot(cA3,c='k')
ax_cD3.plot(cD3,c='k')

ax_chirp.set_title('Chirp signal')
ax_cA1.set_title('Approximation coefficients')
ax_cD1.set_title('Detail coefficients')

ax_cA1.set_ylabel('Level 01')
ax_cA2.set_ylabel('Level 02')
ax_cA3.set_ylabel('Level 03')

ax_chirp.set_yticks([])
ax_cA1.set_yticks([])
ax_cA2.set_yticks([])
ax_cA3.set_yticks([])

ax_cD1.set_yticks([])
ax_cD2.set_yticks([])
ax_cD3.set_yticks([])
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/dwt_chirp.pdf')


#%% Plot DWT Coefficients [French]
fig = plt.figure(constrained_layout=True,figsize=(6,3.5))
gs = GridSpec(4, 2, figure=fig)

ax_chirp = fig.add_subplot(gs[0, :])
ax_cA1 = fig.add_subplot(gs[1, 0])
ax_cD1 = fig.add_subplot(gs[1, 1])
ax_cA2 = fig.add_subplot(gs[2, 0])
ax_cD2 = fig.add_subplot(gs[2, 1])
ax_cA3 = fig.add_subplot(gs[3, 0])
ax_cD3 = fig.add_subplot(gs[3, 1])

ax_chirp.plot(chirp,c='k')

ax_cA1.plot(cA1,c='k')
ax_cD1.plot(cD1,c='k')
ax_cA2.plot(cA2,c='k')
ax_cD2.plot(cD2,c='k')
ax_cA3.plot(cA3,c='k')
ax_cD3.plot(cD3,c='k')

ax_chirp.set_title('Le signal (Chirp)')
ax_cA1.set_title("Approximation")
ax_cD1.set_title("Détails")

ax_cA1.set_ylabel('Niveau 01')
ax_cA2.set_ylabel('Niveau 02')
ax_cA3.set_ylabel('Niveau 03')

ax_chirp.set_yticks([])
ax_cA1.set_yticks([])
ax_cA2.set_yticks([])
ax_cA3.set_yticks([])

ax_cD1.set_yticks([])
ax_cD2.set_yticks([])
ax_cD3.set_yticks([])

fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/dwt_chirp_fr.pdf')
