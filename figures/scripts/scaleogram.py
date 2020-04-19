#%%
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import pywt

#%%

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('axes', titlesize=12)

#%%
file = pd.read_csv('C:/Users/abdel/Documents/Workspace/FEMTOBearingDataSet/TRAIN/Bearing1_1/acc_00020.csv', header=None)
vibrations = file[4].values

#%%
vibrations_frequency = 25.6*(10**3)
vibrations_sampling_period = 1/vibrations_frequency

scales = arange(1,1025)
wavelet_name = 'morl'

#%%
[coef, freq] = pywt.cwt(vibrations, scales, wavelet_name, sampling_period= vibrations_sampling_period)
time = linspace(0,0.025,2560//4)
coef = coef[:,:2560//4]
time = linspace(0,25,2560//4)


#%%
fig, axis = plt.subplots(1,1,figsize=(5.5,3))
cntr = axis.contourf(time, freq, abs(coef), levels=50, cmap='inferno')
axis.set_xlabel('Time (ms)')
axis.set_ylabel('Frequency (Hz)')

axis.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

for c in cntr.collections:
    c.set_edgecolor("face")
    c.set_rasterized(True)

fig.colorbar(cntr, ax=axis)
# %%
fig.tight_layout()
fig.savefig('D:\\Thesis\\Document\\figures\\scaleogram.pdf')