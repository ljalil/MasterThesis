from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import pywt

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('axes', titlesize=12)

file = pd.read_csv('/home/abdeljalil/Workspace/FEMTODATA/Bearing1_1/acc_02500.csv', header=None)

vibrations = file[4].values

vibrations_frequency = 25.6*(10**3)
vibrations_sampling_period = 1/vibrations_frequency

scales = arange(1,257)
wavelet_name = 'morl'

[coef, freq] = pywt.cwt(vibrations, scales, wavelet_name, sampling_period= vibrations_sampling_period)
time = linspace(0,0.1,2560)


#%% Plot [English]
fig, axis = plt.subplots(1,1,figsize=(5.5,3))
cntr = axis.contourf(time, freq[1:], abs(coef)[1:, :], levels=80, cmap='viridis')
axis.set_xlabel('Time (s)')
axis.set_ylabel('Frequency (Hz)')
axis.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

#for c in cntr.collections:
#    c.set_edgecolor("face")
#    c.set_linewidth(0.000000000001)
#    c.set_rasterized(True)
fig.colorbar(cntr, ax=axis)
fig.tight_layout()
#fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/scaleogram.pdf')
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/scaleogram.png')

#%% Plot [French]
fig, axis = plt.subplots(1,1,figsize=(5.5,3))
cntr = axis.contourf(time, freq[1:], abs(coef)[1:, :], levels=80, cmap='viridis')
axis.set_xlabel('Temps (s)')
axis.set_ylabel('Fréquence (Hz)')
axis.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

#for c in cntr.collections:
#    c.set_edgecolor("face")
#    c.set_rasterized(True)

fig.colorbar(cntr, ax=axis)
#axis.set_rasterization_zorder(-20)
fig.tight_layout()
#fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/scaleogram_fr.pdf')
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/scaleogram_fr.png')


