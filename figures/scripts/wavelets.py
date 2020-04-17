import matplotlib.pyplot as plt
import numpy as np
import pywt

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('axes', titlesize=12)
plt.rc('text', usetex=True)

discrete_wavelets = ['db5', 'sym5', 'coif5', 'bior2.4']
continuous_wavelets = ['mexh', 'morl', 'shan', 'gaus5']

fig, axes = plt.subplots(1,4,figsize=(6,1.8))

for continuous_wavelet in enumerate(continuous_wavelets):
    wavelet = pywt.ContinuousWavelet(continuous_wavelet[1])
    w, x = wavelet.wavefun()
    #Some wavelets names have 'wavelets' instead of 'wavelet' and it leaves an 'S'
    title = wavelet.family_name.replace('wavelets','')
    title = title.replace('wavelet','')
    axes[continuous_wavelet[0]].plot(x,w, c='k',lw=1)
    axes[continuous_wavelet[0]].set_title(title)
    


for axis in axes.ravel():
    axis.set_xticks([])
    axis.set_yticks([])
fig.tight_layout()
fig.savefig('D:/Thesis\Document/figures/plots/wavelets.pdf')