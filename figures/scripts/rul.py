# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:47:21 2020

@author: Abdeljalil
"""

from numpy import *
import matplotlib.pyplot as plt
#%% Matplotlib LaTeX Settings
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('axes', titlesize=12)
plt.rc('text', usetex=True)

#%%
def RUL_modeler(life : int, kind : str, classes = 5):
    if kind == 'linear':
        return flip(arange(life)).reshape(-1,1)

    elif kind == 'nonlinear':
        x = array([0, 1.4*life/2, life])
        y = array([life, 1.2*life/2, 0])
    
        polynom = poly1d(polyfit(x,y,3))
    
        return polynom(arange(0,life)).reshape(-1,1)
    
    elif kind == 'piecewise':
        x = ones((100,))*life
        x = concatenate((x, linspace(life,0, life-100)))
        return x
    
life = 192
rul_linear = RUL_modeler(life, kind='linear')
rul_piecewise = RUL_modeler(life, kind='piecewise')
rul_nonlinear = RUL_modeler(life, kind='nonlinear')

#%% Plot RUL Models
fig, axes = plt.subplots(1,3, figsize=(5.5,1.8),sharey=True)

axes[0].plot(rul_linear, c='k')
axes[0].set_title('Linear')
axes[0].set_ylabel('Remaining time')
axes[1].plot(rul_piecewise, c='k')
axes[1].set_title('Piecewise')
axes[2].plot(rul_nonlinear, c='k')
axes[2].set_title('Polynomial')

for axis in axes:
    axis.grid(ls=':')
    axis.set_xticks([0,50,100,150,200])
    axis.set_yticks([0,50,100,150,200])
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_xlim([0,200])
    axis.set_ylim([0,250])
    axis.grid(True,ls=':')
    axis.xaxis.set_ticklabels([])
    axis.xaxis.set_ticks_position('none')
    axis.yaxis.set_ticks_position('none')
    axis.set_xlabel('Time')
    
fig.tight_layout()
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/igures/rul_models.pdf', bbox_inches = 'tight')
#%%
x = ones((100,))*life
x = concatenate((x, (linspace(life,0, life-100))))
plt.plot(x)


#%% Plot RUL Models [French]
fig, axes = plt.subplots(1,3, figsize=(5.5,1.8),sharey=True)

axes[0].plot(rul_linear, c='k')
axes[0].set_title('Linéaire')
axes[0].set_ylabel('Temps restant')
axes[1].plot(rul_piecewise, c='k')
axes[1].set_title('Par morceaux')
axes[2].plot(rul_nonlinear, c='k')
axes[2].set_title('Polynôme')

for axis in axes:
    axis.grid(ls=':')
    axis.set_xticks([0,50,100,150,200])
    axis.set_yticks([0,50,100,150,200])
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_xlim([0,200])
    axis.set_ylim([0,250])
    axis.grid(True,ls=':')
    axis.xaxis.set_ticklabels([])
    axis.xaxis.set_ticks_position('none')
    axis.yaxis.set_ticks_position('none')
    axis.set_xlabel('Temps')

fig.tight_layout()
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/rul_models_fr.pdf', bbox_inches = 'tight')
