# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:47:21 2020

@author: Abdeljalil
"""

from numpy import *
import matplotlib.pyplot as plt

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
#%%
fig, axes = plt.subplots(1,3, figsize=(6,1.4),sharey=True)

axes[0].plot(rul_linear, c='k')
axes[1].plot(rul_piecewise, c='k')
axes[2].plot(rul_nonlinear, c='k')

for axis in axes:
    axis.grid(ls=':')
    axis.set_xticks([0,50,100,150,200])
    axis.set_yticks([0,50,100,150,200])
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_xlim([0,200])
    axis.set_ylim([0,250])
    axis.grid(ls=':')
    
#%%
x = ones((100,))*life
x = concatenate((x, (linspace(life,0, life-100))))
plt.plot(x)