# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:38:05 2020

@author: Abdeljalil
"""

#%% Load Libraries
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Bbox

#%%
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('axes', titlesize=12)
plt.rc('text', usetex=True)

#%%
x = linspace(0.001,1.2,100) 
y = x*log(x)

fig, axis = plt.subplots(1,1,figsize=(4,3))
ax = 0.1
ay = ax*log(ax)
bx = 1
by = bx*log(bx)

axis.plot(x,y, c='k',lw=1, label='$xlog(x)$')
ylim = -0.4
axis.plot([ax,ax],[ylim,ay], '--',c='gray', lw='1')
axis.plot([bx,bx],[ylim,by], '--',c='gray', lw='1')
axis.plot([ax,bx],[ay,by],c='blue', lw='1')
lambda_val=.5
lambda_x=lambda_val*ax+(1-lambda_val)*bx
lambda_y = lambda_val*(ax*log(ax))+(1-lambda_val)*(bx*log(bx))
axis.plot([lambda_x, lambda_x],[ylim,lambda_y],c='gray',lw=1)
axis.set_ylim([ylim,.3])
axis.set_xticks([ax,lambda_x,bx])

arrowprops = dict(
    arrowstyle = "->",
    connectionstyle = "angle,angleA=0,angleB=90,rad=5", lw='.8')
bbox = dict(boxstyle="round", fc="1", lw=.8)

axis.annotate('$chord$',(lambda_x-.15,lambda_y-0.05),(lambda_x-0.05,lambda_y+.1), arrowprops=arrowprops, bbox=bbox)
axis.set_xticklabels(['$a$','$x_\lambda$','$b$'])
axis.set_yticks([])
axis.set_yticklabels([])
axis.set_xlabel('$x$')
axis.set_ylabel('$xlog(x)$')
fig.tight_layout()
#fig.savefig('D:\\Thesis\\Document\\figures\\convex_function.pdf',bbox_inches="tight")
#%%
x1 = linspace(-5,5,100)
y1 = linspace(-5,5,100)
X1, Y1 = meshgrid(x1,y1)
Z1 = X1**2 +Y1**2
x2 = linspace(-3,3,100)
y2 = linspace(-3,3,100)
X2, Y2 = meshgrid(x2,y2)
Z2 = sin(X2)*sin(Y2)
#%% Plot
fig = plt.figure(figsize=(6,2.5))
axis1 = fig.add_subplot(1,2,1, projection='3d')
axis1.plot_surface(X1,Y1,Z1, cmap='inferno')
axis1.contour(X1,Y1,Z1, zdir='z', offset=0, cmap='inferno')
axis1.set_title('Convex function')

axis2 = fig.add_subplot(1,2,2, projection='3d')
axis2.plot_surface(X2,Y2,Z2, cmap='inferno')
axis2.contour(X2,Y2,Z2, zdir='z', offset=-1, cmap='inferno')
axis2.set_title('Non-convex function')
for axis in [axis1, axis2]:
    axis.set_proj_type('ortho')
    axis.xaxis.pane.fill = False
    axis.yaxis.pane.fill = False
    axis.zaxis.pane.fill = False
    axis.grid(False)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_zticklabels([])
    axis.yaxis._axinfo['label']['space_factor'] = 1.0
    axis.view_init(20, 30)
fig.tight_layout(pad=0)

#fig.savefig('D:\\Thesis\\Document\\figures\\gradient_descent.pdf', bbox_inches=Bbox([[0, .15], [6, 2.6]]))
plt.show()

#%% Plot [French]
fig = plt.figure(figsize=(6,2.5))
axis1 = fig.add_subplot(1,2,1, projection='3d')
axis1.plot_surface(X1,Y1,Z1, cmap='inferno')
axis1.contour(X1,Y1,Z1, zdir='z', offset=0, cmap='inferno')
axis1.set_title('Fonction convexe')

axis2 = fig.add_subplot(1,2,2, projection='3d')
axis2.plot_surface(X2,Y2,Z2, cmap='inferno')
axis2.contour(X2,Y2,Z2, zdir='z', offset=-1, cmap='inferno')
axis2.set_title('Fonction non convexe')
for axis in [axis1, axis2]:
    axis.set_proj_type('ortho')
    axis.xaxis.pane.fill = False
    axis.yaxis.pane.fill = False
    axis.zaxis.pane.fill = False
    axis.grid(False)
    
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_zticklabels([])
    axis.yaxis._axinfo['label']['space_factor'] = 1.0
    axis.view_init(20, 30)

fig.tight_layout(pad=0)

fig.savefig('D:\\Thesis\\Document\\figures\\gradient_descent_fr.pdf', bbox_inches=Bbox([[0, .15], [6, 2.6]]))
plt.show()

#%% Saddle Point

x3 = linspace(-2,2,100)
y3 = linspace(-1,1,100)
X3, Y3 = meshgrid(x3,y3)
Z3 = sin(X3)*sin(Y3)


fig = plt.figure(figsize=(3.5,3))
axis3 = fig.add_subplot(1,1,1, projection='3d')
axis3.set_proj_type('ortho')

axis3.plot_surface(X3,Y3,Z3, cmap='inferno')
axis3.contour(X3,Y3,Z3, zdir='z', offset=-0.9, cmap='inferno')
#axis3.scatter([-0.1],[-0.1],[1],c='green',s=30)
axis3.xaxis.pane.fill = False
axis3.yaxis.pane.fill = False
axis3.zaxis.pane.fill = False
axis3.grid(False)
#axis3.text(.5, .8, -0.05, "Saddle point", size=9,zorder=100000, color='k',bbox={'facecolor':'white', 'alpha':0.8, 'pad':2})
axis3.set_xticklabels([])
axis3.set_yticklabels([])
axis3.set_zticklabels([])
#axis3.set_xlabel('x')
axis3.yaxis._axinfo['label']['space_factor'] = 1.0
#axis3.view_init(0, 180)
axis3.view_init(25, 110)
axis3.autoscale_view('tight')

fig.tight_layout(pad=0)

fig.savefig('D:\\Thesis\\Document\\figures\\saddle_point.pdf', bbox_inches=Bbox([[0.3, 0.2], [3.3, 2.7]]))
plt.show()

#%% Global local minima

x4 = linspace(0,6,100)
y4 = linspace(0,6,100)
X4, Y4 = meshgrid(x4,y4)
Z4 = X4*sin(X4)*sin(Y4)


fig = plt.figure(figsize=(4.5,3.5))
axis4 = fig.add_subplot(1,1,1, projection='3d')
axis4.set_proj_type('ortho')

axis4.plot_surface(X4,Y4,Z4, cmap='inferno')
axis4.contour(X4,Y4,Z4, zdir='z',offset=-4.9, levels=10, cmap='inferno')

#axis4.scatter([4.909, 2],[1.575, 4.7272],[-4.7,-1.9],c='green',marker='x',s=25)
axis4.xaxis.pane.fill = False
axis4.yaxis.pane.fill = False
axis4.zaxis.pane.fill = False
axis4.grid(False)
axis4.set_xlim([0,6])
axis4.set_ylim([0,6])

axis4.set_xticklabels([])
axis4.set_yticklabels([])
axis4.set_zticklabels([])


axis4.yaxis._axinfo['label']['space_factor'] = 1.0
axis4.view_init(25, 60)
#axis4.text(6.2, 1.575, -5.5, "Global minimum", size=9,zorder=100000, color='k',bbox={'facecolor':'white', 'alpha':0.8, 'pad':2})
#axis4.text(1.5, 1.575, -5.6, "Local minimum", size=9,zorder=100000, color='k',bbox={'facecolor':'white', 'alpha':0.8, 'pad':2})

fig.tight_layout(pad=0)
#fig.savefig('D:\\Thesis\\Document\\figures\\global_local_minima.pdf', bbox_inches=Bbox([[0.3, .2], [4.2, 3]]))
plt.show()