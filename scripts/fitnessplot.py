import numpy as np
import matplotlib.pyplot as plt
#%%

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('axes', titlesize=12)
plt.rc('text', usetex=True)

#%%
monot = np.array([0.486,0.481,0.059,0.035,0.481,0.287])
monot_cum = np.array([1,1,0.998,1,1,1])
trend = np.array([0.987,0.989,0.985,0.994,0.988,0.990])
trend_cum = np.array([0.993,0.995,0.890,0.976,0.993,0.996])

#%%

fig, axis = plt.subplots(1,1, figsize=(4,3.5))

axis.scatter(monot, trend, c='k', marker='^')
axis.scatter(monot_cum, trend_cum, c='k', marker='o')
axis.legend(['Features','Cumulative Features'], framealpha=.5,loc='lower center')
axis.set_ylabel('Trendability')
axis.set_xlabel('Monotonicity')
axis.set_xlim([0,1.1])
axis.set_ylim([0,1.1])
axis.grid(ls=':')
fig.tight_layout()

fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/featuresfitness.pdf')
#%%


fig, axis = plt.subplots(1,1, figsize=(4,3.5))

axis.scatter(monot, trend, c='k', marker='^')
axis.scatter(monot_cum, trend_cum, c='k', marker='o')
axis.legend(['Caractéristiques','Caractéristiques Cumulatives'], framealpha=.5,loc='lower center')
axis.set_ylabel('Trendabilité')
axis.set_xlabel('Monotonie')
axis.set_xlim([0,1.1])
axis.set_ylim([0,1.1])
axis.grid(ls=':')
fig.tight_layout()
fig.savefig('/home/abdeljalil/Workspace/MasterThesis/figures/featuresfitness_fr.pdf')
