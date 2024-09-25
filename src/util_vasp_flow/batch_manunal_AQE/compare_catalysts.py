import os, sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import pandas as pd


# Read landscape output from different catalyst compositions
labels = ['CuZn','CuCo','NiGa','CuPd']
colors = [plt.cm.rainbow(n) for n in np.linspace(0,1,len(labels)) ]
colors = ['r','g','b','orange']

jobdirs = [
    'task02_henry_geometry_0819/dft_structures/',
    'task04_Cu_Co_COtoMethanol_0918/',
    'task05_Ni_Ga_COtoMethanol_0918/',
    'task06_Cu_Pd_COtoMethanol_0918/',
]

jobdirs = [ os.path.join(j,'Best_surf_pathway.csv') for j in jobdirs ]

fig, axs = plt.subplots(1,1,figsize=(4,3),tight_layout=True,dpi=100)
xdata = np.arange(7)

for n,j in enumerate(jobdirs):
    df = pd.read_csv( j, index_col=0)
    axs.plot(xdata, df[ df.columns[0] ], lw=1, ls='-', marker='o', ms=3, markerfacecolor='none', color=colors[n], label=labels[n], )
    for c in df.columns[1:10]:
        axs.plot(xdata, df[c], lw=1, ls='-', marker='o', ms=3, markerfacecolor='none', color=colors[n] )


axs.tick_params(direction='in',labelsize=8)
axs.set_xlabel('Reaction states ',fontsize=10) ## input X name
axs.set_ylabel('Energy landscape (eV)',fontsize=10) ## input Y name

axs.legend(fontsize=6, frameon=False, loc='upper right', ncol=1, columnspacing=1)#, bbox_to_anchor=(1.6, 0.1) )

fig.savefig("results_compared.png", dpi=400)
exit()
