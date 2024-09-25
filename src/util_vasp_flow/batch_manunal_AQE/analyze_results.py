from ase.io import read
from ase.visualize import view
from ase.visualize.plot import plot_atoms

import os, sys, ast

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import pandas as pd


# Read energy output
path_energy_csv = './'

tot  = pd.read_csv( os.path.join(path_energy_csv,'energy_tot.csv'), index_col=0)
slab = pd.read_csv( os.path.join(path_energy_csv,'energy_slab.csv') )#, index_col=0)

energy_slab = { k[:-7]:v for k,v in zip(slab['name'],slab['energy']) } 
energy_CO  =  -14.48438332
energy_methanol  =  -29.940914

# What pathways to be considered
#labels = sorted( set([ ''.join(m.split('__')[1:]) for m in list( tot.index ) ]) )
labels = sorted( set([ m.split('_')[-1] for m in list( tot.index ) ]) )
#for n,l in enumerate(labels):
#    print(n,l)

pathway_index = [
    [5, 6, 3, 0, 9],
    [5, 2, 3, 0, 9],
    [5, 4, 3, 0, 9],
    [5, 4, 1, 8, 9],
]

#for p in pathway_index:
#    print( ' -> '.join([labels[n].ljust(10,' ') for n in p]) )


# We need to know ID --> surface info
try:
    lookup_table = pd.read_csv('lookup_ID_table.csv')#, index_col=0)
    lookup_table = lookup_table.set_index('id')[['millers','symbols','bulk','composition']].to_dict()
    #print( lookup_table )
    # Got a dict (key='millers'...) of dict (key=id)
except:
    lookup_table = None


## Summarize all pathways: slab ID is key, value is a list of pathways
all_energy_paths = {}
know_slabs = []
for k in energy_slab.keys():
    row_in_tot = [ k+'_'+b for b in labels ]  ## '_' or '__'
    energy = np.array( tot.loc[ row_in_tot ] ).flatten() # adsorbed species' energy

    energy_surface = energy_slab[k]
    energy_H_ad = energy[7]

    # CO binding to surface
    energy_CO_bind = energy[5] - energy_CO - energy_surface

    # Methanol desorb from surface
    energy_methanol_desorb = energy_methanol + energy_surface - energy[9]

    energy_paths = []
    for path in pathway_index:
        reaction_energy = np.diff( energy[path] ) + energy_surface - energy_H_ad
        energy_paths.append( [energy_CO_bind] + list(reaction_energy) + [energy_methanol_desorb] )

    all_energy_paths[k] = energy_paths

    k = k.replace("job_", "")
    k = k.replace("job-", "")

    if lookup_table is not None:
        b = lookup_table['millers'][k]
        a = lookup_table['composition'][k]
        #d = [ f'{i}$_{int(float(j))}$'.strip() for i,j in zip(['Cu','Zn'],d.split(',')) ]
        this_label = a.strip() + b.strip()
    else:
        this_label = 'ID'
    know_slabs.append( this_label )

#print(all_energy_paths)
#from collections import Counter
#know_slabs = Counter(know_slabs)
#for k,v in know_slabs.items():
#    print(k, v)
#exit()

## Rank all pathways
df_data = []
for k,v in all_energy_paths.items():
    for n in range(len(pathway_index)): # 0-4 in the pathway
        df_data.append( [ k+f'/{n}', np.max(v[n][:]) ] )  ## Shall we consider the first and last step? CO adsorption and methanol desorption

df_energy_paths = pd.DataFrame( df_data, columns=['name','dE_max'] )
df_energy_paths = df_energy_paths.sort_values(by='dE_max')
print( '', len(df_energy_paths) )

# Plot all surfaces
fig, axs = plt.subplots(1,2,figsize=(3*2,3),tight_layout=True,dpi=100)

for n,(surf,energy_path) in enumerate(all_energy_paths.items()):
    xdata = list(np.arange( len(energy_path[0]) )+1)
    # The path that are most exothermic: Find the biggest reaction energy in each path, and pick one with the lowest value
    for n,e in enumerate(energy_path):
        axs[0].plot(xdata, e, lw=2, ls='-', marker='o', ms=8, markerfacecolor='none')# color=colors[n], label=labels[n], )
        #ax_max = np.max( [ ax_max, np.max(e) ] )
        #ax_min = np.min( [ ax_min, np.min(e) ] )
        # Energy landscape
        axs[1].plot( [0]+xdata, [0]+list(np.cumsum(e)), lw=2, ls='-', marker='o', ms=6, markerfacecolor='none')# color=colors[n], label=labels[n], )

for n in range(2):
    axs[n].tick_params(direction='in',labelsize=8)
    #axs[n].set_ylim((ax_min-0.1,ax_max+0.1))
    #axs[n].set_xlim(left=0)

axs[0].set_xlabel('Reaction step ',fontsize=10) ## input X name
axs[1].set_xlabel('Reaction states ',fontsize=10) ## input X name
axs[0].set_ylabel('Reaction energy at each step (eV)',fontsize=10) ## input Y name
axs[1].set_ylabel('Energy landscape(eV)',fontsize=10) ## input Y name

fig.savefig("All_surf_pathway.png", dpi=400)
#plt.show()


# Plot selected pathways
selected_paths = df_energy_paths.head(20)
print( selected_paths )

fig, axs = plt.subplots(1,2,figsize=(3.5*2,3),tight_layout=True,dpi=100)
colors = [plt.cm.rainbow(n) for n in np.linspace(0,1,len(selected_paths)) ]

landscape_selected_paths = {}
for n,name in enumerate(selected_paths['name']):
    surface_key = name.split('/')[0] ## id number
    path_index = int(name.split('/')[1])
    
    if lookup_table is not None:
        a = lookup_table['symbols'][surface_key]
        b = lookup_table['bulk'][surface_key]
        c = lookup_table['millers'][surface_key]
        d = lookup_table['composition'][surface_key]
        d = [ f'{i}$_{int(float(j))}$'.strip() for i,j in zip(['Cu','Zn'],d.split(',')) ]
        this_label = ''.join(d) + c.strip()
        #print( surface_key, this_label )
    else:
        this_label = str(n)
    
    e = all_energy_paths[surface_key][path_index]
    xdata = list(np.arange(len(e))+1)

    xlabel = [ labels[i] for i in pathway_index[path_index] ]
    xlabel = ['CO(g)']+xlabel+['CH3OH(g)']
    xlabel_reaction = [ xlabel[i] +r' $\rightarrow$ '+ xlabel[i+1] for i in range(len(xlabel)-1)]
    
    axs[0].plot(xdata, e, lw=2, ls='-', marker='o', ms=4, markerfacecolor='none', color=colors[n])#, label=labels[n], )
    landscape = [0]+list(np.cumsum(e))
    axs[1].plot( [0]+xdata, landscape, lw=2, ls='-', marker='o', ms=4, markerfacecolor='none', color=colors[n], label=this_label, )
    landscape_selected_paths[surface_key] = landscape
    
    axs[0].set_xticks( xdata )
    axs[0].set_xticklabels( xlabel_reaction, fontsize=8, rotation=-90 )
    axs[1].set_xticks( [0]+xdata )
    axs[1].set_xticklabels( xlabel, fontsize=8, rotation=-90 )

for n in range(2):
    axs[n].tick_params(direction='in',labelsize=8)
    #axs[n].set_ylim((ax_min-0.1,ax_max+0.1))
    #axs[n].set_xlim(left=0)

axs[0].set_xlabel('Reaction steps ',fontsize=8) ## input X name
axs[1].set_xlabel('Reaction states ',fontsize=8) ## input X name
axs[0].set_ylabel('Reaction energy \nat each step (eV)',fontsize=8) ## input Y name
axs[1].set_ylabel('Energy landscape(eV)',fontsize=8) ## input Y name

axs[1].legend(fontsize=6, frameon=False, loc='center right', ncol=1, columnspacing=1, bbox_to_anchor=(1.6, 0.1) )

fig.savefig("Best_surf_pathway.png", dpi=400)
#plt.show()
landscape_selected_paths = pd.DataFrame.from_dict( landscape_selected_paths )
landscape_selected_paths.to_csv('Best_surf_pathway.csv')

## The best one:
name = selected_paths['name'].iloc[0]
surface_key = name.split('/')[0] ## id number
path_index = int(name.split('/')[1])
#print( surface_key, lookup_table['millers'][surface_key], lookup_table['bulk'][surface_key], lookup_table['composition'][surface_key])
#print( ' -> '.join([labels[n].ljust(10,' ') for n in pathway_index[path_index]]) )
#print( ' -> '.join([ str(n).ljust(10,' ') for n in pathway_index[path_index]]) )

