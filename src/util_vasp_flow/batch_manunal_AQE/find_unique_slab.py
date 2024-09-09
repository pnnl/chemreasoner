from ase.io import read
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def subgraph_finder(graph):
    subgraph = [graph.subgraph(c) for c in nx.connected_components(graph)]
    nodes = [ list(subg.nodes) for subg in subgraph ]
    return nodes

def write_calculate_instruction(jobdir, energy_reference_path=''):
    with open( os.path.join(jobdir,'energy_reference.txt') , 'w' ) as f1:
        f1.write( energy_reference_path )
        print('Writing', energy_reference_path, ' to ', jobdir)


# all slabs' folder names #
slabs = [ s for s in os.listdir('.') ]
atoms = [ read(os.path.join(s,'data-in.xyz')) for s in slabs ]

# Use graph to cluster identical traj. We use num of atom, element order and coord.
#Note: XYZ coord are not 100% different after a few decimals. So coords needs to be modified before comparing
natoms, elements, coords = [],[],[]
for i,at in enumerate(atoms):
    natoms.append( len(atoms) )
    elements.append( at.get_chemical_symbols() )
    coords.append( np.round(at.get_positions(), decimals=1) )

# Every traj is a node
graph = nx.Graph()
for n in range(len(atoms)):
    graph.add_node(n, name=slabs[n])

# Now find which two nodes are identical. We add an edge if they are the same.
    for i in range(0, len(atoms)-1):
        for j in range(1, len(atoms)):
            # 1: if two nodes have different num of atom, they cannot be the same
            # 2: For the same num of atom , do they have the same element order?
            # 3: If same element, do they have the same coord?
            if natoms[i]==natoms[j]:
                if elements[i]==elements[j] and np.all(coords[i]==coords[j]):
                    graph.add_edge(i,j, bon_type=0)

# Show graph
fig, axs = plt.subplots(1,1, figsize=(4,3), tight_layout=True, dpi=200)
pos = nx.kamada_kawai_layout(graph, )#pos=pos)      
pos = nx.spring_layout(graph, k=0.3, pos=pos, seed=12345)
nx.draw_networkx(graph, pos, with_labels=False, ax=axs,
                 node_size=50, font_size=8, font_weight='bold', node_color='lightblue',# cmap=plt.cm.rainbow, 
                 edge_color='k', #width=edge_width,
                )
               
plt.savefig("../slab_graph.png", dpi=200)

# print results
clusters = subgraph_finder(graph)
for cluster in clusters:
    print( cluster )
    cluster = [ graph.nodes[c]['name'] for c in cluster ]

    #for c in cluster:
    #    print( c )
 
    # Write instruction log to each traj   
    energy_ref = cluster[0] # Only calculating the first traj is needed
    #write_calculate_instruction( cluster[0] )
    for c in cluster[1:]:
        write_calculate_instruction(c, energy_reference_path=energy_ref )

    """
    # To prove we get the right results
    xyz = [ read(os.path.join(c,'data-in.xyz')) for c in cluster ]
    xyz = [ x.get_positions() for x in xyz ]
    mse = [ np.square(xyz[n]-xyz[0]).mean() for n in range(len(xyz)) ]
    print( mse )
    """

exit()

