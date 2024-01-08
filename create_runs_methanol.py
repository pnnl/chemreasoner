import pickle
import pandas as pd
# CuZn, CuGa, PdSn, NiZn, PtGa
cu_cat = [
    ["Cu", "Zn"],
    # ["Cu","Ga"],
    # ["Pd","Sn"],
    ["Ni","Zn"],
    # ["Pt","Ga"],
    # ["Cu","Ni"],
    # ["Cu","Sn"],
    # ["Cu","In"]
]

computational_pathways_methanol = [
   ["CO2", "*OCHO", "*CHOH", "*OHCH3"],
   # ["CO2", "*COOH", "*CO", "*OHCH3"],
   ["CO2", "*CO", "*CHO", "*CH2*O", "*OHCH3"],
]

slab_pathways = {}
for slab_name in cu_cat:
    
    slab_fname = ''.join(slab_name)
    pathways={}
    for ip, path in enumerate(computational_pathways_methanol):
    
 
        step_list=[]
        for adsorbate in path:
            current_step_list = [ slab_name, adsorbate, slab_fname ]
            step_list.append(current_step_list)
            
            
        pathways[slab_fname+"_"+str(ip)] = step_list
 
        

    slab_pathways[slab_fname] = pathways


with open("methanol_pathways.pkl", "wb") as f:
    pickle.dump(slab_pathways, f)