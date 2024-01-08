import pickle
import pandas as pd
# CuZn, CuGa, PdSn, NiZn, PtGa

cu_cat = [
    ["Cu","Fe"],
["Cu","Ag"],
["Cu","Ni"],
["Cu","Zn"],
# ["Cu", "Sn"],
# ["Cu","In"],
# ["Cu","Ga"],
["Pd","Zn"]
]

# computational_pathways_methanol = [
#    ["CO2", "*OCHO", "*CHOH", "*OHCH3"],
#    # ["CO2", "*COOH", "*CO", "*OHCH3"],
#    ["CO2", "*CO", "*CHO", "*CH2*O", "*OHCH3"],
# ]

computational_pathways_ethanol = [
    ["CO2", "*CO", "*COOH", "*CHOH", "*OCH2CH3"],
    ["CO2", "*CO", "*CH2*O", "*OCH2CH3"],
]

slab_pathways = {}
for slab_name in cu_cat:
    
    slab_fname = ''.join(slab_name)
    pathways={}
    for ip, path in enumerate(computational_pathways_ethanol):
    
        step_list=[]
        for adsorbate in path:
            current_step_list = [ slab_name, adsorbate, slab_fname ]
            step_list.append(current_step_list)
            
            
        pathways[slab_fname+"_"+str(ip)] = step_list
        

    slab_pathways[slab_fname] = pathways

with open("ethanol_pathways.pkl", "wb") as f:
    pickle.dump(slab_pathways, f)