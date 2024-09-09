import os, sys
import numpy as np
import pandas as pd


def get_final_energy(fin):
    with open(fin,'r') as f1:
        lines = [ l for l in f1.readlines() if 'E0=' in l]
        lines = [ float(line.split()[4]) for line in lines ]
    return  lines

def get_energy_link(fin):
    with open(fin,'r') as f1: 
        lines = f1.readlines()
    return lines[0]

def get_natom_from_xyz(fin):
    with open(fin,'r') as f1:
        lines = f1.readlines()
    return len(lines)-2

def get_results_from_outcar(fin,num_atom):
    with open(fin,'r') as f1:
        lines = f1.readlines() 
    # The index line of position and force
    line_idx = [ n for n in range(len(lines)) if 'TOTAL-FORCE (eV/Angst)' in lines[n] ]
    frames = [ [l.split() for l in lines[n+2: n+2+num_atom]] for n in line_idx ]
    # The index line of energy
    line_idx = [ n for n in range(len(lines)) if 'energy  without entropy=' in lines[n] ]
    energies = [ float(lines[n].split()[-1]) for n in line_idx ]

    return np.array(frames,dtype=float), energies 


jobdirs = [ s for s in os.listdir('.') ]
#jobdirs = ['./']

energy_results = []  # The last energy of every system
energy_txt = 'energy_reference.txt'
# If energy_reference.txt in the job, then find its energy from its link, otherwise from job.log
for n in range( len(jobdirs) ):
    filelist = os.listdir( jobdirs[n] )
    if energy_txt in filelist:
        energy_source = get_energy_link( os.path.join(jobdirs[n], energy_txt) )
        # Just a sanity check
        if 'job.log'in filelist:
            print('This is no right')
    else:
        energy_source = jobdirs[n]

    # natom #
    natoms = get_natom_from_xyz( os.path.join(energy_source,'data-in.xyz') )
    # energy changes #
    #energy = get_final_energy( os.path.join(energy_source,'job.log') )
    # pos and force # and energy
    pos_f,energy = get_results_from_outcar( os.path.join(energy_source,'OUTCAR'), natoms )

    energy_results.append( [jobdirs[n], energy[-1]] )

    ## Save the energy and traj under each job, only if it was computed.
    if energy_source == jobdirs[n]:  
        np.save( os.path.join(energy_source,'track_energy.npy'), energy)
        np.save( os.path.join(energy_source,'track_pos_force.npy'), pos_f)
    #print( pos_f )

energy_results = np.array( energy_results, dtype='str' )
print( energy_results )

# Save results to csv
fout = sys.argv[1] ## e.g.,'../energy_cp2k.csv'
df = pd.DataFrame(energy_results, columns=['name','energy'])
print(df.shape)
df.to_csv(fout, index=False)

