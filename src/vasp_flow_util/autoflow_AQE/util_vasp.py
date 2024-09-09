import numpy as np
from ase.calculators.vasp import Vasp
from ase.io import read,write
from ase.constraints import FixAtoms
import sys, os
import pandas as pd


'''
For generating VASP input, based on a XYZ file with box info. Typically from ASE output.
Part of the codes are from Open Catalyst:https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/data/oc/utils/vasp.py
'''
# NOTE: this is the setting for slab and adslab
VASP_FLAGS = {
    "ibrion": 2,
    "nsw": 1,
    "isif": 0,
    "isym": 0,
    "lreal": "Auto",
    "ediffg": -0.03,
    "symprec": 1e-10,
    "encut": 350.0,
    "laechg": True,
    "lwave": False,
    "ncore": 12,
    "gga": "RP",
    "pp": "PBE",
    "xc": "PBE",
}


def clean_up_inputs(atoms, vasp_flags):
    """
    Parses the inputs and makes sure some things are straightened out.

    Arg:
        atoms       `ase.Atoms` object of the structure we want to relax
        vasp_flags  A dictionary of settings we want to pass to the `Vasp`
                    calculator
    Returns:
        atoms       `ase.Atoms` object of the structure we want to relax, but
                    with the unit vectors fixed (if needed)
        vasp_flags  A modified version of the 'vasp_flags' argument
    """
    # Make a copy of the vasp_flags so we don't modify the original
    vasp_flags = vasp_flags.copy()
    # Check that the unit vectors obey the right-hand rule, (X x Y points in
    # Z). If not, then flip the order of X and Y to enforce this so that VASP
    # is happy.
    if np.dot(np.cross(atoms.cell[0], atoms.cell[1]), atoms.cell[2]) < 0:
        atoms.set_cell(atoms.cell[[1, 0, 2], :])

    # Calculate and set the k points
    if "kpts" not in vasp_flags:
        k_pts = calculate_surface_k_points(atoms)
        vasp_flags["kpts"] = k_pts

    return atoms, vasp_flags


def calculate_surface_k_points(atoms):
    """
    For surface calculations, it's a good practice to calculate the k-point
    mesh given the unit cell size. We do that on-the-spot here.

    Arg:
        atoms   `ase.Atoms` object of the structure we want to relax
    Returns:
        k_pts   A 3-tuple of integers indicating the k-point mesh to use
    """
    cell = atoms.get_cell()
    order = np.inf
    a0 = np.linalg.norm(cell[0], ord=order)
    b0 = np.linalg.norm(cell[1], ord=order)
    multiplier = 40
    return (
        max(1, int(round(multiplier / a0))),
        max(1, int(round(multiplier / b0))),
        1,
    )


def write_vasp_input_files(atoms, outdir=".", vasp_flags=None):
    """
    Effectively goes through the same motions as the `run_vasp` function,
    except it only writes the input files instead of running.

    Args:
        atoms       `ase.Atoms` object that we want to relax.
        outdir      A string indicating where you want to save the input files.
                    Defaults to '.'
        vasp_flags  A dictionary of settings we want to pass to the `Vasp`
                    calculator. Defaults to a standerd set of values if `None`
    """
    if vasp_flags is None:  # Immutable default
        vasp_flags = VASP_FLAGS

    atoms, vasp_flags = clean_up_inputs(atoms, vasp_flags.copy())
    calc = Vasp(directory=outdir, **vasp_flags)
    calc.write_input(atoms)


'''
For post-run analysis of pos, energy, force. Need OUTCAR file and CONTCAR.
'''
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

########################## run part #################################
try:
    run_mode = int( sys.argv[1] )
except:
    print( 'Arg needs to be either 0 or 1. 0=generating VASP input. 1=analyzing VASP output' )

if run_mode == 0:
    # Read .xyz input
    print('Reading')
    file_name_input = 'data-in.xyz'
    atom1 = read(file_name_input)

    # Set frozen atoms: tag 0 = frozen
    print('Freezing')
    frozen_idx = np.argwhere( atom1.get_array('tags')==0 ).flatten()
    c = FixAtoms(indices=frozen_idx)
    atom1.set_constraint(c)
    ##write( 'POSCAR.vasp', atom1 )

    # Write POSCAR with frozen information, and INCAR, POT, KPOINTS, etc
    print('Writting')
    write_vasp_input_files( atom1, outdir=".", vasp_flags=VASP_FLAGS )
    #print( atom1 )

elif run_mode == 1:
    energy_txt = 'energy_reference.txt'
    jobdir = './'
    # If energy_reference.txt in the job, then find its energy from its link, otherwise from job.log/OUTCAR
    filelist = os.listdir( jobdir )
    if energy_txt in filelist:
        energy_source = get_energy_link( os.path.join(jobdir, energy_txt) )
    else:
        energy_source = jobdir

    # Total number of atoms
    natoms = get_natom_from_xyz( os.path.join(energy_source,'data-in.xyz') )
    # pos and force # and energy
    pos_f,energy = get_results_from_outcar( os.path.join(energy_source,'OUTCAR'), natoms )

    ## Save the energy and traj under each job, only if it was computed.
    if energy_source == jobdir:  
        np.save( os.path.join(energy_source,'track_energy.npy'), energy)
        np.save( os.path.join(energy_source,'track_pos_force.npy'), pos_f)

    print('Optimized structure energy: ',energy[-1], ' for optimized structure: CONTCAR' )

