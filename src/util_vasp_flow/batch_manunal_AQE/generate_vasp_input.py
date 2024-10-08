import numpy as np
from ase.calculators.vasp import Vasp
from ase.io import read,write
from ase.constraints import FixAtoms
import sys, os


# NOTE: this is the setting for slab and adslab
VASP_FLAGS = {
    "ibrion": 2,
    "nsw": 10,  ## This time we change 2000 to 200
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


# Read .xyz input
#print('Reading')
file_name_input = 'data-in.xyz'
#file_name_input = sys.argv[1]
atom1 = read(file_name_input)

# Set frozen atoms: tag 0 = frozen
#print('Freezing')
frozen_idx = np.argwhere( atom1.get_array('tags')==0 ).flatten()
c = FixAtoms(indices=frozen_idx)
atom1.set_constraint(c)
##write( 'POSCAR.vasp', atom1 )

# Write POSCAR with frozen information
#print('Writting')
write_vasp_input_files( atom1, outdir=".", vasp_flags=VASP_FLAGS )
#print( atom1 )


