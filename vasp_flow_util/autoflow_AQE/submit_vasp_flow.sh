#!/bin/bash
#SBATCH -p bsc120c                                                                                                 
#SBATCH -o slurm-%J.out                                                                                                 
#SBATCH -e slurm-%J.err                                                            
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=120
#SBATCH -t 1:00:00    
#SBATCH -J test_vasp                                                                                                                                                 
 
date

. /anfhome/.profile

module purge
module load intel-oneapi-mkl
module load mpi/impi-2021

export PATH=$PATH:/anfhome/shared/chemreasoner/vasp.6.4.3/bin
export VASP_PP_PATH=/anfhome/shared/chemreasoner/vasp.6.4.3/pot
export ASE_VASP_VDW=/anfhome/shared/chemreasoner/vasp.6.4.3/pot/vdw_kernel.bindat

## Change this to any env with ASE installed
conda activate /anfhome/difan.zhang/anaconda3

## Convert xyz input to POSCAR, and generate POTCAR, KP, etc...
python  util_vasp.py  0

## Running VASP
mpirun  vasp_std_old > job.log

## Post-process to get position, force, energy
python  util_vasp.py  1

date
 
