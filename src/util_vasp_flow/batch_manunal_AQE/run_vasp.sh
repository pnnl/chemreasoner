#!/bin/bash
#SBATCH -p bsc120c                                                                                                 
#SBATCH -o slurm-%J.out                                                                                                 
#SBATCH -e slurm-%J.err                                                            
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=120
#SBATCH -t 24:00:00    
#SBATCH -J geo_vasp                                                                                                                                                 
 
date

. /anfhome/.profile

module purge
module load intel-oneapi-mkl
module load mpi/impi-2021
  
#export OMP_NUM_THREADS=2
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

mpirun  vasp_std > job.log
#srun -n 120 -c 2  vasp_std > job.log

date
 
