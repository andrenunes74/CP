#!/bin/bash
#SBATCH --time=1:00
#SBATCH --ntasks=4
#SBATCH --partition=cpar
mpirun -np 4 ~/TP3_MPI/bin/k_means