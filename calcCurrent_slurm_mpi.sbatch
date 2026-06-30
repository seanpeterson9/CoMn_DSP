#!/bin/bash

#SBATCH --partition=priority
#SBATCH --account=priority-mmosquera
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10000MB
#SBATCH --time=6-00:00:00
#SBATCH --output=calcCurrent_mpi_%j.out
#SBATCH --no-requeue

module load mpi4py/4.0.1-gompi-2024a

export SNPSLMD_LICENSE_FILE="27020@chestnut.msu.montana.edu"
export MPI_EXE=mpiexec
export MKL_DYNAMIC=TRUE
export OMP_NUM_THREADS=4

${MPI_EXE} -n $(( SLURM_NTASKS )) python3 calcCurrentDenstiyFromBandsDatFilesMPI.py


##When running this script you need to load the mpi4py module as well as activate 
## the venv in the terminal it's being run in
## the associated venv was created by:
##	uv venv myenv
##	source myenv/bin/activate
##	uv pip install numpy scipy matplotlib csv
