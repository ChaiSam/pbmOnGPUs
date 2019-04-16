#!/bin/bash

## The submit script for GPU CUDA executables, for example FDTD3d. 
## 1. Specify the job name in '--job-name='. 
## 2. Specify the number of requested GPU units in '--gres=', maximum 2. 
## 3. Load the cuda module: "module purge; module load cuda" 
## 4. Submit the script to the cluster through SLURM: "sbatch cuda_gpu_batch.sh" 

#SBATCH -N 1            # 1 node
#SBATCH -t 0-03:00:00   # requested 1 day and 3 hours
#SBATCH -p SOE_main   # partition name
#SBATCH --job-name=MPI_test_case
#SBATCH --time=0:45:0
#SBATCH --mem-per-cpu=1000M
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH -J gpu_test_pbm1          # sensible name for the job
###SBATCH --gres=gpu:1         # number of GPUs required
###SBATCH --account=cuda   # need to reference cuda account


myrun=model.out                         # executable to run

#export OMP_NUM_THREADS=1  # assign the number of threads
MYHDIR=$SLURM_SUBMIT_DIR            # directory with input/output files 
MYTMP="/tmp/$USER/$SLURM_JOB_ID"    # local scratch directory on the node
mkdir -p $MYTMP                     # create scratch directory on the node  
cp $MYHDIR/$myrun $MYTMP
cp $MYHDIR/PBM_Input.in $MYTMP
cp -r $MYHDIR/sampledumpfiles $MYTMP
cp -r $MYHDIR/csvDump $MYTMP
#cp -rf  $MYHDIR/*  $MYTMP                # copy all input files into the scratch
cd $MYTMP                           # run tasks in the scratch 

mpirun ./$myrun PBM_Input.in 128 200 0.0 > run.out

rm $myrun                           # the executable doesn't need to be copied back
cp $MYTMP/* $MYHDIR                 # copy everything back into the home dir
rm -r  $MYTMP                      # remove scratch directory

