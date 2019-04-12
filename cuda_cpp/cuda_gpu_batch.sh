#!/bin/bash

## The submit script for GPU CUDA executables, for example FDTD3d. 
## 1. Specify the job name in '--job-name='. 
## 2. Specify the number of requested GPU units in '--gres=', maximum 2. 
## 3. Load the cuda module: "module purge; module load cuda" 
## 4. Submit the script to the cluster through SLURM: "sbatch cuda_gpu_batch.sh" 

#SBATCH -N 1            # 1 node
#SBATCH -t 1-03:00:00   # requested 1 day and 3 hours
#SBATCH -p SOE_interactive   # partition name
#SBATCH -J gpu_test          # sensible name for the job
#SBATCH --gres=gpu:1         # number of GPUs required

myrun=FDTD3d                         # executable to run

export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE  # assign the number of threads
MYHDIR=$SLURM_SUBMIT_DIR            # directory with input/output files 
MYTMP="/tmp/$USER/$SLURM_JOB_ID"    # local scratch directory on the node
mkdir -p $MYTMP                     # create scratch directory on the node  
cp $MYHDIR/$myrun  $MYTMP                # copy all input files into the scratch
cd $MYTMP                           # run tasks in the scratch 

./$myrun > run.out

rm $myrun                           # the executable doesn't need to be copied back
cp $MYTMP/* $MYHDIR                 # copy everything back into the home dir
rm -rf  $MYTMP                      # remove scratch directory

