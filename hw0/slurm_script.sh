#!/bin/bash

# to run this slurm script, put it in the same directory as the primetest.cu code
# and execute "sbatch slurm_script.sh" in the shell.
#
# if you want to find out what slurm jobs you have running, type "squeue" in the shell.
#
# if you want to cancel a running job, use "scancel" after you have the job id from
# squeue.


# The -n<#> flag specifies the number of processes
#SBATCH -n1

# The --gres=gpu:<X> flag specifies the number of GPUs per node to assign
#SBATCH --gres=gpu:2

# The -J flag specifies the job name
#SBATCH -J gpu_class_compile_or_run

# The --mail-type=ALL flag turns on email notifications.
#SBATCH --mail-type=ALL
# If you want to receive email at your real email address, put the address in
# a file called $HOME/.forward
# If you don’t want any mail at all, change the above type from ALL to NONE

# The -o flag controls the name of the output file.
# By default, a file called "slurm-%j.out" is used, where %j is the job id.
#SBATCH -o "job_gpu_compilerun-%j.out"

echo
echo "This job will compile or execute on a single GPU."
echo "You can have this print out whatever you want. "
echo "This is job $SLURM_JOB_ID running on `srun hostname`"
echo

# print each command before it’s executed:
set -x

# uncomment this, if you want to “make” (i.e. there is a makefile 
# in the current directory
# srun make

# uncomment this, if you want to compile the file “myprog.cu” in the
# current directory
nvcc copV.cu -o copV -arch=sm_20

# uncomment this if you want to run “myprog”
# note that the “-P singleton” argument states that this should run ONLY
# AFTER the previous commands (which are compiles) have finished
srun ./copV 10000000

