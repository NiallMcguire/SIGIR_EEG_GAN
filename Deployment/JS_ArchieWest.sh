#!/bin/bash

#=================================================================
#
# Job script for running a job on a single GPU (any available GPU)
#
#=================================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the gpu partition (queue) with any GPU
#SBATCH --partition=gpu --gres=gpu:A100 --mem-per-cpu=9600
#
# Specify project account (replace as required)
#SBATCH --account=moshfeghi-pmwc
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=05:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
# Job name
#SBATCH --job-name=gpu_test
#
# Output file
#SBATCH --output=slurm-%j.out
#======================================================

module purge
module load nvidia/sdk/22.3
module load anaconda/python-3.9.7/2021.11

#Uncomment the following if you are running multi-threaded
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#
#=========================================================
# Prologue script to record job details
# Do not change the line below
#=====================
#----------------------------------------------------------

#Modify the line below to run your program. This is an example
#=========================================================
python /users/gxb18167/SIGIR_EEG_GAN/Deployment/JS-Distance.py --model WGAN_v2_Text --type normal --generator_path Generation_size_Word_Level_batch_size_64_g_d_learning_rate2e-05_2e-05_word_embedding_dim_50_z_size_100_num_epochs_100_device_cuda:0_model_final.pt


# Do not change the line below
#=========================================================
/opt/software/scripts/job_epilogue.sh
#----------------------------------------------------------
