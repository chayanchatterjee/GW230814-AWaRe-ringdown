#!/bin/bash
#SBATCH --job-name=train_recons_temporal             # Job name
#SBATCH --output=train_temporal_output.log        # Standard output log file
#SBATCH --error=train_temporal_error.log          # Standard error log file
#SBATCH --partition=interactive_gpu         # Partition
#SBATCH --account=dsi_dgx_iacc
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --time=48:00:00                   # Time limit (hh:mm:ss)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=6  
#SBATCH --mem=80GB 

# Load Singularity module if required by your cluster
#module load singularity

# Define the Singularity container path
CONTAINER_PATH="/data/p_dsi/ligo/gw_container.simg"

#singularity shell $CONTAINER_PATH


#singularity exec --bind /nobackup/user/chattec:/nobackup/user/chattec --bind /data/p_dsi/ligo:/data/p_dsi/ligo  --nv  $CONTAINER_PATH python3 train_recons_temporal_atten.py --train_hdf /data/p_dsi/ligo/GW230814/Train_GW230814.hdf --test_hdf /data/p_dsi/ligo/GW230814/Test_GW230814.hdf --extract_attention --test_calibration

singularity exec --bind /nobackup/user/chattec:/nobackup/user/chattec --bind /data/p_dsi/ligo:/data/p_dsi/ligo  --nv  $CONTAINER_PATH python3 train_recons_temporal_atten.py --train_hdf /data/p_dsi/ligo/GW230814/Train_GW230814_ringdown_1024Hz_highpass_16kHz_sr.hdf --test_hdf /data/p_dsi/ligo/GW230814/Test_GW230814_ringdown_1024Hz_highpass_16kHz_sr.hdf --extract_attention --test_calibration

