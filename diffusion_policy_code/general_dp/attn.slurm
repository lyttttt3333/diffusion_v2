#!/bin/bash
#SBATCH --job-name="attn_battery_180_v1"
#SBATCH --output="/projects/bcfs/ywang41/baseline/slurm_output/%x/%j.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=32  
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcyd-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 16:00:00

export HYDRA_FULL_ERROR=1

source /u/ywang41/miniforge3/etc/profile.d/conda.sh
conda activate robodiff_bl
python train.py --config-dir=config --config-name=pack_battery_attn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'