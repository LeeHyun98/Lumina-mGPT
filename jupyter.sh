#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=./slurm_log/S-%x.%j.out
#SBATCH --nodelist=node01 # 계산 노드 (a100/a6000/a3000/a5000/a4000)

ml purge
ml load cuda/12.1
eval "$(conda shell.bash hook)"
conda activate Lumina-mGPT
srun jupyter notebook --no-browser --port=30620