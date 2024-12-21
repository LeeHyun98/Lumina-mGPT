#!/bin/bash

#SBATCH --job-name=Lumina_mGPT # 작업 이름
#SBATCH --partition=a5000 # 계산 노드 (a100/a6000/a3000/a5000/a4000)
#SBATCH --gres=gpu:1 # GPU 개수
#SBATCH --time=0-00:30:00 # 최대 수행 시간 (d-hh:mm:ss 형식)
#SBATCH --mem=16G # RAM 크기
#SBATCH --cpus-per-task=4 # CPU 개수 (4개 권장)
#SBATCH --output=./slurm_log/S-%x.%j.out # 실행 결과 std output을 저장할 파일
#SBATCH --nodelist=node04 # 계산 노드 (a100/a6000/a3000/a5000/a4000)

ml purge
ml load cuda/12.1 # CUDA 로드
eval "$(conda shell.bash hook)"
conda activate Lumina-mGPT # 가상 환경 활성화
srun python inference.py # 수행할 작업
