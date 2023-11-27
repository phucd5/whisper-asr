#!/bin/bash
#SBATCH --job-name=whisper_training_a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=whisper_train_a100%j.txt

module load miniconda CUDAcore/11.3.1 cuDNN/8.2.1.32-CUDA-11.3.1
conda activate whisper_asr
python asr.py
