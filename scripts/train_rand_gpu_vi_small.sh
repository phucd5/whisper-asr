#!/bin/bash
#SBATCH --job-name=whisper_training_vi_small
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=whisper_train_vi_small_%j.txt


module load miniconda CUDAcore/11.3.1 cuDNN/8.2.1.32-CUDA-11.3.1
conda activate whisper_asr
python ../run.py --model_name openai/whisper-small --language Vietnamese --language_code vi --save_to_hf --output_dir ../models/whisper-small-vi