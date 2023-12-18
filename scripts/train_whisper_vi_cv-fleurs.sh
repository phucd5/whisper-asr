#!/bin/bash
#SBATCH --job-name=train_whisper_vi_cv-fleurs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=whisper_train_vi_better_%j.txt

module load miniconda CUDAcore/11.3.1 cuDNN/8.2.1.32-CUDA-11.3.1
conda activate whisper_asr
python ../train.py \
    --model_name ../working-models/whisper-vi-cv \
    --language Vietnamese \
    --language_code vi_vn \
    --output_dir ../models/whisper-vi-cv-fleurs \
    --dataset_name google/fleurs \
    --ref_key raw_transcription \
    --save_to_hf \
    --existing_model

