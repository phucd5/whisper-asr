import torch
import subprocess
import argparse
from WhisperASR import WhisperASR

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Train WhisperASR model with specified language settings.')
parser.add_argument('--language', type=str, required=True,
                    help='Language for the Whisper ASR model')
parser.add_argument('--language_code', type=str, required=True,
                    help='Language code for the Whisper ASR model')
args = parser.parse_args()

# check if GPU is available
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)

# print GPU info
try:
    gpu_info = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
except subprocess.CalledProcessError as e:
    if e.returncode == 1:
        print('Not connected to a GPU')
    else:
        print('An error occurred: ', e)
except FileNotFoundError:
    print('nvidia-smi not found. Is the Nvidia driver installed?')
else:
    print(gpu_info)

# train the model with command-line arguments
model = WhisperASR(model_name="openai/whisper-small",
                   language=args.language, language_code=args.language_code)
model.train()

