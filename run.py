import torch
import subprocess
from WhisperASR import WhisperASR

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

# train the model
model = WhisperASR()
model.train()
