# Fine-tuning OpenAI's Whisper for Multilingual ASR with Transformers

## Author
Phuc Duong and Sophia Kang 

CPSC 488: AI Foundational Models

## Overview
In this project, we investigate and improve on OpenAI's Whisper model, as detailed in the paper "Robust Speech Recognition via Large-Scale Weak Supervision," to focus on accurate recognition and transcription of Vietnamese and Korean. These languages present unique linguistic challenges in the speech recognition space: Vietnamese, with its specific tonal nature, dialectical variation, and prevalence of monosyllabic words, and Korean, with word segmentation. 

Through the Common Voice and FLEURS datasets and leveraging OpenAI's Whisper API and Hugging Face's transformers library to train and load the model, we aim to finetune Whisper's performance in order to reduce transcription errors and improve adaptability in regards to the model ability to handle the complexity of these two languages.

## Setup and Computing Infrastructure

We used Python 3.10.13 and Pytorch 1.12.1 to train and test our models. However, our model is expected to be compatible with Python 3.9-3.11 and recent Pytorch versions (although not explicilty verified). We also used Hugging Face transformer library to interface with the models. We used an NVIDIA RTX A5000 graphics card for training and evaluation on the Yale High Performance Computing (HPC) clusters, running the Red Hat Enterprise Linux OS with version 8.8, codenamed Ootpa. We used an Intel Xeon Gold 6326 CPU with 64 CPU cores, with 64 GB of RAM available on the cluster, and 4 CPUs allocated per task.

A full list of the dependencies and their versions can be found in `environment.yml`.

To install all the dependencies please do the following command

```
conda env create -f environment.yml
conda activate env_whisper_asr
```

## Files Description

This technique is adapted from Hugging Face's methodology from [Hugging Face's article](https://huggingface.co/blog/fine-tune-whisper) on fine-tuning Whisper.

- train.py: Initiates the model, and ensure the environment is set up for training. 

- WhisperASR.py: Interface with the Hugging Face transformers library to load data, prepare the model for training and train the model.

- MetricsEval.py: Utilized in the WhisperASR class to specify the evaluation metric of the model and compute the metric for evaluation. The default metric is Word Error Rate (WER).

- DataCollatorSpeechSeq2SeqWithPadding.py: Preparing batches of data during the training of the model. Convert input features to batched Pytorch tensors and pad labels while ensuring its not taken into account when computing lost. 

## Training

### Configuration
Before running the code, please make sure to update the following constants in the `WhisperASR.py` file to match your specific configuration:

1. `OUTPUT_DIR`: Set this to the directory where you want to save the model and related files.
2. `HF_API_KEY`: Replace `"api_key"` with your Hugging Face API key.
3. `BASE_MODEL`: Specify the base model that you want to train on. The base models are below:


|  Size  | Parameters | English-only model | Multilingual model |  
|:------:|:----------:|:------------------:|:------------------:|
|  tiny  |    39 M    |         ✓          |         ✓          |
|  base  |    74 M    |         ✓          |         ✓          |
| small  |   244 M    |         ✓          |         ✓          |
| medium |   769 M    |         ✓          |         ✓          |
| large  |   1550 M   |                    |         ✓          |

Official model information is available at Open AI's Whisper [repository](https://github.com/openai/whisper/blob/main/model-card.md).

In addition, please update the training parameters as you wish in `WhisperASR.py`.

## Command-line usage

To train the model you use the script train.py with the following CLI arguments. 

**Note:** The train script was built to interface with Hugging Face transformers and datasets available on Hugging Face website. If using a custom Whisper model or custom dataset, please make sure it's compaitable with Hugging Face libaries.

Required Parameters:

- `--language`: Language to train the Whisper ASR model with (ex: `Vietnamese`)
- `--language_code`: Language code for the Whisper ASR model based on the data set. (ex: `vi` for Common Voice)
- `--output_dir`: Directory to save the trained model.

Optional Parameters:

- `--model_name`: Model name from Hugging Face or custom path (default: `openai/whisper-small`).
- `--existing_model`: Flag to indicate an existing model is being used (default: False).
- `--save_to_hf`: Flag to indicate saving the model to Hugging Face Hub (default: False).
- `--dataset_name`: Dataset name to train on (default: `mozilla-foundation/common_voice_13_0`).
- `--ref_key`: Key in dataset for reference data (default: `sentence` - matches with Common Voice).

### Examples


Training a pre-trained local model on the Vietnamese language with the google/fleurs dataset

```shell
python ../train.py \
    --model_name ../working-models/whisper-vi-cv \
    --language Vietnamese \
    --language_code vi_vn \
    --output_dir ../models/whisper-vi-cv-fleurs \
    --dataset_name google/fleurs \
    --ref_key transcription \
    --save_to_hf \
    --existing_model
```

Training Hugging Face openai/whisper-small on Vietnamese using the mozilla-foundation/common_voice_13_0 dataset.

```shell
python ../train.py --language Vietnamese --language_code vi --output_dir ../models/whisper-small-vi --save_to_hf
```

In addition, the `scripts` folder will also contain Slurm batch scripts that we used to train the model on Yale High Performance Computing (HPC) clusters for references. 

## Evaluation

### Configuration
Before running the code, please make sure to update the following constants in the `evaluate_model.py` file to match your specific configuration:

1. `HF_API_KEY`: Replace `"api_key"` with your Hugging Face API key.

### Command-line usage

To evaluate the model on a Hugging Face dataset you use the script evaluate_model.py with the following CLI arguments. 

**Note:**  The evaluate script was built to interface with Hugging Face transformers and datasets available on Hugging Face website. If using a custom Whisper model please make sure it's compaitable with Hugging Face libaries.

Functional Parameters:

- `--model_name`: Hugging Face Whisper model name, or the path to a local model directory. (default: `openai/whisper-small`)
- `--dataset_name`: Name of the dataset from Hugging Face (default: `mozilla-foundation/common_voice_13_0`)
- `--config`: The configuration of the dataset (default: 'vi' for Vietnamese for Common Voice)
- `--language`: Language code for transcription (default: 'vi' for Vietnamese for Common Voice)
- `--device`: GPU ID for using a GPU (e.g., 0), or -1 to use CPU (default: 0).
- `--split`: The dataset split to evaluate on (default: 'test').
- `--output_file`: Name of the file to save the evaluation results.
- `--ref_key`: Key in the dataset for reference data (default: 'sentence' - matches with Common Voice).

Flag Parameters:

- `--save_transcript`: Flag to save the transcript to a file (default: False).
- `--cer`: Flag to calculate the Character Error Rate (default: False).
- `--spacing_er`: Flag to calculate spacing accuracy rate (default: False)

### Examples

Evaluting standard openai/whisper-small on commmon voice dataset for the Korean language using WER and Spacing Accuracy metric.

```bash
python evaluate_model.py --language ko --config ko --save_transcript --output_file eval-ko-cv-standard --dataset_name mozilla-foundation/common_voice_13_0 --ref_key sentence --spacing_er
```

Evaluting standard local finetuned model on Google Fleurs dataset for the Vietnamese language using WER and CER metric.

```bash
python evaluate_model.py --language vietnamese --config vi_vn --save_transcript --output_file eval-vi-fleurs-finetuned --model_name ../working-models/whisper-vi-cv/ --dataset_name google/fleurs --ref_key transcription --cer 
```

All three of WER, CER, and Spacing Accuracy metrics are included in the tables for our final report.

## Evaluation with Industry Standard Models

We evaluated our finetuned model on Google's Speech-to-Text and IBM Watson's Speech-to-Text

### Configuration
Before running the code, please make sure to update the following constants in the `evaluate_model.py` file to match your specific configuration:

1. `HF_API_KEY`: Replace `"api_key"` with your Hugging Face API key.
2. `WATSON_API_URL`: Replace `"api_url"` with your WATSON API URL.
3. `WATSON_API_KEY`: Replace `"api_key"` with your WATSON API key.

In addition, make sure to have a "key.json" file that represent Google's Cloud Credential found [here](https://console.cloud.google.com/apis/credentials).

### Command-line usage

To perform the same evaluation we created a script `evaluate_industry_models` that can be ran with any Hugging Face dataset with with the following CLI arguments. 


Required Parameters:

- `--model_type`: STT model to evaluate on. (google, watson) - (default: `google`)

Functional Parameters:

- `--dataset_name`: Name of the dataset from Hugging Face (default: `mozilla-foundation/common_voice_13_0`)
- `--config`: The configuration of the dataset (default: 'vi' for Vietnamese for Common Voice)
- `--language`: Language code for transcription (default: 'vi' for Vietnamese for Common Voice)
- `--split`: The dataset split to evaluate on (default: 'test').
- `--output_file`: Name of the file to save the evaluation results.
- `--ref_key`: Key in the dataset for reference data (default: 'sentence' - matches with Common Voice).

Flag Parameters:

- `--save_transcript`: Flag to save the transcript to a file (default: False).
- `--cer`: Flag to calculate the Character Error Rate (default: False).
- `--spacing_er`: Flag to calculate spacing accuracy rate (default: False)

### Examples

Evaluating Google's Speech-to-Text model on mozilla-foundation/common_voice_13_0" dataset with spacing error enabled

```bash
python evaluate_industry_model.py --model_type google --dataset_name google/fleurs --language ko-KR --config ko_kr --spacing_er
```

### Evaluations
Our evaluation results (text files) are included in the evaluations > evaluation-results folder.
There are 6 files for Korean, 6 files for Vietnamese, and 3 files for industry models.

The following illustrates how to interpret filenames.
- `eval-ko-cv-standard`: pretrained whisper for Korean, evaluated on CV test set.
- `eval-ko-cv-finetuned`: fine-tuned whisper for Korean on CV, evaluated on CV test set.
- `eval-ko-cv-finetuned-cv+fleurs`: fine-tuned whisper for Korean on CV + FLEURS, evaluated on CV test set.
- `eval-ko-fleurs-standard`: pretrained whisper for Korean, evaluated on FLEURS test set.
- `eval-ko-fleurs-finetuned`: fine-tuned whisper for Korean on CV, evaluated on FLEURS test set.
- `eval-ko-fleurs-finetuned-cv+fleurs`: fine-tuned whisper for Korean on CV + FLEURS, evaluated on FLEURS test set.

The same file naming convention applies for the six Vietnamese files.

There are 3 files for evaluation of industry models. Industry models were all evaluated on the CV test set, for reasons we detail in the report.
- `eval-google-ko-cv`: Google STT model for ko-KR (Korean)
- `eval-google-vi-cv`: Google STT model for vi-VN (Vietnamese)
- `eval-watson-ko-cv`: IBM Watson STT model for ko-KR (Korean)

## References
1. M. Ardila et al. “Common Voice: A Massively-Multilingual Speech Corpus” Mozilla. [Link](https://huggingface.co/datasets/common_voice)
2. A. Radford et al. “Robust Speech Recognition via Large-Scale Weak Supervision.” arXiv preprint arXiv:2212.04356, 2022. [Link](https://arxiv.org/abs/2212.04356)
