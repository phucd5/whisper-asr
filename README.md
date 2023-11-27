# Fine-tuning OpenAI's Whisper for Multilingual ASR with Transformers

## Overview
In this project, we investigate and improve on OpenAI's Whisper model, as detailed in the paper "Robust Speech Recognition via Large-Scale Weak Supervision," to focus on accurate recognition and transcription of Vietnamese and Korean. These languages present unique linguistic challenges in the speech recognition space: Vietnamese, with its specific tonal nature, dialectical variation, and prevalence of monosyllabic words, and Korean, with word segmentation. 

Through the Common Voice Project, an expansive dataset with extensive coverage on both languages and leveraging OpenAI's Whisper API and HuggingFace's transformers library to train and load the model, we aim to fine-tune Whisper's performance in order to reduce transcription errors and improve adaptability in regards to the model ability to handle the complexity of these two languages.

## Author
Phuc Duong and Sophia Kang 

CPSC 480: AI Foundational Models

## References
1. M. Ardila et al. “Common Voice: A Massively-Multilingual Speech Corpus” Mozilla. [Link](https://huggingface.co/datasets/common_voice)
2. A. Radford et al. “Robust Speech Recognition via Large-Scale Weak Supervision.” arXiv preprint arXiv:2212.04356, 2022. [Link](https://arxiv.org/abs/2212.04356)
