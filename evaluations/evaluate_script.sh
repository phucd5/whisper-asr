#!/bin/bash

echo "Common Voice dataset evaluation for Korean - standard and fine-tuned"
python evaluate_model.py --language ko --config ko --save_transcript --output_file eval-ko-cv-standard --dataset_name mozilla-foundation/common_voice_13_0 --cer --ref_key sentence --spacing_er
python evaluate_model.py --language ko --config ko --save_transcript --output_file eval-ko-cv-finetuned --model_name ../working-models/whisper-ko-cv/ --dataset_name mozilla-foundation/common_voice_13_0 --cer --ref_key sentence --spacing_er

echo "Common Voice dataset evaluation for Vietnamese - standard and fine-tuned"
python evaluate_model.py --language vi --config vi --save_transcript --output_file eval-vi-cv-standard  --dataset_name mozilla-foundation/common_voice_13_0 --cer --ref_key sentence --spacing_er
python evaluate_model.py --language vi --config vi --save_transcript --output_file eval-vi-cv-finetuned --model_name ../working-models/whisper-vi-cv/ --dataset_name mozilla-foundation/common_voice_13_0 --cer --ref_key sentence --spacing_er

echo "FLEURS dataset evaluation for Korean - standard and fine-tuned"
python evaluate_model.py --language korean --config ko_kr --save_transcript --output_file eval-ko-fleurs-standard --dataset_name google/fleurs --cer --ref_key raw_transcription --spacing_er
python evaluate_model.py --language korean --config ko_kr --save_transcript --output_file eval-ko-fleurs-finetuned --model_name ../working-models/whisper-ko-cv/ --dataset_name google/fleurs --cer --ref_key raw_transcription --spacing_er

echo "FLEURS dataset evaluation for Vietnamese - standard and fine-tuned"
python evaluate_model.py --language vietnamese --config vi_vn --save_transcript --output_file eval-vi-fleurs-standard --dataset_name google/fleurs --cer --ref_key raw_transcription --spacing_er
python evaluate_model.py --language vietnamese --config vi_vn --save_transcript --output_file eval-vi-fleurs-finetuned --model_name ../working-models/whisper-vi-cv/ --dataset_name google/fleurs --cer --ref_key raw_transcription --spacing_er

echo "Evaluation of CV+FLEURS finetuned model for Korean and Vietnamese using Common Voice and FLEURS datasets"
python evaluate_model.py --language ko --config ko --save_transcript --output_file eval-ko-cv-finetuned-cv+fleurs  --model_name ../working-models/whisper-ko-cv-fleurs/ --dataset_name mozilla-foundation/common_voice_13_0 --cer --ref_key sentence --spacing_er
python evaluate_model.py --language vi --config vi --save_transcript --output_file eval-vi-cv-finetuned-cv+fleurs --model_name ../working-models/whisper-vi-cv-fleurs/ --dataset_name mozilla-foundation/common_voice_13_0 --cer --ref_key sentence --spacing_er
python evaluate_model.py --language korean --config ko_kr --save_transcript --output_file eval-ko-fleurs-finetuned-cv+fleurs --model_name ../working-models/whisper-ko-cv-fleurs/ --dataset_name google/fleurs --cer --ref_key transcription --spacing_er
python evaluate_model.py --language vietnamese --config vi_vn --save_transcript --output_file eval-vi-fleurs-finetuned-cv+fleurs  --model_name ../working-models/whisper-vi-cv-fleurs/ --dataset_name google/fleurs --cer --ref_key transcription --spacing_er
