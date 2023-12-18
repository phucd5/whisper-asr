import argparse
import os

from datasets import Audio, load_dataset
import evaluate
from tqdm import tqdm
from transformers import pipeline

from utils import compute_spacing, remove_punctuation

HF_API_KEY = "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"

def transcribe(whisper_asr, audio):
    """
    Transcribe the given audio using the Whisper ASR model.

    Args:
        whisper_asr (transformers.Pipeline): The Whisper ASR Hugging Face pipeline
        audio (dict): The audio data to transcribe

    Returns:
        text (str): The  model output text from the audio
    """

    text = whisper_asr(audio)["text"]
    return text

def evaluation(dataset, whisper_asr):
    """
    Evaluate the model on a dataset

    Args:
        dataset (datasets.Dataset): The dataset containing audio samples and transcriptions.
        whisper_asr (transformers.Pipeline): The Whisper ASR Hugging Face pipeline.

    Returns:
        predictions (list of str): List of transcriptions generated by the model.
        eferences (list of str): List of reference transcriptions from the dataset.
    """

    predictions, references = [], []

    # loop through each sample and get the output and ref
    for item in tqdm(dataset, desc='Evaluating Progress'):
        audio_data = item["audio"]
        text = transcribe(whisper_asr, audio_data)
        predictions.append(text)
        references.append(item[args.ref_key])
    
    return predictions, references

def main(args):
    print(f"[INFO] Evaluting {args.model_name} with {args.dataset_name} with language {args.language}/{args.config}")

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")


    # load model
    whisper_asr = pipeline(
        "automatic-speech-recognition", model=args.model_name, device=args.device
    )

    whisper_asr.model.config.forced_decoder_ids = (
        whisper_asr.tokenizer.get_decoder_prompt_ids(
            language=args.language, task="transcribe"
        )
    )

    # load the dataset and downsample audio data to 16kHz
    dataset = load_dataset(args.dataset_name, args.config, split="test", token=HF_API_KEY)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # evaluate
    predictions, references = evaluation(dataset, whisper_asr)
    wer = wer_metric.compute(references=references, predictions=predictions)

    # determine whether to calculate additional metrics
    if args.cer:
        cer = cer_metric.compute(references=references, predictions=predictions)

    if args.spacing_er:
        spacing_er = compute_spacing(references=references, predictions=predictions)
    
    os.makedirs("evaluation-results", exist_ok=True)
    # output results to a file
    with open(os.path.join("evaluation-results", args.output_file), 'w') as file:
        file.write(f"[INFO] Evaluated {args.model_name} with {args.dataset_name} with language {args.language}/{args.config}\n\n")
        file.write(f"WER : {round(100 * wer, 4)}\n\n")

        # determine whether to print additional metrics
        if args.cer:
            file.write(f"CER : {round(100 * cer, 4)}\n\n")

        if args.spacing_er:
            file.write(f"Spacing: {round(100 * spacing_er, 4)}\n\n")

        if args.save_transcript:
            for ref, pred in zip(references, predictions):
                file.write(f"Reference: {ref}\nPrediction: {pred}\n{'-' * 40}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper ASR Evaluation")

    parser.add_argument("--model_name", type=str, default="openai/whisper-small", help="Hugging Face Whisper model name, or the path to a local model directory. (default: 'openai/whisper-small)")
    parser.add_argument("--dataset_name", type=str, default="mozilla-foundation/common_voice_13_0", help="Name of the dataset from Hugging Face (default: 'mozilla-foundation/common_voice_13_0')")
    parser.add_argument("--config", type=str, default="vi", help="Configuration of the dataset (default: 'vi' for Vietnamese for Common Voice).")
    parser.add_argument("--language", type=str, default="vi", help="Language code for transcription (default: 'vi' for Vietnamese for Common Voice)")
    parser.add_argument("--device", type=int, default=0, help="GPU ID for using a GPU (e.g., 0), or -1 to use CPU. (default: 0)")
    parser.add_argument("--split", type=str, default="test", help="The dataset split to evaluate on (default: 'test').")
    parser.add_argument("--output_file", type=str, default="whisper_eval", help="Name of the file to save the evaluation results.")
    parser.add_argument("--ref_key", type=str, default="sentence", help="Key in the dataset for reference data (default: 'sentence' - matches with Common Voice)")

    parser.add_argument("--save_transcript", action='store_true', help="Flag to save the transcript to a file (default: False).")
    parser.add_argument("--cer", action='store_true', help="Flag to calculate the Character Error Rate (default: False)")
    parser.add_argument("--spacing_er", action='store_true', help="Flag to calculate spacing error rate (default: False)")
    args = parser.parse_args()
    
    main(args)