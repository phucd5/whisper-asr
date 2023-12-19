import os
import argparse

from transformers import pipeline
from datasets import load_dataset, Audio
import evaluate
from tqdm import tqdm
from google.cloud import speech
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from utils import remove_punctuation, compute_spacing

HF_API_KEY = "api_key"
WATSON_API_URL = "api_url"
WATSON_API_KEY = "api_key"


def transcribe_google(audio, language):
    """
    Transcribe audio using Google's Speech-to-Text API.

    Args:
        audio (bytes): the audio from mp3 file in bytes
        language (str): language code for transcription

    Returns:
        str: transcribed text from the model
    """

    client = speech.SpeechClient.from_service_account_file('key.json')

    audio_for_google = speech.RecognitionAudio(content=audio)

    config = speech.RecognitionConfig(
        sample_rate_hertz=16000,
        language_code=language,
        enable_automatic_punctuation=True,
    )

    response = client.recognize(config=config, audio=audio_for_google)
    return response.results[0].alternatives[0].transcript if response.results else ""


def transcribe_watson(audio, language):
    """
    Transcribe audio using using IBM Watson's Speech-to-Text API

    Args:
        audio (bytes): the audio from mp3 file in bytes
        language(str): language code for transcription

    Returns:
        str: transcribed text from the model
    """

    auth = IAMAuthenticator(WATSON_API_KEY)
    SpeechToText = SpeechToTextV1(authenticator=auth)
    SpeechToText.set_service_url(WATSON_API_URL)

    response = SpeechToText.recognize(
        audio=audio,
        model=language)

    return response.result['results'][0]['alternatives'][0]['transcript']


def evaluate_audio(language, dataset, transcribe_fn):
    """
    Evaluate an audio dataset with the specified transcrption function model
    and then write evlauation to file

    Args:
        language (str): language code for transcription
        dataset (dict): dataset to evaluate
        transcribe_fn (function): transcrption function based on model selected
    """

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    predictions, references = [], []

    # loop through the dataset and evaluate
    for sample in tqdm(dataset, desc='Evaluating Progress'):
        with open(sample["audio"]["path"], 'rb') as f:
            audio = f.read()

        text = transcribe_fn(audio, language)
        predictions.append(text)
        references.append(sample[args.ref_key])

    wer = wer_metric.compute(references=references, predictions=predictions)

    # determine whether to calculate additional metrics
    if args.cer:
        cer = cer_metric.compute(
            references=references, predictions=predictions)

    if args.spacing_er:
        spacing_er = compute_spacing(
            references=references, predictions=predictions)

    os.makedirs("evaluation-results", exist_ok=True)
    # output results to a file
    with open(os.path.join("evaluation-results", args.output_file), 'w') as file:
        file.write(
            f"[INFO] Evaluated {args.model_type} Speech-to-Text with {args.dataset_name} with language {args.language}/{args.config}\n\n")
        file.write(f"WER : {round(100 * wer, 4)}\n\n")

        # determine whether to print additional metrics
        if args.cer:
            file.write(f"CER : {round(100 * cer, 4)}\n\n")

        if args.spacing_er:
            file.write(f"Spacing: {round(100 * spacing_er, 4)}\n\n")

        if args.save_transcript:
            for ref, pred in zip(references, predictions):
                file.write(
                    f"Reference: {ref}\nPrediction: {pred}\n{'-' * 40}\n")


if __name__ == "__main__":

    # parse input
    parser = argparse.ArgumentParser(
        description="Whisper ASR Evaluation - Industry Speech to Text")

    parser.add_argument("--model_type", type=str, required=True, default="google",
                        help="STT model to evaluate on. (google, watson) - (default: google)")

    parser.add_argument("--dataset_name", type=str, default="mozilla-foundation/common_voice_13_0",
                        help="Name of the dataset from Hugging Face (default: 'mozilla-foundation/common_voice_13_0')")
    parser.add_argument("--config", type=str, default="ko",
                        help="Configuration of the dataset (default: 'vi' for Vietnamese for Common Voice).")
    parser.add_argument("--language", type=str, default="ko-KR",
                        help="Language code for model API (default: 'vi-VN' for Vietnamese for Google STT)")
    parser.add_argument("--split", type=str, default="test",
                        help="The dataset split to evaluate on (default: 'test').")
    parser.add_argument("--output_file", type=str, default="whisper_eval_stt",
                        help="Name of the file to save the evaluation results.")
    parser.add_argument("--ref_key", type=str, default="sentence",
                        help="Key in the dataset for reference data (default: 'sentence' - matches with Common Voice)")

    parser.add_argument("--save_transcript", action='store_true',
                        help="Flag to save the transcript to a file (default: False).")
    parser.add_argument("--cer", action='store_true',
                        help="Flag to calculate the Character Error Rate (default: False)")
    parser.add_argument("--spacing_er", action='store_true',
                        help="Flag to calculate spacing error rate (default: False)")

    args = parser.parse_args()

    # load data
    dataset = load_dataset(args.dataset_name, args.config,
                           split="test", token=HF_API_KEY)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # make sure its a supported model
    if args.model_type not in ["google", "watson"]:
        raise ValueError(
            "Model not supported. Please choose (azure, google or watson)")

    model_map = {
        "google": transcribe_google,
        "watson": transcribe_watson
    }

    evaluate_audio(language=args.language, dataset=dataset,
                   transcribe_fn=model_map[args.model_type])
