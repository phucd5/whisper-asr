
from datasets import load_dataset, Audio
import argparse
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

HF_API_KEY = "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"
WATSON_API_KEY = "wiiYD82U_jjMINMwBbkmkGHM2zdoDvvVEyd5lsHA-M66"

parser = argparse.ArgumentParser(description="Whisper ASR Evaluation - Google")

parser.add_argument("--dataset_name", type=str, default="mozilla-foundation/common_voice_13_0", help="Name of the dataset from Hugging Face (default: 'mozilla-foundation/common_voice_13_0')")
parser.add_argument("--config", type=str, default="ko", help="Configuration of the dataset (default: 'vi' for Vietnamese for Common Voice).")
parser.add_argument("--language", type=str, default="ko", help="Language code for transcription (default: 'vi' for Vietnamese for Common Voice)")
parser.add_argument("--split", type=str, default="test", help="The dataset split to evaluate on (default: 'test').")
parser.add_argument("--output_file", type=str, default="whisper_eval", help="Name of the file to save the evaluation results.")
parser.add_argument("--ref_key", type=str, default="sentence", help="Key in the dataset for reference data (default: 'sentence' - matches with Common Voice)")

args = parser.parse_args()

dataset = load_dataset(args.dataset_name, args.config, split="test", token=HF_API_KEY)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Select a single entry from the dataset
sample = dataset[0]
audio_data = sample["audio"]["path"] # Get the audio array


with open(audio_data, 'rb') as f:
    audio_mp3_data = f.read()


api_url = "https://api.us-east.speech-to-text.watson.cloud.ibm.com/instances/df31e263-65c3-4dd5-8be4-08c565c8cdd5"

auth = IAMAuthenticator(WATSON_API_KEY)
SpeechToText = SpeechToTextV1(authenticator = auth)
SpeechToText.set_service_url(api_url)

response = SpeechToText.recognize(
    audio=audio_mp3_data,
    content_type="audio/mp3",
    model='ko-KR_BroadbandModel'
)

print(response.result['results'][0]['alternatives'][0]['transcript'])

# # Initialize the Google Cloud Speech-to-Text client
# client = speech.SpeechClient.from_service_account_file('key.json')


# # Prepare the audio for Google Speech-to-Text
# audio_for_google = speech.RecognitionAudio(content=audio_mp3_data)


# # Configure the request
# config = speech.RecognitionConfig(
#     sample_rate_hertz=16000,
#     language_code='ko-KR',
#     enable_automatic_punctuation=True,
# )

# # Perform speech recognition
# response = client.recognize(config=config, audio=audio_for_google)

# # Print the transcription
# for result in response.results:
#     print("Transcript: {}".format(result.alternatives[0].transcript))