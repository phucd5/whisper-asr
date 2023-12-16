
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

from datasets import load_dataset, DatasetDict
from datasets import Audio

from DataCollatorSpeechSeq2SeqWithPadding import DataCollatorSpeechSeq2SeqWithPadding
from MetricsEval import MetricsEval

from huggingface_hub import HfFolder

OUTPUT_DIR = "./whisper-small-hi"
SAVE_DIR = "./models/"

TRAIN_BATCH_SIZE = 16
GRAIDENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
MAX_STEPS = 4000
EVAL_BATCH_SIZE = 8
SAVE_STEPS = 1000
EVAL_STEPS = 1000
LOGGING_STEPS = 25


class WhisperASR:
    """Whisper Model for Automatic Speech Recognition (ASR) using Hugging Face's Transformers library."""

    def __init__(self, model_name="openai/whisper-small", language="Hindi", language_code="hi"):
        """
        Initialize the model and load the data. 
        The default config is the small model trained on the Common Voice dataset for Hindi.

        Args:
            model_name (str, optional): The model name. Ex: "openai/whisper-small".
            language (str, optional): The language of the model. Ex: "Hindi".
            language_code (str, optional): The language code of the model. Ex: "hi".

        """

        # initialize model and tokenizer
        self.model_name = model_name
        self.language = language
        self.language_code = language_code

        self.train_split = "train+validation"
        self.test_split = "test"

        HfFolder.save_token("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")

        # initalize feature extractor, tokenizer and processor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            self.model_name, language=language, task="transcribe")
        self.processor = WhisperProcessor.from_pretrained(
            self.model_name, language=language, task="transcribe")

        # load model
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name)
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        # load data
        self.data = DatasetDict()
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            self.processor)
        self._load_data()

        self.OUTPUT_DIR = f"./whisper-small-hi/{language_code}"
        self.SAVE_DIR = f"./models/{language_code}"

    def _load_data(self):
        """Load the data from the Common Voice dataset and prepare it for training."""

        print("Preparing data...")

        # load data from Common Voice dataset
        self.data["train"] = load_dataset(
            "mozilla-foundation/common_voice_13_0", self.language_code, split=self.train_split, token='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')
        self.data["test"] = load_dataset(
           "mozilla-foundation/common_voice_13_0", self.language_code, split=self.test_split, token='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')

        print(self.data)

        # remove unnecessary columns
        self.data = self.data.remove_columns(
            ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

        print(self.data)
        print(self.data["train"][0])

        # downsample audio data to 16kHz
        self.data = self.data.cast_column("audio", Audio(sampling_rate=16000))
        self.data = self.data.map(
            self._prepare_data, remove_columns=self.data.column_names["train"], num_proc=2)

    def _prepare_data(self, batch):
        """Converts audio files to the model's input feature format and encodes the target texts.

        Args:
            batch (dict): A batch of audio and text data.
        """

        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch

    def train(self):
        """Train the model. Set the training arguments here and using Seq2SeqTrainer. 
        After training, save the model to the specified directory.
        """

        # metric evaluation for training
        eval_fn = MetricsEval(self.tokenizer)

        # configure training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.OUTPUT_DIR,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=SAVE_STEPS,
            eval_steps=EVAL_STEPS,
            logging_steps=LOGGING_STEPS,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=True,
        )
        

        # initialize trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.data["train"],
            eval_dataset=self.data["test"],
            data_collator=self.data_collator,
            compute_metrics=eval_fn.compute,
            tokenizer=self.processor.feature_extractor,
        )

        self.processor.save_pretrained(training_args.output_dir)

        # start training
        print("Starting training...")
        trainer.train()

        # save model to model directory
        trainer.save_model(self.SAVE_DIR)
        print("Model saved in the current directory")

        kwargs = {
            "dataset_tags": "mozilla-foundation/common_voice_11_0",
            "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
            "dataset_args": "config: hi, split: test",
            "language": "hi",
            "model_name": "Whisper Small Hi - Sanchit Gandhi",  # a 'pretty' name for our model
            "finetuned_from": "openai/whisper-small",
            "tasks": "automatic-speech-recognition",
            "tags": "hf-asr-leaderboard",
        }
        trainer.push_to_hub(**kwargs)
        print("MOdel saved to hub")
