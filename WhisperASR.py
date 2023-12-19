from datasets import Audio, DatasetDict, load_dataset
from huggingface_hub import HfFolder
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    WhisperFeatureExtractor, WhisperForConditionalGeneration,
    WhisperProcessor, WhisperTokenizer
)

from DataCollatorSpeechSeq2SeqWithPadding import DataCollatorSpeechSeq2SeqWithPadding
from MetricsEval import MetricsEval

# change constants as applicable
OUTPUT_DIR = "../models"
HF_API_KEY = "api_key"
BASE_MODEL = "openai/whisper-small"

# training constants
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

    def __init__(self, model_name="openai/whisper-small", dataset_name="mozilla-foundation/common_voice_13_0", existing_model=False, language="Korean", language_code="ko", save_to_hf=False, output_dir="./models/whisper", ref_key="sentence"):
        """
        Initialize the model and load the data. 
        The default config is the small model trained on the Common Voice dataset for Korean.

        Args:
            model_name (str): The model name from Hugging Face or custom path.
            If 'existing_model' is True, this should be the path to the pre-trained model. Ex: "openai/whisper-small".
            
            existing_model (bool): Flag to indicate whether to load an existing model from the specified 
            'model_name' path. If False, a new model is initialized.
            
            language (str): The language of the model. Ex: "Korean".
            language_code (str): The language code of the model. Must match the language. Ex: "ko"
            output_dir (str): The output directory of the model to save to
            save_to_hf (bool): Whether to push to Hugging Face Repo
            ref_key (str): The key to the reference data in the dataset
        """
        # setting up to save to hugging face repo
        self.save_to_hf = save_to_hf
        if save_to_hf:
            HfFolder.save_token(HF_API_KEY) # token to save to HF
        
        self.dataset_name = dataset_name
        self.ref_key = ref_key

        # initialize model and tokenizer
        self.model_name = model_name
        self.language = language
        self.language_code = language_code
        self.existing_model = existing_model

        self.train_split = "train+validation"
        self.test_split = "test"

        # initalize feature extractor, tokenizer and processor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            self.model_name, language=language, task="transcribe")
        self.processor = WhisperProcessor.from_pretrained(
            self.model_name, language=language, task="transcribe")

        # load correct model
        if existing_model:
            print(f"[INFO] Loading {self.model_name} model from existing model...")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        else:
            print(f"[INFO] Loading {self.model_name} from hugging face library...")
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        # load data
        self.data = DatasetDict()
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            self.processor)
        self._load_data()
        self.OUTPUT_DIR = output_dir

    def _load_data(self):
        """Load the data from the Common Voice dataset and prepare it for training."""

        print(f"[INFO] Preparing {self.dataset_name} data for training phase...")

        # load data from Common Voice dataset
        self.data["train"] = load_dataset(
            self.dataset_name, self.language_code, split=self.train_split, token=HF_API_KEY)
        self.data["test"] = load_dataset(
           self.dataset_name, self.language_code, split=self.test_split, token=HF_API_KEY)
        
        print("[INFO] Structure of the loaded data:")
        print(self.data)

        print("[INFO] Sample entry from the training dataset: ")
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
        batch["labels"] = self.tokenizer(batch[self.ref_key]).input_ids
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
            push_to_hub=self.save_to_hf,
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

        # # start training
        print("[INFO] Starting training...: ")
        trainer.train()
        
        # training finished and save model to model directory
        print(f"[INFO] Training finished and model saved to {self.OUTPUT_DIR}")

        if self.save_to_hf:
            kwargs = {
                "language": f"{self.language_code}",
                "model_name": f"Whisper - {self.language} Model",
                "finetuned_from": f"{BASE_MODEL}",
                "tasks": "automatic-speech-recognition",
            }

            trainer.push_to_hub(**kwargs)
            print(f"[INFO] Model saved to Hugging Face Hub")
