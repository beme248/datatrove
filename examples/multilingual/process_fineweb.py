import json
import os

from nltk.corpus import stopwords

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import GopherRepetitionFilter, ListFilter, MultilingualGopherQualityFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter


HF_TOKEN = os.environ.get("HF_TOKEN", "")  # HuggingFace token to access datasets (gated)
EXECUTOR = os.environ.get("EXECUTOR", "slurm")  # local/slurm
DATASET = "HuggingFaceFW/fineweb_german_extract"
MAIN_OUTPUT_PATH = "./fineweb_processed"
STATS_FILE = "./lang_stats.json"
DOC_LIMIT = 1000  # Limit number of documents per dataset

# Load language-specific statistics and stopwords
with open(STATS_FILE, "r") as f:
    language_stats = json.load(f)

min_avg_word_lengths = {k: v["min_avg_word_length"] for k, v in language_stats.items()}
max_avg_word_lengths = {k: v["max_avg_word_length"] for k, v in language_stats.items()}
stopwords = {
    "de": stopwords.words("german"),
    "fr": stopwords.words("french"),
}

# Process dataset
pipeline = [
    HuggingFaceDatasetReader(DATASET, dataset_options={"token": HF_TOKEN, "split": "train"}, limit=DOC_LIMIT),
    GopherRepetitionFilter(),
    MultilingualGopherQualityFilter(
        max_avg_word_lengths=max_avg_word_lengths,
        min_avg_word_lengths=min_avg_word_lengths,
        stop_words=stopwords,
    ),
    ListFilter(),
    JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DATASET}"),
]

executor = {
    "slurm": SlurmPipelineExecutor(
        pipeline=pipeline,
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/{DATASET}",
        tasks=2,
        time="00:05:00",
        partition="clariden",
    ),
    "local": LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/{DATASET}",
    ),
}[EXECUTOR]

executor.run()
