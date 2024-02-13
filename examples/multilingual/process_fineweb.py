import json
import os

from nltk.corpus import stopwords

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import GopherRepetitionFilter, ListFilter, MultilingualGopherQualityFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter


DATASET = "HuggingFaceFW/fineweb_german_extract"
MAIN_OUTPUT_PATH = "./processed_data"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
EXECUTOR = os.environ.get("EXECUTOR", "slurm")
STATS_FILE = "./stats_20000_10.json"

with open(STATS_FILE, "r") as f:
    language_stats = json.loads(f.read())

min_avg_word_lengths = {k: round(v["word_length_mean"] - v["word_length_std"]) for k, v in language_stats.items()}
max_avg_word_lengths = {k: round(v["word_length_mean"] + v["word_length_std"]) for k, v in language_stats.items()}
stopwords = {
    "de": stopwords.words("german"),
    "fr": stopwords.words("french"),
}

pipeline = [
    HuggingFaceDatasetReader(DATASET, dataset_options={"token": HF_TOKEN, "split": "train[:1000]"}),
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
