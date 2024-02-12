import json
import sys

from nltk.corpus import stopwords

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import GopherRepetitionFilter, ListFilter, MultilingualGopherQualityFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter


MAIN_OUTPUT_PATH = "./processed_data"
STATS_FILE = "./stats_20000_10.json"
DATASET = "HuggingFaceFW/fineweb_german_extract"

if len(sys.argv) == 2 and sys.argv[1] in ["local", "slurm"]:
    EXECUTOR = sys.argv[1]
else:
    print("Wrong executor provided (use either 'local' or 'slurm')")
    exit(1)

with open(STATS_FILE, "r") as f:
    language_stats = json.loads(f.read())

min_avg_word_lengths = {k: round(v["word_length_mean"] - v["word_length_std"]) for k, v in language_stats.items()}
max_avg_word_lengths = {k: round(v["word_length_mean"] + v["word_length_std"]) for k, v in language_stats.items()}
stopwords = {
    "de": stopwords.words("german"),
    "fr": stopwords.words("french"),
}

pipeline = [
    HuggingFaceDatasetReader(DATASET, dataset_options={"use_auth_token": True, "split": "train"}, limit=5000),
    GopherRepetitionFilter(),
    MultilingualGopherQualityFilter(
        max_avg_word_lengths=max_avg_word_lengths,
        min_avg_word_lengths=min_avg_word_lengths,
        stop_words=stopwords,
    ),
    ListFilter(),
    JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DATASET}"),
]

if EXECUTOR == "local":
    executor = LocalPipelineExecutor(pipeline=pipeline, logging_dir=f"{MAIN_OUTPUT_PATH}/logs/{DATASET}")
elif EXECUTOR == "slurm":
    executor = SlurmPipelineExecutor(
        pipeline=pipeline,
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/{DATASET}",
        tasks=2,
        time="00:05:00",
        partition="clariden",
    )

executor.run()
