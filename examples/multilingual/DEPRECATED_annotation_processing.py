import sys

import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
import yaml

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import (
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
)
from datatrove.pipeline.readers import ShuffledHFDatasetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

if len(sys.argv) != 2 or sys.argv[1] not in [
    "statistics",
    "filters_q",
    "filters_meanstd",
    "raw",
]:
    print("First argument should be: 'statistics', 'filters_q', 'filters_meanstd' or 'raw'.")
    print("Use 'statistics' to generate statistics of the Wikipedia documents.")
    print("Use 'filters' to only generate the filter values for multilingual Gopher quality filter.")
    exit(1)

RUN_MODE = sys.argv[1]
FILTERS_FOLDER = f"./{RUN_MODE}_filters"
RUN_NAME = f"{RUN_MODE}_filters"
LANGUAGES = [
    "en",
    "de",
    "hr",
    "pt",
    "cs",
    "zh",
    "fr",
    "ru",
    "tr",
    "ar",
    "th",
    "hi",
    "sw",
    "te",
    "ja",
]

DOC_LIMIT = 1_000
NUM_TASKS = 100
NUM_WORKERS = 100
EXECUTOR = "local"
DUMP_TO_PROCESS = "CC-MAIN-2023-23"

# Load language-specific statistics and stopwords
def load_filters(folder_path, languages):
    filters = {}
    for language in languages:
        with open(f"{folder_path}/{language}.yml", "r") as f:
            language_filter = yaml.safe_load(f)
            filters[language] = language_filter
    return filters


filters = load_filters(FILTERS_FOLDER, LANGUAGES)

min_avg_word_lengths = {k: v["min_avg_word_length"] for k, v in filters.items()}
max_avg_word_lengths = {k: v["max_avg_word_length"] for k, v in filters.items()}
stopwords = {k: v["stopwords"] for k, v in filters.items()}
max_non_alpha_words_ratio = {k: v["max_non_alpha_words_ratio"] for k, v in filters.items()}
min_stop_words = {k: 2 for k, _ in filters.items()}

char_duplicates_ratio = {k: v["char_duplicates_ratio"] for k, v in filters.items()}
line_punct_thr = {k: v["line_punct_thr"] for k, v in filters.items()}
new_line_ratio = {k: v["new_line_ratio"] for k, v in filters.items()}
short_line_thr = {k: v["short_line_thr"] for k, v in filters.items()}

if __name__ == "__main__":
    for language in LANGUAGES:
        FILTERING_OUTPUT_PATH = f"data/datatrove/multilingual_{RUN_MODE}/{DUMP_TO_PROCESS}/base_processing/{language}"
        LOGS = f"logs/datatrove/multilingual_{RUN_MODE}/{DUMP_TO_PROCESS}/base_processing/{language}"
        data_path = f"s3://fineweb-data-processing-us-east-1/base_processing/non_english/{language}/{DUMP_TO_PROCESS}"
        pipeline =[
            ShuffledHFDatasetReader(
                "ZR0zNqSGMI/curated_dataset",
                dataset_options={
                    "name": f"{language}",
                },
                limit=DOC_LIMIT,
                default_metadata={"language": language},
            ),
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/3_gopher_rep/{DUMP_TO_PROCESS}"),
                language=language
            ),
            GopherQualityFilter(
                max_avg_word_length=max_avg_word_lengths[language],
                min_avg_word_length=min_avg_word_lengths[language],
                stop_words=stopwords[language],
                min_stop_words=min_stop_words[language],
                max_non_alpha_words_ratio=max_non_alpha_words_ratio[language],
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/4_gopher_qual/{DUMP_TO_PROCESS}"),
                language=language
            ),
            FineWebQualityFilter(
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/6_fineweb_qual/{DUMP_TO_PROCESS}"),
                line_punct_thr=line_punct_thr[language],
                short_line_thr=short_line_thr[language],
                char_duplicates_ratio=char_duplicates_ratio[language],
                new_line_ratio=new_line_ratio[language],
                language=language,
            ),
            JsonlWriter(f"{FILTERING_OUTPUT_PATH}/output/{DUMP_TO_PROCESS}"),
        ]
        executor = {
            "local": LocalPipelineExecutor(
                pipeline=pipeline,
                logging_dir=f"{LOGS}",
                tasks=NUM_TASKS,
                workers=NUM_WORKERS,
            ),
            "slurm": SlurmPipelineExecutor(
                pipeline=pipeline,
                logging_dir=f"{LOGS}",
                tasks=NUM_TASKS,
                time="06:00:00",
                partition="normal",
                workers=NUM_WORKERS,
            ),
        }[EXECUTOR]
        executor.run()