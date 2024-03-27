import os
import sys

import yaml

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import (
    GopherRepetitionFilter,
    ListFilter,
    MultilingualGopherQualityFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter


# DUMP should be given as an argument. Example: CC-MAIN-2023-23
if len(sys.argv) != 2:
    print("Argument required: dump name")
    sys.exit(-1)

DUMP = sys.argv[1]
DOC_LIMIT = 5000
NUM_TASKS = 20
FILTERS_FOLDER = "./filters"
RUN_NAME = "top50_stopwords_p_thresh_0_008_spacy_v2"
LANGUAGES = [
    "en",
    "de",
    "ru",
    "fr",
    "ja",
    "es",
    "zh",
    "it",
    "nl",
    "pl",
    "pt",
    "cs",
    "vi",
    "id",
    "tr",
    "sv",
    "fa",
    "ko",
    "hu",
    "ar",
    "el",
    "ro",
    "da",
    "fi",
    "th",
    "uk",
    "sk",
    "no",
    "bg",
    "ca",
    "hr",
    "la",
    "sr",
    "hi",
    "sl",
    "lt",
    "et",
    "he",
    "bn",
    "lv",
    "sh",
    "sq",
    "az",
    "ta",
    "is",
    "mk",
    "ka",
    "gl",
    "hy",
    "eu",
    "ms",
    "ur",
    "ne",
    "mr",
    "ml",
    "kk",
    "te",
    "mn",
    "be",
    "gu",
    "kn",
    "tl",
    "my",
    "eo",
    "uz",
    "km",
    "tg",
    "cy",
    "nn",
    "bs",
    "si",
    "sw",
    "pa",
    "tt",
    "ckb",
    "af",
    "or",
    "ky",
    "ga",
    "am",
    "oc",
    "ku",
    "lo",
    "lb",
    "ba",
    "ceb",
    "fy",
    "ps",
    "mt",
    "br",
    "as",
    "mg",
    "war",
    "dv",
    "yi",
    "so",
    "sa",
    "sd",
    "azb",
    "tk",
]

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
stop_words = {k: v["stopwords"] for k, v in filters.items()}
alpha_ratio = {k: v["max_non_alpha_words_ratio"] for k, v in filters.items()}
min_stop_words = {k: 2 for k, _ in filters.items()}

for lang in LANGUAGES:
    data_path = f"s3://fineweb-data-processing-us-east-1/base_processing/non_english/{lang}/{DUMP}"

    MAIN_OUTPUT_PATH = f"data/datatrove/multi_lingual_{DOC_LIMIT}_{RUN_NAME}/base_processing/{lang}"
    SLURM_LOGS = f"logs/datatrove/multi_lingual_{DOC_LIMIT}_{RUN_NAME}/base_processing/{lang}"

    executor = SlurmPipelineExecutor(
        job_name=f"cc_{DUMP}_{lang}",
        pipeline=[
            JsonlReader(
                data_path,
                default_metadata={"dump": DUMP},
                limit=DOC_LIMIT,
                text_key="content",
            ),
            GopherRepetitionFilter(exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/repetitive/{DUMP}")),
            MultilingualGopherQualityFilter(
                max_avg_word_lengths=max_avg_word_lengths,
                min_avg_word_lengths=min_avg_word_lengths,
                stop_words=stop_words,
                min_stop_words=min_stop_words,
                max_non_alpha_words_ratio=alpha_ratio,
                exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/quality/{DUMP}"),
            ),
            ListFilter(exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/list/{DUMP}")),
            JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DUMP}"),
        ],
        tasks=NUM_TASKS,
        time="04:00:00",
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing/{DUMP}",
        slurm_logs_folder=f"{SLURM_LOGS}/{DUMP}/slurm_logs",
        randomize_start=True,
        mem_per_cpu_gb=2,
        partition="normal",
    )
    executor.run()
