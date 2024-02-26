import json
import os

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import MultilingualGopherQualityFilter
from datatrove.pipeline.readers import ShuffledHFDatasetReader
from datatrove.pipeline.writers import JsonlWriter


# Top 10 languages in CommonCrawl according to (https://arxiv.org/pdf/2306.01116.pdf) + English
LANGUAGES = ["en", "ru", "de", "es", "ja", "fr", "zh", "it", "pt", "nl", "pl"]
MAIN_OUTPUT_PATH = "./sanity_check_wiki"
WIKI_VERSION = "20231101"  # See https://huggingface.co/datasets/wikimedia/wikipedia
DOC_LIMIT = 2000  # Limit number of documents per dataset
TASKS = 10
EXECUTOR = os.environ.get("EXECUTOR", "slurm")  # local/slurm
STATS_FILE = "./wiki_lang_stats.json"
STOPWORDS = "stopwords_top_n", "8"  # key for getting stopwords from statistics

# Load language-specific statistics and stopwords
with open(STATS_FILE, "r") as f:
    language_stats = json.load(f)

# Set-up language specific statistics
min_avg_word_lengths = {k: v["min_avg_word_length"] for k, v in language_stats.items()}
max_avg_word_lengths = {k: v["max_avg_word_length"] for k, v in language_stats.items()}
stopwords = {k: v[STOPWORDS[0]][STOPWORDS[1]] for k, v in language_stats.items()}
min_stop_words = {k: len(v) // 4 for k, v in language_stats.items()}  # TODO: is // 4 good for min. stopwords?

if __name__ == "__main__":
    readers = [
        ShuffledHFDatasetReader(
            "wikimedia/wikipedia",
            limit=DOC_LIMIT,
            dataset_options={
                "name": f"{WIKI_VERSION}.{language}",
                "split": "train",
            },
            default_metadata={"language": language},
        )
        for language in LANGUAGES
    ]

    pipeline = [
        *readers,
        MultilingualGopherQualityFilter(  # turn off all other filters except stopwords
            stop_words=stopwords,
            min_stop_words=min_stop_words,
            max_avg_word_lengths=None,
            min_avg_word_lengths=None,
            min_doc_words=None,
            max_doc_words=None,
            max_symbol_word_ratio=None,
            max_bullet_lines_ratio=None,
            max_ellipsis_lines_ratio=None,
            max_non_alpha_words_ratio=None,
            exclusion_writer=JsonlWriter(
                f"{MAIN_OUTPUT_PATH}/removed",
                output_filename="${language}/${rank}.jsonl.gz",
            ),
        ),
    ]

    executor = {
        "slurm": SlurmPipelineExecutor(
            pipeline=pipeline,
            logging_dir=f"{MAIN_OUTPUT_PATH}/logs/",
            tasks=TASKS,
            time="06:00:00",
            partition="normal",
        ),
        "local": LocalPipelineExecutor(
            pipeline=pipeline,
            logging_dir=f"{MAIN_OUTPUT_PATH}/logs/",
            tasks=TASKS,
        ),
    }[EXECUTOR]

    executor.run()
