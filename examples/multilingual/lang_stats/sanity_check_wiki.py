import json
import os
from math import e, floor, log

from nltk.corpus import stopwords as nltk_stopwords

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import MultilingualGopherQualityFilter
from datatrove.pipeline.readers import ShuffledHFDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.stats import DocumentStats, DocumentStatsReducer

LANGUAGES = ["en", "ru", "de", "es", "ja", "fr", "zh", "it", "pt", "nl", "pl"] # Top 10 languages in CommonCrawl according to (https://arxiv.org/pdf/2306.01116.pdf) + English
WIKI_OUTPUT_PATH = "./sanity_check_wiki"
WIKI_VERSION = "20231101"  # See https://huggingface.co/datasets/wikimedia/wikipedia
PAPERS_OUTPUT_PATH = "./sanity_check_papers"
DOC_LIMIT = 40000  # Limit number of documents per dataset
TASKS = 10
EXECUTOR = os.environ.get("EXECUTOR", "slurm")  # local/slurm
STATS_FILE = "./wiki_lang_stats.json"

# Load language-specific statistics and stopwords
with open(STATS_FILE, "r") as f:
    language_stats = json.load(f)

# Stopwords: intersection of wiki top 15 and NLTK stopwords
nltk_stopwords = {
    "en": nltk_stopwords.words("english"),
    "fr": nltk_stopwords.words("french"),
    "de": nltk_stopwords.words("german"),
    "es": nltk_stopwords.words("spanish"),
}
stopwords = {
    k: list(set(v).intersection(set(language_stats[k]["stopwords_top_n"]["15"])))
    for k, v in nltk_stopwords.items()
}
# Min. stopwords: floor(ln(num_stopwords))
min_stop_words = {k: floor(log(len(v), e)) for k, v in stopwords.items()}

# Arguments that will be passed to the MultilingualGopherQualityFilter
gopher_args = {
    "stop_words": stopwords,
    "min_stop_words": min_stop_words,
    "max_avg_word_lengths": None,
    "min_avg_word_lengths": None,
    "min_doc_words": None,
    "max_doc_words": None,
    "max_symbol_word_ratio": None,
    "max_bullet_lines_ratio": None,
    "max_ellipsis_lines_ratio": None,
    "max_non_alpha_words_ratio": None,
}
print(gopher_args)

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
            shuffle_seed=43,
        )
        for language in LANGUAGES
    ]

    pipeline = [
        *readers,
        MultilingualGopherQualityFilter(
            exclusion_writer=JsonlWriter(
                f"{WIKI_OUTPUT_PATH}/removed",
                output_filename="${language}/${rank}.jsonl.gz",
            ),
            **gopher_args,
        ),
        DocumentStats(output_folder=f"{WIKI_OUTPUT_PATH}/doc_stats/"),
    ]

    executor = {
        "slurm": SlurmPipelineExecutor(
            pipeline=pipeline,
            logging_dir=f"{WIKI_OUTPUT_PATH}/logs/",
            tasks=TASKS,
            time="06:00:00",
            partition="normal",
        ),
        "local": LocalPipelineExecutor(
            pipeline=pipeline,
            logging_dir=f"{WIKI_OUTPUT_PATH}/logs/",
            tasks=TASKS,
        ),
    }[EXECUTOR]

    executor.run()

    pipeline_reduce = [
        DocumentStatsReducer(
            f"{WIKI_OUTPUT_PATH}/doc_stats/", f"{WIKI_OUTPUT_PATH}", "doc_stats.json"
        )
    ]

    executor_reduce = {
        "slurm": SlurmPipelineExecutor(
            pipeline=pipeline_reduce,
            logging_dir=f"{WIKI_OUTPUT_PATH}/logs_reduce/",
            tasks=1,
            time="12:00:00",
            partition="normal",
            depends=executor,
        ),
        "local": LocalPipelineExecutor(
            pipeline=pipeline_reduce,
            logging_dir=f"{WIKI_OUTPUT_PATH}/logs_reduce/",
            tasks=1,
        ),
    }[EXECUTOR]

    executor_reduce.run()


    # Scientific papers (en) pipeline
    pipeline = [
        ShuffledHFDatasetReader(
            "scientific_papers",
            limit=DOC_LIMIT,
            dataset_options={
                "name": "arxiv",
                "split": "train",
            },
            default_metadata={"language": "en"},
            shuffle_seed=43,
            text_key="article",
        ),
        MultilingualGopherQualityFilter(
            exclusion_writer=JsonlWriter(
                f"{PAPERS_OUTPUT_PATH}/removed",
                output_filename="${language}/${rank}.jsonl.gz",
            ),
            **gopher_args,
        ),
    ]

    executor = {
        "slurm": SlurmPipelineExecutor(
            pipeline=pipeline,
            logging_dir=f"{PAPERS_OUTPUT_PATH}/logs/",
            tasks=TASKS,
            time="06:00:00",
            partition="normal",
        ),
        "local": LocalPipelineExecutor(
            pipeline=pipeline,
            logging_dir=f"{PAPERS_OUTPUT_PATH}/logs/",
            tasks=TASKS,
        ),
    }[EXECUTOR]

    executor.run()


    pipeline_reduce = [
        DocumentStatsReducer(
            f"{PAPERS_OUTPUT_PATH}/doc_stats/", f"{PAPERS_OUTPUT_PATH}", "doc_stats.json"
        )
    ]

    executor_reduce = {
        "slurm": SlurmPipelineExecutor(
            pipeline=pipeline_reduce,
            logging_dir=f"{PAPERS_OUTPUT_PATH}/logs_reduce/",
            tasks=1,
            time="06:00:00",
            partition="normal",
            depends=executor,
        ),
        "local": LocalPipelineExecutor(
            pipeline=pipeline_reduce,
            logging_dir=f"{PAPERS_OUTPUT_PATH}/logs_reduce/",
            tasks=1,
        ),
    }[EXECUTOR]

    executor_reduce.run()

