import json
import os

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import MultilingualGopherQualityFilter
from datatrove.pipeline.readers import ShuffledHFDatasetReader
from datatrove.pipeline.writers import JsonlWriter

from nltk.corpus import stopwords as nltk_stopwords
from math import sqrt, floor, log, e


# Top 10 languages in CommonCrawl according to (https://arxiv.org/pdf/2306.01116.pdf) + English
# LANGUAGES = ["en", "ru", "de", "es", "ja", "fr", "zh", "it", "pt", "nl", "pl"]
LANGUAGES = ["es"]
WIKI_OUTPUT_PATH = "./sanity_check_wiki"
WIKI_VERSION = "20231101"  # See https://huggingface.co/datasets/wikimedia/wikipedia
PAPERS_OUTPUT_PATH = "./sanity_check_papers"
DOC_LIMIT = 4000  # Limit number of documents per dataset
TASKS = 10
EXECUTOR = os.environ.get("EXECUTOR", "slurm")  # local/slurm
STATS_FILE = "./wiki_lang_stats.json"
STOPWORDS = "stopwords_top_n", "15"  # key for getting stopwords from statistics

# Load language-specific statistics and stopwords
with open(STATS_FILE, "r") as f:
    language_stats = json.load(f)

# Set-up language specific statistics
min_avg_word_lengths = {k: v["min_avg_word_length"] for k, v in language_stats.items()}
max_avg_word_lengths = {k: v["max_avg_word_length"] for k, v in language_stats.items()}

# Language stats stopwords
# stopwords = {k: v[STOPWORDS[0]][STOPWORDS[1]] for k, v in language_stats.items()}

# Gopher stopwords for english
# stopwords = {"en": ["the", "be", "to", "of", "and", "that", "have", "with"]}

# NLTK stopwords
en_stopwords = nltk_stopwords.words("english")
fr_stopwords = nltk_stopwords.words("french")
de_stopwords = nltk_stopwords.words("german")
es_stopwords = nltk_stopwords.words("spanish")
nltk_stopwords = {"en": en_stopwords, "fr": fr_stopwords, "de": de_stopwords, "es": es_stopwords}
# stopwords = nltk_stopwords

# Intersection between NLTK and language stats stopwords
stopwords = {k: list(set(v).intersection(set(language_stats[k][STOPWORDS[0]][STOPWORDS[1]]))) for k, v in nltk_stopwords.items()} 

# Min. stopwords strategies
# min_stop_words = {k: 2 for k, v in stopwords.items()}
# min_stop_words = {k: len(v) // 4 for k, v in stopwords.items()}
# min_stop_words = {k: floor(sqrt(len(v))) for k,v in stopwords.items()}
min_stop_words = {k: floor(log(len(v), e)) for k,v in stopwords.items()}

print(stopwords)
print({k: len(v) for k,v in stopwords.items()})
print(min_stop_words)

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
                f"{WIKI_OUTPUT_PATH}/removed",
                output_filename="${language}/${rank}.jsonl.gz",
            ),
        ),
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


    # Scientific papers (en) pipeline
    # pipeline = [
    #     ShuffledHFDatasetReader(
    #         "scientific_papers",
    #         limit=DOC_LIMIT,
    #         dataset_options={
    #             "name": "arxiv",
    #             "split": "train",
    #         },
    #         default_metadata={"language": "en"},
    #         shuffle_seed=43,
    #         text_key="article",
    #     ),
    #     MultilingualGopherQualityFilter(  # turn off all other filters except stopwords
    #         stop_words=stopwords,
    #         min_stop_words=min_stop_words,
    #         max_avg_word_lengths=None,
    #         min_avg_word_lengths=None,
    #         min_doc_words=None,
    #         max_doc_words=None,
    #         max_symbol_word_ratio=None,
    #         max_bullet_lines_ratio=None,
    #         max_ellipsis_lines_ratio=None,
    #         max_non_alpha_words_ratio=None,
    #         exclusion_writer=JsonlWriter(
    #             f"{PAPERS_OUTPUT_PATH}/removed",
    #             output_filename="${language}/${rank}.jsonl.gz",
    #         ),
    #     ),
    # ]

    # executor = {
    #     "slurm": SlurmPipelineExecutor(
    #         pipeline=pipeline,
    #         logging_dir=f"{PAPERS_OUTPUT_PATH}/logs/",
    #         tasks=TASKS,
    #         time="06:00:00",
    #         partition="normal",
    #     ),
    #     "local": LocalPipelineExecutor(
    #         pipeline=pipeline,
    #         logging_dir=f"{PAPERS_OUTPUT_PATH}/logs/",
    #         tasks=TASKS,
    #     ),
    # }[EXECUTOR]

    # executor.run()
