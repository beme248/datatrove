import numpy as np

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import LanguageFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.stats import LanguageStats, LanguageStatsReducer


# Top 10 languages other than English in CommonCrawl according to (https://arxiv.org/pdf/2306.01116.pdf)
# TODO: ja, zh don't have good tokenizers
# LANGUAGES = ["ru", "de", "es", "ja", "fr", "zh", "it", "pt", "nl", "pl"]
LANGUAGES = ["fr", "de"]
MAIN_OUTPUT_PATH = "./language_stats"
WIKI_VERSION = "20231101"  # See https://huggingface.co/datasets/wikimedia/wikipedia
DOC_LIMIT = 20000  # Limit number of documents per dataset

# Count token lengths
readers = [
    HuggingFaceDatasetReader(
        "wikimedia/wikipedia",
        limit=DOC_LIMIT,
        dataset_options={
            "name": f"{WIKI_VERSION}.{language}",
            "split": "train",
        },
    )
    for language in LANGUAGES
]

pipeline = [
    *readers,
    LanguageFilter(languages=(LANGUAGES)),
    LanguageStats(output_folder=f"{MAIN_OUTPUT_PATH}/lang_stats/"),
]
executor = LocalPipelineExecutor(pipeline=pipeline, logging_dir=f"{MAIN_OUTPUT_PATH}/logs/")
executor.run()


# Reduce token lengths into lang_stats.json file
def length_counter_reducer(length_counter):
    lengths = list(length_counter.keys())
    freqs = list(length_counter.values())

    word_length_mean = np.average(lengths, weights=freqs)
    word_length_std = np.sqrt(np.cov(lengths, fweights=freqs))

    return {
        "min_avg_word_length": round(word_length_mean - word_length_std),
        "max_avg_word_length": round(word_length_mean + word_length_std),
    }


pipeline_reduce = [
    LanguageStatsReducer(
        input_folder=f"{MAIN_OUTPUT_PATH}/lang_stats/",
        output_folder=".",
        output_file_name="lang_stats.json",
        length_counter_reducer=length_counter_reducer,
    )
]
executor_reduce = LocalPipelineExecutor(pipeline=pipeline_reduce, logging_dir=f"{MAIN_OUTPUT_PATH}/logs_reduce/")
executor_reduce.run()
