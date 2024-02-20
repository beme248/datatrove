import os

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import LanguageFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.stats import LanguageStats, LanguageStatsReducer


# Top 10 languages other than English in CommonCrawl according to (https://arxiv.org/pdf/2306.01116.pdf)
LANGUAGES = ["ru", "de", "es", "ja", "fr", "zh", "it", "pt", "nl", "pl"]
MAIN_OUTPUT_PATH = "./language_stats"
WIKI_VERSION = "20231101"  # See https://huggingface.co/datasets/wikimedia/wikipedia
DOC_LIMIT = 50000  # Limit number of documents per dataset
EXECUTOR = os.environ.get("EXECUTOR", "slurm")  # local/slurm

if __name__ == "__main__":
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
    executor = {
        "local": LocalPipelineExecutor(pipeline=pipeline, logging_dir=f"{MAIN_OUTPUT_PATH}/logs/", tasks=8),
        "slurm": SlurmPipelineExecutor(
            pipeline=pipeline,
            logging_dir=f"{MAIN_OUTPUT_PATH}/logs/",
            tasks=2,
            time="00:30:00",
            partition="clariden",
        ),
    }[EXECUTOR]
    executor.run()


    # Reduce token lengths into lang_stats.json file
    def length_counter_reducer(length_counter):
        # Make sure to import np here for slurm executor
        import numpy as np

        lengths = list(length_counter.keys())
        freqs = list(length_counter.values())

        word_length_mean = np.average(lengths, weights=freqs)
        word_length_std = np.sqrt(np.cov(lengths, fweights=freqs))

        return {
            "min_avg_word_length": round(word_length_mean - word_length_std),
            "max_avg_word_length": round(word_length_mean + word_length_std),
            'word_length_mean': word_length_mean,
            'word_length_std': word_length_std,
        }


    pipeline_reduce = [
        LanguageStatsReducer(
            input_folder=f"{MAIN_OUTPUT_PATH}/lang_stats/",
            output_folder=".",
            output_file_name="lang_stats.json",
            length_counter_reducer=length_counter_reducer,
        )
    ]
    executor_reduce = {
        "local": LocalPipelineExecutor(pipeline=pipeline_reduce, logging_dir=f"{MAIN_OUTPUT_PATH}/logs_reduce/"),
        "slurm": SlurmPipelineExecutor(
            pipeline=pipeline_reduce,
            logging_dir=f"{MAIN_OUTPUT_PATH}/logs_reduce/",
            tasks=1,
            time="00:30:00",
            partition="clariden",
            depends=executor,
        ),
    }[EXECUTOR]
    executor_reduce.run()
