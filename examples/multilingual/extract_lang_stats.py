import os

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import LanguageFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.stats import LanguageStats, LanguageStatsReducer


# Top 10 languages in CommonCrawl according to (https://arxiv.org/pdf/2306.01116.pdf) + English
LANGUAGES = ["en", "ru", "de", "es", "ja", "fr", "zh", "it", "pt", "nl", "pl"]
MAIN_OUTPUT_PATH = "./language_stats"
WIKI_VERSION = "20231101"  # See https://huggingface.co/datasets/wikimedia/wikipedia
DOC_LIMIT = 5000  # Limit number of documents per dataset
TASKS = 8
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
        "local": LocalPipelineExecutor(pipeline=pipeline, logging_dir=f"{MAIN_OUTPUT_PATH}/logs/", tasks=TASKS),
        "slurm": SlurmPipelineExecutor(
            pipeline=pipeline,
            logging_dir=f"{MAIN_OUTPUT_PATH}/logs/",
            tasks=TASKS,
            time="06:00:00",
            partition="normal",
        ),
    }[EXECUTOR]
    executor.run()

    # Additional statistics to be added into lang_stats.json file
    def stat_reducer(language_stats):
        # Make sure to import np here for slurm executor
        import numpy as np

        def q(counts, q):
            counts_sorted = sorted(counts)
            xs = [d[0] for d in counts_sorted]
            ys = [d[1] for d in counts_sorted]
            ys_cumsum = np.cumsum(ys)
            index = np.sum(ys_cumsum < q * ys_cumsum[-1])
            return xs[index]

        length_counter = language_stats["length_counter"]

        lengths = list(length_counter.keys())
        freqs = list(length_counter.values())

        word_length_mean = np.average(lengths, weights=freqs)
        word_length_std = np.sqrt(np.cov(lengths, fweights=freqs))
        word_length_q = {f"{i/20:.2f}": q(length_counter.items(), i / 20) for i in range(21)}

        top_8_words = dict(language_stats["word_counter"].most_common(8))

        return {
            "min_avg_word_length": round(word_length_mean - word_length_std),
            "max_avg_word_length": round(word_length_mean + word_length_std),
            "word_length_mean": word_length_mean,
            "word_length_std": word_length_std,
            "word_length_q": word_length_q,
            "top_8_words": top_8_words,
        }

    pipeline_reduce = [
        LanguageStatsReducer(
            input_folder=f"{MAIN_OUTPUT_PATH}/lang_stats/",
            output_folder=".",
            output_file_name="lang_stats.json",
            reduce_fn=stat_reducer,
            word_common_prune=10000,
        )
    ]
    executor_reduce = {
        "local": LocalPipelineExecutor(pipeline=pipeline_reduce, logging_dir=f"{MAIN_OUTPUT_PATH}/logs_reduce/"),
        "slurm": SlurmPipelineExecutor(
            pipeline=pipeline_reduce,
            logging_dir=f"{MAIN_OUTPUT_PATH}/logs_reduce/",
            tasks=1,
            time="00:30:00",
            partition="normal",
            depends=executor,
        ),
    }[EXECUTOR]
    executor_reduce.run()
