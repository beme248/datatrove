import os

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import ShuffledHFDatasetReader
from datatrove.pipeline.stats import LanguageStats, LanguageStatsReducer


LANGUAGES_TOP50 = [
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
]
LANGUAGES_PLUS50 = [
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

MAIN_OUTPUT_PATH = "./wiki_language_stats"
WIKI_VERSION = "20231101"  # See https://huggingface.co/datasets/wikimedia/wikipedia
DOC_LIMIT = 4000
TASKS = 10
EXECUTOR = os.environ.get("EXECUTOR", "slurm")  # local/slurm

if __name__ == "__main__":
    # Count token lengths
    readers = [
        ShuffledHFDatasetReader(  # Use shuffled dataset when using DOC_LIMIT
            "wikimedia/wikipedia",
            dataset_options={
                "name": f"{WIKI_VERSION}.{language}",
                "split": "train",
            },
            limit=DOC_LIMIT,
            default_metadata={"language": language},
        )
        for language in LANGUAGES_TOP50
    ]

    pipeline = [
        *readers,
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

    def stat_mapper(language_stats):
        # Make sure to import np here for slurm executor
        import numpy as np

        def q_lengths(counts, q):
            counts_sorted = sorted(counts)
            xs = [d[0] for d in counts_sorted]
            ys = [d[1] for d in counts_sorted]
            ys_cumsum = np.cumsum(ys)
            index = np.sum(ys_cumsum < q * ys_cumsum[-1])
            return xs[index]

        def q_words(counts, q):
            counts_sorted = sorted(counts, key=lambda x: -x[1])
            xs = [d[0] for d in counts_sorted]
            ys = [d[1] for d in counts_sorted]
            ys_cumsum = np.cumsum(ys)
            index = np.sum(ys_cumsum < q * ys_cumsum[-1])
            return xs[:index]

        def p_thresh_words(counts, p):
            counts_sorted = sorted(counts, key=lambda x: -x[1])
            xs = [d[0] for d in counts_sorted]
            ys = [d[1] for d in counts_sorted]
            ys_cumsum = np.cumsum(ys)
            index = np.sum(ys > p * ys_cumsum[-1])
            return xs[:index]

        length_counter = language_stats["length_counter"]
        word_counter = language_stats["word_counter"]

        lengths = list(length_counter.keys())
        freqs = list(length_counter.values())

        word_length_mean = np.average(lengths, weights=freqs)
        word_length_std = np.sqrt(np.cov(lengths, fweights=freqs))
        word_length_q = {f"{i/20:.2f}": q_lengths(length_counter.items(), i / 20) for i in range(21)}

        alpha_ratio_mean = language_stats["alpha_ratio_mean"]
        alpha_ratio_std = language_stats["alpha_ratio_std"]

        stopwords_q = {f"{q:.2f}": q_words(word_counter.items(), q) for q in [0.15, 0.2, 0.25, 0.3]}
        stopwords_p_thresh = {
            f"{p:.3f}": p_thresh_words(word_counter.items(), p)
            for p in [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012]
        }
        stopwords_top_n = {
            f"{n}": list(dict(language_stats["word_counter"].most_common(n)).keys()) for n in [5, 8, 10, 15]
        }

        return language_stats | {
            "min_avg_word_length": round(word_length_mean - word_length_std),
            "max_avg_word_length": round(word_length_mean + word_length_std),
            "max_non_alpha_words_ratio": round(alpha_ratio_mean - 3 * alpha_ratio_std, 1),
            "word_length_mean": word_length_mean,
            "word_length_std": word_length_std,
            "word_length_q": word_length_q,
            "stopwords_q": stopwords_q,
            "stopwords_p_thresh": stopwords_p_thresh,
            "stopwords_top_n": stopwords_top_n,
        }

    pipeline_reduce = [
        LanguageStatsReducer(
            input_folder=f"{MAIN_OUTPUT_PATH}/lang_stats/",
            output_folder="language_statistics",
            map_fn=stat_mapper,
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
