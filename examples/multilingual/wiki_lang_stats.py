import sys
sys.path.append('/mloscratch/homes/bmessmer/code/copied/datatrove/src')

import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import ShuffledHFDatasetReader
from datatrove.pipeline.stats import LanguageStatistics, LanguageStatsCalculator, LanguageStatsReducer

if len(sys.argv) != 2 or sys.argv[1] not in ["statistics", "filters"]:
    print("First argument should be: 'statistics' or 'filters'.")
    print("Use 'statistics' to generate statistics of the Wikipedia documents.")
    print("Use 'filters' to only generate the filter values for multilingual Gopher quality filter.")
    exit(1)

# high resource:
# chinese (105M docs, 13 benchmarks)
# french (135M docs, 10 benchmarks)
# russian (160M docs, 10 benchmarks)
# medium resource:
# turkish (21M docs, 5 benchmarks)
# arabic (17M docs, 12 benchmarks)
# thai (11M docs, 6 benchmarks)
# hindi (5M docs, 9 benchmarks)
# low resource:
# swahili (200k docs, 8 benchmarks)
# telugu (500k docs, 4 benchmarks)

RUN_MODE = sys.argv[1]
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
]

MAIN_OUTPUT_PATH = "./wiki_stats_pipeline_2"
WIKI_VERSION = "20231101"  # See https://huggingface.co/datasets/wikimedia/wikipedia
DOC_LIMIT = 4000
NUM_TASKS = 10
NUM_WORKERS = 10
# EXECUTOR = os.environ.get("EXECUTOR", "slurm")  # local/slurm
EXECUTOR = "local"

if __name__ == "__main__":
    for language in LANGUAGES:
        pipeline = [
            ShuffledHFDatasetReader(  # Use shuffled dataset when using DOC_LIMIT
                "wikimedia/wikipedia",
                dataset_options={
                    "name": f"{WIKI_VERSION}.{language}",
                    "split": "train",
                    "cache_dir": "./hf_cache"
                },
                limit=DOC_LIMIT,
                default_metadata={"language": language},
            ),
            LanguageStatsCalculator(output_folder=f"{MAIN_OUTPUT_PATH}/lang_stats_per_rank/"),
        ]
        executor = {
            "local": LocalPipelineExecutor(
                pipeline=pipeline,
                logging_dir=f"{MAIN_OUTPUT_PATH}/logs/{language}",
                tasks=NUM_TASKS,
                workers=NUM_WORKERS,
            ),
            "slurm": SlurmPipelineExecutor(
                pipeline=pipeline,
                logging_dir=f"{MAIN_OUTPUT_PATH}/logs/{language}",
                tasks=NUM_TASKS,
                time="06:00:00",
                partition="normal",
                workers=NUM_WORKERS,
            ),
        }[EXECUTOR]
        executor.run()

        # Compute language filter parameters
        def filters_mapper(language_stats: LanguageStatistics):
            # Make sure to import np here for slurm executor
            import numpy as np

            def p_thresh_words(counts, p):
                counts_sorted = sorted(counts, key=lambda x: -x[1])
                xs = [d[0] for d in counts_sorted]
                ys = [d[1] for d in counts_sorted]
                ys_cumsum = np.cumsum(ys)
                index = np.sum(ys > p * ys_cumsum[-1])
                return xs[:index]

            length_counter = language_stats.length_counter
            word_counter = language_stats.word_counter

            lengths = list(length_counter.keys())
            freqs = list(length_counter.values())

            word_length_mean = np.average(lengths, weights=freqs)
            word_length_std = np.sqrt(np.cov(lengths, fweights=freqs))

            alpha_ratio_mean = language_stats.alpha_ratio.mean
            alpha_ratio_std = language_stats.alpha_ratio.std

            line_punct_ratio_mean = language_stats.line_punct_ratio.mean
            line_punct_ratio_std = language_stats.line_punct_ratio.std

            short_line_ratio_mean = language_stats.short_line_ratio.mean
            short_line_ratio_std = language_stats.short_line_ratio.std

            new_line_ratio_mean = language_stats.new_line_ratio.mean
            new_line_ratio_std = language_stats.new_line_ratio.std

            def is_clean(word):
                word = word.strip()
                return (
                    word != "–"
                    and word != "—"
                    and word != "’"
                    and word != "’’"
                    and word != "||"
                    and word != "|"
                    and word != "।"
                    and word != "''"
                    and word != "'"
                    and word != "``"
                    and word != "`"
                    and word != "‘"
                    and word != "„"
                    and word != "“"
                    and word != "”"
                    and word != "«"
                    and word != "»"
                    and word != "|-"
                    and word != ":"
                    and word != "："
                    and word != "《"
                    and word != "》"
                    and word != "，"
                    and word != "("
                    and word != ")"
                    and word != "（"
                    and word != "）"
                    and word != "//"
                    and word != "/"
                    and word != "\\"
                    and word != "\\\\"
                    and "=" not in word
                    and "\u200d" not in word
                    and "align" != word
                    and not word.isdigit()
                )

            def to_clean(stopwords):
                return [w for w in stopwords if is_clean(w)]

            def to_clean_stopwords(lang, word_counter):
                stopwords = to_clean(p_thresh_words(word_counter, 0.008))
                if len(stopwords) < 8 or lang == "sr":
                    stopwords = p_thresh_words(word_counter, 0.003)
                return stopwords

            return {
                "min_avg_word_length": round(word_length_mean - word_length_std),
                "max_avg_word_length": round(word_length_mean + word_length_std),
                "max_non_alpha_words_ratio": round(alpha_ratio_mean - 3 * alpha_ratio_std, 1),
                "stopwords": to_clean_stopwords(language_stats.language, word_counter),
                "line_punct_thr": max(round(line_punct_ratio_mean - line_punct_ratio_std, 2), 0),
                "short_line_thr": round(short_line_ratio_mean + short_line_ratio_std, 2),
                "new_line_ratio": min(round(new_line_ratio_mean + 2 * new_line_ratio_std, 2), 1),
                "char_duplicates_ratio": 0.01,
            }

        pipeline_reduce = {
            "filters": [
                LanguageStatsReducer(
                    input_folder=f"{MAIN_OUTPUT_PATH}/lang_stats_per_rank/",
                    output_folder="./filters",
                    map_fn=filters_mapper,
                )
            ],
            "statistics": [
                LanguageStatsReducer(
                    input_folder=f"{MAIN_OUTPUT_PATH}/lang_stats_per_rank/",
                    output_folder="./statistics",
                )
            ],
        }

        executor_reduce = {
            "local": LocalPipelineExecutor(
                pipeline=pipeline_reduce[RUN_MODE], logging_dir=f"{MAIN_OUTPUT_PATH}/logs_{RUN_MODE}/{language}"
            ),
            "slurm": SlurmPipelineExecutor(
                pipeline=pipeline_reduce[RUN_MODE],
                logging_dir=f"{MAIN_OUTPUT_PATH}/logs_{RUN_MODE}/{language}",
                tasks=1,
                time="00:30:00",
                partition="normal",
                depends=executor,
            ),
        }[EXECUTOR]
        executor_reduce.run()
        print(f"Done '{language}'.")
