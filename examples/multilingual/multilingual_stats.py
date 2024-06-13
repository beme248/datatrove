import sys

import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader, ShuffledHFDatasetReader
from datatrove.pipeline.stats import LanguageStatistics, LanguageStatsCollector, LanguageStatsReducer


if len(sys.argv) < 2 or sys.argv[1] not in ["statistics", "filters_q", "filters_meanstd", "raw"]:
    print("First argument should be: 'statistics' or 'filters'.")
    print("Use 'statistics' to generate statistics of the Wikipedia documents.")
    print("Use 'filters' to only generate the filter values for multilingual Gopher quality filter.")
    exit(1)

if len(sys.argv) < 3 or sys.argv[2] not in ["cc", "wiki"]:
    print("Second argument should be: 'cc' or 'wiki'.")
    print("Use 'cc' to generate statistics of CC documents.")
    print("Use 'wiki' to generate statistics of CC documents.")
    exit(1)

RUN_MODE = sys.argv[1]
DATASET_MODE = sys.argv[2]
LANGUAGES = [
    # "en",
    "de",
    # "hr",
    # "pt",
    # "cs",
    # "zh",
    # "fr",
    # "ru",
    # "tr",
    # "ar",
    # "th",
    # "hi",
    # "sw",
    # "te",
    # "ja",
]

MAIN_OUTPUT_PATH = f"./{DATASET_MODE}_stats_pipeline_{RUN_MODE}"
DOC_LIMIT = 4000
NUM_TASKS = 10
NUM_WORKERS = 10
EXECUTOR = "local" # os.environ.get("EXECUTOR", "slurm")  # local/slurm
DUMP_TO_PROCESS = "CC-MAIN-2023-23"
WIKI_VERSION = "20231101"  # See https://huggingface.co/datasets/wikimedia/wikipedia

if __name__ == "__main__":
    for language in LANGUAGES:
        data_path = f"s3://fineweb-data-processing-us-east-1/base_processing/non_english/{language}/{DUMP_TO_PROCESS}"
        readers = {
        'cc': JsonlReader(
                data_path,
                default_metadata={"dump": DUMP_TO_PROCESS},
                limit=DOC_LIMIT,
                text_key="content",
            ),
        'wiki': ShuffledHFDatasetReader(  # Use shuffled dataset when using DOC_LIMIT
                "wikimedia/wikipedia",
                dataset_options={
                    "name": f"{WIKI_VERSION}.{language}",
                    "split": "train",
                },
                limit=DOC_LIMIT,
                default_metadata={"language": language},
            ),
        
        }
        pipeline = [
            readers[DATASET_MODE],
            LanguageStatsCollector(
                output_folder=f"{MAIN_OUTPUT_PATH}/lang_stats_per_rank/{language}", language=language
            ),
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
        def filters_meanstd_mapper(language_stats: LanguageStatistics):
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

            alpha_ratio_mean = float(np.mean(language_stats.alpha_ratio))
            alpha_ratio_std = float(np.std(language_stats.alpha_ratio))

            line_punct_ratio_mean = float(np.mean(language_stats.line_punct_ratio))
            line_punct_ratio_std = float(np.std(language_stats.line_punct_ratio))

            short_line_ratio_mean = float(np.mean(language_stats.short_line_ratio))
            short_line_ratio_std = float(np.std(language_stats.short_line_ratio))

            new_line_ratio_mean = float(np.mean(language_stats.new_line_ratio))
            new_line_ratio_std = float(np.std(language_stats.new_line_ratio))

            language_score_mean = float(np.mean(language_stats.language_score))
            language_score_std = float(np.std(language_stats.language_score))

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
                if len(stopwords) < 8:
                    stopwords = p_thresh_words(word_counter, 0.002)
                return stopwords

            return {
                "min_avg_word_length": max(round(word_length_mean - word_length_std), 0),
                "max_avg_word_length": round(word_length_mean + word_length_std),
                "max_non_alpha_words_ratio": round(alpha_ratio_mean - 0.5 * alpha_ratio_std, 2),
                "stopwords": to_clean_stopwords(language, word_counter),
                "line_punct_thr": max(round(line_punct_ratio_mean - line_punct_ratio_std, 2), 0),
                "short_line_thr": round(short_line_ratio_mean + short_line_ratio_std, 2),
                "new_line_ratio": min(round(new_line_ratio_mean + 2 * new_line_ratio_std, 2), 1),
                "char_duplicates_ratio": 0.01,
                "language_score_thr": max(round(float(language_score_mean - 3 * language_score_std), 2), 0),
            }

        # Compute language filter parameters
        def filters_q_mapper(language_stats: LanguageStatistics):
            # Make sure to import np here for slurm executor
            import numpy as np

            def p_thresh_words(counts, p):
                counts_sorted = sorted(counts, key=lambda x: -x[1])
                xs = [d[0] for d in counts_sorted]
                ys = [d[1] for d in counts_sorted]
                ys_cumsum = np.cumsum(ys)
                index = np.sum(ys > p * ys_cumsum[-1])
                return xs[:index]

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
                if len(stopwords) < 8:
                    stopwords = p_thresh_words(word_counter, 0.002)
                return stopwords

            return {
                "min_avg_word_length": round(float(np.quantile(language_stats.avg_word_doc_length, 0.0001))),
                "max_avg_word_length": round(float(np.quantile(language_stats.avg_word_doc_length, 0.9999))),
                "max_non_alpha_words_ratio": round(float(np.quantile(language_stats.alpha_ratio, 0.25)), 2),
                "stopwords": to_clean_stopwords(language, language_stats.word_counter),
                "line_punct_thr": round(float(float(np.quantile(language_stats.line_punct_ratio, 0.2))), 2),
                "short_line_thr": round(float(np.quantile(language_stats.short_line_ratio, 0.8)), 2),
                "new_line_ratio": min(round(float(np.quantile(language_stats.new_line_ratio, 0.97)), 2), 1),
                "char_duplicates_ratio": 0.01,
                "language_score_thr": round(float(np.quantile(language_stats.language_score, 0.02)), 2),
            }

        # Compute language statistics
        def statistics_mapper(language_stats: LanguageStatistics):
            # Make sure to import np here for slurm executor
            import numpy as np

            from datatrove.pipeline.stats.lang_stats import STATS_KEYS

            ls = language_stats.to_dict()
            return {
                "length_counter": dict(language_stats.length_counter),
                "word_counter": dict(language_stats.word_counter),
                "total_words": int(language_stats.total_words),
                "total_docs": int(language_stats.total_docs),
                "total_bytes": int(language_stats.total_bytes),
                **{
                    key: {
                        "mean": float(np.mean(ls[key])),
                        "std": float(np.std(ls[key])),
                    }
                    for key in STATS_KEYS
                },
            }

        pipeline_reduce = {
            "filters_q": [
                LanguageStatsReducer(
                    input_folder=f"{MAIN_OUTPUT_PATH}/lang_stats_per_rank/{language}",
                    output_folder=f"./{DATASET_MODE}_{RUN_MODE}",
                    map_fn=filters_q_mapper,
                    output_filename=f"{language}.yml",
                )
            ],
            "filters_meanstd": [
                LanguageStatsReducer(
                    input_folder=f"{MAIN_OUTPUT_PATH}/lang_stats_per_rank/{language}",
                    output_folder=f"./{DATASET_MODE}_{RUN_MODE}",
                    map_fn=filters_meanstd_mapper,
                    output_filename=f"{language}.yml",
                )
            ],
            "statistics": [
                LanguageStatsReducer(
                    input_folder=f"{MAIN_OUTPUT_PATH}/lang_stats_per_rank/{language}",
                    output_folder=f"./{DATASET_MODE}_{RUN_MODE}",
                    map_fn=statistics_mapper,
                    output_filename=f"{language}.yml",
                )
            ],
            "raw": [
                LanguageStatsReducer(
                    input_folder=f"{MAIN_OUTPUT_PATH}/lang_stats_per_rank/{language}",
                    output_folder=f"./{DATASET_MODE}_{RUN_MODE}",
                    output_filename=f"{language}.yml",
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