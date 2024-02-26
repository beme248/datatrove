import os

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import LanguageFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.stats import LanguageStats, LanguageStatsReducer

from util import stat_reducer


# See https://huggingface.co/datasets/mlsum
LANGUAGES = ["de", "es", "fr", "ru"]
MAIN_OUTPUT_PATH = "./mlsum_language_stats"
DOC_LIMIT = 5000  # Limit number of documents per dataset
TASKS = 8
EXECUTOR = os.environ.get("EXECUTOR", "slurm")  # local/slurm

if __name__ == "__main__":
    # Count token lengths
    readers = [
        HuggingFaceDatasetReader(
            "mlsum",
            limit=DOC_LIMIT,
            dataset_options={
                "name": f"{language}",
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

    pipeline_reduce = [
        LanguageStatsReducer(
            input_folder=f"{MAIN_OUTPUT_PATH}/lang_stats/",
            output_folder=".",
            output_file_name="mlsum_lang_stats.json",
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
