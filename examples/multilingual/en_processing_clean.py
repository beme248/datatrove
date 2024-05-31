import sys
import fire

import datasets
import datetime
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
import yaml

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.writers.jsonl import JsonlWriter

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

LANGUAGES = [
    "en",
]

# Load language-specific statistics and stopwords
def load_filters(folder_path, languages):
    filters = {}
    for language in languages:
        with open(f"{folder_path}/{language}.yml", "r") as f:
            language_filter = yaml.safe_load(f)
            filters[language] = language_filter
    return filters


def process_data(
    dump_to_process: str = "CC-MAIN-2023-23",
    doc_limit: int = 500 * 100,
    num_tasks: int = 100,
    num_workers: int = 100,
    executor_mode: str = "local"):

    dataset_mode: str = 'cc',
    run_name = f"{dataset_mode}_clean"

    for language in LANGUAGES:
        filtering_output_path = f"processing/en_{run_name}/data/{dump_to_process}/{language}"
        logs_path = f"processing/en_{run_name}/logs/{dump_to_process}/{language}"
        pipeline =[
            WarcReader(
                f"s3://commoncrawl/crawl-data/{dump_to_process}/segments/",
                glob_pattern="*/warc/*",  # we want the warc files
                default_metadata={"dump": dump_to_process},
                limit=doc_limit
            ),
            URLFilter(exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/url/{dump_to_process}")),
            Trafilatura(favour_precision=True, timeout=5),
            LanguageFilter(
                exclusion_writer=JsonlWriter(
                    f"{filtering_output_path}/non_english/{dump_to_process}",
                    output_filename="${language}/" + dump_to_process + "/${rank}.jsonl.gz",  # folder structure: language/dump/file
                )
            ),
            JsonlWriter(f"{filtering_output_path}/output/{dump_to_process}"),
        ]
        executor = {
            "local": LocalPipelineExecutor(
                pipeline=pipeline,
                logging_dir=f"{logs_path}",
                tasks=num_tasks,
                workers=num_workers,
            ),
            "slurm": SlurmPipelineExecutor(
                pipeline=pipeline,
                logging_dir=f"{logs_path}",
                tasks=num_tasks,
                time="06:00:00",
                partition="normal",
                workers=num_workers,
            ),
        }[executor_mode]
        executor.run()

if __name__ == '__main__':
  fire.Fire(process_data)