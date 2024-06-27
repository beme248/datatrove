import sys
import fire

import datasets
import datetime

import datasets
import fire
import yaml

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LambdaFilter,
    FastTextClassifierFilter,
)
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

def process_data(
    doc_limit: int = -1,
    num_tasks: int = 100,
    num_workers: int = 40,
    executor_mode: str = "local"):
    filtering_output_path = f"fasttext_processing_100BT/v1/data"
    logs_path = f"fasttext_processing_100BT/v1/logs"

    pipeline = [
        HuggingFaceDatasetReader(
            "HuggingFaceFW/fineweb",
            dataset_options={
                "name": "sample-100BT",
                "split": "train",
                "cache_dir": "./hf_cache"
            },
            limit=doc_limit,
        ),
        FastTextClassifierFilter(
            model_url = "model.bin",
            keep_labels = [ ("positive", 0.5) ],
            save_labels_in_metadata = True,
            exclusion_writer = JsonlWriter(f"{filtering_output_path}/removed/fasttext_classifier"),
            newline_replacement = "",
        ),
        JsonlWriter(f"{filtering_output_path}/output")
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
