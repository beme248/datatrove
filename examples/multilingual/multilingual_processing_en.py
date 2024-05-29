import sys
import fire

import datasets
import datetime
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
import yaml

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import (
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
)
from datasets import Features, ClassLabel, Value, load_dataset
from datatrove.pipeline.readers import JsonlReader, HuggingFaceDatasetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    GopherQualityFilter,
    GopherRepetitionFilter,
    C4QualityFilter,
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
    filter_mode: str = 'wiki',
    dataset_mode: str = 'cc',
    dump_to_process: str = "CC-MAIN-2023-23",
    doc_limit: int = 100,
    num_tasks: int = 1,
    num_workers: int = 1,
    executor_mode: str = "local"):

    assert filter_mode == 'wiki' or filter_mode == 'cc', \
        f"run_mode must either be 'wiki' or 'cc', to use filters computed on wiki-data or cc-data."
    assert dataset_mode == 'cc' or dataset_mode == 'curated', \
        f"dataset_mode must either be 'cc' or 'curated', to filter cc data or the curated data annotation data."

    filters_folder = f"./{filter_mode}_filters"
    run_name = f"{dataset_mode}_with {filter_mode}_filters"
    filters = load_filters(filters_folder, LANGUAGES)

    min_avg_word_lengths = {k: v["min_avg_word_length"] for k, v in filters.items()}
    max_avg_word_lengths = {k: v["max_avg_word_length"] for k, v in filters.items()}
    stopwords = {k: v["stopwords"] for k, v in filters.items()}
    max_non_alpha_words_ratio = {k: v["max_non_alpha_words_ratio"] for k, v in filters.items()}
    min_stop_words = {k: 2 for k, _ in filters.items()}

    char_duplicates_ratio = {k: v["char_duplicates_ratio"] for k, v in filters.items()}
    line_punct_thr = {k: v["line_punct_thr"] for k, v in filters.items()}
    new_line_ratio = {k: v["new_line_ratio"] for k, v in filters.items()}
    short_line_thr = {k: v["short_line_thr"] for k, v in filters.items()}

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
            Trafilatura(favour_precision=True, timeout=10),
            LanguageFilter(
                exclusion_writer=JsonlWriter(
                    f"{dump_to_process}/non_english/",
                    output_filename="${language}/" + dump_to_process + "/${rank}.jsonl.gz",  # folder structure: language/dump/file
                )
            ),
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/3_gopher_rep/{dump_to_process}"),
                language=language
            ),
            GopherQualityFilter(
                max_avg_word_length=max_avg_word_lengths[language],
                min_avg_word_length=min_avg_word_lengths[language],
                stop_words=stopwords[language],
                min_stop_words=min_stop_words[language],
                max_non_alpha_words_ratio=max_non_alpha_words_ratio[language],
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/4_gopher_qual/{dump_to_process}"),
                language=language
            ),
            C4QualityFilter(
                filter_no_terminal_punct=False,
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/5_c4/{dump_to_process}"),
            ),
            FineWebQualityFilter(
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/6_fineweb_qual/{dump_to_process}"),
                line_punct_thr=line_punct_thr[language],
                short_line_thr=short_line_thr[language],
                char_duplicates_ratio=char_duplicates_ratio[language],
                new_line_ratio=new_line_ratio[language],
                language=language,
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