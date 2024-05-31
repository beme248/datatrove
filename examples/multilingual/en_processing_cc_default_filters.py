import fire

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import (
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import (
    GopherQualityFilter,
    GopherRepetitionFilter,
    C4QualityFilter,
)
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

LANGUAGES = [
    "en",
]

def process_data(
    dump_to_process: str = "CC-MAIN-2023-23",
    doc_limit: int = 10_000,
    num_tasks: int = 100,
    num_workers: int = 100,
    executor_mode: str = "local"):

    filter_mode = 'default'
    dataset_mode = 'cc'
    run_name = f"{dataset_mode}_with_{filter_mode}_filters"

    for language in LANGUAGES:
        data_path = f"clean_cc_en/v5/data/{dump_to_process}/{language}/output/{dump_to_process}"
        filtering_output_path = f"processing/multilingual_{run_name}/data/{dump_to_process}/{language}"
        logs_path = f"processing/multilingual_{run_name}/logs/{dump_to_process}/{language}"
        pipeline =[
            JsonlReader(
                data_path,
                default_metadata={"dump": dump_to_process},
                limit=doc_limit,
                text_key="text",
            ),
            # URLFilter(exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/url/{dump_to_process}")),
            # Trafilatura(favour_precision=True, timeout=5),
            # LanguageFilter(
            #     exclusion_writer=JsonlWriter(
            #         f"{filtering_output_path}/non_english/{dump_to_process}",
            #         output_filename="${language}/" + dump_to_process + "/${rank}.jsonl.gz",  # folder structure: language/dump/file
            #     )
            # ),
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/3_gopher_rep/{dump_to_process}"),
                language=language
            ),
            GopherQualityFilter(
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/4_gopher_qual/{dump_to_process}"),
                language=language
            ),
            C4QualityFilter(
                filter_no_terminal_punct=False,
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/5_c4/{dump_to_process}"),
            ),
            FineWebQualityFilter(
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/6_fineweb_qual/{dump_to_process}"),
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