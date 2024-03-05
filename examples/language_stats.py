import sys

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.stats import DocumentStats, DocumentStatsReducer
from datatrove.pipeline.readers import JsonlReader


# 176 languages supported by fasttext, see https://fasttext.cc/docs/en/language-identification.html
LANGUAGES = "af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce ceb ckb co cs cv cy da de diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez li lmo lo lrc lt lv mai mg mhr min mk ml mn mr mrj ms mt mwl my myv mzn nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec vep vi vls vo wa war wuu xal xmf yi yo yue zh".split(
    " "
)
TASKS = 20

# DUMP should be given as an argument. Example: CC-MAIN-2023-23
if len(sys.argv) != 2:
    print("Argument required: dump name")
    sys.exit(-1)
DUMP = sys.argv[1]

for lang in LANGUAGES[:4]:
    additional_data_path = f'test/data/datatrove/multi_lingual_stats/base_processing/{lang}'
    additional_logs_path = f'test/logs/datatrove/multi_lingual_stats/base_processing/{lang}'
    data_path = f"s3://fineweb-data-processing-us-east-1/base_processing/non_english/{lang}/{DUMP}"

    executor = SlurmPipelineExecutor(
        job_name=f"stats_cc_{DUMP}_{lang}",
        pipeline=[
            JsonlReader(
                data_path,
                default_metadata={"dump": DUMP},
                limit=10,
                text_key="content",
            ),
            DocumentStats(output_folder=f"{additional_data_path}/stats/{DUMP}"),
        ],
        tasks=TASKS,
        time="24:00:00",
        logging_dir=f"{additional_logs_path}/stats/{DUMP}/",
        randomize_start=True,
        mem_per_cpu_gb=2,
        partition="normal",
    )
    executor.run()

    executor_reduce = SlurmPipelineExecutor(
        job_name=f"cc_stats_{DUMP}",
        pipeline=[
            DocumentStatsReducer(
                f"{additional_data_path}/stats/",
                output_folder=f"{additional_data_path}/reduce/{DUMP}",
                output_file_name=f"{DUMP}.json",
            )
        ],
        tasks=1,
        time="01:00:00",
        logging_dir=f"{additional_logs_path}/reduce/{DUMP}/",
        partition="normal",
        depends=executor,
    )
    executor_reduce.run()