import os
import sys

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.stats import DocumentStats, DocumentStatsReducer


# 176 languages supported by fasttext, see https://fasttext.cc/docs/en/language-identification.html
# LANGUAGES = "af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce ceb ckb co cs cv cy da de diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez li lmo lo lrc lt lv mai mg mhr min mk ml mn mr mrj ms mt mwl my myv mzn nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec vep vi vls vo wa war wuu xal xmf yi yo yue zh".split(
#     " "
# )
LANGUAGES = "en"
TASKS = 10

# DUMP should be given as an argument. Example: CC-MAIN-2023-23
if len(sys.argv) != 2:
    print("Argument required: dump name")
    sys.exit(-1)
DUMP = sys.argv[1]

scratch = os.getenv("SCRATCH")

for lang in LANGUAGES:
    additional_data_path = f"data/datatrove/multi_lingual_stats/base_processing/{lang}"
    additional_logs_path = f"logs/datatrove/multi_lingual_stats/base_processing/{lang}"
    data_path = f"s3://fineweb-data-processing-us-east-1/base_processing/non_english/{lang}/{DUMP}"
    MAIN_OUTPUT_PATH = os.path.join(scratch, additional_data_path)
    SLURM_LOGS = os.path.join(scratch, additional_logs_path)

    executor = LocalPipelineExecutor(
        # job_name=f"cc_stats_{DUMP}_{lang}",
        pipeline=[
            JsonlReader(
                data_path,
                default_metadata={"dump": DUMP},
                text_key="content",
                limit=2,
            ),
            DocumentStats(output_folder=f"{MAIN_OUTPUT_PATH}/stats/{DUMP}"),
        ],
        tasks=TASKS,
        workers=TASKS,
        logging_dir=additional_logs_path)
    executor.run()

    executor_reduce = LocalPipelineExecutor(
        # job_name=f"cc_stats_reduce_{DUMP}_{lang}",
        pipeline=[
            DocumentStatsReducer(
                f"{MAIN_OUTPUT_PATH}/stats/{DUMP}",
                output_folder=f"{MAIN_OUTPUT_PATH}/reduce/{DUMP}",
                output_file_name=f"{DUMP}.json",
                limit=2,
            )
        ],
        tasks=1,
        workers=TASKS,
        logging_dir=additional_logs_path,
        depends=executor
    )
    executor_reduce.run()
