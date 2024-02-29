import sys

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.stats import LanguageStats, LanguageStatsReducer


# 176 languages supported by fasttext, see https://fasttext.cc/docs/en/language-identification.html
LANGUAGES = "af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce ceb ckb co cs cv cy da de diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez li lmo lo lrc lt lv mai mg mhr min mk ml mn mr mrj ms mt mwl my myv mzn nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec vep vi vls vo wa war wuu xal xmf yi yo yue zh".split(
    " "
)

# DUMP should be given as an argument. Example: CC-MAIN-2023-23
if len(sys.argv) != 2:
    print("Argument required: dump name")
    sys.exit(-1)
DUMP = sys.argv[1]

MAIN_OUTPUT_PATH = "./commoncrawl_stats/"

executor = SlurmPipelineExecutor(
    job_name=f"cc_stats_{DUMP}",
    pipeline=[
        WarcReader(
            f"s3://commoncrawl/crawl-data/{DUMP}/segments/",
            glob_pattern="*/warc/*",  # we want the warc files
            default_metadata={"dump": DUMP},
        ),
        URLFilter(),
        Trafilatura(favour_precision=True),
        LanguageFilter(languages=LANGUAGES),
        LanguageStats(output_folder=f"{MAIN_OUTPUT_PATH}/stats/"),
    ],
    tasks=100,
    time="24:00:00",
    logging_dir=f"{MAIN_OUTPUT_PATH}/logs/{DUMP}/",
    randomize_start=True,
    mem_per_cpu_gb=2,
    partition="normal",
)
executor.run()


executor_reduce = SlurmPipelineExecutor(
    job_name=f"cc_stats_{DUMP}",
    pipeline=[
        LanguageStatsReducer(
            f"{MAIN_OUTPUT_PATH}/stats/",
            output_folder=f"{MAIN_OUTPUT_PATH}/stats_reduce/",
            output_file_name=f"{DUMP}.json",
        )
    ],
    tasks=1,
    time="01:00:00",
    logging_dir=f"{MAIN_OUTPUT_PATH}/logs_reduce/{DUMP}/",
    partition="normal",
    depends=[executor],
)
executor_reduce.run()
