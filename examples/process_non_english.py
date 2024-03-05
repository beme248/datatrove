import sys

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import (
    MultilingualGopherQualityFilter,
    GopherRepetitionFilter,
    ListFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
import os
import json
from nltk.corpus import stopwords
from datatrove.utils.typeshelper import Languages


# DUMP should be given as an argument. Example: CC-MAIN-2023-23
if len(sys.argv) != 2:
    print("Argument required: dump name")
    sys.exit(-1)

DUMP = sys.argv[1]
DOC_LIMIT = 50000
NUM_TASKS = 20
STATS_FILE = "./examples/multilingual/lang_stats/wiki_lang_stats.json"
# LANGUAGES = [Languages.russian, Languages.german, Languages.spanish, "ja", Languages.french, "zh", Languages.italian, Languages.polish, Languages.dutch, "pl"]
# LANGUAGES = [ Languages.russian, Languages.german, Languages.spanish, Languages.french, Languages.italian, Languages.portuguese, Languages.dutch, Languages.polish ]
LANGUAGES = [ Languages.russian, Languages.german, Languages.spanish, Languages.french, Languages.italian, Languages.portuguese, Languages.dutch ]
EN_LANGUAGES = {
    Languages.english: "english",
    Languages.russian: "russian",
    Languages.german: "german",
    Languages.spanish: "spanish",
    Languages.french: "french",
    Languages.italian: "italian",
    Languages.portuguese: "portuguese",
    Languages.dutch: "dutch"
}

# STOP_WORDS = "stopwords_p_thresh_0_008"
STOP_WORDS = "stopwords_nltk"

scratch = os.getenv('SCRATCH')

# Load language-specific statistics and stopwords
with open(STATS_FILE, "r") as f:
    language_stats = json.load(f)

def to_clean_stopwords(k, v):
    if k not in EN_LANGUAGES:
        return []
    # clean_stopwords = stopwords.words(EN_LANGUAGES[k])
    # return [ word for word in v["stopwords_p_thresh"]["0.008"] if word in clean_stopwords ]
    return stopwords.words(EN_LANGUAGES[k])

min_avg_word_lengths = {k: v["min_avg_word_length"] for k, v in language_stats.items()}
max_avg_word_lengths = {k: v["max_avg_word_length"] for k, v in language_stats.items()}
stop_words = {k: to_clean_stopwords(k, v) for k, v in language_stats.items()}
min_stop_words = {k: 2 for k, _ in language_stats.items()}

print(f"======> {stop_words}")

breakpoint()

for lang in LANGUAGES:
    additional_data_path = f'data/datatrove/multi_lingual_{DOC_LIMIT}_{STOP_WORDS}/base_processing/{lang}'
    additional_logs_path = f'logs/datatrove/multi_lingual_{DOC_LIMIT}_{STOP_WORDS}/base_processing/{lang}'
    data_path = f"s3://fineweb-data-processing-us-east-1/base_processing/non_english/{lang}/{DUMP}"

    MAIN_OUTPUT_PATH = os.path.join(scratch, additional_data_path)
    SLURM_LOGS = os.path.join(scratch, additional_logs_path)

    executor = SlurmPipelineExecutor(
        job_name=f"cc_{DUMP}",
        pipeline=[
            JsonlReader(
                data_path,
                default_metadata={"dump": DUMP},
                limit=DOC_LIMIT,
                text_key="content",
            ),
            GopherRepetitionFilter(exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/repetitive/{DUMP}")),
            MultilingualGopherQualityFilter(
                max_avg_word_lengths=max_avg_word_lengths,
                min_avg_word_lengths=min_avg_word_lengths,
                stop_words=stop_words,
                min_stop_words=min_stop_words,
                exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/quality/{DUMP}"),
            ),
            ListFilter(exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/list/{DUMP}")),
            JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DUMP}"),
        ],
        tasks=NUM_TASKS,
        time="00:30:00",
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing/{DUMP}",
        slurm_logs_folder=f"{SLURM_LOGS}/{DUMP}/slurm_logs",
        randomize_start=True,
        mem_per_cpu_gb=2,
        partition="normal",
    )
    executor.run()

