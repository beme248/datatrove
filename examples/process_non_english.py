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
DOC_LIMIT = 5000
NUM_TASKS = 20
STATS_FILE = "./examples/multilingual/lang_stats/wiki_lang_stats_100.json"
TOP50_LANGUAGES = [ Languages.english, Languages.spanish, Languages.portuguese, Languages.italian, Languages.french, Languages.romanian, Languages.german, Languages.latin, Languages.czech, Languages.danish, Languages.finnish, Languages.greek, Languages.norwegian, Languages.polish, Languages.russian, Languages.slovenian, Languages.swedish, Languages.turkish, Languages.dutch, Languages.chinese, Languages.japanese, Languages.vietnamese, Languages.indonesian, Languages.persian, Languages.korean, Languages.arabic, Languages.thai, Languages.hindi, Languages.bengali, Languages.tamil, Languages.hungarian, Languages.ukrainian, Languages.slovak, Languages.bulgarian, Languages.catalan, Languages.croatian, Languages.serbian, Languages.lithuanian, Languages.estonian, Languages.hebrew, Languages.latvian, Languages.serbocroatian, Languages.albanian, Languages.azerbaijani, Languages.icelandic, Languages.macedonian, Languages.georgian, Languages.galician, Languages.armenian, Languages.basque ]
TOP100_LANGUAGES = [ Languages.swahili, Languages.malay, Languages.tagalog, Languages.javanese, Languages.punjabi, Languages.bihari,Languages.gujarati, Languages.yoruba, Languages.marathi, Languages.urdu, Languages.amharic, Languages.telugu, Languages.malayalam, Languages.kannada, Languages.nepali, Languages.kazakh, Languages.belarusian, Languages.burmese, Languages.esperanto, Languages.uzbek, Languages.khmer, Languages.tajik, Languages.welsh, Languages.norwegian_nynorsk, Languages.bosnian, Languages.sinhala, Languages.tatar, Languages.afrikaans, Languages.oriya, Languages.kirghiz, Languages.irish, Languages.occitan, Languages.kurdish, Languages.lao, Languages.luxembourgish, Languages.bashkir, Languages.western_frisian, Languages.pashto, Languages.maltese, Languages.breton, Languages.assamese, Languages.malagasy, Languages.divehi, Languages.yiddish, Languages.somali, Languages.sanskrit, Languages.sindhi, Languages.turkmen, Languages.south_azerbaijani, Languages.sorani, Languages.cebuano, Languages.war ]
RUN_NAME = "top50_stopwords_p_thresh_0_008_spacy_v2"

scratch = os.getenv('SCRATCH')

# Load language-specific statistics and stopwords
with open(STATS_FILE, "r") as f:
    language_stats = json.load(f)

def is_clean(word):
    word = word.strip() 
    return  word != '–' and \
            word != '—' and \
            word != '’' and \
            word != '’’' and \
            word != '||' and \
            word != '|' and \
            word != '।' and \
            word != "''" and \
            word != "'" and \
            word != '``' and \
            word != '`' and \
            word != '‘' and \
            word != '„' and \
            word != '“' and \
            word != '”' and \
            word != '«' and \
            word != '»' and \
            word != '|-' and \
            word != ':' and \
            word != '：' and \
            word != '《' and \
            word != '》' and \
            word != '，' and \
            word != '(' and \
            word != ')' and \
            word != '（' and \
            word != '）' and \
            word != '//' and \
            word != '/' and \
            word != '\\' and \
            word != '\\\\' and \
            "=" not in word and \
            "\u200d" not in word and \
            "align" != word and \
            not word.isdigit()

def to_clean(stopwords):
    return [ w for w in stopwords if is_clean(w) ]

def to_clean_stopwords(k, v):
    stopwords = to_clean(v["stopwords_p_thresh"]["0.008"])
    if len(stopwords) < 8 or k == "sr":
        print(f"====> {k}: {stopwords}")
        stopwords = to_clean(v["stopwords_p_thresh"]["0.003"])
        print(f"----> {k}: {stopwords}")
    return stopwords

def to_alpha_ratio(_, v):
    return v["max_non_alpha_words_ratio"]

min_avg_word_lengths = {k: v["min_avg_word_length"] for k, v in language_stats.items()}
max_avg_word_lengths = {k: v["max_avg_word_length"] for k, v in language_stats.items()}
stop_words = { k: to_clean_stopwords(k, v) for k, v in language_stats.items() }
alpha_ratio = { k: to_alpha_ratio(k, v) for k, v in language_stats.items() }
min_stop_words = { k: 2 for k, _ in language_stats.items() }

breakpoint()

for lang in TOP50_LANGUAGES + TOP100_LANGUAGES:
    additional_data_path = f'data/datatrove/multi_lingual_{DOC_LIMIT}_{RUN_NAME}/base_processing/{lang}'
    additional_logs_path = f'logs/datatrove/multi_lingual_{DOC_LIMIT}_{RUN_NAME}/base_processing/{lang}'
    data_path = f"s3://fineweb-data-processing-us-east-1/base_processing/non_english/{lang}/{DUMP}"

    MAIN_OUTPUT_PATH = os.path.join(scratch, additional_data_path)
    SLURM_LOGS = os.path.join(scratch, additional_logs_path)

    executor = SlurmPipelineExecutor(
        job_name=f"cc_{DUMP}_{lang}",
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
                max_non_alpha_words_ratio=alpha_ratio,
                exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/quality/{DUMP}"),
            ),
            ListFilter(exclusion_writer=JsonlWriter(f"{MAIN_OUTPUT_PATH}/removed/list/{DUMP}")),
            JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DUMP}"),
        ],
        tasks=NUM_TASKS,
        time="04:00:00",
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing/{DUMP}",
        slurm_logs_folder=f"{SLURM_LOGS}/{DUMP}/slurm_logs",
        randomize_start=True,
        mem_per_cpu_gb=2,
        partition="normal",
        mail_user="bettina.messmer@epfl.ch"
    )
    executor.run()

