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
)
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter


datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

END_PUNCTUATION = (
    ".",
    "'",
    '"',
    "!",
    "?",
    "։",
    "؟",
    "۔",
    "܀",
    "܁",
    "܂",
    "߹",
    "।",
    "॥",
    "၊",
    "။",
    "።",
    "፧",
    "፨",
    "᙮",
    "᜵",
    "᜶",
    "᠃",
    "᠉",
    "᥄",
    "᥅",
    "᪨",
    "᪩",
    "᪪",
    "᪫",
    "᭚",
    "᭛",
    "᭞",
    "᭟",
    "᰻",
    "᰼",
    "᱾",
    "᱿",
    "‼",
    "‽",
    "⁇",
    "⁈",
    "⁉",
    "⸮",
    "⸼",
    "꓿",
    "꘎",
    "꘏",
    "꛳",
    "꛷",
    "꡶",
    "꡷",
    "꣎",
    "꣏",
    "꤯",
    "꧈",
    "꧉",
    "꩝",
    "꩞",
    "꩟",
    "꫰",
    "꫱",
    "꯫",
    "﹒",
    "﹖",
    "﹗",
    "！",
    "．",
    "？",
    "𐩖",
    "𐩗",
    "𑁇",
    "𑁈",
    "𑂾",
    "𑂿",
    "𑃀",
    "𑃁",
    "𑅁",
    "𑅂",
    "𑅃",
    "𑇅",
    "𑇆",
    "𑇍",
    "𑇞",
    "𑇟",
    "𑈸",
    "𑈹",
    "𑈻",
    "𑈼",
    "𑊩",
    "𑑋",
    "𑑌",
    "𑗂",
    "𑗃",
    "𑗉",
    "𑗊",
    "𑗋",
    "𑗌",
    "𑗍",
    "𑗎",
    "𑗏",
    "𑗐",
    "𑗑",
    "𑗒",
    "𑗓",
    "𑗔",
    "𑗕",
    "𑗖",
    "𑗗",
    "𑙁",
    "𑙂",
    "𑜼",
    "𑜽",
    "𑜾",
    "𑩂",
    "𑩃",
    "𑪛",
    "𑪜",
    "𑱁",
    "𑱂",
    "𖩮",
    "𖩯",
    "𖫵",
    "𖬷",
    "𖬸",
    "𖭄",
    "𛲟",
    "𝪈",
    "｡",
    "。",
    "ល",
    "។",
    "៕",
    "៖",
    "៙",
    "៚",
)  # FineWeb + Spacy sentencizer stop chars + Khmer puncts

LANGUAGES = [
    "en",
    "de",
    "ru",
    "fr",
    "ja",
    "es",
    "zh",
    "it",
    "nl",
    "pl",
    "pt",
    "cs",
    "vi",
    "id",
    "tr",
    "sv",
    "fa",
    "ko",
    "hu",
    "ar",
    "el",
    "ro",
    "da",
    "fi",
    "th",
    "uk",
    "sk",
    "no",
    "bg",
    "ca",
    "hr",
    "la",
    "sr",
    "hi",
    "sl",
    "lt",
    "et",
    "he",
    "bn",
    "lv",
    "sh",
    "sq",
    "az",
    "ta",
    "is",
    "mk",
    # "ka",
    "gl",
    "hy",
    "eu",
    "ms",
    "ur",
    "ne",
    "mr",
    "ml",
    "kk",
    "te",
    # "mn",
    "be",
    "gu",
    "kn",
    "tl",
    # "my",
    "eo",
    "uz",
    # "km",
    "tg",
    "cy",
    "nn",
    "bs",
    "si",
    "sw",
    "pa",
    "tt",
    "ckb",
    "af",
    "or",
    "ky",
    "ga",
    "am",
    "oc",
    "ku",
    # "lo",
    "lb",
    "ba",
    "ceb",
    "fy",
    "ps",
    "mt",
    # "br",
    "as",
    "mg",
    "war",
    # "dv",
    "yi",
    "so",
    "sa",
    "sd",
    "azb",
    "tk",
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
    filters_folder: str = "./filters_meanstd",
    dataset_mode: str = "cc",
    dump_to_process: str = "CC-MAIN-2023-23",
    doc_limit: int = 1000,
    num_tasks: int = 100,
    num_workers: int = 16,
    executor_mode: str = "local",
):
    assert (
        dataset_mode == "cc" or dataset_mode == "curated"
    ), "dataset_mode must either be 'cc' or 'curated', to filter cc data or the curated data annotation data."

    def to_reader(language, data_path):
        # Use default adapter, but replace datetime of metadata
        def hf_adapter(ctx, data: dict, path: str, id_in_file: int | str):
            default_metadata = data.pop("metadata", {})
            if "date" in default_metadata and isinstance(default_metadata["date"], datetime.datetime):
                default_metadata["date"] = default_metadata["date"].isoformat()
            return {
                "text": data.pop(ctx.text_key, ""),
                "id": data.pop(ctx.id_key, f"{path}/{id_in_file}"),
                "media": data.pop("media", []),
                "metadata": default_metadata | data,  # remaining data goes into metadata
            }

        readers = {
            "cc": JsonlReader(
                data_path,
                default_metadata={"dump": dump_to_process},
                limit=doc_limit,
                text_key="content",
            ),
            "curated": HuggingFaceDatasetReader(
                "ZR0zNqSGMI/curated_dataset",
                dataset_options={
                    "data_files": f"{language}/*.jsonl.gz",
                    "split": "train",
                    "cache_dir": "./hf_cache00",
                },
                limit=doc_limit,
                default_metadata={"language": language},
                adapter=hf_adapter,
            ),
        }
        return readers[dataset_mode]

    run_name = f"{dataset_mode}_with_{filters_folder}"
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
    language_score_thr = {k: v["language_score_thr"] for k, v in filters.items()}

    for language in LANGUAGES:
        filtering_output_path = f"processing/multilingual_{run_name}/data/{dump_to_process}/{language}"
        logs_path = f"processing/multilingual_{run_name}/logs/{dump_to_process}/{language}"
        data_path = f"s3://fineweb-data-processing-us-east-1/base_processing/non_english/{language}/{dump_to_process}"
        pipeline = [
            to_reader(language, data_path),
            LambdaFilter(
                lambda doc: doc.metadata["language_score"] > language_score_thr[language],
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/2_lang_score/{dump_to_process}"),
            ),
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/3_gopher_rep/{dump_to_process}"),
                language=language,
            ),
            GopherQualityFilter(
                max_avg_word_length=max_avg_word_lengths[language],
                min_avg_word_length=min_avg_word_lengths[language],
                stop_words=stopwords[language],
                min_stop_words=min_stop_words[language],
                max_non_alpha_words_ratio=max_non_alpha_words_ratio[language],
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/4_gopher_qual/{dump_to_process}"),
                language=language,
            ),
            C4QualityFilter(
                filter_no_terminal_punct=False,
                end_punctuation=END_PUNCTUATION,
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/5_c4/{dump_to_process}"),
                language=language,
            ),
            FineWebQualityFilter(
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/removed/6_fineweb_qual/{dump_to_process}"),
                line_punct_thr=line_punct_thr[language],
                short_line_thr=short_line_thr[language],
                stop_chars=END_PUNCTUATION,
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


if __name__ == "__main__":
    fire.Fire(process_data)
