import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
import yaml

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import DocumentsPipeline, PipelineStep
from datatrove.pipeline.filters.gopher_repetition_filter import (
    find_all_duplicate,
    find_duplicates,
    find_top_duplicate,
    get_n_grams,
)
from datatrove.tools.word_tokenizers import default_tokenizer


MEAN_STD_KEYS = [
    "hash_word_ratio",
    "ellipsis_word_ratio",
    "bullet_start_ratio",
    "ellipsis_end_ratio",
    "alpha_ratio",
    "line_punct_ratio",
    "short_line_ratio",
    "duplicate_line_ratio",
    "new_line_ratio",
    "dup_para_frac",
    "dup_para_char_frac",
    "dup_line_frac",
    "dup_line_char_frac",
    "top_2_gram",
    "top_3_gram",
    "top_4_gram",
    "duplicated_5_grams",
    "duplicated_6_grams",
    "duplicated_7_grams",
    "duplicated_8_grams",
    "duplicated_9_grams",
    "duplicated_10_grams",
]


@dataclass
class MeanStdFloat:
    mean: float
    std: float


@dataclass
class LanguageStatistics:
    language: str
    word_counter: Counter
    length_counter: Counter
    total_words: int
    total_docs: int
    total_bytes: int
    hash_word_ratio: MeanStdFloat
    ellipsis_word_ratio: MeanStdFloat
    bullet_start_ratio: MeanStdFloat
    ellipsis_end_ratio: MeanStdFloat
    alpha_ratio: MeanStdFloat
    line_punct_ratio: MeanStdFloat
    short_line_ratio: MeanStdFloat
    duplicate_line_ratio: MeanStdFloat
    new_line_ratio: MeanStdFloat
    dup_para_frac: MeanStdFloat
    dup_para_char_frac: MeanStdFloat
    dup_line_frac: MeanStdFloat
    dup_line_char_frac: MeanStdFloat
    top_2_gram: MeanStdFloat
    top_3_gram: MeanStdFloat
    top_4_gram: MeanStdFloat
    duplicated_5_grams: MeanStdFloat
    duplicated_6_grams: MeanStdFloat
    duplicated_7_grams: MeanStdFloat
    duplicated_8_grams: MeanStdFloat
    duplicated_9_grams: MeanStdFloat
    duplicated_10_grams: MeanStdFloat

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "word_counter": dict(self.word_counter),
            "length_counter": dict(self.length_counter),
            "total_words": self.total_words,
            "total_docs": self.total_docs,
            "total_bytes": self.total_bytes,
            "hash_word_ratio": {"mean": self.hash_word_ratio.mean, "std": self.hash_word_ratio.std},
            "ellipsis_word_ratio": {"mean": self.ellipsis_word_ratio.mean, "std": self.ellipsis_word_ratio.std},
            "bullet_start_ratio": {"mean": self.bullet_start_ratio.mean, "std": self.bullet_start_ratio.std},
            "ellipsis_end_ratio": {"mean": self.ellipsis_end_ratio.mean, "std": self.ellipsis_end_ratio.std},
            "alpha_ratio": {"mean": self.alpha_ratio.mean, "std": self.alpha_ratio.std},
            "line_punct_ratio": {"mean": self.line_punct_ratio.mean, "std": self.line_punct_ratio.std},
            "short_line_ratio": {"mean": self.short_line_ratio.mean, "std": self.short_line_ratio.std},
            "duplicate_line_ratio": {"mean": self.duplicate_line_ratio.mean, "std": self.duplicate_line_ratio.std},
            "new_line_ratio": {"mean": self.new_line_ratio.mean, "std": self.new_line_ratio.std},
            "dup_para_frac": {"mean": self.dup_para_frac.mean, "std": self.dup_para_frac.std},
            "dup_para_char_frac": {"mean": self.dup_para_char_frac.mean, "std": self.dup_para_char_frac.std},
            "dup_line_frac": {"mean": self.dup_line_frac.mean, "std": self.dup_line_frac.std},
            "dup_line_char_frac": {"mean": self.dup_line_char_frac.mean, "std": self.dup_line_char_frac.std},
            "top_2_gram": {"mean": self.top_2_gram.mean, "std": self.top_2_gram.std},
            "top_3_gram": {"mean": self.top_3_gram.mean, "std": self.top_3_gram.std},
            "top_4_gram": {"mean": self.top_4_gram.mean, "std": self.top_4_gram.std},
            "duplicated_5_grams": {"mean": self.duplicated_5_grams.mean, "std": self.duplicated_5_grams.std},
            "duplicated_6_grams": {"mean": self.duplicated_6_grams.mean, "std": self.duplicated_6_grams.std},
            "duplicated_7_grams": {"mean": self.duplicated_7_grams.mean, "std": self.duplicated_7_grams.std},
            "duplicated_8_grams": {"mean": self.duplicated_8_grams.mean, "std": self.duplicated_8_grams.std},
            "duplicated_9_grams": {"mean": self.duplicated_9_grams.mean, "std": self.duplicated_9_grams.std},
            "duplicated_10_grams": {"mean": self.duplicated_10_grams.mean, "std": self.duplicated_10_grams.std},
        }


class LanguageStatsCalculator(PipelineStep):
    type = "📊 - STATS"
    name = "🌐 Languages"

    def __init__(
        self,
        output_folder: DataFolderLike,
        language_field: str = "language",
        word_count_prune=2,
    ):
        super().__init__()
        self.language_field = language_field
        self.output_folder = get_datafolder(output_folder)
        self.word_count_prune = word_count_prune
        self.paragraph_exp = re.compile(r"\n{2,}")
        self._line_splitter = re.compile("\n+")

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        stats = {}

        # map and produce one output file per rank
        for doc in data:
            language = doc.metadata.get(self.language_field)
            if language not in stats:
                stats[language] = {
                    "length_counter": Counter(),
                    "word_counter": Counter(),
                    "total_words": 0,
                    "total_docs": 0,
                    "total_bytes": 0,
                    **{k: [] for k in MEAN_STD_KEYS},
                }

            text = doc.text
            words_punct = default_tokenizer.tokenize(doc.text, language)
            words = [w for w in words_punct if w not in string.punctuation]
            n_words = len(words)

            stats[language]["total_docs"] += 1
            stats[language]["total_words"] += n_words
            stats[language]["total_bytes"] += len(text.encode("utf-8"))

            ## Gopher quality filters
            # Distribution of word lengths
            for word in words:
                stats[language]["length_counter"][len(word)] += 1
                stats[language]["word_counter"][word.lower()] += 1

            # Compute hash to word ratio and ellipsis to word ratio
            hash_word_ratio = (text.count("#") / n_words) if n_words > 0 else 0
            stats[language]["hash_word_ratio"].append(hash_word_ratio)
            ellipsis_word_ratio = ((text.count("...") + text.count("…")) / n_words) if n_words > 0 else 0
            stats[language]["ellipsis_word_ratio"].append(ellipsis_word_ratio)

            # Compute ratio of lines starting with a bullet and ratio of lines ending in an ellipsis
            lines = text.splitlines()
            n_lines = len(lines)
            bullet_start_ratio = (
                (sum(s.lstrip().startswith("•") or s.lstrip().startswith("-") for s in lines) / n_lines)
                if n_lines > 0
                else 0
            )
            stats[language]["bullet_start_ratio"].append(bullet_start_ratio)
            ellipsis_end_ratio = (
                (sum(s.rstrip().endswith("...") or s.rstrip().endswith("…") for s in lines) / n_lines)
                if n_lines > 0
                else 0
            )
            stats[language]["ellipsis_end_ratio"].append(ellipsis_end_ratio)

            # Compute ratio of words in the document that contain at least one alphabetic character
            alpha_ratio = (sum([any((c.isalpha() for c in w)) for w in words]) / n_words) if n_words > 0 else 0
            stats[language]["alpha_ratio"].append(alpha_ratio)

            ## FineWeb filters
            lines = doc.text.split("\n")
            lines = [line for line in lines if line.strip() != ""]
            n_lines = len(lines)

            # Compute ratio ratio of lines ending in punctuation
            stop_chars = (
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
            )  # FineWeb + Spacy sentencizer stop chars
            line_punct_ratio = sum(1 for line in lines if line.endswith(stop_chars)) / n_lines if n_lines > 0 else 0
            stats[language]["line_punct_ratio"].append(line_punct_ratio)

            # Compute ratio of "short" lines (<=30 characters)
            short_line_ratio = sum(1 for line in lines if len(line) <= 30) / n_lines if n_lines > 0 else 0
            stats[language]["short_line_ratio"].append(short_line_ratio)

            # Compute ratio of duplicated lines
            n_chars = len(doc.text.replace("\n", ""))
            duplicate_line_ratio = find_duplicates(lines)[1] / n_chars if n_chars > 0 else 0
            stats[language]["duplicate_line_ratio"].append(duplicate_line_ratio)

            # Compute ratio of newlines to words
            new_line = doc.text.count("\n")
            n_words_punct = len(words_punct)
            new_line_ratio = new_line / n_words_punct if n_words_punct > 0 else 0
            stats[language]["new_line_ratio"].append(new_line_ratio)

            ## Gopher repetition filters

            # Compute paragraph repetition ratios
            n_text = len(text)

            paragraphs = self.paragraph_exp.split(text.strip())
            paragraphs_duplicates, char_duplicates = find_duplicates(paragraphs)
            n_paragraphs = len(paragraphs)
            stats[language]["dup_para_frac"] = paragraphs_duplicates / n_paragraphs if n_paragraphs > 0 else 0
            stats[language]["dup_para_char_frac"] = char_duplicates / n_text if n_text > 0 else 0

            # Compute line repetition ratios
            lines = self._line_splitter.split(text)
            line_duplicates, char_duplicates = find_duplicates(lines)
            n_lines = len(lines)
            stats[language]["dup_line_frac"] = line_duplicates / n_lines if n_lines > 0 else 0
            stats[language]["dup_line_char_frac"] = char_duplicates / n_text if n_text > 0 else 0

            # Compute top n-gram repetition ratios
            for n in [2, 3, 4]:
                n_grams = get_n_grams(words_punct, n)
                if not n_grams:
                    continue
                top_char_length = find_top_duplicate(n_grams)
                stats[language][f"top_{n}_gram"] = top_char_length / n_text if n_text > 0 else 0

            # Compute duplicated n-gram repetition ratios
            for n in [5, 6, 7, 8, 9, 10]:
                n_duplicates_char = find_all_duplicate(words_punct, n)
                stats[language][f"duplicated_{n}_grams"] = n_duplicates_char / n_text if n_text > 0 else 0

            yield doc

        for language in stats:
            # Calculate local mean and mean of squares
            for key in MEAN_STD_KEYS:
                values = np.array(stats[language][key])
                stats[language][f"{key}_mean"] = np.mean(values)
                stats[language][f"{key}_sq_mean"] = np.mean(values**2)
                del stats[language][key]
            # Prune word counter (include only words that appear at least word_count_prune times)
            if self.word_count_prune is not None:
                word_counter = stats[language]["word_counter"]
                stats[language]["word_counter"] = Counter(
                    {k: v for k, v in word_counter.items() if v >= self.word_count_prune}
                )

        # save to disk
        for language in stats:
            with self.output_folder.open(f"/{language}/{rank:05d}.json", "wt") as f:
                json.dump(
                    {language: stats[language]},
                    f,
                )


class LanguageStatsReducer(PipelineStep):
    type = "📊 - STATS"
    name = "🌐 Language stats reducer"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        map_fn: Callable[[LanguageStatistics], dict] = lambda ls: ls.to_dict(),
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.map_fn = map_fn

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        stats = {}
        doc_count = {}

        # combine all json files with stats
        assert world_size == 1, "world_size must be 1 when getting the input from an input_folder"
        for file in self.input_folder.list_files(glob_pattern="**/*.json"):
            with self.input_folder.open(file, "rt") as f:
                file_data = json.load(f)
                for language in file_data:
                    if language not in stats:
                        stats[language] = {
                            "length_counter": Counter(),
                            "word_counter": Counter(),
                            "total_words": 0,
                            "total_docs": 0,
                            "total_bytes": 0,
                        }
                        for key in MEAN_STD_KEYS:
                            stats[language][f"{key}_mean"] = []
                            stats[language][f"{key}_std"] = []
                    if language not in doc_count:
                        doc_count[language] = []
                    stats[language]["total_words"] += file_data[language]["total_words"]
                    stats[language]["total_docs"] += file_data[language]["total_docs"]
                    stats[language]["total_bytes"] += file_data[language]["total_bytes"]
                    length_counter = Counter({int(k): v for k, v in file_data[language]["length_counter"].items()})
                    stats[language]["length_counter"] += length_counter
                    stats[language]["word_counter"] += file_data[language]["word_counter"]
                    for key in MEAN_STD_KEYS:
                        stats[language][f"{key}_mean"].append(file_data[language][f"{key}_mean"])
                        stats[language][f"{key}_std"].append(file_data[language][f"{key}_sq_mean"])
                    doc_count[language].append(file_data[language]["total_docs"])

        for language in stats:
            # Average statistics over documents
            for key in MEAN_STD_KEYS:
                E_X = np.average(stats[language][f"{key}_mean"], weights=doc_count[language])
                E_X2 = np.average(stats[language][f"{key}_std"], weights=doc_count[language])
                stats[language][f"{key}_mean"] = E_X
                stats[language][f"{key}_std"] = np.sqrt(E_X2 - (E_X**2))

        stats = {
            k: LanguageStatistics(
                language=k,
                word_counter=Counter(v["word_counter"]).most_common(
                    10_000
                ),  # 10000 most common words pruning, TODO: remove
                length_counter=Counter(v["length_counter"]),
                total_bytes=int(v["total_bytes"]),
                total_docs=int(v["total_docs"]),
                total_words=int(v["total_words"]),
                **{
                    key: MeanStdFloat(mean=float(v[f"{key}_mean"]), std=float(v[f"{key}_std"]))
                    for key in MEAN_STD_KEYS
                },
            )
            for k, v in stats.items()
        }

        # Apply mapping function
        stats = {k: self.map_fn(v) for k, v in stats.items()}

        # Save stats
        for language in stats:
            with self.output_folder.open(f"{language}.yml", "wt") as f:
                yaml.safe_dump(
                    stats[language],
                    f,
                )
