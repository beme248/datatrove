import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
import yaml

from datatrove.io import DataFolderLike, cached_asset_path_or_download, get_datafolder
from datatrove.pipeline.base import DocumentsPipeline, PipelineStep
from datatrove.pipeline.filters.gopher_repetition_filter import (
    find_all_duplicate,
    find_duplicates,
    find_top_duplicate,
    get_n_grams,
)
from datatrove.utils.text import PUNCTUATION_SET
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer


LANGUAGE_ID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

STATS_KEYS = [
    "avg_word_doc_length",
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
    "language_score",
]


@dataclass
class LanguageStatistics:
    word_counter: Counter
    length_counter: Counter
    total_words: int
    total_docs: int
    total_bytes: int
    avg_word_doc_length: list[float]
    hash_word_ratio: list[float]
    ellipsis_word_ratio: list[float]
    bullet_start_ratio: list[float]
    ellipsis_end_ratio: list[float]
    alpha_ratio: list[float]
    line_punct_ratio: list[float]
    short_line_ratio: list[float]
    duplicate_line_ratio: list[float]
    new_line_ratio: list[float]
    dup_para_frac: list[float]
    dup_para_char_frac: list[float]
    dup_line_frac: list[float]
    dup_line_char_frac: list[float]
    top_2_gram: list[float]
    top_3_gram: list[float]
    top_4_gram: list[float]
    duplicated_5_grams: list[float]
    duplicated_6_grams: list[float]
    duplicated_7_grams: list[float]
    duplicated_8_grams: list[float]
    duplicated_9_grams: list[float]
    duplicated_10_grams: list[float]
    language_score: list[float]

    def to_dict(self) -> dict:
        return {
            "word_counter": dict(self.word_counter),
            "length_counter": dict(self.length_counter),
            "total_words": self.total_words,
            "total_docs": self.total_docs,
            "total_bytes": self.total_bytes,
            **{k: self.__getattribute__(k) for k in STATS_KEYS},
        }


STOP_CHARS = (
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


class LanguageStatsCollector(PipelineStep):
    type = "📊 - STATS"
    name = "🌐 Stats collector"

    def __init__(
        self,
        output_folder: DataFolderLike,
        language: str = Languages.english,
        word_count_prune=2,
        stop_chars=STOP_CHARS,
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.word_count_prune = word_count_prune
        self.paragraph_exp = re.compile(r"\n{2,}")
        self._line_splitter = re.compile("\n+")
        self.tokenizer = load_word_tokenizer(language)
        self.language = language
        self._fasttext_model = None
        self.stop_chars = stop_chars

    @property
    def fasttext_model(self):
        if not self._fasttext_model:
            from fasttext.FastText import _FastText

            model_file = cached_asset_path_or_download(
                LANGUAGE_ID_MODEL_URL,
                namespace="filters",
                subfolder="language_filter",
                desc="fast-text language identifier model",
            )
            self._fasttext_model = _FastText(model_file)
        return self._fasttext_model

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        stats = {
            "length_counter": Counter(),
            "word_counter": Counter(),
            "total_words": 0,
            "total_docs": 0,
            "total_bytes": 0,
            **{k: [] for k in STATS_KEYS},
        }

        # map and produce one output file per rank
        for doc in data:
            text = doc.text
            words_symbols = self.tokenizer.word_tokenize(doc.text)
            words = [w for w in words_symbols if any(ch not in PUNCTUATION_SET for ch in w)]
            n_words = len(words)
            n_words_symbols = len(words_symbols)

            stats["total_docs"] += 1
            stats["total_words"] += n_words
            stats["total_bytes"] += len(text.encode("utf-8"))

            ## Gopher quality filters
            # Distribution of word lengths
            for word in words:
                stats["length_counter"][len(word)] += 1
                stats["word_counter"][word.lower()] += 1

            # Compute average word length
            avg_n_words = np.mean([len(w) for w in words]) if n_words > 0 else 0
            stats["avg_word_doc_length"].append(avg_n_words)

            # Compute hash to word ratio and ellipsis to word ratio
            hash_word_ratio = (text.count("#") / n_words_symbols) if n_words_symbols > 0 else 0
            stats["hash_word_ratio"].append(hash_word_ratio)
            ellipsis_word_ratio = (
                ((text.count("...") + text.count("…")) / n_words_symbols) if n_words_symbols > 0 else 0
            )
            stats["ellipsis_word_ratio"].append(ellipsis_word_ratio)

            # Compute ratio of lines starting with a bullet and ratio of lines ending in an ellipsis
            lines = text.splitlines()
            n_lines = len(lines)
            bullet_start_ratio = (
                (sum(s.lstrip().startswith("•") or s.lstrip().startswith("-") for s in lines) / n_lines)
                if n_lines > 0
                else 0
            )
            stats["bullet_start_ratio"].append(bullet_start_ratio)
            ellipsis_end_ratio = (
                (sum(s.rstrip().endswith("...") or s.rstrip().endswith("…") for s in lines) / n_lines)
                if n_lines > 0
                else 0
            )
            stats["ellipsis_end_ratio"].append(ellipsis_end_ratio)

            # Compute ratio of words in the document that contain at least one alphabetic character
            alpha_ratio = (
                (sum([any((c.isalpha() for c in w)) for w in words_symbols]) / n_words_symbols)
                if n_words_symbols > 0
                else 0
            )
            stats["alpha_ratio"].append(alpha_ratio)

            ## FineWeb filters
            lines = doc.text.split("\n")
            lines = [line for line in lines if line.strip() != ""]
            n_lines = len(lines)

            # Compute ratio ratio of lines ending in punctuation
            stop_chars = self.stop_chars
            line_punct_ratio = sum(1 for line in lines if line.endswith(stop_chars)) / n_lines if n_lines > 0 else 0
            stats["line_punct_ratio"].append(line_punct_ratio)

            # Compute ratio of "short" lines (<=30 characters)
            short_line_ratio = sum(1 for line in lines if len(line) <= 30) / n_lines if n_lines > 0 else 0
            stats["short_line_ratio"].append(short_line_ratio)

            # Compute ratio of duplicated lines
            n_chars = len(doc.text.replace("\n", ""))
            duplicate_line_ratio = find_duplicates(lines)[1] / n_chars if n_chars > 0 else 0
            stats["duplicate_line_ratio"].append(duplicate_line_ratio)

            # Compute ratio of newlines to words
            new_line = doc.text.count("\n")
            n_words_symbols = len(words_symbols)
            new_line_ratio = new_line / n_words_symbols if n_words_symbols > 0 else 0
            stats["new_line_ratio"].append(new_line_ratio)

            ## Gopher repetition filters

            # Compute paragraph repetition ratios
            n_text = len(text)

            paragraphs = self.paragraph_exp.split(text.strip())
            paragraphs_duplicates, char_duplicates = find_duplicates(paragraphs)
            n_paragraphs = len(paragraphs)
            stats["dup_para_frac"].append(paragraphs_duplicates / n_paragraphs if n_paragraphs > 0 else 0)
            stats["dup_para_char_frac"].append(char_duplicates / n_text if n_text > 0 else 0)

            # Compute line repetition ratios
            lines = self._line_splitter.split(text)
            line_duplicates, char_duplicates = find_duplicates(lines)
            n_lines = len(lines)
            stats["dup_line_frac"].append(line_duplicates / n_lines if n_lines > 0 else 0)
            stats["dup_line_char_frac"].append(char_duplicates / n_text if n_text > 0 else 0)

            # Compute top n-gram repetition ratios
            for n in [2, 3, 4]:
                n_grams = get_n_grams(words_symbols, n)
                top_char_length = find_top_duplicate(n_grams) if n_grams else 0
                stats[f"top_{n}_gram"].append(top_char_length / n_text if n_text > 0 else 0)

            # Compute duplicated n-gram repetition ratios
            for n in [5, 6, 7, 8, 9, 10]:
                n_duplicates_char = find_all_duplicate(words_symbols, n)
                stats[f"duplicated_{n}_grams"].append(n_duplicates_char / n_text if n_text > 0 else 0)

            # Get language score
            labels, scores = self.fasttext_model.predict(
                doc.text.replace("\n", ""), k=len(self.fasttext_model.labels), threshold=-1
            )
            language_label_index = labels.index(f"__label__{self.language}")
            language_score = scores[language_label_index]
            stats["language_score"].append(language_score)

            yield doc

        # Prune word counter (include only words that appear at least word_count_prune times)
        if self.word_count_prune is not None:
            word_counter = stats["word_counter"]
            stats["word_counter"] = Counter({k: v for k, v in word_counter.items() if v >= self.word_count_prune})

        # save to disk
        with self.output_folder.open(f"{rank:05d}.json", "wt") as f:
            json.dump(
                stats,
                f,
            )


class LanguageStatsReducer(PipelineStep):
    type = "📊 - STATS"
    name = "🌐 Language stats reducer"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        output_filename: str,
        map_fn: Callable[[LanguageStatistics], dict] = lambda ls: ls.to_dict(),
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.output_filename = output_filename
        self.map_fn = map_fn

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        stats = {
            "length_counter": Counter(),
            "word_counter": Counter(),
            "total_words": 0,
            "total_docs": 0,
            "total_bytes": 0,
            **{k: [] for k in STATS_KEYS},
        }

        # combine all json files with stats
        assert world_size == 1, "world_size must be 1 when getting the input from an input_folder"
        for file in self.input_folder.list_files(glob_pattern="**/*.json"):
            with self.input_folder.open(file, "rt") as f:
                file_data = json.load(f)
                stats["total_words"] += file_data["total_words"]
                stats["total_docs"] += file_data["total_docs"]
                stats["total_bytes"] += file_data["total_bytes"]
                length_counter = Counter({int(k): v for k, v in file_data["length_counter"].items()})
                stats["length_counter"] += length_counter
                stats["word_counter"] += file_data["word_counter"]
                for key in STATS_KEYS:
                    stats[key] += file_data[key]

        stats = LanguageStatistics(
            word_counter=Counter(stats["word_counter"]).most_common(
                10_000
            ),  # 10000 most common words pruning, TODO: remove and retweak stopword threshold after
            length_counter=Counter(stats["length_counter"]),
            total_bytes=int(stats["total_bytes"]),
            total_docs=int(stats["total_docs"]),
            total_words=int(stats["total_words"]),
            **{key: stats[key] for key in STATS_KEYS},
        )

        # Apply mapping function
        stats = self.map_fn(stats)

        # Save stats
        with self.output_folder.open(self.output_filename, "wt") as f:
            yaml.safe_dump(
                stats,
                f,
            )
