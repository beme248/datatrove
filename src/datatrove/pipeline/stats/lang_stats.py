import json
import string
from collections import Counter

import numpy as np

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import DocumentsPipeline, PipelineStep
from datatrove.tools.word_tokenizers import get_word_tokenizer


class LanguageStats(PipelineStep):
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
                    "hash_word_ratio": [],
                    "ellipsis_word_ratio": [],
                    "bullet_start_ratio": [],
                    "ellipsis_end_ratio": [],
                    "alpha_ratio": [],
                }

            tokenizer = get_word_tokenizer(language)
            text = doc.text
            words = tokenizer.tokenize(text)
            words = [w for w in words if w not in string.punctuation]
            n_words = len(words)

            stats[language]["total_docs"] += 1
            stats[language]["total_words"] += n_words

            # Distribution of word lengths
            for word in words:
                stats[language]["length_counter"][len(word)] += 1
                stats[language]["word_counter"][word.lower()] += 1

            # Compute hash to word ratio and ellipsis to word ratio
            hash_word_ratio = text.count("#") / n_words
            stats[language]["hash_word_ratio"].append(hash_word_ratio)
            ellipsis_word_ratio = (text.count("...") + text.count("…")) / n_words
            stats[language]["ellipsis_word_ratio"].append(ellipsis_word_ratio)

            # Compute ratio of lines starting with a bullet and ratio of lines ending in an ellipsis
            lines = text.splitlines()
            bullet_start_ratio = sum(s.lstrip().startswith("•") or s.lstrip().startswith("-") for s in lines) / len(
                lines
            )
            stats[language]["bullet_start_ratio"].append(bullet_start_ratio)
            ellipsis_end_ratio = sum(s.rstrip().endswith("...") or s.rstrip().endswith("…") for s in lines) / len(
                lines
            )
            stats[language]["ellipsis_end_ratio"].append(ellipsis_end_ratio)

            # Compute ratio of words in the document that contain at least one alphabetic character
            alpha_ratio = sum([any((c.isalpha() for c in w)) for w in words]) / n_words
            stats[language]["alpha_ratio"].append(alpha_ratio)

            yield doc

        for language in stats:
            stats[language]["hash_word_ratio"] = np.mean(stats[language]["hash_word_ratio"])
            stats[language]["ellipsis_word_ratio"] = np.mean(stats[language]["ellipsis_word_ratio"])
            stats[language]["bullet_start_ratio"] = np.mean(stats[language]["bullet_start_ratio"])
            stats[language]["ellipsis_end_ratio"] = np.mean(stats[language]["ellipsis_end_ratio"])
            stats[language]["alpha_ratio"] = np.mean(stats[language]["alpha_ratio"])
            # Prune word counter (include only words that appear at least word_count_prune times)
            if self.word_count_prune is not None:
                word_counter = stats[language]["word_counter"]
                stats[language]["word_counter"] = Counter(
                    {k: v for k, v in word_counter.items() if v >= self.word_count_prune}
                )

        # save to disk
        with self.output_folder.open(f"{rank:05d}_lang_stats.json", "wt") as f:
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
        output_file_name: str,
        reduce_fn=None,  # (stats of language: dict) -> dict
        word_common_prune=500,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.output_file_name = output_file_name
        self.reduce_fn = reduce_fn
        self.word_common_prune = word_common_prune

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        stats = {}
        doc_count = {}

        # combine all json files with stats
        assert world_size == 1, "world_size must be 1 when getting the input from an input_folder"
        for file in self.input_folder.list_files(glob_pattern="**.json"):
            with self.input_folder.open(file, "rt") as f:
                file_data = json.load(f)
                for language in file_data:
                    if language not in stats:
                        stats[language] = {
                            "length_counter": Counter(),
                            "word_counter": Counter(),
                            "total_words": 0,
                            "total_docs": 0,
                            "hash_word_ratio": [],
                            "ellipsis_word_ratio": [],
                            "bullet_start_ratio": [],
                            "ellipsis_end_ratio": [],
                            "alpha_ratio": [],
                        }
                    if language not in doc_count:
                        doc_count[language] = []
                    stats[language]["total_words"] += file_data[language]["total_words"]
                    stats[language]["total_docs"] += file_data[language]["total_docs"]
                    length_counter = Counter({int(k): v for k, v in file_data[language]["length_counter"].items()})
                    stats[language]["length_counter"] += length_counter
                    stats[language]["word_counter"] += file_data[language]["word_counter"]
                    stats[language]["hash_word_ratio"].append((file_data[language]["hash_word_ratio"]))
                    stats[language]["ellipsis_word_ratio"].append((file_data[language]["ellipsis_word_ratio"]))
                    stats[language]["bullet_start_ratio"].append((file_data[language]["bullet_start_ratio"]))
                    stats[language]["ellipsis_end_ratio"].append((file_data[language]["ellipsis_end_ratio"]))
                    stats[language]["alpha_ratio"].append((file_data[language]["alpha_ratio"]))
                    doc_count[language].append(file_data[language]["total_docs"])

        for language in stats:
            # Average statistics over documents
            stats[language]["hash_word_ratio"] = np.average(
                stats[language]["hash_word_ratio"], weights=doc_count[language]
            )
            stats[language]["ellipsis_word_ratio"] = np.average(
                stats[language]["ellipsis_word_ratio"], weights=doc_count[language]
            )
            stats[language]["bullet_start_ratio"] = np.average(
                stats[language]["bullet_start_ratio"], weights=doc_count[language]
            )
            stats[language]["ellipsis_end_ratio"] = np.average(
                stats[language]["ellipsis_end_ratio"], weights=doc_count[language]
            )
            stats[language]["alpha_ratio"] = np.average(stats[language]["alpha_ratio"], weights=doc_count[language])
            # Prune word counter (include only top word_common_prune words)
            if self.word_common_prune is not None:
                word_counter = stats[language]["word_counter"]
                stats[language]["word_counter"] = Counter(dict(word_counter.most_common(self.word_common_prune)))

        # Apply reduction function
        if self.reduce_fn is not None:
            for language in stats:
                stats[language] = stats[language] | self.reduce_fn(stats[language])

        # save stats
        with self.output_folder.open(self.output_file_name, "wt") as f:
            json.dump(
                stats,
                f,
            )
