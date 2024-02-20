import json
import string
from collections import Counter

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
    ):
        super().__init__()
        self.language_field = language_field
        self.output_folder = get_datafolder(output_folder)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        stats = {}

        # map and produce one output file per rank
        for doc in data:
            language = doc.metadata.get(self.language_field)
            if language not in stats:
                stats[language] = {
                    "length_counter": Counter(),
                    "total_tokens": 0,
                    "total_docs": 0,
                }
            tokenizer = get_word_tokenizer(language)
            words = tokenizer.tokenize(doc.text)
            words = [w for w in words if w not in string.punctuation]
            stats[language]["total_docs"] += 1
            stats[language]["total_tokens"] += len(words)
            for word in words:
                stats[language]["length_counter"][len(word)] += 1
            yield doc

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
        length_counter_reducer,  # (Counter) -> dict
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.output_file_name = output_file_name
        self.length_counter_reducer = length_counter_reducer

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        stats = {}

        # combine all json files with stats
        assert world_size == 1, "world_size must be 1 when getting the input from an input_folder"
        for file in self.input_folder.list_files(glob_pattern="**.json"):
            with self.input_folder.open(file, "rt") as f:
                file_data = json.load(f)
                for language in file_data:
                    if language not in stats:
                        stats[language] = {
                            "length_counter": Counter(),
                            "total_tokens": 0,
                            "total_docs": 0,
                        }
                    stats[language]["total_tokens"] += file_data[language]["total_tokens"]
                    stats[language]["total_docs"] += file_data[language]["total_docs"]
                    length_counter = Counter({int(k): v for k, v in file_data[language]["length_counter"].items()})
                    stats[language]["length_counter"] += length_counter

        # reduce collected stats
        out_stats = {}
        for language in stats:
            length_counter = stats[language]["length_counter"]
            out_stats[language] = self.length_counter_reducer(length_counter)

        # save stats
        with self.output_folder.open(self.output_file_name, "wt") as f:
            json.dump(
                out_stats,
                f,
            )
