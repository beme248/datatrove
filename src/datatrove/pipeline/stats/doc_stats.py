import json

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import DocumentsPipeline, PipelineStep


class DocumentStats(PipelineStep):
    type = "📊 - STATS"
    name = "🌐 Document statistics"

    def __init__(self, output_folder: DataFolderLike, language_field: str = "language"):
        super().__init__()
        self.language_field = language_field
        self.output_folder = get_datafolder(output_folder)

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        stats = {}

        # map and produce one output file per rank
        for doc in data:
            language = doc.metadata.get(self.language_field)
            if language not in stats:
                stats[language] = {
                    "total_docs": 0,
                    "total_bytes": 0,
                }

            text = doc.text
            stats[language]["total_docs"] += 1
            stats[language]["total_bytes"] += len(text.encode("utf-8"))

        # save to disk
        with self.output_folder.open(f"{rank:05d}_lang_stats.json", "wt") as f:
            json.dump(
                stats,
                f,
            )


class DocumentStatsReducer(PipelineStep):
    type = "📊 - STATS"
    name = "🌐 Document statistics reducer"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        output_file_name: str,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.output_file_name = output_file_name

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        stats = {}

        # combine all json files with stats
        assert (
            world_size == 1
        ), "world_size must be 1 when getting the input from an input_folder"
        for file in self.input_folder.list_files(glob_pattern="**.json"):
            with self.input_folder.open(file, "rt") as f:
                file_data = json.load(f)
                for language in file_data:
                    if language not in stats:
                        stats[language] = {
                            "total_docs": 0,
                            "total_bytes": 0,
                        }
                    stats[language]["total_docs"] += file_data[language]["total_docs"]
                    stats[language]["total_bytes"] += file_data[language]["total_bytes"]

        # save stats
        with self.output_folder.open(self.output_file_name, "wt") as f:
            json.dump(
                stats,
                f,
            )
