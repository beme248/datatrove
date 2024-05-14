from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.tools.word_tokenizers import MultilingualTokenizer, default_tokenizer


class ListFilter(BaseFilter):
    """
    Checks the ratio of number of lines to number of words.
    Equivalent to around a min of 3.333 words per line

    """

    name = "🎅 List"
    _requires_dependencies = ["nltk"]

    def __init__(
        self,
        new_line_ratio: float | None = 0.3,
        exclusion_writer: DiskWriter = None,
        tokenizer: MultilingualTokenizer = default_tokenizer,
    ):  # TODO better tune
        """ """
        super().__init__(exclusion_writer)
        self.new_line_ratio = new_line_ratio
        self.tokenizer = tokenizer

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Applies heuristic rules to decide if a document should be REMOVED
        Args:
            doc

        Returns:
            False if sample.text is a list
        """

        text = doc.text
        lang = doc.metadata["language"]

        words = self.tokenizer.word_tokenize(text, lang)
        new_line = text.count("\n")
        if new_line / len(words) > self.new_line_ratio:
            return False, "Suspected list"

        return True
