from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.filters.gopher_repetition_filter import find_duplicates
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.tools.word_tokenizers import MultilingualTokenizer, default_tokenizer


class MultilingualFineWebQualityFilter(BaseFilter):
    name = "🍷 Multilingual FineWeb Quality"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
        line_punct_thr: dict[str, float] | None = None,
        line_punct_exclude_zero: dict[str, bool] | None = None,
        short_line_thr: dict[str, float] | None = None,
        short_line_length: dict[str, int] | None = None,
        char_duplicates_ratio: dict[str, float] | None = None,
        new_line_ratio: dict[str, float] | None = None,
        tokenizer: MultilingualTokenizer = default_tokenizer,
    ):
        super().__init__(exclusion_writer)
        self.line_punct_thr = line_punct_thr
        self.line_punct_exclude_zero = line_punct_exclude_zero
        self.short_line_threshold = short_line_thr
        self.short_line_length = short_line_length
        self.char_duplicates_ratio = char_duplicates_ratio
        self.new_line_ratio = new_line_ratio
        self.tokenizer = tokenizer

    def filter(self, doc) -> bool | tuple[bool, str]:
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

        language = doc.metadata["language"]

        lines = doc.text.split("\n")
        lines = [line for line in lines if line.strip() != ""]
        ratio = sum(1 for line in lines if line.endswith(stop_chars)) / len(lines)
        if (
            self.line_punct_thr
            and language in self.line_punct_thr
            and ratio <= self.line_punct_thr[language]
            and not (ratio == 0 and self.line_punct_exclude_zero and self.line_punct_exclude_zero.get(language, False))
        ):
            return False, "line_punct_ratio"

        short_line_length = self.short_line_length.get(language, 30) if self.short_line_length else 30
        ratio = sum(1 for line in lines if len(line) <= short_line_length) / len(lines)
        if (
            self.short_line_threshold
            and language in self.short_line_threshold
            and ratio >= self.short_line_threshold[language]
        ):
            return False, "short_line_ratio"

        ratio = find_duplicates(lines)[1] / len(doc.text.replace("\n", ""))

        if (
            self.char_duplicates_ratio
            and language in self.char_duplicates_ratio
            and ratio >= self.char_duplicates_ratio[language]
        ):
            return False, "char_dup_ratio"

        words = self.tokenizer.word_tokenize(doc.text, language)
        new_line = doc.text.count("\n")
        if (
            self.new_line_ratio
            and language in self.new_line_ratio
            and new_line / len(words) > self.new_line_ratio[language]
        ):
            return False, "list_ratio"

        return True
