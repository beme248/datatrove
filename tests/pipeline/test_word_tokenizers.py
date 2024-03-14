import unittest
from datatrove.tools.word_tokenizers import WORD_TOKENIZERS

sample_text = "Hello world! ქართული " * 500

class TestWordTokenizers(unittest.TestCase):
    def test_word_tokenizers(self):
        for language in WORD_TOKENIZERS:
            tokenizer = WORD_TOKENIZERS[language]
            assert len(tokenizer.tokenize(sample_text)) >= 1, f"'{language}' tokenizer assertion failed"

