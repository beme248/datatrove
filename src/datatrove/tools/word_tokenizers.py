from abc import ABC, abstractmethod

from nltk.tokenize import word_tokenize

from datatrove.utils.typeshelper import Languages

import stanza


class WordTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass


class StanzaWordTokenizer(WordTokenizer):
    def __init__(self, stanza_language: str):
        self.tokenizer = stanza.Pipeline(
            stanza_language, processors="tokenize")

    def tokenize(self, text) -> list[str]:
        doc = self.tokenizer(text)
        tokens = [
            token.text for sentence in doc.sentences for token in sentence.tokens]
        return tokens


class NLTKTokenizer(WordTokenizer):
    def __init__(self, punkt_language: str):
        self.punkt_language = punkt_language

    def tokenize(self, text) -> list[str]:
        return word_tokenize(text, language=self.punkt_language)


WORD_TOKENIZERS = {
    Languages.english: NLTKTokenizer("english"),
    Languages.german: NLTKTokenizer("german"),
    Languages.french: NLTKTokenizer("french"),
    Languages.czech: NLTKTokenizer("czech"),
    Languages.danish: NLTKTokenizer("danish"),
    Languages.dutch: NLTKTokenizer("dutch"),
    Languages.estonian: NLTKTokenizer("estonian"),
    Languages.finnish: NLTKTokenizer("finnish"),
    Languages.greek: NLTKTokenizer("greek"),
    Languages.italian: NLTKTokenizer("italian"),
    Languages.malayalam: NLTKTokenizer("malayalam"),
    Languages.norwegian: NLTKTokenizer("norwegian"),
    Languages.polish: NLTKTokenizer("polish"),
    Languages.portuguese: NLTKTokenizer("portugese"),
    Languages.russian: NLTKTokenizer("russian"),
    Languages.slovenian: NLTKTokenizer("slovene"),
    Languages.spanish: NLTKTokenizer("spanish"),
    Languages.swedish: NLTKTokenizer("swedish"),
    Languages.turkish: NLTKTokenizer("turkish"),
    Languages.chinese: StanzaWordTokenizer('zh'),# TODO: lazy load stanza tokenizers
    Languages.japanese: StanzaWordTokenizer('ja') # TODO: lazy load stanza tokenizers
}


def get_word_tokenizer(language: Languages | str):
    if language in WORD_TOKENIZERS:
        return WORD_TOKENIZERS[language]
    else:
        return WORD_TOKENIZERS[Languages.english]
