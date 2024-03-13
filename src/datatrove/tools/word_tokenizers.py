from abc import ABC, abstractmethod

import stanza
from anbani.nlp.preprocessing import word_tokenize as ka_word_tokenize
from etnltk.tokenize.am import word_tokenize as am_word_tokenize
from nltk.tokenize import word_tokenize

from datatrove.utils.typeshelper import Languages


class WordTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass


class StanzaWordTokenizer(WordTokenizer):
    def __init__(self, stanza_language: str):
        self.stanza_language = stanza_language
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = stanza.Pipeline(self.stanza_language, processors="tokenize")
        return self._tokenizer

    def tokenize(self, text) -> list[str]:
        doc = self.tokenizer(text)
        tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
        return tokens


class NLTKTokenizer(WordTokenizer):
    def __init__(self, punkt_language: str):
        self.punkt_language = punkt_language

    def tokenize(self, text) -> list[str]:
        return word_tokenize(text, language=self.punkt_language)


class AmharicTokenizer(WordTokenizer):
    def tokenize(self, text) -> list[str]:
        return am_word_tokenize(text)


class GeorgianTokenizer(WordTokenizer):
    def tokenize(self, text) -> list[str]:
        return ka_word_tokenize(text)


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
    Languages.portuguese: NLTKTokenizer("portuguese"),
    Languages.russian: NLTKTokenizer("russian"),
    Languages.slovenian: NLTKTokenizer("slovene"),
    Languages.spanish: NLTKTokenizer("spanish"),
    Languages.swedish: NLTKTokenizer("swedish"),
    Languages.turkish: NLTKTokenizer("turkish"),
    Languages.chinese: StanzaWordTokenizer("zh"),
    Languages.japanese: StanzaWordTokenizer("ja"),
    Languages.vietnamese: StanzaWordTokenizer("vi"),
    Languages.indonesian: StanzaWordTokenizer("id"),
    Languages.persian: StanzaWordTokenizer("fa"),
    Languages.korean: StanzaWordTokenizer("ko"),
    Languages.arabic: StanzaWordTokenizer("ar"),
    Languages.hindi: StanzaWordTokenizer("hi"),
    Languages.tamil: StanzaWordTokenizer("ta"),
    Languages.urdu: StanzaWordTokenizer("ur"),
    Languages.marathi: StanzaWordTokenizer("mr"),
    Languages.telugu: StanzaWordTokenizer("te"),
    Languages.hungarian: StanzaWordTokenizer("hu"),
    Languages.romanian: StanzaWordTokenizer("ro"),
    Languages.ukrainian: StanzaWordTokenizer("uk"),
    Languages.slovak: StanzaWordTokenizer("sk"),
    Languages.bulgarian: StanzaWordTokenizer("bg"),
    Languages.catalan: StanzaWordTokenizer("ca"),
    Languages.croatian: StanzaWordTokenizer("hr"),
    Languages.latin: StanzaWordTokenizer("la"),
    Languages.serbian: StanzaWordTokenizer("sr"),
    Languages.lithuanian: StanzaWordTokenizer("lt"),
    Languages.hebrew: StanzaWordTokenizer("he"),
    Languages.latvian: StanzaWordTokenizer("lv"),
    Languages.serbocroatian: StanzaWordTokenizer("sr"),
    Languages.icelandic: StanzaWordTokenizer("is"),
    Languages.galician: StanzaWordTokenizer("gl"),
    Languages.armenian: StanzaWordTokenizer("hy"),
    Languages.basque: StanzaWordTokenizer("eu"),
    Languages.thai: StanzaWordTokenizer("th"),
    Languages.tagalog: NLTKTokenizer("english"),  # Proxy language
    Languages.albanian: StanzaWordTokenizer("sr"),  # Proxy language
    Languages.macedonian: StanzaWordTokenizer("sr"),  # Proxy language
    Languages.azerbaijani: NLTKTokenizer("turkish"),  # Proxy language
    Languages.georgian: GeorgianTokenizer(),
    Languages.amharic: AmharicTokenizer(),
}
# Missing: bn, gu, kn, pa
# Missing: sw, yo
# Missing: jv, ms
# Missing: bh (deprecated in ISO-639)


def get_word_tokenizer(language: Languages | str):
    if language in WORD_TOKENIZERS:
        return WORD_TOKENIZERS[language]
    else:
        return WORD_TOKENIZERS[Languages.english]
