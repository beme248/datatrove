from abc import ABC, abstractmethod
from typing import Callable

from datatrove.utils.typeshelper import Languages


class WordTokenizer(ABC):
    @abstractmethod
    def word_tokenize(self, text: str) -> list[str]:
        pass


class StanzaWordTokenizer(WordTokenizer):
    def __init__(self, stanza_language: str):
        self.stanza_language = stanza_language
        self._tokenizer = None

    @property
    def tokenizer(self):
        import stanza

        if self._tokenizer is None:
            self._tokenizer = stanza.Pipeline(self.stanza_language, processors="tokenize")
        return self._tokenizer

    def word_tokenize(self, text) -> list[str]:
        doc = self.tokenizer(text)
        tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
        return tokens


class NLTKTokenizer(WordTokenizer):
    def __init__(self, punkt_language: str):
        super().__init__()
        self.punkt_language = punkt_language

    def word_tokenize(self, text) -> list[str]:
        from nltk.tokenize import word_tokenize

        return word_tokenize(text, language=self.punkt_language)


class SpaCyTokenizer(WordTokenizer):
    def __init__(self, spacy_language: str, config=None):
        super().__init__()
        import spacy

        if config is None:
            self.tokenizer = spacy.blank(spacy_language)
        else:
            self.tokenizer = spacy.blank(spacy_language, config=config)

    def word_tokenize(self, text) -> list[str]:
        self.tokenizer.max_length = len(text) + 10
        return [
            token.text
            for token in self.tokenizer(text, disable=["parser", "tagger", "ner"])
            if len(token.text.strip()) > 0
        ]


class KiwiTokenizer(WordTokenizer):
    def __init__(self, model_type="sbg"):
        from kiwipiepy import Kiwi

        self.kiwi = Kiwi(model_type=model_type)

    def word_tokenize(self, text) -> list[str]:
        return [token.form for token in self.kiwi.tokenize(text)]


class ThaiTokenizer(WordTokenizer):
    def word_tokenize(self, text) -> list[str]:
        from pythainlp.tokenize import word_tokenize as th_word_tokenize

        return [
            token.strip()
            for token in th_word_tokenize(text, keep_whitespace=False, engine="newmm-safe")
            if len(token.strip()) > 0
        ]


class GeorgianTokenizer(WordTokenizer):
    def word_tokenize(self, text) -> list[str]:
        from anbani.nlp.preprocessing import word_tokenize as ka_word_tokenize

        return ka_word_tokenize(text)


class PashtoTokenizer(WordTokenizer):
    def __init__(self):
        from nlpashto import Tokenizer as PsTokenizer

        self.tokenizer = PsTokenizer()

    def word_tokenize(self, text) -> list[str]:
        return [tok for sent in self.tokenizer.tokenize(text) for tok in sent]


class IndicNLPTokenizer(WordTokenizer):
    def __init__(self, language: str):
        self.language = language

    def word_tokenize(self, text) -> list[str]:
        from indicnlp.tokenize.indic_tokenize import trivial_tokenize as indicnlp_trivial_tokenize

        return [token.strip() for token in indicnlp_trivial_tokenize(text, self.language) if len(token.strip()) > 0]


class MultilingualTokenizer:
    def __init__(self, factory_dict: dict[str, Callable[[], WordTokenizer]]):
        self._factory_dict = factory_dict
        self._tokenizers = {}

    def _get_tokenizer(self, language: str) -> WordTokenizer:
        if language not in self._tokenizers:
            if language not in self._factory_dict:
                raise ValueError(f"'{language}' tokenizer is not set.")
            tokenizer = self._factory_dict[language]()
            self._tokenizers[language] = tokenizer
        return self._tokenizers[language]

    @property
    def languages(self) -> list[str]:
        return list(self._factory_dict.keys())

    def word_tokenize(self, text: str, language: str) -> list[str]:
        return self._get_tokenizer(language).word_tokenize(text)


WORD_TOKENIZER_FACTORY: dict[str, Callable[[], WordTokenizer]] = {
    Languages.english: lambda: NLTKTokenizer("english"),
    Languages.german: lambda: NLTKTokenizer("german"),
    Languages.french: lambda: NLTKTokenizer("french"),
    Languages.czech: lambda: NLTKTokenizer("czech"),
    Languages.danish: lambda: NLTKTokenizer("danish"),
    Languages.dutch: lambda: NLTKTokenizer("dutch"),
    Languages.estonian: lambda: NLTKTokenizer("estonian"),
    Languages.finnish: lambda: NLTKTokenizer("finnish"),
    Languages.greek: lambda: NLTKTokenizer("greek"),
    Languages.italian: lambda: NLTKTokenizer("italian"),
    Languages.malayalam: lambda: NLTKTokenizer("malayalam"),
    Languages.norwegian: lambda: NLTKTokenizer("norwegian"),
    Languages.polish: lambda: NLTKTokenizer("polish"),
    Languages.portuguese: lambda: NLTKTokenizer("portuguese"),
    Languages.russian: lambda: NLTKTokenizer("russian"),
    Languages.slovenian: lambda: NLTKTokenizer("slovene"),
    Languages.spanish: lambda: NLTKTokenizer("spanish"),
    Languages.swedish: lambda: NLTKTokenizer("swedish"),
    Languages.turkish: lambda: NLTKTokenizer("turkish"),
    Languages.chinese: lambda: SpaCyTokenizer("zh", {"nlp": {"tokenizer": {"segmenter": "jieba"}}}),
    Languages.japanese: lambda: StanzaWordTokenizer("ja"),
    Languages.vietnamese: lambda: SpaCyTokenizer("vi"),
    Languages.indonesian: lambda: SpaCyTokenizer("id"),
    Languages.persian: lambda: SpaCyTokenizer("fa"),
    Languages.korean: lambda: KiwiTokenizer(),
    Languages.arabic: lambda: SpaCyTokenizer("ar"),
    Languages.hindi: lambda: SpaCyTokenizer("hi"),
    Languages.tamil: lambda: SpaCyTokenizer("ta"),
    Languages.urdu: lambda: SpaCyTokenizer("ur"),
    Languages.marathi: lambda: SpaCyTokenizer("mr"),
    Languages.telugu: lambda: SpaCyTokenizer("te"),
    Languages.hungarian: lambda: SpaCyTokenizer("hu"),
    Languages.romanian: lambda: SpaCyTokenizer("ro"),
    Languages.ukrainian: lambda: SpaCyTokenizer("uk"),
    Languages.slovak: lambda: SpaCyTokenizer("sk"),
    Languages.bulgarian: lambda: SpaCyTokenizer("bg"),
    Languages.catalan: lambda: SpaCyTokenizer("ca"),
    Languages.croatian: lambda: SpaCyTokenizer("hr"),
    Languages.latin: lambda: SpaCyTokenizer("la"),
    Languages.serbian: lambda: SpaCyTokenizer("sr"),
    Languages.lithuanian: lambda: SpaCyTokenizer("lt"),
    Languages.hebrew: lambda: SpaCyTokenizer("he"),
    Languages.latvian: lambda: SpaCyTokenizer("lv"),
    Languages.icelandic: lambda: SpaCyTokenizer("is"),
    Languages.armenian: lambda: SpaCyTokenizer("hy"),
    Languages.basque: lambda: SpaCyTokenizer("eu"),
    Languages.thai: lambda: ThaiTokenizer(),
    Languages.georgian: lambda: GeorgianTokenizer(),
    Languages.tagalog: lambda: SpaCyTokenizer("tl"),
    Languages.albanian: lambda: SpaCyTokenizer("sq"),
    Languages.macedonian: lambda: SpaCyTokenizer("mk"),
    Languages.azerbaijani: lambda: SpaCyTokenizer("az"),
    Languages.amharic: lambda: SpaCyTokenizer("am"),
    Languages.bengali: lambda: SpaCyTokenizer("bn"),
    Languages.malay: lambda: SpaCyTokenizer("ms"),
    Languages.urdu: lambda: SpaCyTokenizer("ur"),
    Languages.nepali: lambda: SpaCyTokenizer("ne"),
    Languages.kazakh: lambda: StanzaWordTokenizer("kk"),
    Languages.gujarati: lambda: SpaCyTokenizer("gu"),
    Languages.kannada: lambda: SpaCyTokenizer("kn"),
    Languages.welsh: lambda: StanzaWordTokenizer("cy"),
    Languages.norwegian_nynorsk: lambda: NLTKTokenizer("norwegian"),
    Languages.sinhala: lambda: SpaCyTokenizer("si"),
    Languages.tatar: lambda: SpaCyTokenizer("tt"),
    Languages.afrikaans: lambda: SpaCyTokenizer("af"),
    Languages.kirghiz: lambda: SpaCyTokenizer("ky"),
    Languages.irish: lambda: SpaCyTokenizer("ga"),
    Languages.luxembourgish: lambda: SpaCyTokenizer("lb"),
    Languages.maltese: lambda: StanzaWordTokenizer("mt"),
    Languages.sanskrit: lambda: SpaCyTokenizer("sa"),
    Languages.yoruba: lambda: SpaCyTokenizer("yo"),
    Languages.pashto: lambda: PashtoTokenizer(),
    Languages.serbocroatian: lambda: SpaCyTokenizer("sr"),
    Languages.oriya: lambda: IndicNLPTokenizer("or"),
    Languages.punjabi: lambda: IndicNLPTokenizer("sa"),
    Languages.assamese: lambda: IndicNLPTokenizer("as"),
    Languages.war: lambda: IndicNLPTokenizer("war"),
    Languages.sindhi: lambda: IndicNLPTokenizer("sd"),
    Languages.bosnian: lambda: SpaCyTokenizer("hr"),  # Proxy
    Languages.belarusian: lambda: SpaCyTokenizer("uk"),  # Proxy
    Languages.galician: lambda: NLTKTokenizer("portuguese"),  # Proxy
    Languages.esperanto: lambda: NLTKTokenizer("english"),  # Proxy
    Languages.occitan: lambda: NLTKTokenizer("italian"),  # Proxy
    Languages.cebuano: lambda: NLTKTokenizer("english"),  # Proxy
    Languages.swahili: lambda: NLTKTokenizer("english"),  # Proxy
    Languages.javanese: lambda: NLTKTokenizer("english"),  # Proxy
    Languages.uzbek: lambda: NLTKTokenizer("turkish"),  # Proxy, alternative ru
    Languages.tajik: lambda: SpaCyTokenizer("ru"),  # Proxy
    Languages.kurdish: lambda: NLTKTokenizer("english"),  # Proxy, multiple scripts!
    Languages.sorani: lambda: SpaCyTokenizer("fa"),  # Proxy
    Languages.south_azerbaijani: lambda: SpaCyTokenizer("fa"),  # Proxy
    Languages.bashkir: lambda: SpaCyTokenizer("tt"),  # Proxy
    Languages.western_frisian: lambda: NLTKTokenizer("dutch"),  # Proxy
    Languages.breton: lambda: StanzaWordTokenizer("cy"),  # Proxy
    Languages.malagasy: lambda: NLTKTokenizer("english"),  # Proxy
    Languages.yiddish: lambda: SpaCyTokenizer("he"),  # Proxy
    Languages.somali: lambda: NLTKTokenizer("english"),  # Proxy
    Languages.turkmen: lambda: NLTKTokenizer("turkish"),  # Proxy
}

default_tokenizer = MultilingualTokenizer(WORD_TOKENIZER_FACTORY)
