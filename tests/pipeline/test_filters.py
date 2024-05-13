import unittest

from datatrove.data import Document
from datatrove.pipeline.filters import (
    GopherQualityFilter,
    GopherRepetitionFilter,
    LambdaFilter,
    LanguageFilter,
    MultilingualGopherQualityFilter,
    RegexFilter,
    UnigramLogProbFilter,
    URLFilter,
)

from ..utils import require_fasttext, require_nltk, require_tldextract


TEXT_LF_1 = (
    "I wish it need not have happened in my time,' said Frodo. 'So do I,' said Gandalf, 'and so do all who live to "
    "see such times. But that is not for them to decide. All we have to decide is what to do with the time that is "
    "given us.'"
)

TEXT_LF_2 = (
    "Un magicien n'est jamais en retard Frodon Sacquet. Pas plus qu'il est en avance. Il arrive précisément "
    "à l'heure prévue."
)

TEXT_LF_3 = "Um mago nunca chega tarde, Frodo Bolseiro. Nem cedo. Ele chega precisamente na hora que pretende."

TEXT_LF_4 = (
    "Molti tra i vivi meritano la morte. E parecchi che sono morti avrebbero meritato la vita. Sei forse tu in "
    "grado di dargliela? E allora non essere troppo generoso nel distribuire la morte nei tuoi giudizi: "
    "sappi che nemmeno i più saggi possono vedere tutte le conseguenze."
)


def get_doc(text, url=None, language="en"):
    return Document(text, id="0", metadata={"url": url, "language": language})


class TestFilters(unittest.TestCase):
    def check_filter(self, filter, doc, filter_reason):
        filter_result = filter.filter(doc)
        self.assertEqual(type(filter_result), tuple)
        self.assertEqual(filter_result[1], filter_reason)

    @require_nltk
    def test_gopher_repetition(self):
        gopher_repetition = GopherRepetitionFilter()

        self.check_filter(gopher_repetition, get_doc("I am your father.\n" * 4), "dup_line_frac")
        self.check_filter(gopher_repetition, get_doc("I am your father.\n\n" * 4), "dup_para_frac")
        text = "I am groot.\n\n" + "You are a wizard.\n\n" + "I am your father.\n\n" + f"{'x' * 30}.\n\n" * 2
        self.check_filter(gopher_repetition, get_doc(text), "dup_para_char_frac")
        doc = get_doc("I am groot.\n" + "You are a wizard.\n" + "I am your father.\n" + f"{'x' * 40}.\n" * 2)
        self.check_filter(gopher_repetition, doc, "dup_line_char_frac")
        self.check_filter(gopher_repetition, get_doc("I am Frank, I am Frank, I am Frank"), "top_2_gram")
        doc = get_doc("I am Frank, you are Jhon. I am Frank. I am Frank you are Jhon")
        self.check_filter(gopher_repetition, doc, "top_3_gram")
        doc = get_doc("I am Frank, you are Jhon. I am Frank. I am Frank you are Jhon")
        self.check_filter(gopher_repetition, doc, "top_3_gram")
        doc = get_doc("I am a solo traveller " * 4 + TEXT_LF_1)
        self.check_filter(gopher_repetition, doc, "duplicated_5_n_grams")

    def test_gopher_quality(self):
        gopher_quality = GopherQualityFilter(min_doc_words=10, max_doc_words=1000)
        self.check_filter(gopher_quality, get_doc("I am too small..."), "gopher_short_doc")
        self.check_filter(gopher_quality, get_doc("I am " * 20), "gopher_below_avg_threshold")
        self.check_filter(gopher_quality, get_doc("interconnection " * 20), "gopher_above_avg_threshold")
        self.check_filter(gopher_quality, get_doc("# comment " * 20), "gopher_too_many_hashes")
        self.check_filter(gopher_quality, get_doc("... comment " * 20), "gopher_too_many_ellipsis")
        text = "the ./!*?<><> apple <?////> orange  ++ interconnection !<>??? have" * 20
        self.check_filter(gopher_quality, get_doc(text), "gopher_below_alpha_threshold")

        self.check_filter(gopher_quality, get_doc("Have Have" * 20), "gopher_enough_stop_words")
        self.assertTrue(gopher_quality.filter(get_doc("have " * 20)) is True)
        self.assertTrue(gopher_quality(get_doc(TEXT_LF_1)))

    def test_multilingual_gopher_quality(self):
        min_avg_word_lengths = {"en": 2, "fr": 3}
        max_avg_word_lengths = {"en": 7, "fr": 9}
        stop_words = {"en": ["the", "of", "in"], "fr": ["le", "la", "des", "du", "en", "l'"]}
        min_stop_words = {"en": 1, "fr": 2}
        max_non_alpha_word_ratios = {"en": 0.8, "fr": 0.9}

        gopher_quality = MultilingualGopherQualityFilter(
            min_doc_words=10,
            max_doc_words=1000,
            min_avg_word_lengths=min_avg_word_lengths,
            max_avg_word_lengths=max_avg_word_lengths,
            stop_words=stop_words,
            min_stop_words=min_stop_words,
            max_non_alpha_words_ratio=max_non_alpha_word_ratios,
        )

        self.check_filter(gopher_quality, get_doc("I am too small...", language="en"), "gopher_short_doc")
        self.check_filter(gopher_quality, get_doc("I am too small...", language="fr"), "gopher_short_doc")

        self.check_filter(gopher_quality, get_doc("hi " * 20 + "h", language="en"), "gopher_below_avg_threshold")
        self.check_filter(gopher_quality, get_doc("des " * 20 + "la", language="fr"), "gopher_below_avg_threshold")

        self.check_filter(
            gopher_quality, get_doc("dynamic " * 20 + " computer", language="en"), "gopher_above_avg_threshold"
        )
        self.check_filter(
            gopher_quality, get_doc("mangeront " * 20 + " mangeronts", language="fr"), "gopher_above_avg_threshold"
        )

        self.check_filter(gopher_quality, get_doc("# comment " * 20, language="en"), "gopher_too_many_hashes")
        self.check_filter(gopher_quality, get_doc("# comment " * 20, language="fr"), "gopher_too_many_hashes")

        self.check_filter(gopher_quality, get_doc("... comment " * 20, language="en"), "gopher_too_many_ellipsis")
        self.check_filter(gopher_quality, get_doc("... comment " * 20, language="fr"), "gopher_too_many_ellipsis")

        self.check_filter(gopher_quality, get_doc("• comment\n" * 20, language="en"), "gopher_too_many_bullets")
        self.check_filter(gopher_quality, get_doc("• comment\n" * 20, language="fr"), "gopher_too_many_bullets")
        self.check_filter(gopher_quality, get_doc("- comment\n" * 20, language="en"), "gopher_too_many_bullets")
        self.check_filter(gopher_quality, get_doc("- comment\n" * 20, language="fr"), "gopher_too_many_bullets")

        self.check_filter(
            gopher_quality,
            get_doc("text text text text text text text text text text text text…\n" * 20, language="en"),
            "gopher_too_many_end_ellipsis",
        )
        self.check_filter(
            gopher_quality,
            get_doc(
                "comment comment comment comment comment comment comment comment comment comment comment comment…\n"
                * 20,
                language="fr",
            ),
            "gopher_too_many_end_ellipsis",
        )
        self.check_filter(
            gopher_quality,
            get_doc("text text text text text text text text text text text text...\n" * 20, language="en"),
            "gopher_too_many_end_ellipsis",
        )
        self.check_filter(
            gopher_quality,
            get_doc(
                "comment comment comment comment comment comment comment comment comment comment comment comment...\n"
                * 20,
                language="fr",
            ),
            "gopher_too_many_end_ellipsis",
        )

        text = "the ./!*?<><> apple <?////> orange  ++ interconnection !<>??? have" * 20
        self.check_filter(gopher_quality, get_doc(text, language="en"), "gopher_below_alpha_threshold")
        self.check_filter(gopher_quality, get_doc(text, language="fr"), "gopher_below_alpha_threshold")

        self.check_filter(gopher_quality, get_doc("le la des du " * 20, language="en"), "gopher_enough_stop_words")
        self.check_filter(gopher_quality, get_doc("the of in salut " * 20, language="fr"), "gopher_enough_stop_words")

        self.assertTrue(gopher_quality.filter(get_doc(TEXT_LF_1, language="en")) is True)
        self.assertTrue(gopher_quality.filter(get_doc(TEXT_LF_2, language="fr")) is True)

    def test_lambda(self):
        doc = Document(text=TEXT_LF_1, id="0", metadata={"test": 1})
        lambda_filter = LambdaFilter(filter_function=lambda doc: doc.metadata["test"] > 0)
        self.assertTrue(lambda_filter.filter(doc))
        doc.metadata["test"] = -1
        self.assertFalse(lambda_filter.filter(doc))

    @require_fasttext
    def test_language(self):
        language_filter = LanguageFilter(languages=("en", "it"))

        self.assertTrue(language_filter.filter(Document(text=TEXT_LF_1, id="0")))
        self.assertFalse(language_filter.filter(Document(text=TEXT_LF_2, id="0")))
        self.assertFalse(language_filter.filter(Document(text=TEXT_LF_3, id="0")))
        self.assertTrue(language_filter.filter(Document(text=TEXT_LF_4, id="0")))

    def test_regex(self):
        regex_filter = RegexFilter(regex_exp=r"(?i)copyright")
        self.assertFalse(regex_filter.filter(get_doc(TEXT_LF_1 + "\n\nCoPyRiGhT")))
        self.assertTrue(regex_filter.filter(get_doc(TEXT_LF_1)))

    @require_nltk
    def test_unigram_prob(self):
        unigram_filter = UnigramLogProbFilter(logprobs_threshold=-10)
        self.assertTrue(unigram_filter.filter(Document(text=TEXT_LF_1, id="0")))
        self.assertFalse(unigram_filter.filter(Document(text="Cacophony Pareidolia Serendipity", id="0")))

    @require_tldextract
    def test_url(self):
        url_filter = URLFilter(extra_domains=("blocked.com", "danger.org", "badsubdomain.nice.com"))

        for url, result in (
            ("https://blocked.com/some-sub-url?with=stuff", "domain"),
            ("https://hey.danger.org/some-sub-url?with=stuff", "domain"),
            ("http://hey.danger.org/some-sub-url?with=stuff", "domain"),
            ("http://www.danger.org/some-sub-url?with=stuff", "domain"),
            ("https://nice.com/some-sub-url?with=stuff", True),
            ("https://badsubdomain.nice.com/some-sub-url?with=stuff", "subdomain"),
            ("https://sdsd.badsubdomain.nice.com/some-sub-url?with=stuff", True),
            ("https://blocke.dcom/some-sub-url?with=stuff", True),
        ):
            doc = get_doc(TEXT_LF_1, url)
            if result is True:
                assert url_filter.filter(doc)
            else:
                self.check_filter(url_filter, doc, result)
