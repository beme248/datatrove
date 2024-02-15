from nltk.tokenize import word_tokenize

from datatrove.utils.typeshelper import Languages


WORD_TOKENIZERS = {
    Languages.english: lambda t: word_tokenize(t, language="english"),
    Languages.german: lambda t: word_tokenize(t, language="german"),
    Languages.french: lambda t: word_tokenize(t, language="french"),
    Languages.czech: lambda t: word_tokenize(t, language="czech"),
    Languages.danish: lambda t: word_tokenize(t, language="danish"),
    Languages.dutch: lambda t: word_tokenize(t, language="dutch"),
    Languages.estonian: lambda t: word_tokenize(t, language="estonian"),
    Languages.finnish: lambda t: word_tokenize(t, language="finnish"),
    Languages.greek: lambda t: word_tokenize(t, language="greek"),
    Languages.italian: lambda t: word_tokenize(t, language="italian"),
    Languages.malayalam: lambda t: word_tokenize(t, language="malayalam"),
    Languages.norwegian: lambda t: word_tokenize(t, language="norwegian"),
    Languages.polish: lambda t: word_tokenize(t, language="polish"),
    Languages.portuguese: lambda t: word_tokenize(t, language="portuguese"),
    Languages.russian: lambda t: word_tokenize(t, language="russian"),
    Languages.slovenian: lambda t: word_tokenize(t, language="slovene"),
    Languages.spanish: lambda t: word_tokenize(t, language="spanish"),
    Languages.swedish: lambda t: word_tokenize(t, language="swedish"),
    Languages.turkish: lambda t: word_tokenize(t, language="turkish"),
}


def get_word_tokenizer(language: Languages | str):
    if language in WORD_TOKENIZERS:
        return WORD_TOKENIZERS[language]
    else:
        return WORD_TOKENIZERS[Languages.english]
