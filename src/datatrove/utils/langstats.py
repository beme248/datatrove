import random
import string
from collections import Counter

import numpy as np
from datasets import load_dataset
from nltk.tokenize import word_tokenize


# Wikipedia language -> NLTK Punkt tokenizer language mapping
TOKENIZER_LANGUAGES = {
    "cz": "czech",
    "da": "danish",
    "en": "english",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "it": "italian",
    "ml": "malayalam",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "si": "slovene",
    "es": "spanish",
    "sv": "swedish",
    "tr": "turkish",
}


def tokenize(documents: list[str], language: str) -> list[list[str]]:
    """Tokenizes a list of documents in a given language.

    Args:
        documents (list[str]): List of documents to tokenize
        language (str): language code (en, fr, de, ...)

    Returns:
        list[list[str]]: tokenized documents
    """
    # TODO: ja, zh, etc.
    ret = []
    if language in TOKENIZER_LANGUAGES:
        nltk_language = TOKENIZER_LANGUAGES[language]
    else:
        nltk_language = "english"
        print(f"[WARN] Tokenizer for '{language}' does not exist, using 'english' tokenizer.")

    for document in documents:
        words = word_tokenize(document, nltk_language)
        words = [w for w in words if w not in string.punctuation]
        ret.append(words)
    return ret


def random_sample(documents: list, n_samples: int) -> list:
    """Randomly samples `n_samples` documents.

    Args:
        documents (list): list of documents
        n_samples (int): number of samples

    Returns:
        list: list of randomly sampled documents
    """
    idxs = range(len(documents))
    sample = random.sample(idxs, n_samples)
    return [documents[i] for i in sample]


def compute_stats(tokenized_documents: list[list[str]]) -> dict:
    """_summary_

    Args:
        tokenized_documents (list[list[str]]): tokenized documents

    Returns:
        dict: statistics of the language (keys 'total_words', 'total_documents', 'word_length_mean', 'word_length_std')
    """
    length_counter = Counter()
    total_words = 0
    total_documents = 0
    for document in tokenized_documents:
        total_words += len(document)
        total_documents += 1
        for word in document:
            length_counter[len(word)] += 1

    lengths = list(length_counter.keys())
    freqs = list(length_counter.values())

    word_length_mean = np.average(lengths, weights=freqs)
    word_length_std = np.sqrt(np.cov(lengths, fweights=freqs))

    return {
        "total_words": total_words,
        "total_documents": total_documents,
        "word_length_mean": word_length_mean,
        "word_length_std": word_length_std,
    }


def compute_wikistats(language: str, n_samples: int | None = None, dump_version: str = "20231101") -> dict:
    """Computes language statistics for Wikipedia text of given language.

    Args:
        language (str): language code (en, fr, de, ...)
        n_samples (int | None, optional): number of samples for approximation. Defaults to None.
        dump_version (str, optional): version of Wikipedia dump (see https://huggingface.co/datasets/wikimedia/wikipedia). Defaults to "20231101".

    Returns:
        dict: statistics of the language from Wikipedia texts (keys 'total_words', 'total_documents', 'word_length_mean', 'word_length_std')
    """

    def text_extractor(dataset):
        ret = []
        for data in dataset:
            ret.append(data["text"])
        return ret

    dataset = load_dataset("wikimedia/wikipedia", f"{dump_version}.{language}", split="train")
    if n_samples is None:
        documents = dataset
    else:
        # TODO: Alternative: limit number of words sampled as not all documents have the same length
        documents = random_sample(dataset, n_samples)
    documents = text_extractor(documents)
    documents = tokenize(documents, language)
    return compute_stats(documents)
