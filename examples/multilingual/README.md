# Multilingual CommonCrawl cleaning pipeline

To extend the [RefinedWeb](https://arxiv.org/pdf/2306.01116.pdf) pipeline to support multilingual data, we build on top of the `datatrove` Python library. To effectively process multilingual data, we use per-language word tokenizers and adjust the Gopher quality filter thresholds for each language. Our implementation and filter thresholds are outlined in further sections.


## Language-specific word tokenizers

The filters used in the cleaning pipeline are sensitive to word tokenization, which can impact the results. Therefore, we use several word tokenization libraries to support almost 100 languages. In order to process large volumes of data efficiently, we utilize fast and reliable tokenization libraries: [NLTK](https://www.nltk.org/), [SpaCy](https://spacy.io/), [Stanza](https://stanfordnlp.github.io/stanza/), [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library), [Jieba](https://github.com/fxsjy/jieba), [NLPashto](https://pypi.org/project/nlpashto/), [PyVi](https://pypi.org/project/pyvi/) and [Anbani](https://github.com/Anbani/anbani.py). For languages that aren't officially supported by the libraries, we use a tokenizer of a supported language that is written in the same script and is from a close language family.

To further analyze the implementation of word tokenizers, inspect the [word tokenizer source code](https://github.com/beme248/datatrove/blob/multilingual/src/datatrove/tools/word_tokenizers.py).


## Multilingual Gopher quality filter: language-specific adjustments

In our implementation of the multilingual Gopher quality filter, we make language-specific adjustments based on the statistical analysis of the Wikipedia data for the top 100 high-resource languages.

We [extract the statistics](https://github.com/beme248/datatrove/blob/multilingual/examples/multilingual/lang_stats/wiki_lang_stats.py) for each language from their respective [Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia). We use the [language statistics visualization tool](https://huggingface.co/spaces/ZR0zNqSGMI/mlo-language-statistics) to analyze the statistics. By comparing the statistic values among different languages, we identify that the following filter values should be tweaked per-language: `stop_words`, `min_avg_word_length`, `max_avg_word_length` and `max_non_alpha_words_ratio`.

Further subsections explain the choice for filter threshold values. Other filter threshold values are set to the default values from the original Gopher quality filter.

To further analyze the implementation of the filters, inspect the [Gopher quality filter source code](https://github.com/beme248/datatrove/blob/multilingual/src/datatrove/pipeline/filters/gopher_quality_filter.py) and [multilingual Gopher quality filter source code](https://github.com/beme248/datatrove/blob/multilingual/src/datatrove/pipeline/filters/multilingual_gopher_quality_filter.py).

### `stop_words`

To obtain stop words for each language, we count the occurrences of each word in the Wikipedia dataset. We choose stop word candidates as the highest frequency words. To account for differences among languages (e.g., English uses "the", while German uses "der", "die" and "das"), we select words with a frequency higher than 0.8% of the total word count frequency instead of a fixed number of stop words with the highest frequencies. We also remove whitespaces and symbols (e.g. "«" and "»") from the stop words.

To reduce the risk of overfiltering the data, if there are less than 8 stop words in the cleaned stop words list, we choose words that appear more frequently than 0.3% of the total word count frequency. We remove whitespaces and symbols from them as well.

To further analyze word frequencies, use the [language statistics visualization tool](https://huggingface.co/spaces/ZR0zNqSGMI/mlo-language-statistics) (tab *Word frequency*).


### `min_avg_word_length` and `max_avg_word_length`

We calculate the language-specific thresholds for `min_avg_word_length` and `max_avg_word_length` as one standard deviation below (for minimum) and one standard deviation above (for maximum) the mean word length value rounded to the closest integer. When computed for the English language, these values are similar to the original Gopher quality filter thresholds: 2 (for minimum) and 8 (for maximum).


### `max_non_alpha_words_ratio`

We calculate the `max_non_alpha_words_ratio` filter threshold for each language as three standard deviations below the mean `alpha_ratio` rounded to one decimal place. When computed for the English language, the value is equal to the default Gopher quality filter threshold: 0.8.

# Running the pipeline

## Install conda

Follow [Quick command line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) tutorial for Linux to set up `conda`.

Restart your shell after running `~/miniconda3/bin/conda init bash` to be able to use `conda`.

## Clone the repository

```bash
git clone -b multilingual https://github.com/beme248/datatrove
cd datatrove
```

## Set up conda environment

```bash
conda create -n datatrove python=3.11
conda activate datatrove
pip install -e ".[all]" # Install dependencies
```

## Run the pipeline

To generate language-specific filter thresholds (optional, filter thresholds are already provided in folder `filters`), run
```bash
python wiki_lang_stats.py filters
```

To start the CommonCrawl cleaning pipeline, run
```bash
python process_non_english.py DUMP_NAME
```


<!-- ## Running on the CSCS Slurm cluster

### Set up access to CSCS Clariden cluster

Follow the [tutorial](https://github.com/swiss-ai/documentation/blob/main/getting_started_with_clariden/setup_clariden.md) to set up the access to the Clariden cluster.

By the end of the tutorial, you should be able to `ssh` into your account on the cluster.
```bash
ssh clariden
```

### Install conda

Follow [Quick command line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) tutorial for Linux to set up `conda` under your user on the cluster.

Restart your shell after running `~/miniconda3/bin/conda init bash` to be able to use `conda`.

### Clone the repository

```bash
git clone -b multilingual https://github.com/beme248/datatrove
cd datatrove
```

### Set up conda environment

```bash
conda create -n datatrove python=3.11
conda activate datatrove
pip install -e ".[all]" # Install dependencies
```

### Run the pipeline


```bash
cd examples/multilingual
```

To generate language statistics (optional, language statistics are already provided), run
```bash
export HF_DATASETS_CACHE="$SCRATCH/hf_datasets"
python wiki_lang_stats.py
```

Note that we change the HuggingFace datasets library cache to the `$SCRATCH` directory becuase the datasets will not fit in `$HOME` directory. -->