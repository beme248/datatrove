# Multilingual preprocessing pipeline

## Running locally

### Install conda

Follow [Quick command line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) tutorial for Linux to set up `conda`.

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
pip install -U datasets # Upgrade datasets library to a newer version
```

### Run the pipeline

```bash
cd examples/multilingual
```

To generate language statistics (optional, `wiki_lang_stats.json` is already provided), run
```bash
EXECUTOR=local python wiki_lang_stats.py
```

To start the fineweb preprocessing pipeline, run
```bash
EXECUTOR=local HF_TOKEN=your-huggingface-access-token python process_fineweb.py
```



## Running on the CSCS Slurm cluster

### Set up access to CSCS Clariden cluster

Follow the [tutorial](https://github.com/swiss-ai/documentation/blob/main/getting_started_with_clariden/setup_clariden.md) to set up the access to the Clariden cluster.

By the end of the tutorial, you should be able to `ssh` into your account on the cluster.
```bash
ssh clariden
```

### Install conda

Since launching Slurm jobs in a container environment [isn't supported at the moment](https://confluence.cscs.ch/pages/viewpage.action?pageId=776306695#UsingContainerImagesonClariden(ContainerEngine)-Usingcontainerizedenvironmentsinbatchscripts), we will use `conda` to launch our preprocessing pipeline.

Follow [Quick command line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) tutorial for Linux to set up `conda` under your user on the cluster.

Restart your shell after running `~/miniconda3/bin/conda init bash` to be able to use `conda`.

### Clone the repository

```bash
cd $SCRATCH
git clone -b multilingual https://github.com/beme248/datatrove
cd datatrove
```

### Set up conda environment

```bash
conda create -n datatrove python=3.11
conda activate datatrove
pip install -e ".[all]" # Install dependencies
pip install -U datasets # Upgrade datasets library to a newer version
```

### Run the pipeline


```bash
cd examples/multilingual
```

To generate language statistics (optional, `wiki_lang_stats.json` is already provided), run
```bash
export HF_DATASETS_CACHE="$SCRATCH/hf_datasets"
python wiki_lang_stats.py
```

To start the fineweb preprocessing pipeline, run
```bash
HF_TOKEN=your-huggingface-access-token python process_fineweb.py
```

Note that we change the HuggingFace datasets library cache to the `$SCRATCH` directory becuase the datasets will not fit in `$HOME` directory.


## Results

### fineweb_german ($mean \pm 1\sigma$ german statistics)

#### Pipeline
```python
[
HuggingFaceDatasetReader(DATASET, dataset_options={"token": HF_TOKEN, "split": "train"}, limit=DOC_LIMIT) # DOC_LIMIT = 100000
GopherRepetitionFilter()
MultilingualGopherQualityFilter(
    max_avg_word_lengths=max_avg_word_lengths, # de = 10
    min_avg_word_lengths=min_avg_word_lengths, # de = 2
    stop_words=stopwords), # de = nltk.corpus.stopwords.words('german')
ListFilter()
JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DATASET}")
]
```

#### Insights (based on 00000.jsonl)
- Documents are mostly a mix of blogs/articles and business pages
- There are some "About (business/event/person)"/"Privacy policy" pages that contain information but may not be so useful for language modelling
- Most of the processed dataset seems to be good quality
- Filtered documents make sense to be filtered

#### Issues (based on 00000.jsonl)
- There are multiple pages where the text processed is about cookies (sometimes containing same text in English)
- Sports results table was not filtered (little text, only results)
- NSFW urls might escape the previous filter
- Some documents are login/register/order pages
- Too much ellipsis filter sometimes produces FP (but very rarely)



### fineweb_french ($mean \pm 1\sigma$ french statistics)

#### Pipeline
```python
[
HuggingFaceDatasetReader(DATASET, dataset_options={"token": HF_TOKEN, "split": "train"}, limit=DOC_LIMIT) # DOC_LIMIT = 100000
GopherRepetitionFilter()
MultilingualGopherQualityFilter(
    max_avg_word_lengths=max_avg_word_lengths, # fr = 8
    min_avg_word_lengths=min_avg_word_lengths, # fr = 2
    stop_words=stopwords), # fr = nltk.corpus.stopwords.words('french')
ListFilter()
JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DATASET}")
]
```

#### Insights (based on 00000.jsonl)
- Most of the processed dataset seems to be good quality
- There are some "Contact"/"About me" pages that contain information but may not be so useful for language modelling
- Filtered documents make sense to be filtered

#### Issues (based on 00000.jsonl)
- Similar as in German dataset, there is text about cookies
- NSFW content may be found
- Too much ellipsis filter sometimes produces FP (but very rarely)


### fineweb_german ($mean \pm 2\sigma$ german statistics)

#### Pipeline
```python
[
HuggingFaceDatasetReader(DATASET, dataset_options={"token": HF_TOKEN, "split": "train"}, limit=DOC_LIMIT) # DOC_LIMIT = 100000
GopherRepetitionFilter()
MultilingualGopherQualityFilter(
    max_avg_word_lengths=max_avg_word_lengths, # de = 14
    min_avg_word_lengths=min_avg_word_lengths, # de = -2
    stop_words=stopwords), # de = nltk.corpus.stopwords.words('german')
ListFilter()
JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DATASET}")
]
```


### fineweb_french ($mean \pm 2\sigma$ french statistics)

#### Pipeline
```python
[
HuggingFaceDatasetReader(DATASET, dataset_options={"token": HF_TOKEN, "split": "train"}, limit=DOC_LIMIT) # DOC_LIMIT = 100000
GopherRepetitionFilter()
MultilingualGopherQualityFilter(
    max_avg_word_lengths=max_avg_word_lengths, # fr = 12
    min_avg_word_lengths=min_avg_word_lengths, # fr = -1
    stop_words=stopwords), # fr = nltk.corpus.stopwords.words('french')
ListFilter()
JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DATASET}")
]
```

### fineweb_german ($mean \pm 0.5\sigma$ german statistics)

#### Pipeline
```python
[
HuggingFaceDatasetReader(DATASET, dataset_options={"token": HF_TOKEN, "split": "train"}, limit=DOC_LIMIT) # DOC_LIMIT = 100000
GopherRepetitionFilter()
MultilingualGopherQualityFilter(
    max_avg_word_lengths=max_avg_word_lengths, # de = 8
    min_avg_word_lengths=min_avg_word_lengths, # de = 4
    stop_words=stopwords), # de = nltk.corpus.stopwords.words('german')
ListFilter()
JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DATASET}")
]
```

### fineweb_french ($mean \pm 0.5\sigma$ french statistics)

#### Pipeline
```python
[
HuggingFaceDatasetReader(DATASET, dataset_options={"token": HF_TOKEN, "split": "train"}, limit=DOC_LIMIT) # DOC_LIMIT = 100000
GopherRepetitionFilter()
MultilingualGopherQualityFilter(
    max_avg_word_lengths=max_avg_word_lengths, # fr = 7
    min_avg_word_lengths=min_avg_word_lengths, # fr = 4
    stop_words=stopwords), # fr = nltk.corpus.stopwords.words('french')
ListFilter()
JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DATASET}")
]
```

### Summary

| Dataset | Average word length parameter | Input documents | Output (filtered) documents |
|---------|-------------------------------|-----------------|-----------------------------|
|  German |      $mean \pm 0.5\sigma$     |            800k |                       ~428k |
|  German |       $mean \pm 1\sigma$      |            800k |                       ~436k |
|  German |       $mean \pm 2\sigma$      |            800k |                       ~437k |
|  French |      $mean \pm 0.5\sigma$     |            800k |                       ~444k |
|  French |       $mean \pm 1\sigma$      |            800k |                       ~452k |
|  French |       $mean \pm 2\sigma$      |            800k |                       ~454k |


| Dataset | Average word length parameter | Below avg. word length | Above avg. word length | Not enough stopwords |
|---------|-------------------------------|------------------------|------------------------|----------------------|
|  German |      $mean \pm 0.5\sigma$     |                  ~0.6k |                 ~18.5k |                ~2.2k |
|  German |       $mean \pm 1\sigma$      |                      6 |                  ~3.3k |                ~2.9k |
|  German |       $mean \pm 2\sigma$      |                      0 |                  ~0.8k |                ~3.2k |
|  French |      $mean \pm 0.5\sigma$     |                  ~9.4k |                  ~8.8k |                ~0.8k |
|  French |       $mean \pm 1\sigma$      |                     33 |                  ~4.1k |                ~0.9k |
|  French |       $mean \pm 2\sigma$      |                      0 |                    ~1k |                ~1.5k |