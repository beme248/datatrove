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

To generate language statistics (optional, `lang_stats.json` is already provided), run
```bash
EXECUTOR=local python extract_lang_stats.py
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

To generate language statistics (optional, `lang_stats.json` is already provided), run
```bash
python extract_lang_stats.py
```

To start the fineweb preprocessing pipeline, run
```bash
HF_TOKEN=your-huggingface-access-token python process_fineweb.py
```



## Results

### fineweb_german ($mean \pm 1*std$ german statistics)

#### Pipeline
```python
[
HuggingFaceDatasetReader(DATASET, dataset_options={"token": HF_TOKEN, "split": "train"}, limit=DOC_LIMIT) # DOC_LIMIT = 100000
GopherRepetitionFilter()
MultilingualGopherQualityFilter(
    max_avg_word_lengths=max_avg_word_lengths, # de = 10
    min_avg_word_lengths=min_avg_word_lengths, # de = 2
    stop_words=stopwords # de = nltk.stopwords.words('german')
ListFilter()
JsonlWriter(f"{MAIN_OUTPUT_PATH}/output/{DATASET}")
]
```

Run locally with 8 tasks (input total: 800k documents, output total: ~436k documents)

Rough estimate: MutilingualGopherFilter drops ~3.3k documents that are over the maximum average word length, almost no documents (6) that are under average word length, ~2.9k documents without enough stopwords

#### Insights (based on 00000.jsonl)
- Documents are mostly a mix of blogs/articles and business pages
- There are some "About (business/event/person)"/"Privacy policy" pages that contain information but may not be so useful for language modelling
- Most of the processed dataset seem to be good quality

#### Issues (based on 00000.jsonl)
- There are multiple pages where the text processed is about cookies (sometimes containing same text in English)
- Sports results table was not filtered (little text, only results)
- NSFW urls might escape the previous filter
- Some documents are login/register/order pages

TBA: insights from filtered documents