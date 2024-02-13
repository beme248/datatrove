# Multilingual preprocessing pipeline

## Running locally

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
pip install -U datasets # Upgrade datasets library to a newer version
```

### Run the pipeline

```bash
cd examples/multilingual
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
HF_TOKEN=your-huggingface-access-token python process_fineweb.py
```