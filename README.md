# FANATIC
FAst Noise-Aware TopIc Clustering

Authors: Ari Silburt, Anja Subasic, Evan Thompson, Carmeline Dsilva, Tarec Fares

## General
This repo contains the research code and scripts used in the paper [FANATIC: FAst Noise-Aware TopIc Clustering](), and provides a basic overview of the code structure and its major components. For more questions, please directly contact the authors.

In particular, this repo allows a user to:
- Download the reddit data
- Train a (word2vec) embedding model
- Use `FANATIC` to cluster the reddit data, dumping results for downstream analysis. 

Note that the original paper results used an in-house preprocessor, but a very similar open-source one has been provided (see `fanatic/preprocess/nltk_preprocessor.py`).

## License
Please read the `LICENSE`.

## How-to
### Setup
It is recommended to create a fresh python virual environment and run the following commands from the base repo:
```
pip install --upgrade pip
pip install -r requirements.txt
```
This repo has been tested against `python3.7`.

### Download the Reddit Data
Data can be downloaded from [pushshift](https://files.pushshift.io/reddit/submissions/) using `wget`, e.g. `wget https://files.pushshift.io/reddit/submissions/RS_2017-11.zst`. If data files are downloaded to the `data/` directory, subsequent scripts are already set up to look there. 

### Training a Word2vec Embedding
A new word2vec model can be trained using `entrypoints/embedding_driver.py`, it is recommended to carefully inspect the arguments before running. In particular, a Reddit data file(s) must first be downloaded and specified in the `--data-files` argument. 

### Cluster via FANATIC
Once the data has been downloaded and word2vec model trained, a clustering run can be performed using the `entrypoints/clustering_driver.py` script. All input arguments are specified in the `parse_args` function of `fanatic/arguments.py` including data, label, preprocessing, fanatic and output arguments. 

#### Clustering labels
By default, the data is clustered against `data/subreddit_labels.json`, which indicates:
- what subreddits are considered for clustering (all other subreddits are discarded)
- whether the subreddit is a coherent or noise topic (all noise topics are assigned a `NOISE` label). 
These labels can be substituted for different ones, or removed entirely by setting `--subreddit-labels-file None`. 

In the case where subreddit labels are not provided, each subreddit becomes its own valid topic and the concept of noise is eliminated. 

In the case where subreddit labels are provided but `--subreddit-noise-percentage` is set to `None`, the noise percentage is set by the natural data distribution.

#### Clustering Outputs
After a successful clustering run, the files that are output are:
- `fanatic_<dataset-id>_<seed-run>_labels_and_assignments.json`: this file is generated for each seed run and contains, for each document, the *assignment* (what cluster the document ended up in) and *label* (the label associated with the document). This allows a user to run additional analysis afterwards.
- `fanatic_<dataset-id>_<seed-run>_sample_clusters.txt` - this file is generated for each seed run and contains the first 10 documents from each cluster and associated label. Format is `<text> -> <label>`. This gives the user a qualitative sense of what each cluster contains.
- `fanatic_<dataset-id>_<seed-run>_summary.txt` - this file is generated for each seed run and contains all input parameters and clustering results. It is effectively a summary of the entire clustering run, allowing you to quickly parse results and/or recreate the job. Can be consumed by [configparser](https://docs.python.org/3.7/library/configparser.html).
- `fanatic_<dataset-id>_summary_averaged.txt` - this file is generated once for a dataset-id and contains the input arguments and averaged results across the individual seed runs. 

In addition, the full clustering model can be dumped to pickle for deep investigation if desired by adding `--flag-save-clusteringmodel`. Warning that, especially for large datasets, this file is very large.

### Use a Custom Preprocessor
Results from the paper were generated using an in-house preprocessor that is not available to the public. Using nltk we created a similar preprocessor, housed at `fanatic/preprocess/nltk_preprocessor.py`, and inherits from `fanatic/preprocess/generic_preprocessor.py`. One is free to create a new preprocessor, inheriting from `generic_preprocessor.py`, to use more sophisticated features (e.g. BERT embeddings).

### Use Non-Reddit Data
One is free to substitute or modify `fanatic/preprocess/read_data.py` to read in different kinds of data. In particular, `DATASET_INPUT_FIELD`, `DATASET_LABEL_FIELD` and `DATASET_ID_FIELD` are the essential fields that must be populated for downstream preprocessing and clustering. 

## Tests
Some basic unit tests can be run from the home directory with `python3.7 -m pytest tests/unit/`