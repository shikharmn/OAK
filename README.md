# OAK: Enriching Document Representations using Auxiliary Knowledge for Extreme Classification (ICML '24)
This is the codebase for the project Online Auxiliary Knowledge (OAK). Please refer to the publication for technical details, this README will explain usage. Please reach out to shikharmohan@microsoft.com for any clarifications.

## Overview
The objective in eXtreme Classification (XC) is to find relevant labels for a document from an exceptionally large label space. Most XC application scenarios have rich auxiliary data associated with the input documents, e.g., frequently clicked webpages for search queries in sponsored search. Unfortunately, most of the existing XC methods do not use any auxiliary data. In this paper, we propose a novel framework, Online Auxiliary Knowledge (OAK), which harnesses auxiliary information linked to the document to improve XC accuracy. OAK stores information learnt from the auxiliary data in a knowledge bank and during a forward pass, retrieves relevant auxiliary knowledge embeddings for a given document. An enriched embedding is obtained by fusing these auxiliary knowledge embeddings with the document's embedding, thereby enabling much more precise candidate label selection and final classification. OAK training involves three stages. (1) Training a linker module to link documents to relevant auxiliary data points. (2) Learning an embedding for documents enriched using linked auxiliary information. (3) Using the enriched document embeddings to learn the final classifiers. OAK outperforms current state-of-the-art XC methods by up to 
∼5% on academic datasets, and by ∼3% on an auxiliary data-augmented variant of LF-ORCAS-800K dataset in Precision@1. OAK also demonstrates statistically significant improvements in sponsored search metrics when deployed on a large scale search engine.
## Requirements
Use `oak.yml` to install the required libraries.

## Data Preparation
The codebase expects the public datasets in a specific formats.

```bash
📁 LF-WikiSeeAlsoTitles-320K/
    📄 trn_X_Y.txt/npz # contains mappings from train documents IDs to label IDs
    📄 filter_labels_train.txt # this contains train reciprocal pairs to be ignored in evaluation
    📄 tst_X_Y.txt/npz # contains mappings from test document IDs to label IDs
    📄 filter_labels_test.txt # this contains test reciprocal pairs to be ignored in evaluation
    📄 auxiliary_trn_X_Y.txt/npz # contains mappings from train document IDs to AKP IDs in ground truth
    📄 auxiliary_tst_X_Y.txt/npz # contains mappings from test document IDs to AKP IDs in ground truth
    📄 auxiliary_renee_tst_X_Y.txt/npz # contains mappings from test document IDs to AKP IDs in retrieved (more details on this in Addendum)
    📂 raw_data/
        📄 train.raw.txt # each line contains the raw input train text, this needs to be tokenized
        📄 test.raw.txt # each line contains the raw input test text, this needs to be tokenized
        📄 label.raw.txt # each line contains the raw label text, this needs to be tokenized
        📄 auxiliary.raw.txt # each line contains the raw label text, this needs to be tokenized
```

Use `misc_utils/create_tokenized.py` to tokenize all text documents.

It is expected that for every dataset, there exists a `conf/data/<dataset>.yaml` file and these datasets are present in the directory `cfg.base_dir / cfg.data.data_dir`. Ideally, base_dir corresponds to the mount point of a large storage blob where datasets are stored, and cfg.data.data_dir corresponds to the location of the dataset folder. Please refer to the config files for examples, and refer to the Config Files section of the addendum for more details on config management.

## Training
Execute `run_pub.sh <dataset> <version_name> <0/1 for WandB logging> <0,1,...,n for n(GPUs) used>` for training on public benchmark extreme classification datasets. Appropriate hyperparameters have been stored inside config files in `conf/data/<dataset>.yaml`.

## Appendix
This section contains extra details and documentation around how to use this codebase.

### Configuration Management
For flexibility, we use Hydra for configuration management which supports multiple hierarchical configs with override functionalities. E.g. in `run_pub.sh` if one wants to change the learning rate, simply a new line corresponding to the override needs to be added as follows:

```bash
...
CUDA_VISIBLE_DEVICES=$4 python -W ignore -u main.py \
base_dir=${BASE_DIR} \
version=${VERSION} \
training_devices=[0,1,2,3] \
clustering_devices=[3] \
data.lr=0.00005 \               #<-- Override
data=${DATASET} | tee "${DATA_DIR}/OrganicBERT/${VERSION}/debug.log"
...
```

### Retrieved AKPs and Evaluation
For public benchmark datasets, code will evaluate using two sets of linkages, one ground truth (referred to as golden thread) and one retrieved using a retrieval algorithm (referred to as Linker in the paper). As opposed to obtaining the relevant AKPs at inference time using the linker for every document, we obtain linker predictions (test documents to AKPs) for all test documents at once, dump them into a sparse matrix and evaluate using the same.

### Logging and Reproducibility
This code natively supports WandB, but it is turned off by default. For every run, one provides a version name, which creates a folder `cfg.base_dir/cfg.data.data_dir/OrganicBERT/cfg.version`, which contains test embeddings, label embeddings, predictions (in case of public datasets), logs for the entire training run (`debug.log`) and checkpoint files (`state_dict.pt`).

### Testing Code Changes
Overriding `test_mode` to `True` overfits the model with a batch size of 64 over 1024 points on the given dataset as a basic test.