# Question-Aware Knowledge Graph Prompting

## 1. Datasets

Download the QA datasets and ConceptNet for general domain [here](https://nlp.stanford.edu/projects/myasu/DRAGON/data_preprocessed.zip) and for biomedical domain [here](https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_biomed.zip). Unzip the datasets under `/QAP/`.

## 2. Dependencies

Run the following commands to create a conda environment:

    conda create -y -n qap python=3.11
    conda activate qap
    pip install numpy
    pip install torch
    pip install transformers
    pip install tqdm
    pip install torch-geometric

## 3. Training

To train our model on dataset:

Run

    python train_qap.py

The inference code is together with the training code.
