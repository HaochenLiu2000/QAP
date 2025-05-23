# Question-Aware Knowledge Graph Prompting for Enhancing Large Language Models

The codes are associated with the following paper:

>**Question-Aware Knowledge Graph Prompting for Enhancing Large Language Models**

<p align="center">
<img src="QAP.png" alt="Overview of QAP." width="100%" />
</p>

## 1. Datasets

Our datasets are provided by [QA-GNN](https://arxiv.org/abs/2104.06378) and [Dragon](https://arxiv.org/abs/2210.09338).

Download the QA datasets and ConceptNet for general domain (OBQA, Riddle) [here](https://nlp.stanford.edu/projects/myasu/DRAGON/data_preprocessed.zip) and for biomedical domain (MedQA) [here](https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_biomed.zip). Unzip the datasets under `/QAP/`.

Download the Llama2-chat models [here](https://huggingface.co/meta-llama) and place the model under `/QAP/`.

## 2. Dependencies

Run the following commands to create a conda environment:

    conda create -y -n qap python=3.11
    conda activate qap
    pip install numpy==2.0.1
    pip install torch==2.4.0
    pip install transformers==4.46.2
    pip install tqdm
    pip install torch-geometric==2.7.0

## 3. Training

To train our model on dataset:

Run

    cd <dataset_name>
    python train_qap.py

## 4. Evaluating

To Evaluate the trained model on dataset:

Run

    cd <dataset_name>
    python test_qap.py
