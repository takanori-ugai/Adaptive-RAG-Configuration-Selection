# Adaptive RAG Selection Based on Questions and Documents Considering Cost and Accuracy

## Overview

This repository contains code for extracting features from questions and documents, ranking candidate RAG configurations with a ranking model, and selecting an appropriate RAG configuration based on the weights of the evaluation metrics.
The `pipeline` directories contain scripts that run multiple modules together.

## Overview Figures

<div align="center">
  <img src="images/train.png" alt="Training the ranking model">
  <p>Figure 1: Training the ranking model</p>
</div>

<div align="center">
  <img src="images/inference.png" alt="Inference with the ranking model">
  <p>Figure 2: Inference with the ranking model</p>
</div>

## Usage

### 1. Preparation

Run each RAG configuration with NaiveRAG, LightRAG, and GraphRAG.
For the core code, details, and datasets for each method, refer to the corresponding official GitHub repositories.
Place the code used to generate the data in `make_data/` and the execution results in `data/raw/`.

### 2. Environment Setup

The code was implemented with Python 3.10.12.
Install the required libraries with the following commands:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Training the Ranking Model

`train/pipeline/make_model.py` runs the following steps:

- `train/make_train_data.py`: Extracts features and builds the training data
- `train/make_test_data.py`: Extracts features and builds the test data
- `train/make_ranking_model.py`: Builds the ranking model from the extracted features

```bash
python train/pipeline/make_model.py
```

### 4. Inference with the Ranking Model

`inference/pipeline/use_model.py` runs the following steps:

- `inference/use_ranking_model.py`: Evaluates the ranking model
- `inference/selection_rag.py`: Selects a RAG configuration based on the ranking results and metric weights

```bash
python inference/pipeline/make_model.py
```

## One-off Experiment

Select a RAG configuration adaptively for a new question and document.
`inference/pipeline/pilot_study.py` runs the following steps:

- `inference/one_feature_extraction.py`: Extracts features
- `inference/one_selection_rag.py`: Selects a RAG configuration

```bash
python inference/pipeline/pilot_study.py
```

## Directory Structure

### `data/`

- `data/raw`: QA data generated with each RAG method
- `data/interim`: Intermediate data created during dataset construction
- `data/train`: Training data for the ranking model
- `data/test`: Test data for the ranking model

### `artifacts/`

- `artifacts/models`: Trained ranking models
- `artifacts/registry`: Metadata for the trained ranking models
- `artifacts/results`: Evaluation results
