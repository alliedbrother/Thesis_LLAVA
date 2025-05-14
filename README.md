# LLaVA Keen Probe Project

# Project: Visual Knowledge Estimation with LLaVA

## Project Overview

This project aims to estimate the visual knowledge of the LLaVA model using KEEN methodology. The objective is to evaluate the ability of the model to generate accurate image captions without relying on question-answering tasks. We focus on extracting image and image+text embeddings from various layers of the LLaVA model and use these embeddings to train a probe for evaluating factual accuracy.

## Directory Structure

```
project_root/
├── keen_data/                     # Contains generated datasets and model outputs
├── coco_val2017/                  # COCO validation images
├── annotations/                   # COCO annotations
├── models/                        # Trained probe models
├── utils.py                       # Utility functions and model initialization
├── download_coco_val2017.py        # Script to download COCO dataset
├── download_captions.py            # Script to download annotations
├── 100_download_model.py           # Script to download the model
├── 200_extract_pre_generation_embeddings.py # Extract pre-generation embeddings
├── 300_extract_generation_embeddings_V3.py  # Extract generation embeddings
├── 301_extract_generated_captions_V3.py      # Extract generated captions
├── 302_extract_image_details_V3.py            # Extract detailed image embeddings
├── 400_factual_score.py            # Script to compute factual accuracy
├── 500_build_dataset.py            # Build dataset from extracted features
├── 600_train_probe.py              # Train the KEEN probe
├── 700_eval_probe.py               # Evaluate the trained probe
└── map_captions_to_csv.py          # Map image captions to CSV for easy access
```

## Installation Instructions

1. Clone the repository:

```
git clone <repo_url>
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Download the model:

```
python 100_download_model.py
```

## Data Preparation

1. Download the COCO validation images:

```
python download_coco_val2017.py
```

2. Download the captions:

```
python download_captions.py
```

3. Map captions to CSV:

```
python map_captions_to_csv.py
```

## Model Download and Setup

The model used is LLaVA v1.5-7b. It is downloaded using the script:

```
python 100_download_model.py
```

## Feature Extraction

### Pre-generation Embeddings

Extracted using:

```
python 200_extract_pre_generation_embeddings.py
```

### Generation Embeddings

Extracted using:

```
python 300_extract_generation_embeddings_V3.py
```

## Caption Generation

Generate captions using the extracted embeddings:

```
python 301_extract_generated_captions_V3.py
```

## Evaluation

Evaluate the factual accuracy of generated captions:

```
python 400_factual_score.py
```

## Training the Probe

Train the KEEN probe to classify factual correctness:

```
python 600_train_probe.py
```

## Running Evaluation

Evaluate the probe:

```
python 700_eval_probe.py
```

## Utilities

* `utils.py`: Provides utility functions such as model loading and image processing.
* `map_captions_to_csv.py`: Maps image IDs to captions in a CSV format.

## How to Run the Full Pipeline

1. Prepare data and download model.
2. Extract pre-generation and generation embeddings.
3. Generate captions.
4. Train the probe.
5. Evaluate the probe.

## Acknowledgements

* LLaVA for the vision-language model.
* COCO dataset for image captioning.
* KEEN methodology for knowledge estimation.

