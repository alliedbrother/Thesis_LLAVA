# LLaVA Keen Probe Project

This project implements a probing mechanism to analyze the factual correctness of LLaVA model's image captions using various model embeddings.

## Project Structure

### Main Scripts

- `01_download_model.py`: Downloads the LLaVA model and necessary components
- `02_extract_features.py`: Extracts embeddings from different layers of the LLaVA model
- `03_generate_captions.py`: Generates captions for images using the LLaVA model and create factual scores.
- `04_build_dataset.py`: Creates training datasets from extracted embeddings and factual scores
- `05_train_probe.py`: Trains the probing model on the generated datasets
- `06_eval_probe.py`: Evaluates the trained probe on test data

### Utility Scripts

- `utils.py`: Contains helper functions used across the project
- `create_prompts.py`: Generates prompts for the LLaVA model
- `download_captions.py`: Downloads ground truth captions
- `download_coco_val2017.py`: Downloads COCO validation dataset
- `map_captions_to_csv.py`: Maps captions to CSV format
- `display_image.py`: Utility for displaying images
- `test_display.py`: Tests image display functionality
- `create_datasets.py`: Alternative implementation for dataset creation

### Configuration Files

- `prompts.json`: Contains prompts used for generating captions
- `requirements.txt`: Lists all project dependencies with versions

### Data Directory (`keen_data/`)

- `features.json`: Contains extracted embeddings from different model layers
- `factual_scores.json`: Contains factual correctness scores for generated captions
- `coco_val2017_captions.csv`: Ground truth captions from COCO dataset
- `probe.pt`: Trained probing model weights

#### Subdirectories

- `generated_datasets/`: Contains CSV files of training datasets
  - `dataset_vision_tower_embeddings.csv`
  - `dataset_mm_projector_embeddings.csv`
  - `dataset_language_model_embeddings.csv`
  - `dataset_vision_tower_embeddings_after_projection.csv`
  - `dataset_language_model_embeddings_before_projection.csv`

- `models/`: Stores downloaded model files
- `embeddings/`: Contains extracted embeddings

## Dependencies

The project requires the following packages (see `requirements.txt` for specific versions):
- torch
- transformers
- tokenizers
- pandas
- numpy
- tqdm
- pillow
- sentence-transformers
- accelerate
- bitsandbytes
- huggingface-hub

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the scripts in sequence:
   ```bash
   python 01_download_model.py
   python 02_extract_features.py
   python 03_generate_captions.py
   python 04_build_dataset.py
   python 05_train_probe.py
   python 06_eval_probe.py
   ```

## Dataset Structure

Each generated dataset contains:
- Embedding features from a specific model layer
- Factual correctness labels (0 or 1)

The datasets are created from different embedding layers:
1. Vision tower embeddings
2. MM projector embeddings
3. Language model embeddings
4. Vision tower embeddings after projection
5. Language model embeddings before projection


Verify
