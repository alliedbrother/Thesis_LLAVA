import os
import json
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

# Set up paths
base_dir = os.path.dirname(__file__)
output_dir = os.path.join(base_dir, "keen_data", "generated_datasets")
os.makedirs(output_dir, exist_ok=True)

# Load features and factual scores
features_path = os.path.join(base_dir, "keen_data", "features.json")
with open(features_path, "r") as f:
    features = json.load(f)

factual_scores_path = os.path.join(base_dir, "keen_data", "factual_scores.json")
with open(factual_scores_path, "r") as f:
    factual_scores = json.load(f)

# Create mapping of image_id to factual correctness from factual_scores.json
image_id_to_factual = {item["image_id"]: item["average_similarity"] for item in factual_scores}

# Define all embedding layers available in features.json
embedding_layers = {
    "vision_tower": "vision_embeddings",
    "initial_layer": "initial_layer_embeddings",
    "middle_layer": "middle_layer_embeddings",
    "pre_generation": "pre_generation_embeddings",
    "final_layer": "final_layer_embeddings"
}

# Create datasets for each embedding layer
for layer_name, embedding_key in embedding_layers.items():
    print(f"Creating dataset for {layer_name}...")
    
    # Initialize lists to store data
    image_ids = []
    embeddings = []
    factual_scores = []
    
    # Process each image
    for item in tqdm(features):
        image_id = item["image_id"]
        
        # Get factual correctness score from factual_scores mapping
        factual_score = image_id_to_factual.get(image_id, 0.0)
        
        # Get embeddings directly from features.json
        if embedding_key in item:
            embedding = item[embedding_key]
            
            # Convert to numpy array if it's a list
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            # Store data
            image_ids.append(image_id)
            embeddings.append(embedding)
            factual_scores.append(factual_score)
    
    # Create DataFrame
    df = pd.DataFrame({
        "image_id": image_ids,
        "embedding": embeddings,
        "factual_correctness_score": factual_scores
    })
    
    # Save to CSV
    output_file = os.path.join(output_dir, f"dataset_{layer_name}.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved dataset to {output_file}")

print("Dataset creation complete!")

