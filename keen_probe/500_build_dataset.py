import os
import h5py
import pandas as pd
from tqdm import tqdm
import numpy as np

# Set up paths
base_dir = os.path.dirname(__file__)
output_dir = os.path.join(base_dir, "keen_data", "dataset")
os.makedirs(output_dir, exist_ok=True)

# Path to your h5 file
h5_file_path = "/root/project/Thesis_LLAVA/keen_probe/keen_data/2014_extracted_embeddings/generation_embeddings_V41_part3.h5"

# Initialize lists to store data
image_ids = []
embeddings = []

# Read the h5 file
with h5py.File(h5_file_path, 'r') as f:
    # First, let's print the structure of the first image to understand the data organization
    image_ids = list(f.keys())
    if image_ids:
        first_image = f[image_ids[0]]
        print("\nStructure of first image:")
        print("Keys:", list(first_image.keys()))
        
        # Get the embeddings from the correct path
        for image_id in tqdm(image_ids):
            try:
                # Navigate through the nested groups
                current = f[image_id]
                # Get the embeddings from post_generation/step_1/layer_0/image_embeddings
                embedding = current['post_generation']['step_1']['layer_0']['image_embeddings'][:]
                
                image_ids.append(image_id)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing image {image_id}: {str(e)}")
                continue

# Create DataFrame
df = pd.DataFrame({
    "Image_id": image_ids,
    "Type_of_embeddings": embeddings
})

# Save to CSV
output_file = os.path.join(output_dir, "step1_layer0_imageembeddings.csv")
df.to_csv(output_file, index=False)
print(f"Saved dataset to {output_file}")

print("Dataset creation complete!")