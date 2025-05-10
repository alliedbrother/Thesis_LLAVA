import os
import h5py
import json
from tqdm import tqdm

# Constants
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "keen_data", "generation_embeddings_V4Parallel.h5")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "keen_data", "generated_captions_V4Parallel.json")

def extract_captions():
    """Extract captions and associated metadata from the HDF5 embeddings file."""
    print(f"Reading embeddings from: {EMBEDDINGS_PATH}")
    print(f"Output will be saved to: {OUTPUT_PATH}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Dictionary to store all captions
    all_captions = {}
    
    try:
        with h5py.File(EMBEDDINGS_PATH, 'r') as f:
            # Get all image IDs
            image_ids = list(f.keys())
            print(f"Found {len(image_ids)} images")
            
            # Process each image
            for image_id in tqdm(image_ids, desc="Extracting captions"):
                try:
                    # Get the group for this image
                    group = f[image_id]
                    
                    # Print the structure of the first image for debugging
                    if image_id == image_ids[0]:
                        print("\nStructure of first image:")
                        print("Keys:", list(group.keys()))
                        print("Datasets:", {k: group[k].shape for k in group.keys() if isinstance(group[k], h5py.Dataset)})
                        print("Groups:", {k: list(group[k].keys()) for k in group.keys() if isinstance(group[k], h5py.Group)})
                    
                    # Extract caption and tokens_length directly from the group
                    caption = group['generated_caption'][()].decode('utf-8')
                    tokens_length = int(group['tokens_length'][()])
                
                    # Store in dictionary
                    all_captions[image_id] = {
                        'generated_caption': caption,
                        'tokens_length': tokens_length
                    }
                    
                except Exception as e:
                    print(f"Error processing image {image_id}: {str(e)}")
                    print(f"Available keys: {list(group.keys())}")
                    if len(group.keys()) > 0:
                        print("First key type:", type(group[list(group.keys())[0]]))
                    continue
            
            # Save to JSON file
            print(f"\nSaving {len(all_captions)} captions to {OUTPUT_PATH}")
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(all_captions, f, indent=2)
            
            print("Done!")
    
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        raise

if __name__ == "__main__":
    extract_captions() 