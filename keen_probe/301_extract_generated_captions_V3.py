import os
import h5py
import json
from tqdm import tqdm

def extract_captions_from_h5(h5_path, output_json_path):
    """Extract captions and associated metadata from a single HDF5 embeddings file and save as JSON."""
    print(f"Reading embeddings from: {h5_path}")
    print(f"Output will be saved to: {output_json_path}")
    
    # Dictionary to store all captions
    all_captions = {}
    
    try:
        with h5py.File(h5_path, 'r') as f:
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
            print(f"\nSaving {len(all_captions)} captions to {output_json_path}")
            with open(output_json_path, 'w') as f_json:
                json.dump(all_captions, f_json, indent=2)
            
            print("Done!")
    
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        raise

def extract_all_h5_to_json():
    h5_folder = "/root/projects/Thesis_LLAVA/keen_probe/keen_data/2014_extracted_embeddings"
    output_folder = "/root/projects/Thesis_LLAVA/keen_probe/keen_data/2014_extracted_captions"
    os.makedirs(output_folder, exist_ok=True)
    
    h5_files = [f for f in os.listdir(h5_folder) if f.endswith('.h5')]
    print(f"Found {len(h5_files)} .h5 files in {h5_folder}")
    
    for h5_file in h5_files:
        h5_path = os.path.join(h5_folder, h5_file)
        json_file = os.path.splitext(h5_file)[0] + ".json"
        output_json_path = os.path.join(output_folder, json_file)
        extract_captions_from_h5(h5_path, output_json_path)

if __name__ == "__main__":
    extract_all_h5_to_json() 