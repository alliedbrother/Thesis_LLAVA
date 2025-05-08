import h5py
import json
import numpy as np
from pathlib import Path

def extract_captions():
    # File paths
    h5_path = "/home/luk_lab/Desktop/Luk Lab - AI/projects/thesis/Thesis_LLAVA/keen_probe/keen_data/generation_embeddings_V4.h5"
    output_path = "/home/luk_lab/Desktop/Luk Lab - AI/projects/thesis/Thesis_LLAVA/keen_probe/keen_data/generated_captions_V4.json"
    
    captions_data = {}
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print("\nExtracting captions from HDF5 file...")
            
            # Iterate through all image groups
            for image_id in f.keys():
                image_group = f[image_id]
                
                # Get only the caption
                caption = image_group.attrs.get('caption', '')
                
                # Store in dictionary
                captions_data[image_id] = caption
            
            # Save to JSON file
            with open(output_path, 'w') as outfile:
                json.dump(captions_data, outfile, indent=2)
            
            print(f"\nExtracted {len(captions_data)} captions")
            print(f"Results saved to: {output_path}")
            
            # Print a sample caption
            if captions_data:
                sample_id = list(captions_data.keys())[0]
                print("\nSample caption:")
                print(f"Image ID: {sample_id}")
                print(f"Caption: {captions_data[sample_id]}")
    
    except Exception as e:
        print(f"Error processing HDF5 file: {str(e)}")
        raise

if __name__ == "__main__":
    extract_captions() 