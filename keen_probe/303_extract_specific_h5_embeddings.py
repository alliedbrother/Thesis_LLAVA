import h5py
import json
import numpy as np
from pathlib import Path

def print_group_structure(group, indent=0):
    """Recursively print the structure of an HDF5 group."""
    for key in group:
        print("  " * indent + key)
        item = group[key]
        if isinstance(item, h5py.Group):
            print_group_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print("  " * (indent + 1) + f"Dataset shape: {item.shape}, dtype: {item.dtype}")

def extract_embeddings():
    # File paths
    h5_path = "/home/luk_lab/Desktop/Luk Lab - AI/projects/thesis/Thesis_LLAVA/keen_probe/keen_data/generation_embeddings.h5"
    output_path = "/home/luk_lab/Desktop/Luk Lab - AI/projects/thesis/Thesis_LLAVA/keen_probe/keen_data/specific_embeddings_step_15_token_29.json"
    # Target image ID
    target_image_id = "000000000872"
    
    results = {
        "sequence_embeddings": {},
        "vision_query_embeddings": {}
    }
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print("\nAvailable image IDs:")
            print("-" * 50)
            for key in f.keys():
                print(key)
            
            if target_image_id not in f:
                print(f"\nError: Image ID {target_image_id} not found in the file")
                return
            
            # Get the target image group
            image_group = f[target_image_id]
            
            print(f"\nStructure for image {target_image_id}:")
            print("-" * 50)
            print_group_structure(image_group)
            
            # Get step 0 group
            steps_group = image_group.get('generation_steps')
            if steps_group is None:
                print("\nError: No generation_steps group found")
                return
            
            step_15 = steps_group.get('step_15')
            if step_15 is None:
                print("\nError: step_15 not found")
                return
            
            print("\nStructure of step_15:")
            print("-" * 50)
            print_group_structure(step_15)
            
            # Extract sequence embeddings for layers 0, 1, 16, 31, 32
            seq_emb_group = step_15.get('sequence_embeddings')
            if seq_emb_group is not None:
                for layer in [0, 1, 16, 31, 32]:
                    layer_group = seq_emb_group.get(str(layer))
                    if layer_group is not None:
                        # Find the token position 29 dataset
                        for key in layer_group.keys():
                            if 'token_29' in key or '29' in key:
                                embedding = layer_group[key][()]
                                results["sequence_embeddings"][f"layer_{layer}"] = {
                                    "path": f"{layer}/{key}",
                                    "shape": embedding.shape,
                                    "data": embedding.tolist()
                                }
            
            # Extract vision query embeddings for layers 0, 1, 16, 31, 32
            vision_query_group = step_15.get('vision_query_embeddings')
            if vision_query_group is not None:
                for key in vision_query_group.keys():
                    if '29' in key:  # Look for position 29
                        for layer in [0, 1, 16, 31, 32]:
                            if f"_{layer}_" in key:
                                embedding = vision_query_group[key][()]
                                results["vision_query_embeddings"][f"layer_{layer}"] = {
                                    "path": key,
                                    "shape": embedding.shape,
                                    "data": embedding.tolist()
                                }
            
            # Save results to JSON file
            with open(output_path, 'w') as outfile:
                json.dump(results, outfile, indent=2)
            
            print(f"\nExtracted embeddings summary:")
            print("-" * 50)
            print("Sequence embeddings found for layers:", list(results["sequence_embeddings"].keys()))
            print("Vision query embeddings found for layers:", list(results["vision_query_embeddings"].keys()))
            print(f"\nResults saved to: {output_path}")
    
    except Exception as e:
        print(f"Error processing HDF5 file: {str(e)}")
        raise

if __name__ == "__main__":
    extract_embeddings() 