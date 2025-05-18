import os
import h5py
import json
import numpy as np

def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    return obj

def extract_embeddings(h5_path, image_id, output_path):
    """Extract all embeddings for a single image from the HDF5 file."""
    try:
        with h5py.File(h5_path, 'r') as f:
            if image_id not in f:
                print(f"Image {image_id} not found in the HDF5 file.")
                return
            
            # Create a dictionary to store all data
            data = {}
            
            # Get the group for this image
            img_group = f[image_id]
            
            # Extract basic metadata
            data['image_id'] = convert_to_serializable(img_group['image_id'][()])
            data['generated_caption'] = convert_to_serializable(img_group['generated_caption'][()])
            data['tokens_length'] = convert_to_serializable(img_group['tokens_length'][()])
            
            # Extract vision embeddings
            data['vision_embeddings'] = convert_to_serializable(img_group['vision_embeddings'][()])
            
            # Extract pre-generation embeddings
            data['pre_generation'] = {}
            pre_gen_group = img_group['pre_generation']
            for layer_name in pre_gen_group:
                data['pre_generation'][layer_name] = {
                    'image_embeddings': convert_to_serializable(pre_gen_group[layer_name]['image_embeddings'][()]),
                    'query_embeddings': convert_to_serializable(pre_gen_group[layer_name]['query_embeddings'][()])
                }
            
            # Extract post-generation embeddings
            data['post_generation'] = {}
            post_gen_group = img_group['post_generation']
            for step_name in post_gen_group:
                data['post_generation'][step_name] = {}
                for layer_name in post_gen_group[step_name]:
                    data['post_generation'][step_name][layer_name] = {
                        'image_embeddings': convert_to_serializable(post_gen_group[step_name][layer_name]['image_embeddings'][()]),
                        'query_embeddings': convert_to_serializable(post_gen_group[step_name][layer_name]['query_embeddings'][()]),
                        'post_gen_embeddings': convert_to_serializable(post_gen_group[step_name][layer_name]['post_gen_embeddings'][()])
                    }
            
            # Save to JSON file
            output_file = os.path.join(output_path, f"{image_id}_embeddings.json")
            with open(output_file, 'w') as json_file:
                json.dump(data, json_file, indent=2)
            
            print(f"Successfully extracted embeddings for image {image_id} to {output_file}")
            
    except Exception as e:
        print(f"Error extracting embeddings: {str(e)}")

def main():
    # Define paths
    h5_path = "/root/project/Thesis_LLAVA/keen_probe/keen_data/2014_extracted_embeddings/generation_embeddings_V41_part1.h5"
    output_path = "/root/project/Thesis_LLAVA/keen_probe/keen_data/extracted_image_details/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Read the first image ID from the HDF5 file
    with h5py.File(h5_path, 'r') as f:
        image_ids = list(f.keys())
        if not image_ids:
            print("No images found in the HDF5 file.")
            return
        first_image_id = image_ids[0]
    
    # Extract embeddings for the first image
    extract_embeddings(h5_path, first_image_id, output_path)

if __name__ == "__main__":
    main() 