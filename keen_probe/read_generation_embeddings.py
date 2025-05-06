import h5py
import json
import numpy as np
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def format_embeddings(embeddings):
    """Format embeddings to show size instead of full content."""
    if isinstance(embeddings, np.ndarray):
        return f"Array of shape {embeddings.shape}"
    return embeddings

def read_hdf5_structure(group, indent=0):
    """Recursively read HDF5 structure and format it."""
    result = {}
    
    # Read attributes
    if group.attrs:
        attrs = {}
        for key in group.attrs:
            value = group.attrs[key]
            if isinstance(value, np.ndarray):
                if value.dtype.kind == 'U':  # String array
                    attrs[key] = value.tolist()
                else:
                    attrs[key] = f"Array of shape {value.shape}"
            else:
                attrs[key] = value
        result['attributes'] = attrs
    
    # Read datasets and groups
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            # For datasets, store size and attributes
            dataset_info = {
                'type': 'dataset',
                'shape': item.shape,
                'dtype': str(item.dtype)
            }
            
            # Add dataset attributes if any
            if item.attrs:
                attrs = {}
                for attr_key in item.attrs:
                    value = item.attrs[attr_key]
                    if isinstance(value, np.ndarray):
                        attrs[attr_key] = f"Array of shape {value.shape}"
                    else:
                        attrs[attr_key] = value
                dataset_info['attributes'] = attrs
            
            result[key] = dataset_info
        elif isinstance(item, h5py.Group):
            # For groups, recursively read their structure
            result[key] = read_hdf5_structure(item, indent + 2)
    
    return result

def main():
    # File paths
    h5_path = "/home/luk_lab/Desktop/Luk Lab - AI/projects/thesis/Thesis_LLAVA/keen_probe/keen_data/generation_embeddings.h5"
    output_path = "/home/luk_lab/Desktop/Luk Lab - AI/projects/thesis/Thesis_LLAVA/keen_probe/keen_data/generation_embeddings_readable.json"
    
    print(f"Reading HDF5 file from: {h5_path}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Read the entire structure
            structure = read_hdf5_structure(f)
            
            # Write to JSON file with custom encoder
            with open(output_path, 'w') as outfile:
                json.dump(structure, outfile, indent=2, cls=NumpyEncoder)
            
            print(f"\nSuccessfully read HDF5 file and wrote formatted content to: {output_path}")
            
            # Print some basic statistics
            print("\nBasic Statistics:")
            print("-" * 50)
            print(f"Number of images processed: {len(f.keys())}")
            
            # Get first image group for example
            first_image = next(iter(f.values()))
            if isinstance(first_image, h5py.Group):
                print(f"\nExample structure for first image:")
                print(f"Number of generation steps: {len(first_image['generation_steps'])}")
                if 'attributes' in first_image:
                    print("\nImage attributes:")
                    for key, value in first_image['attributes'].items():
                        print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error reading HDF5 file: {str(e)}")
        raise

if __name__ == "__main__":
    main() 