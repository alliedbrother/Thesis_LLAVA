import json
import os
import sys

def extract_image_data(image_id):
    """Extract data for a specific image ID and save to JSON file."""
    # Get the directory of the current script
    base_dir = os.path.dirname(__file__)
    input_path = os.path.join(base_dir, "keen_data", "pre_generation_embeddings.json")
    output_path = os.path.join(base_dir, "keen_data", f"{image_id}.json")

    # Load embeddings
    with open(input_path, 'r') as f:
        embeddings = json.load(f)

    # Check if image ID exists
    if image_id not in embeddings:
        print(f"Error: Image ID {image_id} not found in embeddings file")
        sys.exit(1)

    # Extract data for the specific image
    image_data = embeddings[image_id]

    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(image_data, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_single_image.py <image_id>")
        sys.exit(1)
    
    image_id = sys.argv[1]
    extract_image_data(image_id) 