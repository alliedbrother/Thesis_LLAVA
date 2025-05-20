import os
import json
from glob import glob

# Directory containing the JSON files
input_dir = '/root/projects/Thesis_LLAVA/keen_probe/keen_data/2014_extracted_captions'
output_file = os.path.join(input_dir, 'merged_all.json')

merged_data = {}

# Get all JSON files in the directory
json_files = glob(os.path.join(input_dir, '*.json'))

for file_path in json_files:
    with open(file_path, 'r') as f:
        data = json.load(f)
        merged_data.update(data)

# Write merged data to a new file
with open(output_file, 'w') as f:
    json.dump(merged_data, f, indent=4)

# Print the total count of unique image IDs
print(f"Total unique image IDs: {len(merged_data)}")
print(f"Merged {len(json_files)} files into {output_file}")