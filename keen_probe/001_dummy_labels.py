import os
import pandas as pd
import numpy as np

# Folder containing images
image_folder = '/root/project/Thesis_LLAVA/coco_val2014'

# Find all image files (assuming .jpg extension; modify as needed)
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# Extract image IDs (remove extension)
image_ids = [os.path.splitext(f)[0] for f in image_files]

# Assign random scores (between 0 and 1)
scores = np.random.rand(len(image_ids))

# Create DataFrame
df = pd.DataFrame({
    'image_id': image_ids,
    'score': scores
})

# Write to CSV
df.to_csv('keen_data/labels.csv', index=False)

print(f"Wrote {len(df)} rows to labels.csv")
