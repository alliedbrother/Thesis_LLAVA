import os
import json
import csv
from collections import defaultdict

# Define paths
base_dir = os.path.dirname(__file__)
ann_path = os.path.abspath(os.path.join(base_dir, "..", "annotations", "captions_val2017.json"))
csv_output_path = os.path.join(base_dir, "keen_data", "coco_val2017_captions.csv")

# Load annotations
with open(ann_path, 'r') as f:
    coco_data = json.load(f)

# Create mapping: image_id → list of captions
id_to_captions = defaultdict(list)
for ann in coco_data["annotations"]:
    image_id = f"{ann['image_id']:012d}.jpg"  # Format: 000000000123.jpg
    id_to_captions[image_id].append(ann["caption"])

# Write to CSV
with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "captions"])  # header
    for img_id, caps in id_to_captions.items():
        joined = " ||| ".join(caps)  # use delimiter to split later
        writer.writerow([img_id, joined])

print(f"✅ Captions written to: {csv_output_path}")

