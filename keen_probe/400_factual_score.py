import json
import os
import sys
import pandas as pd
sys.path.append('coco-caption')

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# Create a directory for results if it doesn't exist
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load ground truth
coco = COCO('/root/project/Thesis_LLAVA/coco-caption/annotations/captions_val2014.json')
print(f"Number of images in ground truth: {len(coco.imgs)}")
print(f"Sample image IDs: {list(coco.imgs.keys())[:5]}")

json_path = "/root/project/Thesis_LLAVA/keen_probe/keen_data/2014_extracted_captions/merged_top_15k.json"

with open(json_path, "r") as f:
    data = json.load(f)

preds = []
for image_id, info in data.items():
    preds.append({
        "image_id": image_id,
        "caption": info["generated_caption"]
    })

# preds is now a list of dicts with image_id and caption
print(preds[:2])  # Show a sample

# Filter out image IDs that don't exist in the dataset
valid_preds = []
for pred in preds:
    img_id = pred['image_id']
    if img_id in coco.imgs:
        valid_preds.append(pred)
    else:
        print(f"Warning: Image ID {img_id} not found in dataset, skipping.")

print(f"Evaluating {len(valid_preds)} captions (filtered from {len(preds)} original captions)")

if not valid_preds:
    print("No valid image IDs found. Exiting.")
    sys.exit(1)

# Write to a temporary results file
results_file = os.path.join(results_dir, 'temp_results.json')
with open(results_file, 'w') as f:
    json.dump(valid_preds, f)

# Initialize COCO evaluation
coco_res = coco.loadRes(results_file)
coco_eval = COCOEvalCap(coco, coco_res)

# Set the image IDs to evaluate
coco_eval.params['image_id'] = [pred['image_id'] for pred in valid_preds]

# Evaluate
coco_eval.evaluate()

# Print results
print("\nResults:")
for metric, score in coco_eval.eval.items():
    print(f"{metric}: {score:.3f}")

# Write per-image results to labels.csv in keen_data
data_rows = []
for eval_img in coco_eval.evalImgs:
    row = {'image_id': eval_img['image_id']}
    for metric in coco_eval.eval.keys():
        row[metric] = eval_img.get(metric, None)
    data_rows.append(row)

labels_csv_path = os.path.join('keen_data', 'labels.csv')
df = pd.DataFrame(data_rows)
df.to_csv(labels_csv_path, index=False)
print(f"Per-image evaluation results saved to {labels_csv_path}")