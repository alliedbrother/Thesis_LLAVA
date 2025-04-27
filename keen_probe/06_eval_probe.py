import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from utils import EmbeddingDataset
from keen_probe import KEENProbe

print("Starting evaluation...")

base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, "keen_data", "dataset.json")

# Load and prepare data
print("\nLoading dataset...")
with open(dataset_path) as f:
    data = json.load(f)

# Calculate split indices
total_samples = len(data)
test_start_idx = int(0.8 * total_samples)
test_data = data[test_start_idx:]
test_set = EmbeddingDataset(test_data)
loader = DataLoader(test_set, batch_size=16)

print(f"Total samples: {total_samples}")
print(f"Test samples: {len(test_data)}")

# Load model
print("\nLoading model...")
model = KEENProbe(4096)  # Running on CPU
model.load_state_dict(torch.load(os.path.join(base_dir, "keen_data", "probe.pt"), map_location=torch.device('cpu')))
model.eval()

# Evaluate
print("\nRunning evaluation...")
true_labels, pred_labels, pred_scores = [], [], []
with torch.no_grad():
    for x, y in loader:
        scores = model(x).squeeze().numpy()
        binary_preds = (scores > 0.5).astype(int)
        pred_scores.extend(scores)
        pred_labels.extend(binary_preds)
        true_labels.extend(y.numpy())

# Convert to numpy arrays for easier handling
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)
pred_scores = np.array(pred_scores)

# Calculate metrics
print("\n✅ Evaluation Results:")
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, digits=3))

print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, pred_labels)
print(cm)

# Calculate additional metrics
accuracy = accuracy_score(true_labels, pred_labels)
if len(np.unique(true_labels)) > 1:  # Only calculate AUC if we have both classes
    auc = roc_auc_score(true_labels, pred_scores)
    print(f"\nROC AUC Score: {auc:.3f}")

print(f"Accuracy: {accuracy:.3f}")

# Error Analysis
print("\nError Analysis:")
errors = np.where(pred_labels != true_labels)[0]
if len(errors) > 0:
    print(f"Found {len(errors)} misclassified samples:")
    for idx in errors:
        print(f"Sample {test_data[idx]['image']}:")
        print(f"  True label: {true_labels[idx]}")
        print(f"  Predicted label: {pred_labels[idx]}")
        print(f"  Confidence score: {pred_scores[idx]:.3f}")
else:
    print("No classification errors found!")

# Distribution of prediction scores
print("\nPrediction Score Distribution:")
print(f"Mean score: {np.mean(pred_scores):.3f}")
print(f"Std score: {np.std(pred_scores):.3f}")
print(f"Min score: {np.min(pred_scores):.3f}")
print(f"Max score: {np.max(pred_scores):.3f}")

print("\nEvaluation completed!")
