import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, 
    confusion_matrix, precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score
)
from torch.utils.data import DataLoader
from keen_probe import KEENProbe
from tqdm import tqdm

# Configuration
FACTUAL_THRESHOLD = 0.80  # Same as training
BATCH_SIZE = 16

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

base_dir = os.path.dirname(__file__)
datasets_dir = os.path.join(base_dir, "keen_data", "generated_datasets")
models_dir = os.path.join(base_dir, "keen_data", "models")
results_dir = os.path.join(base_dir, "keen_data", "results")
os.makedirs(results_dir, exist_ok=True)

# List of dataset files
dataset_files = [
    "dataset_vision_tower.csv",
    "dataset_initial_layer.csv",
    "dataset_middle_layer.csv",
    "dataset_final_layer.csv",
    "dataset_pre_generation.csv"
]

class EmbeddingDataset:
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def load_and_preprocess_dataset(file_path):
    print(f"\nLoading dataset: {file_path}")
    df = pd.read_csv(file_path)
    
    # Convert string embeddings to numpy arrays
    embeddings = []
    for emb_str in df['embedding']:
        emb_str = emb_str.strip('[]').strip()
        emb_array = np.fromstring(emb_str, sep=' ')
        embeddings.append(emb_array)
    
    # Convert to torch tensors
    embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
    
    # Create binary labels based on threshold
    labels = torch.tensor(
        (df['factual_correctness_score'] > FACTUAL_THRESHOLD).astype(int),
        dtype=torch.float32
    )
    
    # Print the number of 1's and 0's
    num_ones = int((labels == 1).sum().item())
    num_zeros = int((labels == 0).sum().item())
    print(f"Label distribution: 1's = {num_ones}, 0's = {num_zeros}")
    
    return embeddings, labels

def plot_precision_recall_curve(y_true, y_scores, model_name, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP={average_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, model_name, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, test_loader, model_name):
model.eval()
    all_preds = []
    all_scores = []
    all_labels = []

with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y = y.unsqueeze(1)
            scores = model(x)
            preds = (scores > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

# Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Only calculate these metrics if we have both classes
    unique_classes = np.unique(all_labels)
    if len(unique_classes) > 1:
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
    else:
        print(f"\nWarning: Only class {unique_classes[0]} present in test set.")
        print("Some metrics cannot be calculated.")
        precision = recall = f1 = auc = ap = None
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Only create PR curve if we have both classes
    if len(unique_classes) > 1:
        # Plot precision-recall curve
        pr_curve_path = os.path.join(results_dir, f"pr_curve_{model_name}.png")
        plot_precision_recall_curve(all_labels, all_scores, model_name, pr_curve_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(results_dir, f"confusion_matrix_{model_name}.png")
    plot_confusion_matrix(cm, model_name, cm_path)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results for {model_name}")
    print(f"{'='*50}")
print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=3))
    print(f"\nAccuracy: {accuracy:.3f}")
    
    if len(unique_classes) > 1:
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"ROC AUC: {auc:.3f}")
        print(f"Average Precision: {ap:.3f}")

    # Error analysis
    errors = np.where(all_preds != all_labels)[0]
if len(errors) > 0:
        print(f"\nError Analysis:")
        print(f"Found {len(errors)} misclassified samples")
        print("\nFirst 5 errors:")
        for i in range(min(5, len(errors))):
            idx = errors[i]
            print(f"Sample {idx}:")
            print(f"  True label: {all_labels[idx]}")
            print(f"  Predicted: {all_preds[idx]}")
            print(f"  Confidence: {float(all_scores[idx][0]):.3f}")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision) if precision is not None else None,
        'recall': float(recall) if recall is not None else None,
        'f1': float(f1) if f1 is not None else None,
        'auc': float(auc) if auc is not None else None,
        'ap': float(ap) if ap is not None else None,
        'confusion_matrix': cm.tolist()
    }

# Main evaluation loop
print("Starting evaluation...")
results = {}

for dataset_file in dataset_files:
    print(f"\n{'='*50}")
    print(f"Evaluating probe for {dataset_file}")
    print(f"{'='*50}")
    
    # Load dataset
    dataset_path = os.path.join(datasets_dir, dataset_file)
    embeddings, labels = load_and_preprocess_dataset(dataset_path)
    
    # Load split indices
    model_name = os.path.splitext(dataset_file)[0]
    split_path = os.path.join(models_dir, f"split_indices_{model_name}.json")
    with open(split_path, 'r') as f:
        split_indices = json.load(f)
    
    # Create test set using test indices
    test_indices = split_indices['test']
    X_test = embeddings[test_indices]
    y_test = labels[test_indices]
    
    test_set = EmbeddingDataset(X_test, y_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    
    print(f"Test set size: {len(X_test)}")
    
    # Load model
    model_path = os.path.join(models_dir, f"probe_{model_name}.pt")
    
    print(f"Loading model from {model_path}")
    model = KEENProbe(input_dim=embeddings.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Evaluate model
    model_results = evaluate_model(model, test_loader, model_name)
    results[model_name] = model_results

# Save all results to JSON
results_path = os.path.join(results_dir, "evaluation_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)

print("\nEvaluation completed!")
print(f"Results saved to {results_path}")
