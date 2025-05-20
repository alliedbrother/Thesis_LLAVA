import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==========================
# CONFIGURATION
# ==========================

CONFIG = {
    # Paths
    "H5_FOLDER": "/root/project/Thesis_LLAVA/keen_probe/keen_data/paligemma_extracted_embeddings",
    "LABELS_CSV": "/root/project/Thesis_LLAVA/keen_probe/keen_data/labels_paligemma_f.csv",
    "SPLITS_DIR": "/root/project/Thesis_LLAVA/keen_probe/keen_data/splits_paligemma",
    "MODEL_DIR": "/root/project/Thesis_LLAVA/keen_probe/keen_data/models_binary_paligemma",
    "RESULTS_DIR": "/root/project/Thesis_LLAVA/keen_probe/keen_data/results_binary_paligemma",
    
    # Model parameters
    "REQ_EMBEDDINGS": "pre_generation/layer_0/image_embeddings",
    "INPUT_DIM": 2048,        # Length of input embeddings
    "LAYER_SIZES": [1024, 516, 256],   # List of hidden layer sizes
    
    # Device
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

class EmbeddingRegressor(nn.Module):
    def __init__(self, input_dim, layer_sizes, dropout_rate=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = size
        layers.append(nn.Linear(prev_dim, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_labels(labels_csv):
    """Load labels from CSV file."""
    labels_df = pd.read_csv(labels_csv)
    return dict(zip(labels_df['image_id'], labels_df['chair_has_hallucination']))

def load_split_ids(split_type):
    """Load image IDs for a specific split (train/val/test)."""
    split_file = os.path.join(CONFIG["SPLITS_DIR"], f'{split_type}_ids.csv')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    split_df = pd.read_csv(split_file)
    return set(split_df['image_id'].values)

def load_embeddings_and_scores(h5_folder, req_embeddings, labels_dict, split_ids):
    """Load embeddings and scores for a specific split."""
    all_embeddings = []
    all_scores = []
    all_ids = []
    total_files = 0
    total_images = 0
    total_matched = 0
    
    print(f"\nLoading embeddings from: {h5_folder}")
    print(f"Looking for path: {req_embeddings}")
    print(f"Number of images in split: {len(split_ids)}")
    
    for filename in os.listdir(h5_folder):
        if filename.endswith('.h5'):
            total_files += 1
            file_path = os.path.join(h5_folder, filename)
            with h5py.File(file_path, 'r') as f:
                file_images = 0
                file_matched = 0
                
                for image_group in f.keys():
                    file_images += 1
                    if image_group in split_ids:
                        try:
                            # Navigate through the nested groups
                            current = f[image_group]
                            for path_part in req_embeddings.split('/'):
                                current = current[path_part]
                            
                            # Get the embeddings
                            emb = current[:]
                            
                            all_embeddings.append(emb)
                            all_scores.append(labels_dict[image_group])
                            all_ids.append(image_group)
                            file_matched += 1
                            total_matched += 1
                        except Exception as e:
                            print(f"Error loading {image_group}: {str(e)}")
                
                total_images += file_images
                print(f"File: {filename}")
                print(f"  Total images: {file_images}")
                print(f"  Matched images: {file_matched}")
    
    print(f"\nProcessing complete:")
    print(f"Total files processed: {total_files}")
    print(f"Total images found: {total_images}")
    print(f"Total images matched: {total_matched}")
    
    if not all_embeddings:
        raise ValueError("No embeddings were successfully loaded. Check if the path exists in the HDF5 files.")
    
    return np.stack(all_embeddings), np.array(all_scores), all_ids

def evaluate_model(model, X, y, split_name):
    """Evaluate model performance on a dataset for binary classification."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(CONFIG["DEVICE"])
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(CONFIG["DEVICE"])
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        y_true = y_tensor.cpu().numpy().flatten().astype(int)

        # Calculate metrics
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_true, probs)
        except Exception:
            roc_auc = float('nan')

        metrics = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc_auc
        }

        print(f"\n{split_name} Set Metrics:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        return metrics, preds, probs

def plot_confusion(y_true, y_pred, split_name, save_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{split_name} Set Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{split_name.lower()}_confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_probs, split_name, save_dir):
    """Plot and save ROC curve."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{split_name} Set ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{split_name.lower()}_roc_curve.png'))
        plt.close()
    except Exception as e:
        print(f"Could not plot ROC curve for {split_name} set: {e}")

def save_evaluation_results(results, save_dir):
    """Save evaluation results to JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(save_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {metrics_path}")

def main():
    """Main function to evaluate the model."""
    print("Starting model evaluation...")
    
    # 1. Load labels
    labels_dict = load_labels(CONFIG["LABELS_CSV"])
    print(f"Loaded {len(labels_dict)} labels")
    
    # 2. Load model
    embedding_name = CONFIG["REQ_EMBEDDINGS"].replace('/', '_')
    model_path = os.path.join(CONFIG["MODEL_DIR"], f'best_model_{embedding_name}.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Initialize model
    checkpoint = torch.load(model_path)
    dropout_rate = checkpoint['hyperparams'].get('dropout_rates', 0.1)
    layer_sizes = checkpoint['hyperparams'].get('layer_sizes', CONFIG["LAYER_SIZES"])
    model = EmbeddingRegressor(
        input_dim=CONFIG["INPUT_DIM"],
        layer_sizes=layer_sizes,
        dropout_rate=dropout_rate
    ).to(CONFIG["DEVICE"])
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Load all splits
    train_ids = load_split_ids('train')
    val_ids = load_split_ids('val')
    test_ids = load_split_ids('test')
    
    print(f"Train set size: {len(train_ids)}")
    print(f"Validation set size: {len(val_ids)}")
    print(f"Test set size: {len(test_ids)}")
    
    # 4. Load embeddings and scores for each split
    X_train, y_train, train_image_ids = load_embeddings_and_scores(
        CONFIG["H5_FOLDER"],
        CONFIG["REQ_EMBEDDINGS"],
        labels_dict,
        train_ids
    )
    
    X_val, y_val, val_image_ids = load_embeddings_and_scores(
        CONFIG["H5_FOLDER"],
        CONFIG["REQ_EMBEDDINGS"],
        labels_dict,
        val_ids
    )
    
    X_test, y_test, test_image_ids = load_embeddings_and_scores(
        CONFIG["H5_FOLDER"],
        CONFIG["REQ_EMBEDDINGS"],
        labels_dict,
        test_ids
    )
    
    # 5. Evaluate on all sets
    train_metrics, train_preds, train_probs = evaluate_model(model, X_train, y_train, "Training")
    val_metrics, val_preds, val_probs = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics, test_preds, test_probs = evaluate_model(model, X_test, y_test, "Test")
    
    # 6. Plot results
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    plot_confusion(y_train, train_preds, "Training", CONFIG["RESULTS_DIR"])
    plot_confusion(y_val, val_preds, "Validation", CONFIG["RESULTS_DIR"])
    plot_confusion(y_test, test_preds, "Test", CONFIG["RESULTS_DIR"])
    plot_roc_curve(y_train, train_probs, "Training", CONFIG["RESULTS_DIR"])
    plot_roc_curve(y_val, val_probs, "Validation", CONFIG["RESULTS_DIR"])
    plot_roc_curve(y_test, test_probs, "Test", CONFIG["RESULTS_DIR"])
    
    # 7. Save results
    results = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'predictions': {
            'train': {
                'image_ids': train_image_ids,
                'true_labels': y_train.tolist(),
                'predicted_labels': train_preds.tolist(),
                'predicted_probs': train_probs.tolist()
            },
            'validation': {
                'image_ids': val_image_ids,
                'true_labels': y_val.tolist(),
                'predicted_labels': val_preds.tolist(),
                'predicted_probs': val_probs.tolist()
            },
            'test': {
                'image_ids': test_image_ids,
                'true_labels': y_test.tolist(),
                'predicted_labels': test_preds.tolist(),
                'predicted_probs': test_probs.tolist()
            }
        },
        'model_info': {
            'embeddings_path': CONFIG["REQ_EMBEDDINGS"],
            'model_path': model_path
        }
    }
    
    # Save results with embedding path in filename
    metrics_path = os.path.join(CONFIG["RESULTS_DIR"], f'evaluation_metrics_{embedding_name}.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {metrics_path}")
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 