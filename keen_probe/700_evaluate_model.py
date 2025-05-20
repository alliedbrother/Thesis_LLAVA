import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==========================
# CONFIGURATION
# ==========================

CONFIG = {
    # Paths
    "H5_FOLDER": "/root/projects/Thesis_LLAVA/keen_probe/keen_data/2014_extracted_embeddings",
    "LABELS_CSV": "/root/projects/Thesis_LLAVA/keen_probe/keen_data/labels.csv",
    "SPLITS_DIR": "/root/projects/Thesis_LLAVA/keen_probe/keen_data/splits",
    "MODEL_DIR": "/root/projects/Thesis_LLAVA/keen_probe/keen_data/models_regression",
    "RESULTS_DIR": "/root/projects/Thesis_LLAVA/keen_probe/keen_data/results",
    
    # Model parameters
    "REQ_EMBEDDINGS": "post_generation/step_final/layer_0/post_gen_embeddings",
    "INPUT_DIM": 4096,        # Length of input embeddings
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
    return dict(zip(labels_df['image_id'], labels_df['chair_regression_score']))

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
    """Evaluate model performance on a dataset."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(CONFIG["DEVICE"])
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(CONFIG["DEVICE"])
        y_pred = model(X_tensor)
        
        # Convert tensors to numpy arrays for metric calculation
        y_np = y_tensor.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        # Calculate metrics
        mse = F.mse_loss(y_pred, y_tensor).item()
        rmse = np.sqrt(mse)
        r2 = r2_score(y_np, y_pred_np)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"\n{split_name} Set Metrics:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²: {r2:.6f}")
        
        return metrics, y_pred_np

def plot_confusion_matrix(y_true, y_pred, split_name, save_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Scores')
    plt.ylabel('Predicted Scores')
    plt.title(f'{split_name} Set: True vs Predicted Scores')
    
    # Add R² value to plot
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{split_name.lower()}_predictions.png'))
    plt.close()

def plot_combined_scatter(y_train, train_preds, y_val, val_preds, y_test, test_preds, save_path):
    """Plot and save combined scatter plot for train, val, and test sets."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, train_preds, alpha=0.5, label='Train', color='blue')
    plt.scatter(y_val, val_preds, alpha=0.5, label='Validation', color='orange')
    plt.scatter(y_test, test_preds, alpha=0.5, label='Test', color='green')
    
    all_y_true = np.concatenate([y_train, y_val, y_test])
    plt.plot([min(all_y_true), max(all_y_true)], [min(all_y_true), max(all_y_true)], 'r--')
    plt.xlabel('True Scores')
    plt.ylabel('Predicted Scores')
    plt.title('True vs Predicted Scores (Train/Val/Test)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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
    train_metrics, train_preds = evaluate_model(model, X_train, y_train, "Training")
    val_metrics, val_preds = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics, test_preds = evaluate_model(model, X_test, y_test, "Test")
    
    # 6. Plot results
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    plot_name = f'combined_predictions_{embedding_name}.png'
    plot_path = os.path.join(CONFIG["RESULTS_DIR"], plot_name)
    plot_combined_scatter(y_train, train_preds, y_val, val_preds, y_test, test_preds, plot_path)
    
    # 7. Save results
    results = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'predictions': {
            'train': {
                'image_ids': train_image_ids,
                'true_scores': y_train.tolist(),
                'predicted_scores': train_preds.tolist()
            },
            'validation': {
                'image_ids': val_image_ids,
                'true_scores': y_val.tolist(),
                'predicted_scores': val_preds.tolist()
            },
            'test': {
                'image_ids': test_image_ids,
                'true_scores': y_test.tolist(),
                'predicted_scores': test_preds.tolist()
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