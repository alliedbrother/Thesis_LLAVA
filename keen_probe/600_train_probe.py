import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
import json
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools
from datetime import datetime
import logging
import sys

# ==========================
# CONFIGURATION
# ==========================

CONFIG = {
    # Paths
    "H5_FOLDER": "/root/project/Thesis_LLAVA/keen_probe/keen_data/paligemma_extracted_embeddings",
    "H5_FILE": "generation_embeddings.h5",  # Use only this file
    "LABELS_CSV": "/root/project/Thesis_LLAVA/keen_probe/keen_data/labels_paligemma_f.csv",
    "SPLITS_DIR": "/root/project/Thesis_LLAVA/keen_probe/keen_data/splits_paligemma",
    "MODEL_SAVE_DIR": "/root/project/Thesis_LLAVA/keen_probe/keen_data/models_regression_paligemma",
    "LOG_DIR": "/root/project/Thesis_LLAVA/keen_probe/keen_data/logs_paligemma",
    
    # Model parameters
    "REQ_EMBEDDINGS": "pre_generation/layer_0/query_embeddings",
    "INPUT_DIM": 2048,        # Length of input embeddings
    
    # Hyperparameter search space
    "HYPERPARAMS": {
        "layer_sizes": [
            [1024, 516, 256],
        ],
        "learning_rates": [0.01,0.05,0.1],
        "batch_sizes": [1000,3000],
        "dropout_rates": [0.1]
    },
    
    # Training parameters
    "EPOCHS": 50,
    "EARLY_STOPPING_PATIENCE": 5,
    "MIN_DELTA": 0.001,
    "LR_PATIENCE": 3,
    "LR_FACTOR": 0.5,
    
    # Cross-validation
    "N_SPLITS": 5,
    
    # Device
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

def setup_logging():
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
    
    # Create a timestamp for the log file
    embedding_name = CONFIG["REQ_EMBEDDINGS"].replace('/', '_')
    log_file = os.path.join(CONFIG["LOG_DIR"], f'training_log_{embedding_name}.txt')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

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

def load_embeddings_and_scores(h5_folder, h5_file, req_embeddings, labels_dict, split_ids):
    """Load embeddings and scores for a specific split, with detailed error reporting."""
    all_embeddings = []
    all_scores = []
    missing_image_ids = []
    missing_embeddings = []

    h5_path = os.path.join(h5_folder, h5_file)
    print(f"Opening H5 file: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        all_h5_ids = set(f.keys())
        for image_group in split_ids:
            if image_group in f:
                try:
                    current = f[image_group]
                    for path_part in req_embeddings.split('/'):
                        current = current[path_part]
                    emb = current[:]
                    all_embeddings.append(emb)
                    all_scores.append(labels_dict[image_group])
                except KeyError as e:
                    print(f"[ERROR] Embedding path missing for {image_group} in file {h5_path}: {e}")
                    missing_embeddings.append(image_group)
                except Exception as e:
                    print(f"[ERROR] Unexpected error for {image_group} in file {h5_path}: {e}")
            else:
                print(f"[ERROR] Image ID {image_group} not found in H5 file {h5_path}.")
                missing_image_ids.append(image_group)

    print(f"Total missing image IDs: {len(missing_image_ids)}")
    print(f"Total missing embeddings: {len(missing_embeddings)}")

    if not all_embeddings:
        raise ValueError("No embeddings were successfully loaded. Check if the path exists in the HDF5 files and if the split IDs match the H5 keys.")

    return np.stack(all_embeddings), np.array(all_scores)

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

def train_model(X_train, y_train, X_val, y_val, hyperparams):
    """Train the model with validation monitoring and learning rate scheduling."""
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_sizes'], shuffle=True)
    
    # Initialize model
    model = EmbeddingRegressor(
        input_dim=CONFIG["INPUT_DIM"],
        layer_sizes=hyperparams['layer_sizes'],
        dropout_rate=hyperparams['dropout_rates']
    ).to(CONFIG["DEVICE"])
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rates'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CONFIG["LR_FACTOR"], 
                                patience=CONFIG["LR_PATIENCE"], verbose=False)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(CONFIG["DEVICE"]), y_batch.to(CONFIG["DEVICE"])
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_dataset)
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_tensor = X_val_tensor.to(CONFIG["DEVICE"])
            y_val_tensor = y_val_tensor.to(CONFIG["DEVICE"])
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Early stopping check
        if val_loss < best_val_loss - CONFIG["MIN_DELTA"]:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["EARLY_STOPPING_PATIENCE"]:
                break
    
    return model, history

def save_checkpoint(checkpoint, hyperparams):
    """Save model checkpoint with hyperparameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparam_str = "_".join([f"{k}_{v}" for k, v in hyperparams.items()])
    checkpoint_path = os.path.join(
        CONFIG["MODEL_SAVE_DIR"], 
        f'checkpoint_{timestamp}_{hyperparam_str}.pt'
    )
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def load_best_model():
    """Load the best model based on validation loss."""
    checkpoints = [f for f in os.listdir(CONFIG["MODEL_SAVE_DIR"]) if f.startswith('checkpoint_')]
    if not checkpoints:
        raise ValueError("No checkpoints found!")
    
    best_val_loss = float('inf')
    best_checkpoint = None
    
    for checkpoint_file in checkpoints:
        checkpoint = torch.load(os.path.join(CONFIG["MODEL_SAVE_DIR"], checkpoint_file))
        if checkpoint['val_loss'] < best_val_loss:
            best_val_loss = checkpoint['val_loss']
            best_checkpoint = checkpoint
    
    # Recreate model with best hyperparameters
    model = EmbeddingRegressor(
        input_dim=CONFIG["INPUT_DIM"],
        layer_sizes=best_checkpoint['hyperparams']['layer_sizes'],
        dropout_rate=best_checkpoint['hyperparams']['dropout_rate']
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    return model, best_checkpoint

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
        
        return metrics

def save_model_and_metrics(model, train_metrics, val_metrics, model_save_dir):
    """Save the model and metrics."""
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Create model filename based on embedding path
    embedding_name = CONFIG["REQ_EMBEDDINGS"].replace('/', '_')
    model_path = os.path.join(model_save_dir, f'probe_model_{embedding_name}.pt')
    
    # Save model
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'model_params': {
            'input_dim': CONFIG["INPUT_DIM"],
            'layer_sizes': CONFIG["HYPERPARAMS"]["layer_sizes"][0],
            'learning_rates': CONFIG["HYPERPARAMS"]["learning_rates"][0],
            'batch_sizes': CONFIG["HYPERPARAMS"]["batch_sizes"][0],
            'dropout_rates': CONFIG["HYPERPARAMS"]["dropout_rates"][0],
            'epochs': CONFIG["EPOCHS"],
            'embeddings_path': CONFIG["REQ_EMBEDDINGS"]
        }
    }
    
    metrics_path = os.path.join(model_save_dir, f'training_metrics_{embedding_name}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")

def log_dataset_info(h5_folder, split_ids):
    """Log dataset information."""
    logging.info("\nDataset Information:")
    logging.info("-" * 50)
    logging.info(f"Number of images in split: {len(split_ids)}")
    
    total_files = 0
    total_images = 0
    total_matched = 0
    
    for filename in os.listdir(h5_folder):
        if filename.endswith('.h5'):
            total_files += 1
            file_path = os.path.join(h5_folder, filename)
            with h5py.File(file_path, 'r') as f:
                file_images = len(f.keys())
                file_matched = sum(1 for img_id in f.keys() if img_id in split_ids)
                
                logging.info(f"File: {filename}")
                logging.info(f"  Total images: {file_images}")
                logging.info(f"  Matched images: {file_matched}")
                
                total_images += file_images
                total_matched += file_matched
    
    logging.info("\nProcessing complete:")
    logging.info(f"Total files processed: {total_files}")
    logging.info(f"Total images found: {total_images}")
    logging.info(f"Total images matched: {total_matched}")
    logging.info("-" * 50)

def save_best_model_info(best_model_path, best_hyperparams, best_val_loss):
    """Save information about the best model to a text file."""
    info_path = os.path.join(CONFIG["MODEL_SAVE_DIR"], "best_model_info.txt")
    with open(info_path, 'w') as f:
        f.write("Best Model Information\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Path: {best_model_path}\n")
        f.write(f"Validation Loss: {best_val_loss:.6f}\n\n")
        f.write("Hyperparameters:\n")
        for param, value in best_hyperparams.items():
            f.write(f"- {param}: {value}\n")
        f.write("\nModel Architecture:\n")
        f.write(f"- Input Dimension: {CONFIG['INPUT_DIM']}\n")
        f.write(f"- Layer Sizes: {best_hyperparams['layer_sizes']}\n")
        f.write(f"- Dropout Rate: {best_hyperparams['dropout_rates']}\n")
    
    logging.info(f"\nBest model information saved to: {info_path}")

def main():
    """Main function to train the probe with hyperparameter tuning."""
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting probe training with hyperparameter tuning...")
    logging.info(f"Log file: {log_file}")
    
    # Print detailed device information
    logging.info("\nDevice Information:")
    logging.info("=" * 50)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logging.info(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        try:
            # Get CUDA device count
            device_count = torch.cuda.device_count()
            logging.info(f"Number of CUDA devices: {device_count}")
            
            # Get current CUDA device
            current_device = torch.cuda.current_device()
            logging.info(f"Current CUDA device index: {current_device}")
            
            # Get device name
            device_name = torch.cuda.get_device_name(current_device)
            logging.info(f"CUDA device name: {device_name}")
            
            # Get CUDA version
            cuda_version = torch.version.cuda
            logging.info(f"CUDA version: {cuda_version}")
            
            # Get PyTorch CUDA version
            pytorch_cuda = torch.__version__
            logging.info(f"PyTorch version: {pytorch_cuda}")
            
            # Set device to GPU
            device = torch.device("cuda")
            torch.cuda.set_device(current_device)
            
            # Test CUDA with a small tensor
            test_tensor = torch.tensor([1.0], device=device)
            logging.info("CUDA test tensor created successfully")
            
        except Exception as e:
            logging.error(f"Error initializing CUDA: {str(e)}")
            logging.info("Falling back to CPU")
            device = torch.device("cpu")
    else:
        logging.warning("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")
    
    logging.info(f"Final device being used: {device}")
    logging.info("=" * 50)
    
    # Update CONFIG with the actual device
    CONFIG["DEVICE"] = device
    
    # 1. Load labels and data
    labels_dict = load_labels(CONFIG["LABELS_CSV"])
    train_ids = load_split_ids('train')
    val_ids = load_split_ids('val')
    
    # Print dataset information once at the start
    logging.info(f"\nLoading embeddings from: {CONFIG['H5_FOLDER']}")
    logging.info(f"Looking for path: {CONFIG['REQ_EMBEDDINGS']}")
    logging.info(f"Number of images in split: {len(train_ids)}")
    
    total_files = 0
    total_images = 0
    total_matched = 0
    
    for filename in os.listdir(CONFIG["H5_FOLDER"]):
        if filename.endswith('.h5'):
            total_files += 1
            file_path = os.path.join(CONFIG["H5_FOLDER"], filename)
            with h5py.File(file_path, 'r') as f:
                file_images = len(f.keys())
                file_matched = sum(1 for img_id in f.keys() if img_id in train_ids)
                
                logging.info(f"File: {filename}")
                logging.info(f"  Total images: {file_images}")
                logging.info(f"  Matched images: {file_matched}")
                
                total_images += file_images
                total_matched += file_matched
    
    logging.info("\nProcessing complete:")
    logging.info(f"Total files processed: {total_files}")
    logging.info(f"Total images found: {total_images}")
    logging.info(f"Total images matched: {total_matched}")
    
    # 2. Generate hyperparameter combinations
    hyperparam_combinations = [
        dict(zip(CONFIG["HYPERPARAMS"].keys(), v)) 
        for v in itertools.product(*CONFIG["HYPERPARAMS"].values())
    ]
    
    # 3. Train models with different hyperparameters
    best_val_loss = float('inf')
    best_model = None
    best_hyperparams = None
    best_model_path = None
    
    logging.info("\nStarting hyperparameter search...")
    logging.info(f"Total combinations to try: {len(hyperparam_combinations)}")
    
    for i, hyperparams in enumerate(hyperparam_combinations, 1):
        logging.info(f"================================================")
        logging.info(f"\nTrying combination {i}/{len(hyperparam_combinations)}")
        logging.info(f"Hyperparameters: {hyperparams}")
        
        try:
            # Load data for this fold
            X_train, y_train = load_embeddings_and_scores(
                CONFIG["H5_FOLDER"],
                CONFIG["H5_FILE"],
                CONFIG["REQ_EMBEDDINGS"],
                labels_dict,
                train_ids
            )
            
            X_val, y_val = load_embeddings_and_scores(
                CONFIG["H5_FOLDER"],
                CONFIG["H5_FILE"],
                CONFIG["REQ_EMBEDDINGS"],
                labels_dict,
                val_ids
            )
            
            # Train model
            model, history = train_model(X_train, y_train, X_val, y_val, hyperparams)
            
            # Evaluate on validation set
            val_metrics = evaluate_model(model, X_val, y_val, "Validation")
            
            # Update best model if needed
            if val_metrics['mse'] < best_val_loss:
                best_val_loss = val_metrics['mse']
                best_model = model
                best_hyperparams = hyperparams
                
                # Save best model
                embedding_name = CONFIG["REQ_EMBEDDINGS"].replace('/', '_')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path = os.path.join(
                    CONFIG["MODEL_SAVE_DIR"], 
                    f'best_model_{embedding_name}.pt'
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'hyperparams': hyperparams,
                    'val_loss': best_val_loss,
                    'history': history
                }, best_model_path)
                
                logging.info(f"\nNew best model found!")
                logging.info(f"Validation MSE: {best_val_loss:.6f}")
                logging.info(f"Saved to: {best_model_path}")
                
        except Exception as e:
            logging.error(f"Error training with hyperparameters {hyperparams}: {str(e)}")
            continue
    
    # 4. Save best model information
    if best_model is not None:
        #save_best_model_info(best_model_path, best_hyperparams, best_val_loss)
        logging.info("\nTraining complete!")
        logging.info("\nBest Model Summary:")
        logging.info("=" * 50)
        logging.info(f"Model saved at: {best_model_path}")
        logging.info(f"Validation Loss: {best_val_loss:.6f}")
        logging.info("\nBest Hyperparameters:")
        for param, value in best_hyperparams.items():
            logging.info(f"- {param}: {value}")
    else:
        logging.error("\nNo successful model training completed!")

if __name__ == "__main__":
    main()
