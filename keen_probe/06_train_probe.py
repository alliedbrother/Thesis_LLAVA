import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from keen_probe import KEENProbe
from tqdm import tqdm

# Configuration
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
PATIENCE = 5
VAL_INTERVAL = 1
FACTUAL_THRESHOLD = 0.80  # Threshold for binary classification

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

base_dir = os.path.dirname(__file__)
datasets_dir = os.path.join(base_dir, "keen_data", "generated_datasets")
model_dir = os.path.join(base_dir, "keen_data", "models")
os.makedirs(model_dir, exist_ok=True)

# List of dataset files
dataset_files = [
    "dataset_vision_tower.csv",
    "dataset_initial_layer.csv",
    "dataset_middle_layer.csv",
    "dataset_final_layer.csv",
    "dataset_pre_generation.csv"
]

class EmbeddingDataset(Dataset):
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

def validate(model, val_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y = y.unsqueeze(1)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            
            predicted = (pred > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# Train a probe for each dataset
for dataset_file in dataset_files:
    print(f"\n{'='*50}")
    print(f"Training probe for {dataset_file}")
    print(f"{'='*50}")
    
    # Load and preprocess dataset
    dataset_path = os.path.join(datasets_dir, dataset_file)
    embeddings, labels = load_and_preprocess_dataset(dataset_path)
    
    # Create indices for splitting
    indices = np.arange(len(embeddings))
    np.random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(0.6 * len(indices))
    val_size = int(0.2 * len(indices))
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Save split indices
    split_indices = {
        'train': train_indices.tolist(),
        'val': val_indices.tolist(),
        'test': test_indices.tolist()
    }
    
    # Save split indices to file
    model_name = os.path.splitext(dataset_file)[0]
    split_path = os.path.join(model_dir, f"split_indices_{model_name}.json")
    with open(split_path, 'w') as f:
        json.dump(split_indices, f)
    
    # Create datasets using the indices
    X_train, y_train = embeddings[train_indices], labels[train_indices]
    X_val, y_val = embeddings[val_indices], labels[val_indices]
    
    # Create dataloaders
    train_set = EmbeddingDataset(X_train, y_train)
    val_set = EmbeddingDataset(X_val, y_val)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(test_indices)}")
    
    # Initialize model
    input_dim = embeddings.shape[1]
    model = KEENProbe(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    loss_fn = nn.BCELoss()

# Training loop
best_val_loss = float('inf')
patience_counter = 0
    best_model_path = os.path.join(model_dir, f"best_probe_{model_name}.pt")

print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    train_correct = 0
    train_total = 0
    
    # Training phase
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)
        y = y.unsqueeze(1)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        predicted = (pred > 0.5).float()
        train_correct += (predicted == y).sum().item()
        train_total += y.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{train_correct/train_total:.4f}'
        })
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = train_correct / train_total
    
    # Validation phase
    if (epoch + 1) % VAL_INTERVAL == 0:
        val_loss, val_accuracy = validate(model, val_loader)
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
            # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

print("\nTraining completed!")
print(f"Best validation loss: {best_val_loss:.4f}")

# Load best model for final save
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
    torch.save(model.state_dict(), os.path.join(model_dir, f"probe_{model_name}.pt"))
print(f"Best model saved with validation loss: {checkpoint['val_loss']:.4f} and accuracy: {checkpoint['val_accuracy']:.4f}")

print("\nAll probes trained successfully!")
