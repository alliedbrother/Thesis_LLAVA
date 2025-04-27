import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import EmbeddingDataset
from keen_probe import KEENProbe
from tqdm import tqdm

# Configuration
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
PATIENCE = 5
VAL_INTERVAL = 1

base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, "keen_data", "dataset.json")
model_dir = os.path.join(base_dir, "keen_data", "models")
os.makedirs(model_dir, exist_ok=True)

print("Loading dataset...")
with open(dataset_path) as f:
    data = json.load(f)

# Split data and create dataloaders
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
train_set = EmbeddingDataset(train_data)
val_set = EmbeddingDataset(val_data)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0)

print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")

# Initialize model and training components
input_dim = model.config.hidden_size * 2  # Combined vision and language embeddings
model = KEENProbe(input_dim=input_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
loss_fn = nn.BCELoss()

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
            
            # Calculate accuracy
            predicted = (pred > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# Training loop
best_val_loss = float('inf')
patience_counter = 0
best_model_path = os.path.join(model_dir, "best_probe.pt")

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
        
        # Update progress bar
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
        
        # Save best model and check for early stopping
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
torch.save(model.state_dict(), os.path.join(model_dir, "probe.pt"))
print(f"Best model saved with validation loss: {checkpoint['val_loss']:.4f} and accuracy: {checkpoint['val_accuracy']:.4f}")
