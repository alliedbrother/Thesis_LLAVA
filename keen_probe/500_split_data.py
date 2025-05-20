import os
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# ==========================
# CONFIGURATION
# ==========================

CONFIG = {
    # Paths
    "H5_FOLDER": "/root/projects/Thesis_LLAVA/keen_probe/keen_data/2014_extracted_embeddings",
    "LABELS_CSV": "/root/projects/Thesis_LLAVA/keen_probe/keen_data/labels.csv",
    "SPLITS_DIR": "/root/projects/Thesis_LLAVA/keen_probe/keen_data/splits",
    
    # Split ratios
    "TRAIN_RATIO": 0.7,
    "VAL_RATIO": 0.15,
    "TEST_RATIO": 0.15,
    
    # Random seed for reproducibility
    "RANDOM_SEED": 42
}

def get_all_image_ids(h5_folder):
    """Get all image IDs from HDF5 files."""
    all_image_ids = []
    
    print(f"Loading image IDs from: {h5_folder}")
    
    for filename in os.listdir(h5_folder):
        if filename.endswith('.h5'):
            file_path = os.path.join(h5_folder, filename)
            with h5py.File(file_path, 'r') as f:
                file_image_ids = list(f.keys())
                all_image_ids.extend(file_image_ids)
                print(f"File: {filename}, Images: {len(file_image_ids)}")
    
    print(f"\nTotal unique images found: {len(all_image_ids)}")
    return all_image_ids

def verify_labels_exist(image_ids, labels_csv):
    """Verify that all image IDs have corresponding labels."""
    labels_df = pd.read_csv(labels_csv)
    labels_dict = dict(zip(labels_df['image_id'], labels_df['METEOR']))
    
    valid_image_ids = []
    missing_labels = []
    
    for img_id in image_ids:
        if img_id in labels_dict:
            valid_image_ids.append(img_id)
        else:
            missing_labels.append(img_id)
    
    print(f"\nLabels verification:")
    print(f"Total images: {len(image_ids)}")
    print(f"Images with labels: {len(valid_image_ids)}")
    print(f"Images missing labels: {len(missing_labels)}")
    
    if missing_labels:
        print("\nFirst 5 missing labels:")
        for img_id in missing_labels[:5]:
            print(f"  {img_id}")
    
    return valid_image_ids

def create_splits(image_ids, config):
    """Create train, validation, and test splits."""
    # First split: separate out test set
    train_val_ids, test_ids = train_test_split(
        image_ids,
        test_size=config["TEST_RATIO"],
        random_state=config["RANDOM_SEED"]
    )
    
    # Second split: separate train and validation sets
    # Adjust the test_size to get the desired validation ratio
    val_ratio_adjusted = config["VAL_RATIO"] / (1 - config["TEST_RATIO"])
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_ratio_adjusted,
        random_state=config["RANDOM_SEED"]
    )
    
    print("\nSplit sizes:")
    print(f"Training set: {len(train_ids)} images")
    print(f"Validation set: {len(val_ids)} images")
    print(f"Test set: {len(test_ids)} images")
    
    return train_ids, val_ids, test_ids

def save_splits(train_ids, val_ids, test_ids, config):
    """Save the splits to files."""
    os.makedirs(config["SPLITS_DIR"], exist_ok=True)
    
    # Save as CSV files
    pd.DataFrame({'image_id': train_ids}).to_csv(
        os.path.join(config["SPLITS_DIR"], 'train_ids.csv'), index=False)
    pd.DataFrame({'image_id': val_ids}).to_csv(
        os.path.join(config["SPLITS_DIR"], 'val_ids.csv'), index=False)
    pd.DataFrame({'image_id': test_ids}).to_csv(
        os.path.join(config["SPLITS_DIR"], 'test_ids.csv'), index=False)
    
    # Save split statistics
    stats = {
        'total_images': len(train_ids) + len(val_ids) + len(test_ids),
        'train_size': len(train_ids),
        'val_size': len(val_ids),
        'test_size': len(test_ids),
        'train_ratio': len(train_ids) / (len(train_ids) + len(val_ids) + len(test_ids)),
        'val_ratio': len(val_ids) / (len(train_ids) + len(val_ids) + len(test_ids)),
        'test_ratio': len(test_ids) / (len(train_ids) + len(val_ids) + len(test_ids)),
        'random_seed': config["RANDOM_SEED"]
    }
    
    with open(os.path.join(config["SPLITS_DIR"], 'split_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nSplits saved to: {config['SPLITS_DIR']}")

def main():
    """Main function to create and save data splits."""
    print("Starting data split creation...")
    
    # 1. Get all image IDs from HDF5 files
    all_image_ids = get_all_image_ids(CONFIG["H5_FOLDER"])
    
    # 2. Verify that all images have labels
    valid_image_ids = verify_labels_exist(all_image_ids, CONFIG["LABELS_CSV"])
    
    # 3. Create splits
    train_ids, val_ids, test_ids = create_splits(valid_image_ids, CONFIG)
    
    # 4. Save splits
    save_splits(train_ids, val_ids, test_ids, CONFIG)
    
    print("\nData split creation complete!")

if __name__ == "__main__":
    main() 