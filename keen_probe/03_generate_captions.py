import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import generate_caption, process_image, image_processor, model, tokenizer, device, extract_embeddings
from tqdm import tqdm
import json
import pandas as pd
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

print("Starting caption generation script...")

# Initialize BERT model for factual accuracy
print("Loading BERT model...")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

base_dir = os.path.dirname(__file__)
image_dir = os.path.abspath(os.path.join(base_dir, "..", "coco_val2017"))
output_dir = os.path.join(base_dir, "keen_data")
os.makedirs(output_dir, exist_ok=True)

# Load previously processed images from features.json
features_path = os.path.join(base_dir, "keen_data", "features.json")
print("Loading previously processed images...")
with open(features_path, 'r') as f:
    processed_images = json.load(f)
processed_image_ids = [img['image_id'] for img in processed_images]

# Load ground truth captions from CSV
csv_path = os.path.join(base_dir, "keen_data", "coco_val2017_captions.csv")
print("Loading ground truth captions from CSV...")
df = pd.read_csv(csv_path)
ground_truth_captions = {}
for _, row in df.iterrows():
    # Remove .jpg extension from image filename to get image_id
    image_id = row['image'].replace('.jpg', '')
    if image_id in processed_image_ids:  # Only load captions for processed images
        captions = row['captions'].split(' ||| ')
        ground_truth_captions[image_id] = captions

print(f"Loaded {len(ground_truth_captions)} ground truth captions for processed images")

def get_bert_embedding(text):
    """Get BERT embedding for a text."""
    with torch.no_grad():
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = bert_model(**inputs)
        # Use CLS token embedding as sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.squeeze(0)

def compute_factual_score(generated_caption, ground_truth_captions):
    """Compute factual score using semantic similarity with ground truth captions."""
    try:
        gen_embedding = get_bert_embedding(generated_caption)
        gt_embeddings = [get_bert_embedding(gt) for gt in ground_truth_captions]
        similarities = [cosine_similarity([gen_embedding], [gt])[0][0] for gt in gt_embeddings]
        return max(similarities)
    except Exception as e:
        print(f"Error in compute_factual_score: {str(e)}")
        return 0.0

def process_image_file(image_path, image_id, ground_truth_captions):
    """Process a single image file and compute factual scores."""
    try:
        # Generate caption using LLaVA
        caption = generate_caption(image_path)
        
        # Get ground truth captions
        gt_captions = ground_truth_captions.get(image_id, [])
        
        # Compute factual score
        factual_score = compute_factual_score(caption, gt_captions)
        
        # Extract embeddings
        vision_emb, text_emb = extract_embeddings(image_path, caption)
        
        return {
            "image_id": image_id,
            "image_path": os.path.basename(image_path),
            "generated_caption": caption,
            "ground_truth_captions": gt_captions,
            "factual_correctness": factual_score,
            "vision_embeddings": vision_emb.tolist(),
            "vision_embeddings_after_projection": vision_emb.tolist(),
            "language_model_embeddings": text_emb.tolist(),
            "language_model_embeddings_before_projection": text_emb.tolist()
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def main():
    results = []
    
    print("Processing images and computing factual scores...")
    for img_data in tqdm(processed_images):
        image_id = img_data['image_id']
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        
        result = process_image_file(image_path, image_id, ground_truth_captions)
        if result:
            results.append(result)
    
    # Save results to JSON
    output_path = os.path.join(base_dir, "keen_data", "features.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Compute and print statistics
    total_images = len(results)
    factual_correct = sum(1 for r in results if r['factual_correctness'] > 0.7)
    avg_similarity = sum(r['factual_correctness'] for r in results) / total_images
    
    print(f"\nResults Summary:")
    print(f"Total images processed: {total_images}")
    print(f"Factually correct captions: {factual_correct} ({factual_correct/total_images*100:.2f}%)")
    print(f"Average similarity score: {avg_similarity:.4f}")
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()

print("Caption generation script completed")
