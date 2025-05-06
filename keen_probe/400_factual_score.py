import sys, os
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

print("Starting caption analysis script...")

# Set device to cuda:0 for consistency
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Initialize BERT model for factual accuracy
print("Loading BERT model...")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()

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
        similarities = [float(cosine_similarity([gen_embedding.cpu()], [gt.cpu()])[0][0]) for gt in gt_embeddings]
        return float(max(similarities))
    except Exception as e:
        print(f"Error in compute_factual_score: {str(e)}")
        return 0.0

def analyze_generation_results(generation_embeddings_path, ground_truth_path):
    """Analyze the generation results and compute factual scores."""
    try:
        # Load generation embeddings
        print("Loading generation embeddings...")
        with open(generation_embeddings_path, 'r') as f:
            generation_results = json.load(f)
        
        # Load ground truth captions
        print("Loading ground truth captions...")
        df = pd.read_csv(ground_truth_path)
        ground_truth_captions = {}
        for _, row in df.iterrows():
            image_id = row['image'].replace('.jpg', '')
            captions = row['captions'].split(' ||| ')
            ground_truth_captions[image_id] = captions
        
        # Analyze results
        analysis_results = []
        for result in tqdm(generation_results):
            image_id = result['image_id']
            caption = result['caption']
            
            # Get ground truth captions
            gt_captions = ground_truth_captions.get(image_id, [])
            
            # Compute factual score
            factual_score = compute_factual_score(caption, gt_captions)
            
            analysis_results.append({
                'image_id': image_id,
                'generated_caption': caption,
                'ground_truth_captions': gt_captions,
                'factual_score': factual_score
            })
        
        return analysis_results
        
    except Exception as e:
        print(f"Error in analyze_generation_results: {str(e)}")
        traceback.print_exc()
        raise

def main():
    print("Starting main analysis...")
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, "keen_data")
    
    # Paths
    generation_embeddings_path = os.path.join(output_dir, "generation_embeddings.json")
    ground_truth_path = os.path.join(output_dir, "coco_val2017_captions.csv")
    
    # Analyze results
    analysis_results = analyze_generation_results(generation_embeddings_path, ground_truth_path)
    
    # Save analysis results
    output_path = os.path.join(output_dir, "generation_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"Analysis results saved to {output_path}")
    
    # Print summary statistics
    factual_scores = [result['factual_score'] for result in analysis_results]
    avg_score = sum(factual_scores) / len(factual_scores)
    print(f"\nSummary Statistics:")
    print(f"Average factual score: {avg_score:.4f}")
    print(f"Number of images analyzed: {len(analysis_results)}")

if __name__ == "__main__":
    main()
