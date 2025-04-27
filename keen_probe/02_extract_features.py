import sys, os
import traceback  # Add traceback import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import process_image, image_processor, model, tokenizer, device, extract_embeddings
import json
import torch
from PIL import Image
from tqdm import tqdm
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token

def extract_multi_layer_embeddings(img_path, prompt):
    """Extract embeddings from different layers of the model."""
    try:
        # Load and process image
        image = Image.open(img_path).convert("RGB")
        processed = image_processor(images=image, return_tensors='pt')
        tensor = processed['pixel_values'].squeeze(0).to(device)  # Remove extra dimension
        
        # Get vision tower embeddings
        with torch.no_grad():
            # Get vision tower output
            vision_embeddings = model.model.vision_tower(tensor.unsqueeze(0))
            # Project vision embeddings
            projected_vision = model.model.mm_projector(vision_embeddings)
            vision_features = projected_vision.mean(dim=1)  # [1, hidden_size]
            
            # Prepare prompt for language model
            conv = conv_templates["llava_v1"].copy()
            if model.config.mm_use_im_start_end:
                conv.append_message(conv.roles[0], DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt)
            else:
                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
            
            # Get outputs with all hidden states
            outputs = model(
                input_ids=input_ids,
                images=tensor.unsqueeze(0),
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract embeddings from different layers
            num_layers = len(outputs.hidden_states)
            
            # Get embeddings from:
            # 1. Initial layer (layer 0)
            # 2. Middle layer (approximately half way)
            # 3. Second to last layer (before generation)
            initial_layer = outputs.hidden_states[0].mean(dim=1)
            middle_layer = outputs.hidden_states[num_layers // 2].mean(dim=1)
            pre_generation_layer = outputs.hidden_states[-2].mean(dim=1)
            
            # Get final layer (for generation)
            final_layer = outputs.hidden_states[-1].mean(dim=1)
            
            return {
                "vision_embeddings": vision_features.squeeze(0).cpu().tolist(),
                "initial_layer_embeddings": initial_layer.squeeze(0).cpu().tolist(),
                "middle_layer_embeddings": middle_layer.squeeze(0).cpu().tolist(),
                "pre_generation_embeddings": pre_generation_layer.squeeze(0).cpu().tolist(),
                "final_layer_embeddings": final_layer.squeeze(0).cpu().tolist()
            }
            
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

def main():
    base_dir = os.path.dirname(__file__)
    image_dir = os.path.abspath(os.path.join(base_dir, "..", "coco_val2017"))
    output_dir = os.path.join(base_dir, "keen_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create embeddings directory
    os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)
    
    # Load prompts from JSON
    prompts_path = os.path.join(base_dir, "keen_data", "prompts.json")
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])[:25]  # Process first 25 images
    
    results = []
    
    print("Extracting features and embeddings...")
    for fname in tqdm(image_files):
        try:
            img_path = os.path.join(image_dir, fname)
            base = fname.replace('.jpg', '')
            
            # Get prompt for this image
            prompt = prompts.get(base, "Generate a concise and accurate caption for this image.")
            
            # Load and process image
            image = Image.open(img_path).convert('RGB')
            processed = image_processor(images=image, return_tensors='pt')
            tensor = processed['pixel_values'].squeeze(0).to(device)  # Remove extra dimension
            
            # Extract combined embeddings
            vision_emb, text_emb = extract_embeddings(tensor.unsqueeze(0), prompt)
            
            # Extract multi-layer embeddings
            layer_embeddings = extract_multi_layer_embeddings(img_path, prompt)
            
            # Save embeddings
            torch.save(vision_emb, os.path.join(output_dir, "embeddings", f"{base}_vision.pt"))
            torch.save(text_emb, os.path.join(output_dir, "embeddings", f"{base}_text.pt"))
            
            # Create result entry
            result = {
                "image_id": base,
                "image_path": fname,
                "prompt": prompt,
                "vision_embedding_path": f"embeddings/{base}_vision.pt",
                "text_embedding_path": f"embeddings/{base}_text.pt",
                **layer_embeddings
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")
            continue
    
    # Save combined results
    with open(os.path.join(output_dir, "features.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Completed processing {len(results)} images")
    print(f"✅ Results saved to {os.path.join(output_dir, 'features.json')}")

if __name__ == "__main__":
    main()
