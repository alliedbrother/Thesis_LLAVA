import sys, os
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import process_image, image_processor, model, tokenizer, device, extract_embeddings
import json
import torch
from PIL import Image
from tqdm import tqdm
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token

def extract_detailed_layer_embeddings(img_path, prompt, image_id):
    """
    Extract detailed embeddings from specific layers, including vision-only and vision+query embeddings.
    Returns a dictionary with embeddings from each layer and token position.
    """
    try:
        # Load and process image
        image = Image.open(img_path).convert("RGB")
        tensor = process_image(image, image_processor).unsqueeze(0).to(device).to(model.dtype)

        # Build prompt
        conv = conv_templates["llava_v1"].copy()
        if model.config.mm_use_im_start_end:
            conv.append_message(conv.roles[0], DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt)
        else:
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
        seq_len = input_ids.shape[1]

        # Get token texts using convert_ids_to_tokens
        token_ids = input_ids[0].cpu().numpy()
        token_texts = []
        for token_id in token_ids:
            try:
                # Try to get the token from the tokenizer
                token_text = tokenizer._convert_id_to_token(int(token_id))
                if token_text is None:
                    token_text = f"<id_{token_id}>"
            except:
                # If conversion fails, use the ID
                token_text = f"<id_{token_id}>"
            token_texts.append(token_text)

        # Run LLaVA forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                images=tensor,
                output_hidden_states=True,
                return_dict=True
            )
        hidden_states = outputs.hidden_states
        image_token_idx = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()

        # Print total number of layers
        total_layers = len(hidden_states)
        print(f"Total number of layers in model: {total_layers}")

        # Extract embeddings
        vision_only_embeddings = {}
        vision_query_embeddings = {}

        # Calculate target layers based on total layers
        target_layers = [0, 1, total_layers//2, total_layers-2, total_layers-1]
        print(f"Extracting embeddings from layers: {target_layers}")

        for layer_idx in target_layers:
            layer_hidden = hidden_states[layer_idx]
            # Vision-only (image token)
            vision_vec = layer_hidden[0, image_token_idx, :].cpu().tolist()
            vision_only_embeddings[str(layer_idx)] = vision_vec

            # Vision + query embeddings
            for token_idx in range(seq_len):
                token_vec = layer_hidden[0, token_idx, :].cpu().tolist()
                # Create a descriptive key with token text
                token_text = token_texts[token_idx]
                key = f"(token='{token_text}', pos={token_idx}, layer={layer_idx})"
                vision_query_embeddings[key] = token_vec

        return {
            "image_id": image_id,
            "vision_only_embeddings": vision_only_embeddings,
            "vision_query_embeddings": vision_query_embeddings,
            "prompt": prompt_text,  # Include the full prompt for reference
            "token_sequence": token_texts  # Include the full token sequence
        }

    except Exception as e:
        print(f"Error in extract_detailed_layer_embeddings: {str(e)}")
        traceback.print_exc()
        return None

def extract_multi_layer_embeddings(img_path, prompt):
    """Extract embeddings from different layers of the model."""
    try:
        # Load and process image
        image = Image.open(img_path).convert("RGB")
        processed = image_processor(images=image, return_tensors='pt')
        tensor = processed['pixel_values'].squeeze(0).to(device).to(model.dtype)
        
        # Get vision tower embeddings
        with torch.no_grad():
            vision_embeddings = model.model.vision_tower(tensor.unsqueeze(0))
            vision_embeddings = vision_embeddings.to(model.dtype)  # Ensure correct dtype
            projected_vision = model.model.mm_projector(vision_embeddings)
            vision_features = projected_vision.mean(dim=1)
            
            # Prepare prompt
            conv = conv_templates["llava_v1"].copy()
            if model.config.mm_use_im_start_end:
                conv.append_message(conv.roles[0], DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt)
            else:
                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
            
            outputs = model(
                input_ids=input_ids,
                images=tensor.unsqueeze(0),
                output_hidden_states=True,
                return_dict=True
            )
            
            num_layers = len(outputs.hidden_states)
            
            initial_layer = outputs.hidden_states[0].mean(dim=1)
            middle_layer = outputs.hidden_states[num_layers // 2].mean(dim=1)
            pre_generation_layer = outputs.hidden_states[-2].mean(dim=1)
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
    print("Starting main function...")
    base_dir = os.path.dirname(__file__)
    image_dir = os.path.abspath(os.path.join(base_dir, "..", "coco_val2017"))
    output_dir = os.path.join(base_dir, "keen_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create embeddings directory
    embeddings_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    print("Loading prompts...")
    # Load prompts from JSON
    prompts_path = os.path.join(base_dir, "keen_data", "prompts.json")
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])[:25]
    print(f"Found {len(image_files)} images to process")
    
    results = []
    all_detailed_embeddings = []
    
    print("Extracting features and embeddings...")
    for fname in tqdm(image_files):
        try:
            print(f"\nProcessing image: {fname}")
            img_path = os.path.join(image_dir, fname)
            base = fname.replace('.jpg', '')
            
            # Get prompt for this image
            prompt = prompts.get(base, "Generate a concise and accurate caption for this image.")
            
            print("Loading and processing image...")
            # Load and process image
            image = Image.open(img_path).convert('RGB')
            processed = image_processor(images=image, return_tensors='pt')
            tensor = processed['pixel_values'].squeeze(0).to(device).to(model.dtype)
            
            print("Extracting embeddings...")
            # Extract combined embeddings
            vision_emb, text_emb = extract_embeddings(tensor.unsqueeze(0), prompt)
            
            print("Extracting multi-layer embeddings...")
            # Extract multi-layer embeddings
            layer_embeddings = extract_multi_layer_embeddings(img_path, prompt)
            
            print("Extracting detailed layer embeddings...")
            # Extract detailed layer embeddings
            detailed_embeddings = extract_detailed_layer_embeddings(img_path, prompt, base)
            if detailed_embeddings:
                all_detailed_embeddings.append(detailed_embeddings)
            
            print("Saving embeddings...")
            # Save basic embeddings
            torch.save(vision_emb, os.path.join(embeddings_dir, f"{base}_vision.pt"))
            torch.save(text_emb, os.path.join(embeddings_dir, f"{base}_text.pt"))
            
            # Create result entry
            result = {
                "image_id": base,
                "image_path": fname,
                "prompt": prompt,
                "vision_embedding_path": f"embeddings/{base}_vision.pt",
                "text_embedding_path": f"embeddings/{base}_text.pt",
                **layer_embeddings
            } if layer_embeddings else None
            
            if result:
            results.append(result)
            print(f"Completed processing {fname}")
            
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")
            traceback.print_exc()
            continue
    
    print("\nSaving final results...")
    # Save combined results
    with open(os.path.join(output_dir, "features.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save all detailed embeddings in a single file
    with open(os.path.join(output_dir, "detailed_embeddings.json"), 'w') as f:
        json.dump(all_detailed_embeddings, f, indent=2)
    
    print(f"\n✅ Completed processing {len(results)} images")
    print(f"✅ Results saved to {os.path.join(output_dir, 'features.json')}")
    print(f"✅ Detailed embeddings saved to {os.path.join(output_dir, 'detailed_embeddings.json')}")
    print("Script execution completed. Exiting...")
    
    # Explicitly clean up CUDA memory if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Ensure all file handles are closed
    try:
        import gc
        gc.collect()
    except Exception as e:
        print(f"Warning: Error during cleanup: {str(e)}")

if __name__ == "__main__":
    try:
    main()
    except Exception as e:
        print(f"Fatal error in main: {str(e)}")
        traceback.print_exc()
    finally:
        print("Script finished. Exiting...")
        sys.exit(0)  # Ensure the script exits
