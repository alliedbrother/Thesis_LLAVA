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

def extract_embeddings_with_layers(img_path, prompt, image_id):
    """Extract embeddings from specific layers of the model."""
    try:
        # Load and process image
        image = Image.open(img_path).convert("RGB")
        tensor = process_image(image, image_processor).squeeze(0).to(device).to(model.dtype)

        # Get vision tower embeddings and project them
        with torch.no_grad():
            vision_embeddings = model.model.vision_tower(tensor.unsqueeze(0))
            vision_embeddings = vision_embeddings.to(model.dtype)
            projected_vision = model.model.mm_projector(vision_embeddings)
            image_embeddings_post_projection = projected_vision.mean(dim=1).squeeze(0).cpu().tolist()

            # Prepare prompt
            conv = conv_templates["llava_v1"].copy()
            if model.config.mm_use_im_start_end:
                conv.append_message(conv.roles[0], DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt)
            else:
                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()

            # Tokenize
            input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
            
            # Get token texts
            token_ids = input_ids[0].cpu().numpy()
            token_texts = []
            for token_id in token_ids:
                try:
                    token_text = tokenizer._convert_id_to_token(int(token_id))
                    if token_text is None:
                        token_text = f"<id_{token_id}>"
                except:
                    token_text = f"<id_{token_id}>"
                token_texts.append(token_text)

            # Run model forward pass
            outputs = model(
                input_ids=input_ids,
                images=tensor.unsqueeze(0),
                output_hidden_states=True,
                return_dict=True
            )

            # Get total number of layers
            num_layers = len(outputs.hidden_states)
            print(f"Total number of layers: {num_layers}")

            # Extract embeddings from specific layers
            # Adjust indices to be within range
            target_layers = [0, 1, 16, num_layers-2, num_layers-1]
            print(f"Using layers: {target_layers}")
            
            image_embeddings_llm = {}
            image_query_embeddings_llm = {}

            for layer_idx in target_layers:
                layer_hidden = outputs.hidden_states[layer_idx]
                
                # Store image token embeddings
                image_token_idx = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()
                image_vec = layer_hidden[0, image_token_idx, :].cpu().tolist()
                image_embeddings_llm[f"layer_{layer_idx}"] = image_vec

                # Store query embeddings for each token
                for token_idx in range(input_ids.shape[1]):
                    token_vec = layer_hidden[0, token_idx, :].cpu().tolist()
                    token_text = token_texts[token_idx]
                    key = f"layer_{layer_idx}_token_{token_idx}_{token_text}"
                    image_query_embeddings_llm[key] = token_vec

            return {
                "image_id": image_id,
                "image_embeddings_post_projection": image_embeddings_post_projection,
                "image_embeddings_llm": image_embeddings_llm,
                "image_query_embeddings_llm": image_query_embeddings_llm
            }

    except Exception as e:
        print(f"Error in extract_embeddings_with_layers: {str(e)}")
        traceback.print_exc()
        return None

def main():
    print("Starting main function...")
    base_dir = os.path.dirname(__file__)
    image_dir = os.path.abspath(os.path.join(base_dir, "..", "coco_val2017"))
    output_dir = os.path.join(base_dir, "keen_data")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading prompts...")
    # Load prompts from JSON
    prompts_path = os.path.join(base_dir, "keen_data", "prompts.json")
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])[:25]
    print(f"Found {len(image_files)} images to process")
    
    results = {}
    
    print("Extracting features and embeddings...")
    for fname in tqdm(image_files):
        try:
            print(f"\nProcessing image: {fname}")
            img_path = os.path.join(image_dir, fname)
            base = fname.replace('.jpg', '')
            
            # Get prompt for this image
            prompt = prompts.get(base, "Generate a concise and accurate caption for this image.")
            
            # Extract embeddings
            embeddings = extract_embeddings_with_layers(img_path, prompt, base)
            if embeddings:
                results[base] = embeddings
                print(f"Completed processing {fname}")
            
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")
            traceback.print_exc()
            continue
    
    print("\nSaving final results...")
    # Save results to pre_generation_embeddings.json
    with open(os.path.join(output_dir, "pre_generation_embeddings_V41.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Completed processing {len(results)} images")
    print(f"✅ Results saved to {os.path.join(output_dir, 'pre_generation_embeddings_V41.json')}")
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
