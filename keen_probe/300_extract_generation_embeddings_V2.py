import sys, os
import traceback
import time
import numpy as np
from collections import defaultdict
import h5py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import process_image, image_processor, model, tokenizer, device
import json
import torch
from PIL import Image
from tqdm import tqdm
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token

# Constants
BATCH_SIZE = 4
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "coco_val2017")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "keen_data", "generation_embeddings_V42.h5")

def load_image_ids():
    """Load image IDs from the pre_generation_embeddings_V4.json file."""
    try:
        pre_gen_path = os.path.join(os.path.dirname(__file__), "keen_data", "pre_generation_embeddings_V4.json")
        if not os.path.exists(pre_gen_path):
            print(f"Error: {pre_gen_path} not found")
            return []
            
        with open(pre_gen_path, 'r') as f:
            data = json.load(f)
            return list(data.keys())
    except Exception as e:
        print(f"Error loading image IDs: {e}")
        return []

class Timer:
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}
    
    def start(self, name):
        self.start_times[name] = time.time()
    
    def stop(self, name):
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.timings[name].append(elapsed)
            del self.start_times[name]
    
    def get_average(self, name):
        if name in self.timings and self.timings[name]:
            return sum(self.timings[name]) / len(self.timings[name])
        return 0
    
    def print_stats(self):
        print("\nTiming Statistics:")
        print("-" * 50)
        for name, times in self.timings.items():
            avg = sum(times) / len(times)
            print(f"{name}: {avg:.3f}s (avg)")

timer = Timer()

def monitor_gpu_memory():
    """Monitor GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def process_image_optimized(image, image_processor):
    """Process a single image using the image processor."""
    try:
        result = image_processor(image, return_tensors="pt")
        # Convert BatchFeature to tensor by accessing the pixel_values
        return result.pixel_values.squeeze(0)  # Remove batch dimension
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def load_images_batch(image_ids, image_dir, batch_size=4):
    """Load and process a batch of images."""
    tensors = []
    for img_id in image_ids:
        try:
            image_path = os.path.join(image_dir, f"{img_id}.jpg")
            if not os.path.exists(image_path):
                print(f"Warning: Image {img_id} not found at {image_path}")
                continue
                
            image = Image.open(image_path).convert('RGB')
            tensor = process_image_optimized(image, image_processor)
            
            if tensor is not None:
                tensor = tensor.unsqueeze(0).to(device).to(model.dtype)
                tensors.append(tensor)
            else:
                print(f"Warning: Failed to process image {img_id}")
                
        except Exception as e:
            print(f"Error loading image {img_id}: {e}")
            continue
            
    if not tensors:
        return None
        
    return torch.cat(tensors, dim=0)

def get_token_texts(input_ids):
    """Get token texts from input IDs - optimized version."""
    timer.start('token_processing')
    # Convert all tokens at once
    token_ids = input_ids[0].cpu().numpy()
    
    # Pre-allocate list
    token_texts = [None] * len(token_ids)
    
    # Process in chunks
    chunk_size = 1000
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i:i + chunk_size]
        for j, token_id in enumerate(chunk):
            try:
                token_text = tokenizer._convert_id_to_token(int(token_id))
                token_texts[i + j] = token_text if token_text is not None else f"<id_{token_id}>"
            except:
                token_texts[i + j] = f"<id_{token_id}>"
    
    timer.stop('token_processing')
    return token_texts

def write_json_chunk(file, image_id, data, is_first=False):
    """Write a single JSON chunk efficiently."""
    timer.start('json_writing')
    if not is_first:
        file.write(',\n')
    json_str = json.dumps({
        'image_id': image_id,
        **data
    }, indent=2)
    file.write(f'"{image_id}": {json_str}')
    timer.stop('json_writing')

def extract_step_embeddings(hidden_states, input_ids, token_texts, step_idx):
    """Extract detailed embeddings for a single generation step."""
    timer.start('embedding_extraction')
    
    vision_only_embeddings = {}
    
    # Find image token position
    image_token_idx = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()
    
    # Process layer by layer to reduce memory usage
    for layer_idx, layer_hidden in hidden_states.items():
        # Vision-only (image token) - convert to CPU and FP32 immediately
        vision_vec = layer_hidden[0, image_token_idx, :].cpu().float().tolist()
        vision_only_embeddings[str(layer_idx)] = vision_vec
        
        # Explicitly delete the layer tensor after processing
        del layer_hidden
    
    # Clear GPU memory more aggressively
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    timer.stop('embedding_extraction')
    return {
        "vision_only_embeddings": vision_only_embeddings
    }

def extract_sequence_embeddings(hidden_states, input_ids, token_texts, step_idx):
    """Extract embeddings for the complete sequence at a given generation step."""
    timer.start('sequence_embedding_extraction')
    
    sequence_embeddings = {}
    
    # Process layer by layer to reduce memory usage
    for layer_idx, layer_hidden in hidden_states.items():
        layer_embeddings = []
        
        # Process all tokens in the sequence
        for token_idx in range(input_ids.shape[1]):
            # Get token embedding and convert to CPU
            token_vec = layer_hidden[0, token_idx, :].cpu().float().tolist()
            
            # Get token text
            if token_idx < len(token_texts):
                token_text = token_texts[token_idx]
            else:
                token_id = input_ids[0, token_idx].item()
                token_text = f"<id_{token_id}>"
            
            # Store the embedding with metadata
            layer_embeddings.append({
                'token': token_text,
                'position': token_idx,
                'embedding': token_vec
            })
        
        sequence_embeddings[str(layer_idx)] = layer_embeddings
        
        # Explicitly delete the layer tensor after processing
        del layer_hidden
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    timer.stop('sequence_embedding_extraction')
    return sequence_embeddings

def process_with_retry(img_path, tensor, max_retries=3):
    """Process an image with retry logic."""
    for attempt in range(max_retries):
        try:
            return generate_caption_with_embeddings(img_path, tensor)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1)  # Wait before retry

def generate_caption_with_embeddings(img_path, tensor):
    """Generate a caption and extract detailed embeddings at each generation step."""
    try:
        print(f"\nGenerating caption for {img_path}")
        
        # Prepare conversation
        timer.start('conversation_prep')
        conv = conv_templates["llava_v1"].copy()
        if model.config.mm_use_im_start_end:
            conv.append_message(conv.roles[0], DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
            conv.append_message(conv.roles[1], "")
            conv.append_message(conv.roles[0], "Generate a concise and accurate caption for this image.")
            conv.append_message(conv.roles[1], "")
        else:
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN)
            conv.append_message(conv.roles[1], "")
            conv.append_message(conv.roles[0], "Generate a concise and accurate caption for this image.")
            conv.append_message(conv.roles[1], "")
        timer.stop('conversation_prep')
        
        # Convert prompt to input IDs
        timer.start('prompt_processing')
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        print("Input IDs shape:", input_ids.shape)
        timer.stop('prompt_processing')
        
        # Get initial token texts
        token_texts = get_token_texts(input_ids)
        initial_sequence_length = len(token_texts)
        print(f"Initial sequence length: {initial_sequence_length}")
        
        # Generate caption
        print("Generating caption...")
        generation_steps = []
        current_input_ids = input_ids
        generated_text = ""
        
        # Define target layers
        target_layers = [0, 1, 16, 31, 32]
        
        with torch.no_grad():
            # Generate tokens one by one
            for step in range(50):  # Limit to 50 tokens max
                step_start_time = time.time()
                
                timer.start('model_forward')
                # Clear GPU cache before forward pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Get model outputs with only target layers
                outputs = model(
                    input_ids=current_input_ids,
                    images=tensor,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Extract only the layers we need
                hidden_states = {
                    layer_idx: outputs.hidden_states[layer_idx]
                    for layer_idx in target_layers
                }
                
                # Generate next token
                next_token = torch.argmax(outputs.logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
                
                # Append to current sequence
                current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                timer.stop('model_forward')
                
                # Update token texts
                timer.start('token_update')
                try:
                    new_token_text = tokenizer._convert_id_to_token(next_token.item())
                    if new_token_text is None:
                        new_token_text = f"<id_{next_token.item()}>"
                    print(f"Step {step} token text:", new_token_text)
                    if not new_token_text.startswith("<") and not new_token_text.startswith("▁"):
                        generated_text += new_token_text
                    elif new_token_text.startswith("▁"):
                        generated_text += " " + new_token_text[1:]
                except:
                    print(f"Step {step} token text conversion failed for id:", next_token.item())
                    new_token_text = f"<id_{next_token.item()}>"
                token_texts.append(new_token_text)
                timer.stop('token_update')
                
                # Extract step embeddings
                timer.start('step_embedding_extraction')
                step_embeddings = extract_step_embeddings(
                    hidden_states,
                    current_input_ids,
                    token_texts,
                    step
                )
                timer.stop('step_embedding_extraction')
                
                # Extract sequence embeddings for the complete sequence at this step
                timer.start('sequence_embedding_extraction')
                sequence_embeddings = extract_sequence_embeddings(
                    hidden_states,
                    current_input_ids,
                    token_texts,
                    step
                )
                timer.stop('sequence_embedding_extraction')
                
                # Store step information
                generation_steps.append({
                    'step': step,
                    'token': new_token_text,
                    'token_id': next_token.item(),
                    'embeddings': step_embeddings,
                    'sequence_embeddings': sequence_embeddings
                })
                
                # Print timing for this step
                step_time = time.time() - step_start_time
                print(f"\nStep {step} timing:")
                print(f"Total step time: {step_time:.3f}s")
                print(f"Model forward: {timer.get_average('model_forward'):.3f}s")
                print(f"Token update: {timer.get_average('token_update'):.3f}s")
                print(f"Step embedding extraction: {timer.get_average('step_embedding_extraction'):.3f}s")
                print(f"Sequence embedding extraction: {timer.get_average('sequence_embedding_extraction'):.3f}s")
                
                # Print sequence length at each step
                print(f"Sequence length at step {step}: {len(token_texts)}")
                print(f"Generated tokens so far: {token_texts[initial_sequence_length:]}")
                
                # Clear GPU memory to avoid accumulation
                del outputs
                del hidden_states
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Check for end of generation
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            # Use our manually constructed caption instead of decoding
            caption = generated_text.strip()
            print(f"Generated caption: {caption}")
            print(f"Final sequence length: {len(token_texts)}")
            print(f"Generated tokens: {token_texts[initial_sequence_length:]}")
            
        # Clear GPU memory before returning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'caption': caption,
            'prompt': prompt,
            'token_sequence': token_texts,
            'initial_sequence_length': initial_sequence_length,
            'generation_steps': generation_steps
        }
        
    except Exception as e:
        print(f"Error in generate_caption_with_embeddings: {str(e)}")
        traceback.print_exc()
        raise

def write_embeddings_hdf5(file, image_id, data, is_first=False):
    """Write embeddings to HDF5 file efficiently."""
    timer.start('hdf5_writing')
    
    # Delete existing group if it exists to avoid 'name already exists' errors
    try:
        if image_id in file:
            del file[image_id]
    except Exception as e:
        print(f"Warning: Could not delete existing group {image_id}: {e}")
    
    # Create group for this image
    timer.start('hdf5_group_creation')
    group = file.create_group(image_id)
    timer.stop('hdf5_group_creation')
    
    # Store metadata
    timer.start('hdf5_metadata')
    group.attrs['caption'] = data['caption']
    group.attrs['prompt'] = data['prompt']
    group.attrs['token_sequence'] = np.array(data['token_sequence'], dtype=h5py.special_dtype(vlen=str))
    group.attrs['initial_sequence_length'] = data['initial_sequence_length']
    group.attrs['total_sequence_length'] = len(data['token_sequence'])
    group.attrs['num_generation_steps'] = len(data['generation_steps'])
    group.attrs['generated_tokens'] = np.array(data['token_sequence'][data['initial_sequence_length']:], dtype=h5py.special_dtype(vlen=str))
    timer.stop('hdf5_metadata')
    
    # Store generation steps
    timer.start('hdf5_steps')
    steps_group = group.create_group('generation_steps')
    
    print(f"DEBUG: Writing {len(data['generation_steps'])} generation steps to HDF5")
    print(f"DEBUG: Initial sequence length: {data['initial_sequence_length']}")
    print(f"DEBUG: Total token sequence length: {len(data['token_sequence'])}")
    print(f"DEBUG: Generated tokens: {data['token_sequence'][data['initial_sequence_length']:]}")
    
    for step_idx, step_data in enumerate(data['generation_steps']):
        step_group = steps_group.create_group(f'step_{step_idx}')
        step_group.attrs['token'] = step_data['token']
        step_group.attrs['token_id'] = step_data['token_id']
        step_group.attrs['step'] = step_idx
        step_group.attrs['sequence_length_at_step'] = len(data['token_sequence'][:data['initial_sequence_length'] + step_idx + 1])
        step_group.attrs['generated_sequence_so_far'] = ' '.join(data['token_sequence'][data['initial_sequence_length']:data['initial_sequence_length'] + step_idx + 1])
        
        # Store vision-only embeddings
        timer.start('hdf5_vision_embeddings')
        vision_group = step_group.create_group('vision_only_embeddings')
        for layer, vec in step_data['embeddings']['vision_only_embeddings'].items():
            ds = vision_group.create_dataset(str(layer), data=np.array(vec), 
                                          compression='gzip', compression_opts=9)
            ds.attrs['layer'] = int(layer)
            ds.attrs['step'] = step_idx
        timer.stop('hdf5_vision_embeddings')
        
        # Store sequence embeddings for this step
        timer.start('hdf5_sequence_embeddings')
        sequence_group = step_group.create_group('sequence_embeddings')
        for layer, embeddings in step_data['sequence_embeddings'].items():
            layer_group = sequence_group.create_group(layer)
            for idx, embedding_data in enumerate(embeddings):
                # Create a unique name for each token embedding
                token = embedding_data['token'].replace('<', '_lt_').replace('>', '_gt_').replace(' ', '_')
                embedding_name = f"token_{idx}_{token}"
                
                # Create dataset with metadata
                ds = layer_group.create_dataset(embedding_name, 
                                              data=np.array(embedding_data['embedding']),
                                              compression='gzip', 
                                              compression_opts=9)
                ds.attrs['token'] = embedding_data['token']
                ds.attrs['position'] = embedding_data['position']
                ds.attrs['layer'] = int(layer)
                ds.attrs['step'] = step_idx
                
                # Add flag to indicate if this is a generated token
                is_generated = idx >= data['initial_sequence_length']
                ds.attrs['is_generated'] = is_generated
                if is_generated:
                    ds.attrs['generation_step'] = idx - data['initial_sequence_length']
                    ds.attrs['generated_sequence_so_far'] = ' '.join(data['token_sequence'][data['initial_sequence_length']:idx+1])
        timer.stop('hdf5_sequence_embeddings')
    timer.stop('hdf5_steps')
    
    timer.stop('hdf5_writing')

def process_single_image(image_id, model, image_processor, tokenizer, device, image_dir):
    """Process a single image and extract embeddings."""
    try:
        timer.start('image_processing')
        
        # Load and preprocess image
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        tensor = process_image_optimized(image, image_processor)
        tensor = tensor.unsqueeze(0).to(device).to(model.dtype)
        
        # Generate caption and get embeddings
        generation_result = process_with_retry(image_path, tensor)
        
        timer.stop('image_processing')
        return generation_result
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing {image_id}: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

def save_embeddings(embeddings):
    """Save embeddings to HDF5 file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        
        # Save embeddings
        with h5py.File(EMBEDDINGS_PATH, 'w') as f:
            for img_id, embedding in embeddings.items():
                f.create_dataset(img_id, data=embedding, compression='gzip')
    except Exception as e:
        print(f"Error saving embeddings: {e}")

def main():
    """Main function to process images and extract embeddings."""
    # Load image IDs
    image_ids = load_image_ids()
    if not image_ids:
        print("No image IDs found. Exiting.")
        return

    print(f"Found {len(image_ids)} image IDs")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    
    # Process images and generate captions with embeddings
    with h5py.File(EMBEDDINGS_PATH, 'w') as f:
        for i, image_id in enumerate(image_ids):
            print(f"\nProcessing image {i+1}/{len(image_ids)}: {image_id}")
            
            try:
                # Process single image
                result = process_single_image(image_id, model, image_processor, tokenizer, device, IMAGE_DIR)
                
                if result:
                    # Write embeddings to HDF5 file
                    write_embeddings_hdf5(f, image_id, result)
                    print(f"Successfully processed and saved embeddings for image {image_id}")
                else:
                    print(f"Failed to process image {image_id}")
                
                # Monitor GPU memory
                monitor_gpu_memory()
                
            except Exception as e:
                print(f"Error processing image {image_id}: {str(e)}")
                traceback.print_exc()
                continue
            
            # Clear GPU memory after each image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nProcessing complete!")
    timer.print_stats()

if __name__ == "__main__":
    main() 