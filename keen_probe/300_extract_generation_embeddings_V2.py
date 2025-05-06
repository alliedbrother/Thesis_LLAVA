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
    """Optimized image processing."""
    timer.start('image_processing')
    # Convert to tensor first
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    # Process in FP16
    result = image_processor(image_tensor.half())
    # Ensure result is a tensor and add batch dimension
    if not isinstance(result, torch.Tensor):
        result = torch.tensor(result)
    result = result.unsqueeze(0)
    timer.stop('image_processing')
    return result

def load_images_batch(image_ids, image_dir, batch_size=4):
    """Load and process multiple images in parallel."""
    timer.start('batch_loading')
    tensors = {}
    for i in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[i:i + batch_size]
        batch_tensors = []
        for image_id in batch_ids:
            img_path = os.path.join(image_dir, f"{image_id}.jpg")
            image = Image.open(img_path).convert("RGB")
            tensor = process_image_optimized(image, image_processor)
            tensor = tensor.to(device).to(model.dtype)  # Removed unsqueeze since it's done in process_image_optimized
            batch_tensors.append((image_id, tensor))
        
        # Process batch tensors
        for image_id, tensor in batch_tensors:
            tensors[image_id] = tensor
            
    timer.stop('batch_loading')
    return tensors

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
        tensor = process_image(image, image_processor)
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

def main():
    # Start timing the entire script
    script_start_time = time.time()
    
    print("Starting generation embeddings extraction...")
    base_dir = os.path.dirname(__file__)
    image_dir = os.path.abspath(os.path.join(base_dir, "..", "coco_val2017"))
    output_dir = os.path.join(base_dir, "keen_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # Cache model configuration
    model_config = {
        'mm_use_im_start_end': model.config.mm_use_im_start_end,
        'dtype': model.dtype,
        'device': device
    }
    
    # Load pre_generation_embeddings.json to get list of processed images
    timer.start('data_loading')
    pre_gen_path = os.path.join(output_dir, "pre_generation_embeddings.json")
    print(f"Loading processed images from {pre_gen_path}")
    with open(pre_gen_path, 'r') as f:
        pre_gen_data = json.load(f)
    timer.stop('data_loading')
    
    # Create a set of processed image IDs
    processed_image_ids = set(pre_gen_data.keys())
    print(f"\nFound {len(processed_image_ids)} images to process:")
    print("Image IDs:", sorted(list(processed_image_ids)))
    print("\n" + "="*50 + "\n")
    
    # Create or load existing results
    output_path = os.path.join(output_dir, "generation_embeddings_V3.h5")
    processed_ids = set()
    
    # Add checkpoint file
    checkpoint_path = os.path.join(output_dir, "generation_checkpoint_V3.json")
    
    # Load checkpoint if exists
    timer.start('checkpoint_loading')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            processed_ids = set(checkpoint.get('processed_ids', []))
            last_image_id = checkpoint.get('last_image_id')
    else:
        processed_ids = set()
        last_image_id = None
    timer.stop('checkpoint_loading')
    
    # For testing, just process a few images
    # Comment this out for full processing
    test_image_ids = list(processed_image_ids)[:5]
    processed_image_ids = set(test_image_ids)
    
    # Remove previous file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing file: {output_path}")
    
    # Open HDF5 file
    with h5py.File(output_path, 'a') as f:
        # Get list of already processed images
        if 'processed_images' in f.attrs:
            processed_ids = set(f.attrs['processed_images'])
        
        # Filter out already processed images
        remaining_image_ids = processed_image_ids - processed_ids
        print(f"Found {len(remaining_image_ids)} remaining images to process")
        
        print("Generating captions and extracting embeddings...")
        try:
            # Process images in batches
            batch_size = 4  # Increased batch size for parallel processing
            for i in range(0, len(remaining_image_ids), batch_size):
                batch_start_time = time.time()
                batch_ids = list(remaining_image_ids)[i:i + batch_size]
                print(f"\nProcessing batch {i//batch_size + 1}/{(len(remaining_image_ids) + batch_size - 1)//batch_size}")
                
                # Load batch of images
                tensors = load_images_batch(batch_ids, image_dir, batch_size)
                
                # Process each image in the batch
                for image_id, tensor in tensors.items():
                    try:
                        image_start_time = time.time()
                        print(f"\n{'='*50}")
                        print(f"Processing image: {image_id}")
                        print(f"{'='*50}\n")
                        
                        timer.start('total_processing')
                        
                        # Process image
                        generation_result = process_with_retry(os.path.join(image_dir, f"{image_id}.jpg"), tensor)
                        if generation_result is None:
                            print(f"Skipping {image_id} due to processing error")
                            continue
                        
                        # Write to HDF5
                        write_embeddings_hdf5(f, image_id, generation_result, i == 0 and not processed_ids)
                        
                        # Update processed images list
                        processed_ids.add(image_id)
                        f.attrs['processed_images'] = list(processed_ids)
                        
                        timer.stop('total_processing')
                        
                        # Print detailed timing for this image
                        image_time = time.time() - image_start_time
                        print(f"\nTiming for image {image_id}:")
                        print(f"Total image processing time: {image_time:.3f}s")
                        print(f"Model forward time: {timer.get_average('model_forward'):.3f}s")
                        print(f"Token update time: {timer.get_average('token_update'):.3f}s")
                        print(f"Step embedding extraction: {timer.get_average('step_embedding_extraction'):.3f}s")
                        print(f"Sequence embedding extraction: {timer.get_average('sequence_embedding_extraction'):.3f}s")
                        print(f"HDF5 writing time: {timer.get_average('hdf5_writing'):.3f}s")
                        monitor_gpu_memory()
                        
                    except Exception as e:
                        print(f"Error processing {image_id}: {str(e)}")
                        print(f"Error type: {type(e).__name__}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        continue
                
                # Print batch timing
                batch_time = time.time() - batch_start_time
                print(f"\nBatch timing:")
                print(f"Total batch time: {batch_time:.3f}s")
                print(f"Average time per image: {batch_time/len(batch_ids):.3f}s")
                
                # Clear GPU memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
        finally:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clean up checkpoint
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            
            # Print final timing statistics
            print("\nFinal Timing Statistics:")
            print("-" * 50)
            timer.print_stats()
    
    # Calculate and print total script execution time
    script_end_time = time.time()
    total_time = script_end_time - script_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    
    print("\n" + "="*50)
    print(f"Total script execution time: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
    print(f"Results saved to {output_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main() 