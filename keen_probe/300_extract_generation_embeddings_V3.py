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
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
import torch.multiprocessing as mp
from functools import partial

# Constants
BATCH_SIZE = 1  # Reduced from 4 to 1
NUM_WORKERS = 0  # Disabled multiprocessing
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "coco_val2017")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "keen_data", "generation_embeddings_V4Parallel.h5")

def load_image_ids():
    """Load image IDs from the pre_generation_embeddings_V6.json file."""
    try:
        pre_gen_path = os.path.join(os.path.dirname(__file__), "keen_data", "pre_generation_embeddings_V41.json")
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
    """Extract embeddings for a single generation step."""
    timer.start('embedding_extraction')
    
    step_embeddings = {}
    
    # Find important token positions
    image_token_idx = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()
    query_token_idx = input_ids.shape[1] - 2  # -2 because the last token is the one we just generated
    final_idx = input_ids.shape[1] - 1  # The generated token
    
    # Define target layers
    target_layers = [0, 1, 16, 30, 31]
    
    # Process layer by layer
    for layer_idx, layer_hidden in enumerate(hidden_states):
        if layer_idx not in target_layers:
            continue
            
        # Extract embeddings for each position
        image_vec = layer_hidden[0, image_token_idx, :].cpu().float().tolist()
        query_vec = layer_hidden[0, query_token_idx, :].cpu().float().tolist()
        post_gen_vec = layer_hidden[0, final_idx, :].cpu().float().tolist()
        
        step_embeddings[f'layer_{layer_idx}'] = {
            'image_embeddings': image_vec,
            'query_embeddings': query_vec,
            'post_gen_embeddings': post_gen_vec
        }
        
        # Explicitly delete the layer tensor after processing
        del layer_hidden
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    timer.stop('embedding_extraction')
    return step_embeddings

def extract_sequence_embeddings(hidden_states, input_ids, token_texts, step_idx, outputs):
    """Extract embeddings only at specific token positions: image token, query token, and final generation."""
    timer.start('sequence_embedding_extraction')
    
    sequence_embeddings = {}
    
    # Find important token positions
    image_token_idx = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()
    query_token_idx = input_ids.shape[1] - 2  # -2 because the last token is the one we just generated
    
    # Define layer names for clarity
    layer_names = {
        0: "input_layer",
        1: "first_layer",
        16: "middle_layer",
        len(hidden_states)-2: "second_last_layer",
        len(hidden_states)-1: "final_layer"
    }
    
    # Process layer by layer to reduce memory usage
    for layer_idx, layer_hidden in enumerate(hidden_states):
        if layer_idx not in layer_names:
            continue
            
        layer_embeddings = []
        
        # Store image token embedding
        image_vec = layer_hidden[0, image_token_idx, :].cpu().float().tolist()
        layer_embeddings.append({
            'token': token_texts[image_token_idx],
            'position': image_token_idx,
            'embedding': image_vec,
            'type': 'image_token'
        })
        
        # Store query token embedding
        query_vec = layer_hidden[0, query_token_idx, :].cpu().float().tolist()
        layer_embeddings.append({
            'token': token_texts[query_token_idx],
            'position': query_token_idx,
            'embedding': query_vec,
            'type': 'query_token'
        })
        
        # Store final token embedding
        final_idx = input_ids.shape[1] - 1
        final_vec = layer_hidden[0, final_idx, :].cpu().float().tolist()
        layer_embeddings.append({
            'token': token_texts[final_idx] if final_idx < len(token_texts) else f"<id_{input_ids[0, final_idx].item()}>",
            'position': final_idx,
            'embedding': final_vec,
            'type': 'final_generation'
        })
        
        sequence_embeddings[layer_names[layer_idx]] = layer_embeddings
        
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

def extract_pre_generation_embeddings(hidden_states, input_ids, token_texts):
    """Extract embeddings after processing the entire prompt sequence."""
    timer.start('pre_generation_extraction')
    
    pre_generation = {}
    
    # Find important token positions
    image_token_idx = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()
    query_token_idx = input_ids.shape[1] - 1  # Last token of the prompt
    
    # Define target layers
    target_layers = [0, 1, 16, 30, 31]
    
    # Process layer by layer
    for layer_idx, layer_hidden in enumerate(hidden_states):
        if layer_idx not in target_layers:
            continue
            
        # Store image token embedding
        image_vec = layer_hidden[0, image_token_idx, :].cpu().float().tolist()
        query_vec = layer_hidden[0, query_token_idx, :].cpu().float().tolist()
        
        pre_generation[f'layer_{layer_idx}'] = {
            'image_embeddings': image_vec,
            'query_embeddings': query_vec
        }
        
        # Explicitly delete the layer tensor after processing
        del layer_hidden
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    timer.stop('pre_generation_extraction')
    return pre_generation

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
        
        # Get pre-generation embeddings
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                images=tensor,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract pre-generation embeddings
            pre_generation_embeddings = extract_pre_generation_embeddings(
                outputs.hidden_states,
                input_ids,
                token_texts
            )
            
            # Get vision embeddings directly from vision tower
            with torch.no_grad():
                vision_embeddings = model.model.vision_tower(tensor).mean(dim=1).cpu().float().tolist()
        
        # Generate caption
        print("Generating caption...")
        post_generation = {}
        current_input_ids = input_ids
        generated_text = ""
        
        with torch.no_grad():
            # Generate tokens one by one
            for step in range(50):  # Limit to 50 tokens max
                timer.start('model_forward')
                # Clear GPU cache before forward pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Get model outputs
                outputs = model(
                    input_ids=current_input_ids,
                    images=tensor,
                    output_hidden_states=True,
                    return_dict=True
                )
                timer.stop('model_forward')
                
                # Generate next token
                next_token = torch.argmax(outputs.logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
                
                # Append to current sequence
                current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                
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
                
                # Only store embeddings for first three steps and final step
                if step < 3 or next_token.item() == tokenizer.eos_token_id:
                    step_embeddings = extract_step_embeddings(
                        outputs.hidden_states,
                        current_input_ids,
                        token_texts,
                        step
                    )
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        post_generation['step_final'] = step_embeddings
                    else:
                        post_generation[f'step_{step + 1}'] = step_embeddings
                
                # Print sequence length at each step
                print(f"Sequence length at step {step}: {len(token_texts)}")
                print(f"Generated tokens so far: {token_texts[initial_sequence_length:]}")
                
                # Clear GPU memory to avoid accumulation
                del outputs
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
            'image_id': os.path.basename(img_path).split('.')[0],
            'generated_caption': caption,
            'tokens_length': len(token_texts),
            'vision_embeddings': vision_embeddings,
            'pre_generation': pre_generation_embeddings,
            'post_generation': post_generation
        }
        
    except Exception as e:
        print(f"Error in generate_caption_with_embeddings: {str(e)}")
        traceback.print_exc()
        return None

def write_embeddings_hdf5(f, image_id, result):
    """Write embeddings and metadata to HDF5 file."""
    try:
        # Create group for this image
        group = f.create_group(image_id)
        
        # Write basic metadata
        group.create_dataset('image_id', data=result['image_id'].encode('utf-8'))
        group.create_dataset('generated_caption', data=result['generated_caption'].encode('utf-8'))
        group.create_dataset('tokens_length', data=result['tokens_length'])
        
        # Write vision embeddings
        vision_emb = np.array(result['vision_embeddings'], dtype=np.float32)
        group.create_dataset('vision_embeddings', 
                           data=vision_emb,
                           chunks=True,
                           compression='gzip')
        
        # Write pre-generation embeddings
        pre_gen_group = group.create_group('pre_generation')
        for layer_name, layer_data in result['pre_generation'].items():
            layer_group = pre_gen_group.create_group(layer_name)
            
            # Write image and query embeddings
            layer_group.create_dataset('image_embeddings',
                                     data=np.array(layer_data['image_embeddings'], dtype=np.float32),
                                     chunks=True,
                                     compression='gzip')
            layer_group.create_dataset('query_embeddings',
                                     data=np.array(layer_data['query_embeddings'], dtype=np.float32),
                                     chunks=True,
                                     compression='gzip')
        
        # Write post-generation embeddings
        post_gen_group = group.create_group('post_generation')
        for step_name, step_data in result['post_generation'].items():
            step_group = post_gen_group.create_group(step_name)
            
            # Write embeddings for each layer in this step
            for layer_name, layer_data in step_data.items():
                layer_group = step_group.create_group(layer_name)
                
                # Write all embeddings for this layer
                layer_group.create_dataset('image_embeddings',
                                         data=np.array(layer_data['image_embeddings'], dtype=np.float32),
                                         chunks=True,
                                         compression='gzip')
                layer_group.create_dataset('query_embeddings',
                                         data=np.array(layer_data['query_embeddings'], dtype=np.float32),
                                         chunks=True,
                                         compression='gzip')
                layer_group.create_dataset('post_gen_embeddings',
                                         data=np.array(layer_data['post_gen_embeddings'], dtype=np.float32),
                                         chunks=True,
                                         compression='gzip')
        
        return True
    except Exception as e:
        print(f"Error writing embeddings for image {image_id}: {str(e)}")
        print("Error details:", traceback.format_exc())
        return False

class ImageDataset(Dataset):
    def __init__(self, image_ids, image_dir, image_processor):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.image_processor = image_processor

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
            return image_id, tensor
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            return image_id, None

def collate_fn(batch):
    """Collate function for DataLoader."""
    return (
        [item[0] for item in batch],
        torch.stack([item[1] for item in batch if item[1] is not None])
    )

def process_batch(batch_data, model, tokenizer, device):
    """Process a batch of images in parallel."""
    image_ids, tensors = batch_data
    results = []
    
    # Move tensors to device
    tensors = tensors.to(device).to(model.dtype)
    
    # Process each image in the batch
    for i, (image_id, tensor) in enumerate(zip(image_ids, tensors)):
        try:
            result = generate_caption_with_embeddings(image_id, tensor.unsqueeze(0))
            if result:
                results.append((image_id, result))
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            continue
    
    return results

def main():
    """Main function to process images and extract embeddings."""
    try:
        # Load image IDs
        image_ids = load_image_ids()
        if not image_ids:
            print("No image IDs found. Exiting.")
            return

        print(f"Found {len(image_ids)} image IDs")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        
        # Create dataset and dataloader
        dataset = ImageDataset(image_ids, IMAGE_DIR, image_processor)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Process images in batches
        with h5py.File(EMBEDDINGS_PATH, 'w') as f:
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
                print(f"\nProcessing batch {batch_idx + 1}")
                
                try:
                    # Process batch
                    batch_results = process_batch(batch_data, model, tokenizer, device)
                    
                    # Save results
                    for image_id, result in batch_results:
                        try:
                            write_embeddings_hdf5(f, image_id, result)
                            print(f"Successfully processed and saved embeddings for image {image_id}")
                        except Exception as e:
                            print(f"Error saving embeddings for image {image_id}: {e}")
                            print("Error details:", traceback.format_exc())
                    
                    # Monitor GPU memory
                    monitor_gpu_memory()
                    
                    # Clear GPU memory after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing batch {batch_idx + 1}: {e}")
                    print("Error details:", traceback.format_exc())
                    continue

        print("\nProcessing complete!")
        timer.print_stats()
        
    except Exception as e:
        print(f"Error in main function: {e}")
        print("Error details:", traceback.format_exc())
        raise

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main() 