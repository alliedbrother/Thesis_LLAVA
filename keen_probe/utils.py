import torch
from PIL import Image
import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from torch.utils.data import Dataset
import traceback

device = 'cpu'  # Force CPU usage
model_path = "liuhaotian/llava-v1.5-7b"

print("Initializing model...")
# Configure for CPU with memory efficiency
kwargs = {
    "device_map": "cpu",
    "torch_dtype": torch.float32,  # Use float32 for CPU
    "load_8bit": False,
    "load_4bit": False
}

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name="llava-v1.5-7b",
    **kwargs
)

print("Model loaded successfully")
print("Configuring components for float32...")

# Configure all components for float32
image_processor.torch_dtype = torch.float32
model.to(torch.float32)
vision_tower = model.get_vision_tower()
vision_tower.to(torch.float32)
model.eval()

print("Components configured successfully")

def process_image(image, processor):
    """Process an image and return a tensor."""
    try:
        # Convert PIL Image to tensor
        processed = processor(images=image, return_tensors='pt')
        return processed['pixel_values']
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        print(traceback.format_exc())
        raise

def extract_image_representation(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        tensor = process_image(image, image_processor).unsqueeze(0).to(device)
        prompt = "USER: <image> Please describe this image.\\nASSISTANT:"
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, images=tensor, output_hidden_states=True, return_dict=True)
        image_token_idx = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        return out.hidden_states[-1][0, image_token_idx, :].squeeze(0).cpu()
    except Exception as e:
        print(f"Error in extract_image_representation: {str(e)}")
        print(traceback.format_exc())
        raise

def generate_caption(img_path):
    try:
        print(f"\nGenerating caption for {img_path}")
        
        # Load and process image
        print("Loading and processing image...")
        image = Image.open(img_path).convert("RGB")
        tensor = process_image(image, image_processor).unsqueeze(0).to(device)
        print("Image processed successfully")
        
        # Prepare conversation
        print("Preparing conversation...")
        conv = conv_templates["llava_v1"].copy()
        if model.config.mm_use_im_start_end:
            conv.append_message(conv.roles[0], DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\nGenerate a concise and accurate caption for this image.")
        else:
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nGenerate a concise and accurate caption for this image.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(f"Prompt: {prompt}")
        
        # Convert prompt to input IDs
        print("Converting prompt to input IDs...")
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
        print("Input IDs shape:", input_ids.shape)
        
        # Generate caption
        print("Generating caption...")
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=tensor,
                do_sample=False,
                max_new_tokens=256,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        print("Generation completed")
        
        # Decode output
        caption = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(f"Caption: {caption}")
        return caption
    except Exception as e:
        print(f"Error in generate_caption: {str(e)}")
        print(traceback.format_exc())
        raise

def is_factual(caption, object_list):
    try:
        caption = caption.lower()
        return all(obj in caption for obj in object_list)
    except Exception as e:
        print(f"Error in is_factual: {str(e)}")
        print(traceback.format_exc())
        raise

def extract_embeddings(tensor, prompt):
    """Extract both vision and language embeddings from LLaVA."""
    try:
        # Get vision embeddings
        with torch.no_grad():
            # Get vision tower output
            vision_embeddings = model.model.vision_tower(tensor)
            # Project vision embeddings
            projected_vision = model.model.mm_projector(vision_embeddings)
            vision_features = projected_vision.mean(dim=1)  # [1, hidden_size]
        
        # Get language embeddings
        conv = conv_templates["llava_v1"].copy()
        if model.config.mm_use_im_start_end:
            conv.append_message(conv.roles[0], DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt)
        else:
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                images=tensor,
                output_hidden_states=True,
                return_dict=True
            )
            text_features = outputs.hidden_states[-1].mean(dim=1)  # [1, hidden_size]
        
        return vision_features.squeeze(0).cpu(), text_features.squeeze(0).cpu()
    except Exception as e:
        print(f"Error in extract_embeddings: {str(e)}")
        print(traceback.format_exc())
        raise

class EmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['image_path']
        prompt = item['prompt']
        factual = item['factual']
        
        # Extract combined embeddings
        embeddings = extract_embeddings(img_path, prompt)
        return embeddings, torch.tensor(factual, dtype=torch.float32)
