import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llava.model.builder import load_pretrained_model
import torch
from transformers import AutoConfig

def test_model_loading():
    print("Initializing model loading...")
    model_path = "liuhaotian/llava-v1.5-7b"

    # Configure for CPU with memory efficiency
    kwargs = {
        "device_map": "cpu",
        "torch_dtype": torch.float32,  # Use float32 for CPU
        "load_8bit": False,
        "load_4bit": False
    }
    
    print("Loading model...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava-v1.5-7b",
        **kwargs
    )
    
    # Set model to evaluation mode
    model.eval()
    print(f"✅ Model loaded successfully")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Context length: {context_len}")

if __name__ == "__main__":
    test_model_loading()
