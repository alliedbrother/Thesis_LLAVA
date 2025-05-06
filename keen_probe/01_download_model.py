import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llava.model.builder import load_pretrained_model
import torch
from transformers import AutoConfig
from huggingface_hub import snapshot_download, HfFolder
import os

def test_model_loading():
    print("Initializing model loading...")
    model_path = "liuhaotian/llava-v1.5-7b"

    # Check if model is already downloaded
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache_path = os.path.join(cache_dir, "models--" + model_path.replace("/", "--"))
    
    if os.path.exists(model_cache_path):
        print(f"✅ Model found in cache at: {model_cache_path}")
        print("Will load from cache instead of downloading.")
    else:
        print("Model not found in cache. Will download from Hugging Face Hub.")
        print("This might take a while depending on your internet connection...")

    # Check CUDA availability
    if torch.cuda.is_available():
        print("GPU is available. Using CUDA.")
        device = "cuda"
        dtype = torch.float16  # Use half precision for GPU
    else:
        print("GPU not available. Using CPU.")
        device = "cpu"
        dtype = torch.float32  # Use full precision for CPU

    # Configure model loading based on device
    kwargs = {
        "device_map": "auto" if device == "cuda" else "cpu",
        "torch_dtype": dtype,
        "load_8bit": False,
        "load_4bit": False
    }
    
    print("\nLoading model...")
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
    print(f"Using dtype: {dtype}")
    print(f"Context length: {context_len}")

if __name__ == "__main__":
    test_model_loading()
