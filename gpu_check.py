import torch

def list_gpus():
    if not torch.cuda.is_available():
        print("No GPU available.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}\n")

    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # in GB
        print(f"GPU {i}: {gpu_name}")
        print(f"  Total Memory: {total_mem:.2f} GB")
        print()

if __name__ == "__main__":
    list_gpus()
