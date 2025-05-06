import json
import os

def format_size(size_in_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"

def analyze_embeddings(file_path):
    print(f"\n📊 Analyzing {file_path}...")
    
    # Get file size
    file_size = os.path.getsize(file_path)
    print(f"\n📁 File size: {format_size(file_size)}")
    
    # Load JSON data
    print("\n⌛ Loading JSON data...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Basic statistics
    num_images = len(data)
    print(f"\n📈 Basic Statistics:")
    print(f"- Number of images processed: {num_images}")
    
    # Analyze first entry in detail
    first_entry = data[0]
    print("\n🔍 Entry Structure:")
    print(f"- Keys in each entry: {list(first_entry.keys())}")
    
    # Analyze vision_only_embeddings
    vision_only = first_entry['vision_only_embeddings']
    num_layers = len(vision_only)
    embedding_dim = len(vision_only['0'])  # dimension of embeddings
    
    print(f"\n🧠 Vision-Only Embeddings Analysis:")
    print(f"- Number of layers: {num_layers}")
    print(f"- Embedding dimension: {embedding_dim}")
    print(f"- Total embeddings per image: {num_layers} layers × {embedding_dim} dimensions")
    
    # Open file for writing
    with open('tokens.txt', 'w') as f:
        f.write("Vision-Only Embeddings:\n")
        f.write(f"Layer 1: {len(vision_only['1'])} dimensions\n")
        f.write(f"Layer 31: {len(vision_only['31'])} dimensions\n\n")
        
        # Analyze vision_query_embeddings
        vision_query = first_entry['vision_query_embeddings']
        
        # Extract unique tokens, positions, and layers
        tokens = set()
        positions = set()
        layers = set()
        
        for key in vision_query.keys():
            # Parse the key format: (token=X, pos=Y, layer=Z)
            token = key.split('token=')[1].split(',')[0]
            pos = int(key.split('pos=')[1].split(',')[0])
            layer = int(key.split('layer=')[1].strip(')'))
            tokens.add(token)
            positions.add(pos)
            layers.add(layer)
        
        f.write("Vision-Query Embeddings:\n")
        # Only show layers 1 and 31
        target_layers = [1, 31]
        for layer in target_layers:
            f.write(f"\nLayer {layer}:\n")
            layer_keys = [k for k in vision_query.keys() if k.endswith(f'layer={layer})')]
            layer_keys.sort(key=lambda x: int(x.split('pos=')[1].split(',')[0]))  # Sort by position
            
            for key in layer_keys:
                token = key.split('token=')[1].split(',')[0]
                pos = key.split('pos=')[1].split(',')[0]
                value_size = len(vision_query[key])
                # Highlight the image token
                if token == '<id_-200>':
                    f.write(f"  Position {pos:2s}: Token '{token}' (IMAGE TOKEN): {value_size} dimensions\n")
                else:
                    f.write(f"  Position {pos:2s}: Token '{token}': {value_size} dimensions\n")
    
    print("\n✅ Analysis complete. Results saved to tokens.txt")

if __name__ == "__main__":
    embeddings_file = "keen_probe/keen_data/detailed_embeddings.json"
    analyze_embeddings(embeddings_file) 