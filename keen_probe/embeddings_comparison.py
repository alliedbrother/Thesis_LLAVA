import json

# Load the JSON data
file_path = '/home/luk_lab/Desktop/Luk Lab - AI/projects/thesis/Thesis_LLAVA/keen_probe/keen_data/000000000139_embeddings.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Access image_embeddings from pre_generation at layer_0
pre_gen_image_embeddings_layer_0 = data.get("pre_generation", {}).get("layer_0", {}).get("query_embeddings", [])
post_gen_image_embeddings_layer_0 = data.get("post_generation", {}).get("step_2", {}).get("layer_0", {}).get("query_embeddings", [])
post_gen_image_embeddings_layer_0 = data.get("post_generation", {}).get("step_2", {}).get("layer_0", {}).get("query_embeddings", [])
# Print or further process the embeddings
print(sum(pre_gen_image_embeddings_layer_0))
print(sum(post_gen_image_embeddings_layer_0))

print(pre_gen_image_embeddings_layer_0 == post_gen_image_embeddings_layer_0)