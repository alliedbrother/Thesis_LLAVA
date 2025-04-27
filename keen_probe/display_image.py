import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

def display_image(image_number):
    # Get the base directory
    base_dir = os.path.dirname(__file__)
    image_dir = os.path.abspath(os.path.join(base_dir, "..", "coco_val2017"))
    
    # List all jpg files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    # Check if the image number is valid
    if image_number < 0 or image_number >= len(image_files):
        print(f"Error: Image number must be between 0 and {len(image_files)-1}")
        return
    
    # Get the image path
    image_path = os.path.join(image_dir, image_files[image_number])
    
    # Load and display the image
    try:
        image = Image.open(image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Image: {image_files[image_number]}")
        plt.show()
    except Exception as e:
        print(f"Error displaying image: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python display_image.py <image_number>")
        print("Example: python display_image.py 0")
        sys.exit(1)
    
    try:
        image_number = int(sys.argv[1])
        display_image(image_number)
    except ValueError:
        print("Error: Image number must be an integer") 