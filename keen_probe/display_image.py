# File: display_image.py

import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

def display_image(image_name):
    try:
        # Get the absolute path to the image directory
    base_dir = os.path.dirname(__file__)
    image_dir = os.path.abspath(os.path.join(base_dir, "..", "coco_val2017"))
        image_path = os.path.join(image_dir, image_name)
        
        print(f"Looking for image: {image_path}")
        
        # Check if image exists
        if not os.path.isfile(image_path):
            print(f"Error: Image {image_name} not found in {image_dir}")
            # List available images
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
            if image_files:
                print("\nAvailable images:")
                for f in image_files[:5]:  # Show first 5 images
                    print(f"  - {f}")
                print(f"... and {len(image_files) - 5} more")
        return
    
        # Try to set a display backend
        try:
            plt.switch_backend('TkAgg')
        except:
            try:
                plt.switch_backend('Agg')
            except:
                print("Warning: Could not set display backend")
    
        # Open and display the image
        print("Loading image...")
        img = Image.open(image_path)
        print(f"Image loaded. Size: {img.size}, Mode: {img.mode}")
        
        # Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(image_name)
        print("Displaying image...")
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python display_image.py <image_name.jpg>")
        print("Example: python display_image.py 000000581615.jpg")
        sys.exit(1)
    
    display_image(sys.argv[1])
