import os
from display_image import display_image

def test_display():
    # Get the base directory
    base_dir = os.path.dirname(__file__)
    image_dir = os.path.abspath(os.path.join(base_dir, "..", "coco_val2017"))
    
    # Check if the image directory exists
    if not os.path.exists(image_dir):
        print(f"❌ Error: Image directory not found at {image_dir}")
        return False
    
    # List all jpg files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    if not image_files:
        print("❌ Error: No jpg files found in the directory")
        return False
    
    print(f"✅ Found {len(image_files)} images in the directory")
    
    # Test with first image
    print("\nTesting display with first image (index 0)...")
    try:
        display_image(0)
        print("✅ Display test successful")
        return True
    except Exception as e:
        print(f"❌ Error during display: {e}")
        return False

if __name__ == "__main__":
    print("Starting display test...")
    success = test_display()
    if success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.") 