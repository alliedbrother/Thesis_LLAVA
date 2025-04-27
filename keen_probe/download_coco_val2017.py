import os
import zipfile
import urllib.request

def download_and_extract_coco(dest_folder):
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(dest_folder, "val2017.zip")
    extract_path = os.path.join(dest_folder, "coco_val2017")

    os.makedirs(dest_folder, exist_ok=True)

    print("⬇️ Downloading MS COCO 2017 val set...")
    urllib.request.urlretrieve(url, zip_path)

    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    
    os.rename(os.path.join(dest_folder, "val2017"), extract_path)
    os.remove(zip_path)
    print(f"✅ Images extracted to: {extract_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    download_and_extract_coco(base_dir)
