import os
import urllib.request
import zipfile

def download_and_extract_annotations(dest_folder):
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = os.path.join(dest_folder, "annotations.zip")
    extract_path = os.path.join(dest_folder, "annotations")

    os.makedirs(dest_folder, exist_ok=True)

    print("⬇️ Downloading COCO 2017 annotations...")
    urllib.request.urlretrieve(url, zip_path)

    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)

    os.rename(os.path.join(dest_folder, "annotations"), extract_path)
    os.remove(zip_path)
    print(f"✅ captions_val2017.json is ready at: {extract_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    download_and_extract_annotations(base_dir)
