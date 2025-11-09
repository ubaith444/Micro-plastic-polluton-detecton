import os
import requests
from zipfile import ZipFile

# Roboflow Dataset URL (replace with your own export link)
DATASET_URL = "https://universe.roboflow.com/ds/YOUR_DATASET_KEY_HERE"
SAVE_PATH = "data/upd/UPD.v1.yolov5pytorch.zip"
EXTRACT_PATH = "data/upd/"

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

print("⬇️ Downloading dataset...")
response = requests.get(DATASET_URL)
with open(SAVE_PATH, "wb") as f:
    f.write(response.content)
print("✅ Download complete!")

print("📦 Extracting files...")
with ZipFile(SAVE_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)
print("🎉 Dataset extracted successfully to:", EXTRACT_PATH)
