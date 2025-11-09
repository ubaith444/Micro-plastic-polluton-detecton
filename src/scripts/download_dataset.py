"""Download Underwater Plastic Dataset from Zenodo"""

import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse


def download_file(url: str, output_path: Path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def main(args):
    print("="*70)
    print("DOWNLOADING UNDERWATER PLASTIC DATASET (UPD)")
    print("="*70)
    print(f"Source: https://zenodo.org/records/6907230")
    print(f"Size: 23.7 MB")
    print(f"Output: {args.output_dir}")
    print("="*70 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download URL
    download_url = "https://zenodo.org/records/6907230/files/UPD.v1.yolov5pytorch.zip"
    zip_path = output_dir / "UPD.v1.yolov5pytorch.zip"
    
    # Download
    print("📥 Downloading dataset...")
    download_file(download_url, zip_path)
    print("✓ Download complete!\n")
    
    # Extract
    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("✓ Extraction complete!\n")
    
    # Remove zip file
    if args.remove_zip:
        zip_path.unlink()
        print("✓ Removed zip file\n")
    
    # Verify structure
    print("📊 Dataset Structure:")
    upd_dir = output_dir / "UPD.v1.yolov5pytorch"
    
    if upd_dir.exists():
        for split in ['train', 'val', 'test']:
            images_dir = upd_dir / split / 'images'
            labels_dir = upd_dir / split / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                num_images = len(list(images_dir.glob('*.*')))
                num_labels = len(list(labels_dir.glob('*.txt')))
                print(f"  {split.upper()}: {num_images} images, {num_labels} labels")
    
    print("\n✅ Dataset ready for training!")
    print(f"   Path: {upd_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download UPD dataset from Zenodo')
    parser.add_argument('--output_dir', default='data/upd', help='Output directory')
    parser.add_argument('--remove_zip', action='store_true', help='Remove zip after extraction')
    
    args = parser.parse_args()
    main(args)
