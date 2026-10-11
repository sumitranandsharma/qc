import os
import io
import random
from PIL import Image
import pandas as pd
from huggingface_hub import hf_hub_download

def prepare_dataset():
    print("Loading Kermany Chest X-ray dataset from Hugging Face...")
    parquet_path = hf_hub_download(
        repo_id='hf-vision/chest-xray-pneumonia',
        filename='data/test-00000-of-00001.parquet',
        repo_type='dataset'
    )
    df = pd.read_parquet(parquet_path)
    print(f"Total images loaded: {len(df)}")
    print(df['label'].value_counts())

    normal_df = df[df['label'] == 0].copy().reset_index(drop=True)
    pneumonia_df = df[df['label'] == 1].copy().reset_index(drop=True)

    # Balance classes: 220 Normal and 220 Pneumonia
    n_samples_per_class = min(len(normal_df), 220)
    print(f"Balancing dataset with {n_samples_per_class} per class...")

    random.seed(42)
    normal_indices = list(range(len(normal_df)))
    pneumonia_indices = list(range(len(pneumonia_df)))
    random.shuffle(normal_indices)
    random.shuffle(pneumonia_indices)

    normal_selected = [normal_df.iloc[i] for i in normal_indices[:n_samples_per_class]]
    pneumonia_selected = [pneumonia_df.iloc[i] for i in pneumonia_indices[:n_samples_per_class]]

    # 1. Save sample showcase images for web application live demo
    samples_dir = os.path.join("C:/SAH", "backend", "data", "samples")
    os.makedirs(samples_dir, exist_ok=True)

    for i in range(4):
        # Save 4 normal samples
        img_bytes = normal_selected[i]['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(os.path.join(samples_dir, f"sample_normal_{i+1:02d}.jpg"), quality=95)

        # Save 4 pneumonia samples
        img_bytes = pneumonia_selected[i]['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(os.path.join(samples_dir, f"sample_pneumonia_{i+1:02d}.jpg"), quality=95)

    print(f"Saved 8 showcase sample scans to {samples_dir}")

    # 2. Structure Train (70%), Val (15%), Test (15%) splits
    # Exclude the 4 showcase samples from training/testing splits
    normal_rem = normal_selected[4:]
    pneumonia_rem = pneumonia_selected[4:]

    n_rem = len(normal_rem) # ~216
    n_train = int(n_rem * 0.70) # ~151 per class -> 302 total
    n_val = int(n_rem * 0.15)   # ~32 per class -> 64 total
    n_test = n_rem - n_train - n_val # ~33 per class -> 66 total

    splits = {
        'train': (normal_rem[:n_train], pneumonia_rem[:n_train]),
        'val': (normal_rem[n_train:n_train+n_val], pneumonia_rem[n_train:n_train+n_val]),
        'test': (normal_rem[n_train+n_val:], pneumonia_rem[n_train+n_val:])
    }

    base_dir = os.path.join("C:/SAH", "backend", "data", "processed")
    for split_name, (norm_items, pneu_items) in splits.items():
        norm_dir = os.path.join(base_dir, split_name, "NORMAL")
        pneu_dir = os.path.join(base_dir, split_name, "PNEUMONIA")
        os.makedirs(norm_dir, exist_ok=True)
        os.makedirs(pneu_dir, exist_ok=True)

        for idx, item in enumerate(norm_items):
            img = Image.open(io.BytesIO(item['image']['bytes'])).convert("RGB")
            img.save(os.path.join(norm_dir, f"norm_{idx:04d}.jpg"), quality=90)

        for idx, item in enumerate(pneu_items):
            img = Image.open(io.BytesIO(item['image']['bytes'])).convert("RGB")
            img.save(os.path.join(pneu_dir, f"pneu_{idx:04d}.jpg"), quality=90)

        print(f"Split {split_name}: {len(norm_items)} Normal, {len(pneu_items)} Pneumonia (Total: {len(norm_items) + len(pneu_items)})")

    print("Dataset preparation complete!")

if __name__ == '__main__':
    prepare_dataset()
