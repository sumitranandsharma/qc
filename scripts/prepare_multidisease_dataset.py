import os
import io
import json
import pandas as pd
from PIL import Image

PARQUET_PATH = r"C:\Users\SUMITRANAND SHARMA\.cache\huggingface\hub\datasets--MaiAShaaban--NIH-Chest-Xray-14-subset\snapshots\f287663e0b9f8c1710feeca22ae24b5da1b2ce4e\data\train-00000-of-00001.parquet"
OUTPUT_DIR = r"C:\SAH\backend\data\multidisease_images"
SAMPLES_DIR = r"C:\SAH\backend\data\samples"
METADATA_PATH = r"C:\SAH\backend\data\multidisease_metadata.json"

TARGET_FINDINGS = [
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Pneumothorax",
    "Pneumonia",
    "Cardiomegaly",
    "Consolidation",
    "No Finding"
]

def main():
    print(f"[Dataset] Loading parquet from {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[Dataset] Loaded {len(df)} records.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    records = []
    
    # Process all 50 images
    for idx, row in df.iterrows():
        img_dict = row["image"]
        img_bytes = img_dict["bytes"]
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        labels = list(row["label"])
        # Filter labels to target findings if applicable
        active_findings = [f for f in labels if f in TARGET_FINDINGS]
        if not active_findings and "No Finding" not in labels:
            active_findings = labels

        patient_id = row.get("Patient ID", idx)
        filename = f"nih_scan_{idx:03d}_pid{patient_id}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        image.save(save_path, format="JPEG", quality=90)

        # Build multi-hot binary vector for the 8 core target findings
        binary_vector = [1 if f in labels else 0 for f in TARGET_FINDINGS]

        records.append({
            "index": idx,
            "filename": filename,
            "image_path": save_path,
            "raw_labels": labels,
            "target_findings": active_findings,
            "binary_vector": binary_vector,
            "patient_age": int(row.get("Patient Age", 0)) if not pd.isna(row.get("Patient Age")) else None,
            "patient_gender": str(row.get("Patient Gender", "")) if not pd.isna(row.get("Patient Gender")) else None,
            "view_position": str(row.get("View Position", "PA"))
        })

    # Save metadata JSON
    with open(METADATA_PATH, "w") as f:
        json.dump({
            "target_findings": TARGET_FINDINGS,
            "total_images": len(records),
            "records": records
        }, f, indent=2)

    print(f"[Dataset] Extracted {len(records)} images to {OUTPUT_DIR}")
    print(f"[Dataset] Saved metadata to {METADATA_PATH}")

    # Now select and save specific multi-disease showcase samples into SAMPLES_DIR
    showcase_candidates = [
        {"criteria": lambda r: "Infiltration" in r["raw_labels"] and "Pneumonia" in r["raw_labels"], "name": "nih_sample_infiltration_pneumonia.jpg", "desc": "Infiltration & Pneumonia"},
        {"criteria": lambda r: "Atelectasis" in r["raw_labels"] and "Effusion" in r["raw_labels"], "name": "nih_sample_atelectasis_effusion.jpg", "desc": "Atelectasis & Effusion"},
        {"criteria": lambda r: "Cardiomegaly" in r["raw_labels"], "name": "nih_sample_cardiomegaly.jpg", "desc": "Cardiomegaly (Enlarged Heart)"},
        {"criteria": lambda r: "Pneumothorax" in r["raw_labels"], "name": "nih_sample_pneumothorax.jpg", "desc": "Pneumothorax (Collapsed Lung)"},
        {"criteria": lambda r: "Consolidation" in r["raw_labels"], "name": "nih_sample_consolidation.jpg", "desc": "Consolidation & Effusion"},
        {"criteria": lambda r: "No Finding" in r["raw_labels"], "name": "nih_sample_no_finding.jpg", "desc": "No Finding (Normal Appearance)"}
    ]

    showcase_metadata = []
    for cand in showcase_candidates:
        match = next((r for r in records if cand["criteria"](r)), None)
        if match:
            sample_dest = os.path.join(SAMPLES_DIR, cand["name"])
            img = Image.open(match["image_path"])
            img.save(sample_dest, format="JPEG", quality=90)
            showcase_metadata.append({
                "id": os.path.splitext(cand["name"])[0],
                "filename": cand["name"],
                "findings": match["raw_labels"],
                "description": cand["desc"],
                "binary_vector": match["binary_vector"]
            })
            print(f"  [Showcase] Created {cand['name']} with findings: {match['raw_labels']}")

    showcase_cache = os.path.join(SAMPLES_DIR, "multidisease_showcase.json")
    with open(showcase_cache, "w") as f:
        json.dump(showcase_metadata, f, indent=2)
    print(f"[Dataset] Saved showcase metadata to {showcase_cache}")

if __name__ == "__main__":
    main()
