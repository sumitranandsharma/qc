import os
import sys
import io
import glob
import json
import base64
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import torch
import numpy as np

# Dynamic paths for local and cloud deployment (e.g., Render, Docker)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from services.pipeline import DiagnosticPipeline

app = FastAPI(
    title="Quantum AI Medical Imaging System",
    description="Modular Multi-Disease Hybrid Classical-Quantum Medical Imaging Prototype",
    version="2.0.0"
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = DiagnosticPipeline()

CACHE_PATH = os.path.join(BASE_DIR, "data", "metrics_cache.json")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
MULTIDISEASE_SHOWCASE_PATH = os.path.join(BASE_DIR, "data", "samples", "multidisease_showcase.json")

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "version": "2.0.0",
        "device": str(pipeline.device),
        "qubit_count": 4,
        "circuit_layers": 2,
        "models_loaded": {
            "cnn_extractor": pipeline.extractor is not None,
            "pca_reducer": pipeline.pca_reducer is not None,
            "quantum_vqc_baseline": pipeline.vqc is not None,
            "classical_baselines": pipeline.baselines is not None,
            "quantum_vqc_multidisease": pipeline.multidisease_vqc is not None,
            "classical_multidisease": pipeline.multidisease_baselines is not None
        },
        "active_modality": "chest_xray",
        "supported_modes": ["multi_disease", "baseline"]
    }

@app.get("/api/modalities")
def get_modalities():
    """
    Returns the modality architecture specifications, highlighting Chest X-Ray as active
    and CT, MRI, Ultrasound as planned modular extensions.
    """
    return {
        "modalities": [
            {
                "id": "chest_xray",
                "name": "Chest Radiograph (X-Ray)",
                "status": "active",
                "badge": "Active Pipeline",
                "icon": "activity",
                "description": "Pediatric & adult thoracic pathology detection (Normal vs Pneumonia & Multi-Finding NIH ChestX-ray14).",
                "qubit_allocation": 4,
                "circuit_ansatz": "2-Layer Parameterized Ry/Rz with Circular CNOT",
                "input_resolution": "224×224×3",
                "extractor": "ResNet-18 (512-dim GAP)",
                "modes": [
                    {
                        "id": "multi_disease",
                        "name": "Multi-Disease Finding Analysis (NIH ChestX-ray14)",
                        "findings": ["Infiltration", "Effusion", "Atelectasis", "Pneumothorax", "Pneumonia", "Cardiomegaly", "Consolidation", "No Finding"]
                    },
                    {
                        "id": "baseline",
                        "name": "Binary Triage Baseline (Normal vs Pneumonia)",
                        "findings": ["Pneumonia", "Normal"]
                    }
                ]
            },
            {
                "id": "ct_scan",
                "name": "Computed Tomography (CT)",
                "status": "planned",
                "badge": "Planned / In Dev",
                "icon": "layers",
                "description": "Volumetric axial pulmonary window analysis for interstitial lung disease and nodule volumetry.",
                "qubit_allocation": 8,
                "circuit_ansatz": "Tree Tensor Network (TTN) Variational Circuit",
                "input_resolution": "512×512×N Volumetric",
                "extractor": "3D-ResNet / MedNeXt",
                "modes": []
            },
            {
                "id": "mri",
                "name": "Magnetic Resonance Imaging (MRI)",
                "status": "planned",
                "badge": "Planned / In Dev",
                "icon": "disc",
                "description": "Multi-sequence T1/T2 soft-tissue mapping for neurological and mediastinal lesion segmentation.",
                "qubit_allocation": 8,
                "circuit_ansatz": "Quantum Convolutional Neural Network (QCNN)",
                "input_resolution": "256×256×N Multi-Slice",
                "extractor": "DenseNet-121 / Swin-UNETR",
                "modes": []
            },
            {
                "id": "ultrasound",
                "name": "Point-of-Care Ultrasound (POCUS)",
                "status": "planned",
                "badge": "Planned / In Dev",
                "icon": "radio",
                "description": "Acoustic pleural line movement, B-line lung profile, and subpleural consolidation classification.",
                "qubit_allocation": 4,
                "circuit_ansatz": "Quantum Kernel Ridge Classifier",
                "input_resolution": "320×240 Frame Sequences",
                "extractor": "MobileNetV3 Temporal Extractor",
                "modes": []
            }
        ],
        "default_modality": "chest_xray"
    }

@app.get("/api/samples")
def get_sample_images():
    """
    Returns pre-loaded showcase chest X-rays for 1-click test runs across both baseline and multi-disease modes.
    """
    samples = []

    # 1. Multi-disease showcase samples
    if os.path.exists(MULTIDISEASE_SHOWCASE_PATH):
        try:
            with open(MULTIDISEASE_SHOWCASE_PATH, "r") as f:
                multi_meta = json.load(f)
            for m in multi_meta:
                path = os.path.join(SAMPLES_DIR, m["filename"])
                if os.path.exists(path):
                    with Image.open(path) as img:
                        thumb = img.copy()
                        thumb.thumbnail((120, 120))
                        buffered = io.BytesIO()
                        thumb.save(buffered, format="JPEG", quality=75)
                        thumb_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    samples.append({
                        "id": m["id"],
                        "filename": m["filename"],
                        "mode": "multi_disease",
                        "dataset": "NIH ChestX-ray14",
                        "findings": m.get("findings", []),
                        "ground_truth": ", ".join(m.get("findings", [])),
                        "thumbnail": thumb_b64,
                        "description": m.get("description", "NIH ChestX-ray14 Multi-Disease Scan")
                    })
        except Exception as e:
            print(f"[Samples] Error loading multi-disease showcase: {e}")

    # 2. Baseline Kermany samples (Normal vs Pneumonia)
    sample_files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "sample_*.jpg")))
    for path in sample_files:
        filename = os.path.basename(path)
        sample_id = os.path.splitext(filename)[0]
        label = "Normal" if "normal" in sample_id else "Pneumonia"

        try:
            with Image.open(path) as img:
                thumb = img.copy()
                thumb.thumbnail((120, 120))
                buffered = io.BytesIO()
                thumb.save(buffered, format="JPEG", quality=75)
                thumb_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception:
            thumb_b64 = None

        samples.append({
            "id": sample_id,
            "filename": filename,
            "mode": "baseline",
            "dataset": "Kermany Pediatric",
            "findings": [label],
            "ground_truth": label,
            "thumbnail": thumb_b64,
            "description": f"Pediatric Chest X-Ray ({label})"
        })

    return {"samples": samples}

@app.post("/api/analyze")
async def analyze_scan(
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    mode: str = Form("multi_disease") # "multi_disease" or "baseline"
):
    """
    Analyzes an uploaded image OR a chosen sample ID through the hybrid pipeline.
    """
    pil_image = None

    if file is not None and file.filename:
        contents = await file.read()
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    elif sample_id:
        sample_path = os.path.join(SAMPLES_DIR, f"{sample_id}.jpg")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(SAMPLES_DIR, f"{sample_id}.jpeg")
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
        try:
            pil_image = Image.open(sample_path).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to open sample: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded file or sample_id.")

    # Determine ground truth and metadata
    ground_truth = None
    source_name = "Uploaded Medical Scan"
    
    if sample_id:
        source_name = f"Benchmark Sample ({sample_id})"
        # Check in multidisease showcase metadata first
        if os.path.exists(MULTIDISEASE_SHOWCASE_PATH):
            try:
                with open(MULTIDISEASE_SHOWCASE_PATH, "r") as f:
                    multi_meta = json.load(f)
                matched = next((m for m in multi_meta if m["id"] == sample_id), None)
                if matched:
                    ground_truth = ", ".join(matched.get("findings", []))
            except Exception:
                pass
        if ground_truth is None:
            ground_truth = "Normal" if "normal" in sample_id.lower() else "Pneumonia"
    elif file and file.filename:
        source_name = file.filename

    try:
        result = pipeline.analyze_image(pil_image, mode=mode)
        result["source_info"] = {
            "source_name": source_name,
            "ground_truth": ground_truth,
            "is_sample": bool(sample_id),
            "mode": mode
        }
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@app.get("/api/circuit")
def get_circuit_details(mode: str = Query("multi_disease")):
    """
    Returns circuit architecture, ASCII diagram, and gate metadata.
    """
    if mode == "multi_disease":
        if pipeline.multidisease_vqc is not None:
            return pipeline.multidisease_vqc.get_circuit_details()
        elif pipeline.vqc is not None:
            return pipeline.vqc.get_circuit_details()
    else:
        if pipeline.vqc is not None:
            return pipeline.vqc.get_circuit_details()
        elif pipeline.multidisease_vqc is not None:
            return pipeline.multidisease_vqc.get_circuit_details()

    raise HTTPException(status_code=503, detail="Quantum VQC model is not initialized.")

@app.get("/api/benchmarks")
def get_benchmarks():
    """
    Returns precomputed held-out test evaluation benchmarks for both binary and multi-disease modes.
    """
    if not os.path.exists(CACHE_PATH):
        raise HTTPException(status_code=404, detail="Benchmark cache not found. Please run training pipeline first.")
    
    with open(CACHE_PATH, "r") as f:
        data = json.load(f)
    return data

# Mount static frontend
frontend_dir = os.path.join(ROOT_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Frontend index.html is being prepared."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
