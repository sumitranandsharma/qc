# Quantum AI-Based Medical Imaging System for Disease Diagnosis from Medical Scans

An educational and research capstone prototype integrating **Deep Classical AI (ResNet-18)** with **Quantum Machine Learning (PennyLane Variational Quantum Classifier)** for chest radiograph classification (Normal vs. Pneumonia).

---

## Quick Start (Run Working Prototype)

### 1. Launch the Web Application
Simply run the launcher script:
```powershell
cd C:\SAH
python run_prototype.py
```
Or double-click `run_prototype.bat`.

The application will start at:  
👉 **`http://localhost:8000`** (and will automatically open in your default browser).

---

## Key Features & Demonstration Flow

1. **Diagnostic Studio:**
   - **Upload any Chest X-Ray** (JPEG/PNG) or select from **8 pre-loaded showcase benchmark scans** (4 Normal, 4 Pneumonia).
   - **Multi-Stage Visual Stepper:**
     - Stage 1: CLAHE Contrast Equalization
     - Stage 2: Deep Convolutional Feature Extraction (ResNet-18, 512 dimensions) + **Grad-CAM attention heatmap overlay**
     - Stage 3: PCA Dimensionality Reduction to 4 qubit rotation angles with explained variance analysis
     - Stage 4: PennyLane 4-Qubit Variational Quantum Classifier (VQC) with Ry Angle embedding, CNOT entanglement, and Pauli-Z expectation readout
     - Stage 5: Classical Baselines Comparison (SVM RBF, Linear SVM, Logistic Regression, MLP)
2. **Quantum Circuit Inspector:**
   - Interactive quantum wire diagram displaying qubits $|q_0\rangle$ through $|q_3\rangle$ with parameterized gates.
   - Live parameter inspection and collapsible PennyLane ASCII circuit diagram.
3. **Model Comparison Lab:**
   - Full performance matrix table across held-out test scans.
   - Interactive ROC Curves and latency comparison charts.
   - Side-by-side Confusion Matrices for Quantum VQC vs. Classical SVM.
4. **Theory & Viva Defense:**
   - Computational complexity breakdown.
   - Viva Voce defense questions and answers for capstone examiners.

---

## Experimental Benchmark Results (Held-Out Test Set: 66 Scans)

| Model | Category | Accuracy | F1-Score | Recall | Precision | Specificity | ROC-AUC | Latency / Scan |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variational Quantum Classifier (VQC)** | **Quantum AI** | **83.3%** | **82.0%** | **75.8%** | **89.3%** | **90.9%** | **89.7%** | **2.60 ms** |
| Support Vector Machine (RBF Kernel) | Classical AI | 83.3% | 82.0% | 75.8% | 89.3% | 90.9% | 91.8% | 1.10 ms |
| Support Vector Machine (Linear Kernel) | Classical AI | 87.9% | 86.7% | 78.8% | 96.3% | 97.0% | 94.6% | 0.99 ms |
| Logistic Regression Baseline | Classical AI | 87.9% | 86.7% | 78.8% | 96.3% | 97.0% | 94.8% | 0.77 ms |
| Multi-Layer Perceptron (4-16-8-1) | Classical AI | 86.4% | 85.2% | 78.8% | 92.9% | 93.9% | 93.4% | 1.21 ms |

---

## Project Structure

```
C:\SAH\
├── backend/
│   ├── app.py                     # FastAPI REST API & Static File Server
│   ├── models/
│   │   ├── cnn_extractor.py       # ResNet-18 Feature Extractor & Grad-CAM Heatmap
│   │   ├── pca_reducer.py         # PCA Dimensionality Reduction Model
│   │   ├── quantum_vqc.py         # PennyLane Variational Quantum Circuit (VQC)
│   │   └── classical_baselines.py # SVM, Logistic Regression, MLP Classifiers
│   ├── services/
│   │   └── pipeline.py            # End-to-end multi-stage inference orchestrator
│   └── data/
│       ├── samples/               # 8 Showcase Chest Radiographs for 1-click test runs
│       ├── weights/               # Serialized models (.pt, .joblib)
│       └── metrics_cache.json     # Precomputed benchmark metrics & ROC coordinates
├── frontend/
│   ├── index.html                 # Single Page Application Dashboard
│   ├── css/style.css              # Custom Medical-Tech & Quantum Styling
│   └── js/app.js                  # Dynamic controller, Chart.js graphs, circuit renderer
├── scripts/
│   ├── prepare_dataset.py         # Balanced dataset extraction from Kermany benchmark
│   └── train_pipeline.py          # End-to-end training and benchmark evaluation script
├── presentation/
│   ├── generate_ppt.py            # Programmatic 10-slide PowerPoint builder
│   └── quantum_medical_imaging.pptx # Final 10-Slide Capstone Presentation
├── report/
│   ├── capstone_report.md         # Full 10-Section Academic Capstone Report
│   ├── capstone_report.html       # Browser-viewable & Printable PDF Capstone Report
│   └── viva_defense_guide.md      # Comprehensive Viva Voce Questions & Answers
├── run_prototype.py               # Launch server & open browser
├── run_prototype.bat              # Double-click launcher for Windows
└── README.md
```
