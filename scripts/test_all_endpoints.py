import os
import sys
import io
from PIL import Image
from fastapi.testclient import TestClient

# Add backend directory
sys.path.append(os.path.abspath("C:/SAH/backend"))
from app import app

client = TestClient(app)

def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health status: {res.status_code}"
    data = res.json()
    assert data["status"] == "online"
    assert data["models_loaded"]["quantum_vqc_multidisease"] is True
    assert data["models_loaded"]["classical_multidisease"] is True
    assert "chest_xray" == data["active_modality"]
    print("[PASS] /api/health:", data["status"], "| Active:", data["active_modality"], "| Models Loaded:", data["models_loaded"])

def test_modalities():
    res = client.get("/api/modalities")
    assert res.status_code == 200
    data = res.json()
    assert "modalities" in data
    assert len(data["modalities"]) == 4
    mod_ids = [m["id"] for m in data["modalities"]]
    assert "chest_xray" in mod_ids
    assert "ct_scan" in mod_ids
    assert "mri" in mod_ids
    assert "ultrasound" in mod_ids
    print(f"[PASS] /api/modalities: 4 modalities found: {mod_ids}")

def test_samples():
    res = client.get("/api/samples")
    assert res.status_code == 200
    samples = res.json()["samples"]
    assert len(samples) >= 8, f"Expected at least 8 samples, got {len(samples)}"
    modes = set(s.get("mode") for s in samples)
    print(f"[PASS] /api/samples: {len(samples)} showcase samples loaded covering modes: {modes}.")

def test_circuit():
    # Test multi-disease circuit
    res_m = client.get("/api/circuit?mode=multi_disease")
    assert res_m.status_code == 200
    cm = res_m.json()
    assert cm["n_qubits"] == 4
    assert cm["total_gates"] == 32  # 4 Ry enc + 24 ansatz (8 Ry + 8 Rz + 8 CNOT) + 4 Pauli-Z measurements
    assert cm["n_findings"] == 8
    print(f"[PASS] /api/circuit (Multi-Disease): {cm['n_qubits']} qubits, {cm['total_gates']} gates, {cm['n_findings']} findings.")

    # Test baseline circuit
    res_b = client.get("/api/circuit?mode=baseline")
    assert res_b.status_code == 200
    cb = res_b.json()
    assert cb["n_qubits"] == 4
    print(f"[PASS] /api/circuit (Baseline): {cb['n_qubits']} qubits, {cb['total_gates']} gates.")

def test_benchmarks():
    res = client.get("/api/benchmarks")
    assert res.status_code == 200
    bench = res.json()
    assert "models" in bench
    assert "multidisease" in bench
    assert bench["models"]["quantum_vqc"]["metrics"]["accuracy"] == 0.8333
    multi_vqc = bench["multidisease"]["quantum_vqc"]
    print(f"[PASS] /api/benchmarks: Baseline Accuracy = {bench['models']['quantum_vqc']['metrics']['accuracy']*100:.1f}%, Multi-Disease Hamming Loss = {multi_vqc['hamming_loss']:.4f}.")

def test_analyze_multidisease_sample():
    # Test Multi-Disease Sample
    res = client.post("/api/analyze", data={"sample_id": "nih_sample_infiltration_pneumonia", "mode": "multi_disease"})
    assert res.status_code == 200
    d = res.json()
    assert d["mode"] == "multi_disease"
    assert "predicted_findings" in d
    assert len(d["predicted_findings"]) == 8
    assert "quantum_prediction" in d
    assert "classical_predictions" in d
    print(f"[PASS] /api/analyze (Multi-Disease): {len(d['predicted_findings'])} findings evaluated. Top finding: {d['predicted_findings'][0]['finding']} ({d['predicted_findings'][0]['score']}%)")

def test_analyze_baseline_sample():
    # Test Baseline Normal Sample
    res = client.post("/api/analyze", data={"sample_id": "sample_normal_01", "mode": "baseline"})
    assert res.status_code == 200
    d = res.json()
    assert d["mode"] == "baseline"
    assert d["quantum_prediction"]["predicted_class"] in ["Normal", "Pneumonia"]
    print(f"[PASS] /api/analyze (Baseline Normal): Pred = {d['quantum_prediction']['predicted_class']} ({d['quantum_prediction']['confidence']*100:.1f}%)")

def test_analyze_upload():
    test_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    test_img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("patient_upload.jpg", buf, "image/jpeg")}
    res = client.post("/api/analyze", files=files, data={"mode": "multi_disease"})
    assert res.status_code == 200
    d = res.json()
    assert d["source_info"]["source_name"] == "patient_upload.jpg"
    assert d["source_info"]["is_sample"] is False
    assert len(d["predicted_findings"]) == 8
    print(f"[PASS] /api/analyze (Uploaded JPEG Multi-Disease): Latency = {d['timing_breakdown']['quantum_vqc_ms']} ms")

def test_frontend_serving():
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "time-cnn" in html
    assert "time-quantum" in html
    assert "browse-btn" in html
    assert "xray-dropzone" in html
    print(f"[PASS] GET / (Frontend): HTML rendered ({len(html)} bytes).")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING COMPREHENSIVE ENDPOINT & REGRESSION TESTS")
    print("=" * 60)
    test_health()
    test_modalities()
    test_samples()
    test_circuit()
    test_benchmarks()
    test_analyze_multidisease_sample()
    test_analyze_baseline_sample()
    test_analyze_upload()
    test_frontend_serving()
    print("=" * 60)
    print("ALL 9 TEST SUITES PASSED WITH ZERO ERRORS!")
    print("=" * 60)
