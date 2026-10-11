// State Management
let currentTab = 'overview';
let currentMode = 'multi_disease'; // 'multi_disease' or 'baseline'
let currentGalleryFilter = 'all';
let sampleList = [];
let benchmarkData = null;
let circuitData = null;
let charts = {};
let selectedUploadFile = null;

// Safe DOM Helper Functions (Prevents "Cannot set properties of null" error)
function getEl(id) {
    return document.getElementById(id);
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) {
        el.innerText = text;
    } else {
        console.warn(`[DOM Safe] Element with id '${id}' not found when setting innerText.`);
    }
}

function setHtml(id, html) {
    const el = document.getElementById(id);
    if (el) {
        el.innerHTML = html;
    } else {
        console.warn(`[DOM Safe] Element with id '${id}' not found when setting innerHTML.`);
    }
}

// Initialization on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    checkHealth();
    loadSamples();
    setupDropzone();
    loadCircuit();
    loadBenchmarks();
});

// Navigation Handling
function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-link');
    navButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const target = btn.dataset.target;
            switchTab(target);
        });
    });
}

function switchTab(tabId) {
    currentTab = tabId;
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.nav-link').forEach(el => {
        el.classList.remove('text-blue-600', 'border-b-2', 'border-blue-600');
        el.classList.add('text-slate-600');
    });

    const activeContent = document.getElementById(`tab-${tabId}`);
    const activeLink = document.querySelector(`[data-target="${tabId}"]`);

    if (activeContent) activeContent.classList.remove('hidden');
    if (activeLink) {
        activeLink.classList.remove('text-slate-600');
        activeLink.classList.add('text-blue-600', 'border-b-2', 'border-blue-600');
    }

    if (tabId === 'benchmarks' && benchmarkData) {
        renderBenchmarkCharts(benchmarkData);
    }
    if (tabId === 'circuit') {
        loadCircuit();
    }
}

// Health Check
async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        if (!res.ok) throw new Error("Health check failed");
        const data = await res.json();
        const badge = getEl('system-status-badge');
        if (badge && data.status === 'online') {
            badge.innerHTML = `<span class="inline-block w-2 h-2 rounded-full bg-emerald-500 mr-2"></span> System Online (${data.device.toUpperCase()} | ${data.qubit_count} Qubits Simulator)`;
            badge.className = 'px-3 py-1 rounded-full text-xs font-medium border border-emerald-200 bg-emerald-50 text-emerald-800 flex items-center';
        }
    } catch (err) {
        console.warn("Backend offline or unreachable:", err);
        const badge = getEl('system-status-badge');
        if (badge) {
            badge.innerHTML = `<span class="inline-block w-2 h-2 rounded-full bg-red-500 mr-2"></span> Backend Offline`;
            badge.className = 'px-3 py-1 rounded-full text-xs font-medium border border-red-200 bg-red-50 text-red-800 flex items-center';
        }
    }
}

// Diagnostic Mode Switcher
function setAnalysisMode(mode) {
    currentMode = mode;
    const btnMulti = getEl('btn-mode-multidisease');
    const btnBase = getEl('btn-mode-baseline');
    const modePill = getEl('active-mode-pill');

    if (mode === 'multi_disease') {
        if (btnMulti) {
            btnMulti.className = 'px-3.5 py-1.5 text-xs font-semibold rounded-md shadow-xs bg-white text-indigo-700 transition-all flex items-center gap-1.5';
        }
        if (btnBase) {
            btnBase.className = 'px-3.5 py-1.5 text-xs font-medium rounded-md text-slate-600 hover:text-slate-900 transition-all flex items-center gap-1.5';
        }
        if (modePill) {
            modePill.innerText = 'Multi-Disease Active';
            modePill.className = 'text-[10px] px-2 py-0.5 rounded-full font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200';
        }
    } else {
        if (btnMulti) {
            btnMulti.className = 'px-3.5 py-1.5 text-xs font-medium rounded-md text-slate-600 hover:text-slate-900 transition-all flex items-center gap-1.5';
        }
        if (btnBase) {
            btnBase.className = 'px-3.5 py-1.5 text-xs font-semibold rounded-md shadow-xs bg-white text-blue-700 transition-all flex items-center gap-1.5';
        }
        if (modePill) {
            modePill.innerText = 'Baseline Triage Active';
            modePill.className = 'text-[10px] px-2 py-0.5 rounded-full font-semibold bg-blue-50 text-blue-700 border border-blue-200';
        }
    }

    // Refresh circuit if circuit tab is open
    loadCircuit();
}

// Modality Roadmap Modal Handlers
const MODALITY_DATA = {
    ct_scan: {
        title: "Computed Tomography (CT)",
        badge: "Planned / In Development",
        icon: "layers",
        qubits: "8 Qubits",
        resolution: "512×512×N Volumetric",
        ansatz: "Tree Tensor Network (TTN) Variational Circuit",
        description: "Volumetric high-resolution CT scans capture fine-grained interstitial opacities and bronchial wall thickening across 3D slice stacks. The planned extension integrates a 3D-ResNet / MedNeXt classical backbone with an 8-qubit Tree Tensor Network ansatz to model axial-sagittal spatial correlations without exponential gate explosion."
    },
    mri: {
        title: "Magnetic Resonance Imaging (MRI)",
        badge: "Planned / In Development",
        icon: "disc",
        qubits: "8 Qubits",
        resolution: "256×256×N Multi-Sequence",
        ansatz: "Quantum Convolutional Neural Network (QCNN)",
        description: "MRI captures multi-parametric soft-tissue contrasts (T1, T2, FLAIR, Diffusion). The planned modular pipeline applies multi-channel feature extraction followed by a Quantum Convolutional Neural Network (QCNN) featuring quantum convolution and pooling circuits for tissue characterization."
    },
    ultrasound: {
        title: "Point-of-Care Ultrasound (POCUS)",
        badge: "Planned / In Development",
        icon: "radio",
        qubits: "4 Qubits",
        resolution: "320×240 Frame Sequences",
        ansatz: "Quantum Kernel Ridge Classifier",
        description: "Bedside pulmonary ultrasound detects pleural sliding, B-line lung profiles, and subpleural consolidations in real time. The planned pipeline utilizes MobileNetV3 temporal feature extraction coupled with a Quantum Kernel Ridge Classifier for rapid portable edge triage."
    }
};

function showModalityRoadmap(modalityId) {
    const modal = getEl('modality-modal');
    const info = MODALITY_DATA[modalityId];
    if (!modal || !info) return;

    setText('modal-title', info.title);
    setText('modal-badge', info.badge);
    setText('modal-description', info.description);
    setText('modal-qubits', info.qubits);
    setText('modal-resolution', info.resolution);
    setText('modal-ansatz', info.ansatz);

    modal.classList.remove('hidden');
    if (window.lucide) lucide.createIcons();
}

function closeModalityModal() {
    const modal = getEl('modality-modal');
    if (modal) modal.classList.add('hidden');
}

// Load Pre-packaged Showcase Samples
async function loadSamples() {
    try {
        const res = await fetch('/api/samples');
        if (!res.ok) throw new Error("Failed to fetch samples");
        const data = await res.json();
        sampleList = data.samples || [];
        renderGalleryItems();
    } catch (err) {
        console.error("Failed to load sample gallery:", err);
    }
}

function filterGallery(filter) {
    currentGalleryFilter = filter;
    ['filter-btn-all', 'filter-btn-multi', 'filter-btn-base'].forEach(id => {
        const btn = getEl(id);
        if (btn) {
            btn.className = 'px-2 py-0.5 text-[11px] font-medium rounded bg-slate-100 text-slate-600 hover:bg-slate-200 transition-colors';
        }
    });

    const activeBtn = filter === 'all' ? getEl('filter-btn-all') :
                     (filter === 'multi_disease' ? getEl('filter-btn-multi') : getEl('filter-btn-base'));
    if (activeBtn) {
        activeBtn.className = 'px-2 py-0.5 text-[11px] font-semibold rounded bg-blue-50 text-blue-700 border border-blue-200 transition-colors';
    }

    renderGalleryItems();
}

function renderGalleryItems() {
    const container = getEl('sample-gallery');
    if (!container) return;

    const filtered = sampleList.filter(s => {
        if (currentGalleryFilter === 'all') return true;
        return s.mode === currentGalleryFilter;
    });

    const countBadge = getEl('gallery-count-badge');
    if (countBadge) countBadge.innerText = `${filtered.length} Scans Shown`;

    container.innerHTML = filtered.map(s => {
        const isMulti = s.mode === 'multi_disease';
        const badgeText = isMulti ? (s.findings ? s.findings.slice(0, 2).join(' + ') : 'NIH Scan') : s.ground_truth;
        const badgeClass = isMulti ? 'badge-quantum' : (s.ground_truth === 'Normal' ? 'badge-normal' : 'badge-pneumonia');
        const datasetTag = isMulti ? 'NIH' : 'Kermany';

        return `
            <div onclick="selectSample('${s.id}', '${s.ground_truth}', '${s.filename}', '${s.mode}')" 
                 class="group relative cursor-pointer overflow-hidden rounded-lg border border-slate-200 bg-white p-1.5 transition-all hover:border-blue-500 hover:shadow-sm">
                <div class="aspect-square w-full overflow-hidden rounded bg-slate-100 relative">
                    <img src="data:image/jpeg;base64,${s.thumbnail}" 
                         alt="${s.description}" 
                         class="h-full w-full object-cover transition-transform duration-200 group-hover:scale-105" />
                    <span class="absolute top-1 right-1 px-1 py-0.2 rounded text-[8px] font-mono font-bold bg-slate-900/70 text-white">
                        ${datasetTag}
                    </span>
                </div>
                <div class="mt-1.5 flex items-center justify-between px-0.5">
                    <span class="text-[9px] font-bold px-1.5 py-0.2 rounded truncate max-w-[85px] ${badgeClass}" title="${s.ground_truth}">
                        ${badgeText}
                    </span>
                    <span class="text-[8px] font-mono text-slate-400 truncate max-w-[40px]">${s.id.replace('nih_sample_', '').replace('sample_', '')}</span>
                </div>
            </div>
        `;
    }).join('');
}

// Dropzone & File Upload Setup
function setupDropzone() {
    const dropzone = getEl('xray-dropzone');
    const fileInput = getEl('file-input');
    const browseBtn = getEl('browse-btn');
    const dropText = getEl('dropzone-text');
    const runBtn = getEl('run-upload-analysis-btn');

    if (!dropzone || !fileInput) return;

    dropzone.addEventListener('click', (e) => {
        if (e.target.closest('#browse-btn')) return;
        fileInput.click();
    });

    if (browseBtn) {
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-over');
            if (dropText) dropText.innerText = "Drop Chest X-Ray Here";
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-over');
            if (dropText) dropText.innerText = selectedUploadFile ? `Selected: ${selectedUploadFile.name}` : "Drag & Drop Chest X-Ray Here";
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleSelectedFile(fileInput.files[0]);
        }
    });

    if (runBtn) {
        runBtn.addEventListener('click', () => {
            if (selectedUploadFile) {
                analyzeScan({ file: selectedUploadFile, fileName: selectedUploadFile.name, mode: currentMode });
            }
        });
    }
}

// Handle Validated Selected File
function handleSelectedFile(file) {
    const errorBox = getEl('upload-error-box');
    const errorText = getEl('upload-error-text');
    const infoBox = getEl('selected-file-info');
    const nameEl = getEl('selected-file-name');
    const sizeEl = getEl('selected-file-size');
    const thumbEl = getEl('selected-preview-thumb');
    const dropText = getEl('dropzone-text');

    if (errorBox) errorBox.classList.add('hidden');

    const validExts = ['.jpg', '.jpeg', '.png', '.webp'];
    const lowerName = file.name.toLowerCase();
    const hasValidExt = validExts.some(ext => lowerName.endsWith(ext));
    const isImageMime = file.type.startsWith('image/');

    if (!hasValidExt && !isImageMime) {
        if (errorBox && errorText) {
            errorText.innerText = `Unsupported file format. Please upload a chest radiograph in JPG, JPEG, PNG, or WEBP format.`;
            errorBox.classList.remove('hidden');
        }
        return;
    }

    const maxSizeBytes = 15 * 1024 * 1024;
    if (file.size > maxSizeBytes) {
        if (errorBox && errorText) {
            errorText.innerText = `File is too large (${(file.size / (1024*1024)).toFixed(1)} MB). Maximum allowed size is 15 MB.`;
            errorBox.classList.remove('hidden');
        }
        return;
    }

    selectedUploadFile = file;

    if (dropText) {
        dropText.innerText = `Selected: ${file.name}`;
    }

    if (nameEl) nameEl.innerText = file.name;
    if (sizeEl) {
        const sizeFormatted = file.size > 1024 * 1024 
            ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` 
            : `${(file.size / 1024).toFixed(1)} KB`;
        sizeEl.innerText = sizeFormatted;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        if (thumbEl) thumbEl.src = e.target.result;
    };
    reader.readAsDataURL(file);

    if (infoBox) infoBox.classList.remove('hidden');

    analyzeScan({ file: file, fileName: file.name, groundTruth: null, mode: currentMode });
}

// Gallery Item Selection
function selectSample(sampleId, groundTruth, filename, sampleMode = 'multi_disease') {
    if (sampleMode && sampleMode !== currentMode) {
        setAnalysisMode(sampleMode);
    }
    analyzeScan({ sampleId: sampleId, fileName: filename || sampleId, groundTruth: groundTruth, mode: sampleMode || currentMode });
}

// Unified Scan Analysis Function (Used for BOTH File Upload & Test Gallery)
async function analyzeScan({ file = null, sampleId = null, fileName = "", groundTruth = null, mode = null }) {
    showLoading(true);
    
    const analysisMode = mode || currentMode;

    const placeholder = getEl('analysis-placeholder');
    if (placeholder) placeholder.classList.add('hidden');

    const resultsSec = getEl('analysis-results-section');
    if (resultsSec) {
        resultsSec.classList.remove('hidden');
        resultsSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    setText('active-scan-label', fileName || (file ? file.name : sampleId) || "Medical Scan");
    
    const gtContainer = getEl('ground-truth-container');
    const gtBadge = getEl('ground-truth-badge');
    if (gtContainer && gtBadge) {
        if (groundTruth) {
            gtBadge.innerText = groundTruth;
            gtBadge.className = groundTruth.includes('Normal') ? 'px-2 py-0.5 rounded text-[11px] font-bold badge-normal' : 'px-2 py-0.5 rounded text-[11px] font-bold badge-detected';
            gtContainer.classList.remove('hidden');
        } else {
            gtBadge.innerText = "Unknown / Patient Scan";
            gtBadge.className = 'px-2 py-0.5 rounded text-[11px] font-medium bg-slate-100 text-slate-600 border border-slate-200';
        }
    }

    const formData = new FormData();
    if (file) {
        formData.append('file', file);
    } else if (sampleId) {
        formData.append('sample_id', sampleId);
    }
    formData.append('mode', analysisMode);

    try {
        const res = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server responded with status ${res.status}`);
        }

        const data = await res.json();
        renderAnalysisResults(data, groundTruth);
    } catch (err) {
        console.error("[Diagnostic Pipeline Error]:", err);
        alert("Unable to process the medical scan: " + err.message);
    } finally {
        showLoading(false);
    }
}

function showLoading(isLoading) {
    const loader = getEl('analysis-loader');
    const content = getEl('analysis-content');
    if (loader) {
        if (isLoading) loader.classList.remove('hidden');
        else loader.classList.add('hidden');
    }
    if (content) {
        if (isLoading) content.classList.add('opacity-40', 'pointer-events-none');
        else content.classList.remove('opacity-40', 'pointer-events-none');
    }
}

// Render Analysis Results to DOM
function renderAnalysisResults(data, knownGroundTruth = null) {
    // 1. Render Preprocessed & Attention Images
    const rawImg = getEl('img-raw-scan');
    const claheImg = getEl('img-clahe-scan');
    const gradcamImg = getEl('img-gradcam');
    const gradcamBox = getEl('gradcam-container');

    if (rawImg && data.images && data.images.raw_scan) {
        rawImg.src = `data:image/jpeg;base64,${data.images.raw_scan}`;
    }
    if (claheImg && data.images && data.images.clahe_scan) {
        claheImg.src = `data:image/jpeg;base64,${data.images.clahe_scan}`;
    }
    if (gradcamImg && data.images && data.images.gradcam_heatmap) {
        gradcamImg.src = `data:image/jpeg;base64,${data.images.gradcam_heatmap}`;
        if (gradcamBox) gradcamBox.classList.remove('hidden');
    }

    // 2. CNN Features Information
    if (data.cnn_features) {
        setText('cnn-dim', `${data.cnn_features.dimension} dims`);
        setText('cnn-l2', `${data.cnn_features.l2_norm}`);
        setText('cnn-range', `[${data.cnn_features.min_val}, ${data.cnn_features.max_val}]`);
    }

    // 3. PCA Dimensionality Reduction
    if (data.pca_reduction) {
        const pca = data.pca_reduction;
        setText('pca-total-var', `${(pca.total_explained_variance * 100).toFixed(1)}%`);
        renderPCAChart(pca.scaled_angles, pca.explained_variance_ratio);
    }

    // Mode-specific rendering
    const multiContainer = getEl('multidisease-results-container');
    const baseContainer = getEl('baseline-results-container');

    if (data.mode === 'multi_disease') {
        if (multiContainer) multiContainer.classList.remove('hidden');
        if (baseContainer) baseContainer.classList.add('hidden');

        renderMultiDiseaseResults(data);
    } else {
        if (multiContainer) multiContainer.classList.add('hidden');
        if (baseContainer) baseContainer.classList.remove('hidden');

        renderBaselineResults(data);
    }

    // 6. Agreement & Consensus Banner
    renderAgreementBanner(data);

    // 7. Execution Timings
    if (data.timing_breakdown) {
        const tb = data.timing_breakdown;
        setText('time-preprocess', `${tb.preprocessing_ms || 0} ms`);
        setText('time-cnn', `${tb.cnn_extraction_ms || 0} ms`);
        setText('time-pca', `${tb.pca_reduction_ms || 0} ms`);
        setText('time-quantum', `${tb.quantum_vqc_ms || 0} ms`);
        
        const total = Object.values(tb).reduce((a, b) => a + b, 0);
        setText('time-total', `${total.toFixed(1)} ms`);
    }
}

// Render Multi-Disease Findings Table
function renderMultiDiseaseResults(data) {
    const tableBody = getEl('findings-table-body');
    const findings = data.predicted_findings || [];
    const q = data.quantum_prediction || {};

    setText('multi-quantum-latency', `${(q.inference_latency_ms || 0).toFixed(2)} ms`);

    if (tableBody) {
        tableBody.innerHTML = findings.map(f => {
            let statusBadge = '';
            let barColor = 'bg-slate-400';

            if (f.status === 'Detected') {
                statusBadge = '<span class="inline-block px-2.5 py-0.5 rounded text-xs font-bold badge-detected">Detected (≥50%)</span>';
                barColor = 'bg-rose-600';
            } else if (f.status === 'Possible') {
                statusBadge = '<span class="inline-block px-2.5 py-0.5 rounded text-xs font-bold badge-possible">Possible (25–49%)</span>';
                barColor = 'bg-amber-500';
            } else {
                statusBadge = '<span class="inline-block px-2.5 py-0.5 rounded text-xs font-medium badge-low">Low (&lt;25%)</span>';
                barColor = 'bg-slate-300';
            }

            return `
                <tr class="hover:bg-slate-50/70 transition-colors">
                    <td class="py-2.5 px-3.5 font-semibold text-slate-800">${f.finding}</td>
                    <td class="py-2.5 px-3.5 text-center">${statusBadge}</td>
                    <td class="py-2.5 px-3.5 text-center font-mono font-bold text-slate-800">${f.score}%</td>
                    <td class="py-2.5 px-3.5">
                        <div class="w-full bg-slate-100 h-2 rounded-full overflow-hidden border border-slate-200">
                            <div class="h-full ${barColor} transition-all duration-500" style="width: ${f.score}%;"></div>
                        </div>
                    </td>
                </tr>
            `;
        }).join('');
    }

    // Observables list
    const obsContainer = getEl('multi-observables-list');
    if (obsContainer && q.observable_expectations) {
        obsContainer.innerHTML = q.observable_expectations.map((ev, i) => `
            <span class="inline-flex items-center px-2 py-0.5 rounded bg-slate-100 border border-slate-200 font-mono text-[11px] text-indigo-900 font-semibold">
                ⟨Z<sub>${i}</sub>⟩: ${ev.toFixed(3)}
            </span>
        `).join('');
    }

    // Classical Multi-Disease Summary
    const clContainer = getEl('multi-classical-summary');
    if (clContainer && data.classical_predictions) {
        const lr = data.classical_predictions.logistic_regression || {};
        const mlp = data.classical_predictions.mlp || {};

        const lrDetected = lr.detected_findings || [];
        const mlpDetected = mlp.detected_findings || [];

        clContainer.innerHTML = `
            <div class="p-3 rounded-lg bg-white border border-slate-200">
                <span class="text-[10px] font-mono uppercase font-bold text-slate-500 block">Multi-Output Logistic Regression</span>
                <p class="mt-1 text-slate-800">
                    Detected Findings: <strong>${lrDetected.length > 0 ? lrDetected.join(', ') : 'None (All &lt;50%)'}</strong>
                </p>
            </div>
            <div class="p-3 rounded-lg bg-white border border-slate-200">
                <span class="text-[10px] font-mono uppercase font-bold text-slate-500 block">Multi-Label MLP Classifier</span>
                <p class="mt-1 text-slate-800">
                    Detected Findings: <strong>${mlpDetected.length > 0 ? mlpDetected.join(', ') : 'None (All &lt;50%)'}</strong>
                </p>
            </div>
        `;
    }
}

// Render Baseline Dual Diagnosis
function renderBaselineResults(data) {
    if (data.quantum_prediction) {
        const q = data.quantum_prediction;
        const qBadge = getEl('quantum-class-badge');
        if (qBadge) {
            qBadge.innerText = q.predicted_class;
            qBadge.className = q.predicted_class === 'Pneumonia' 
                ? 'mt-1 inline-block px-3 py-1 rounded-full text-sm font-bold badge-pneumonia shadow-xs'
                : 'mt-1 inline-block px-3 py-1 rounded-full text-sm font-bold badge-normal shadow-xs';
        }

        setText('quantum-confidence', `${(q.confidence * 100).toFixed(1)}%`);
        
        const meterBar = getEl('quantum-meter-bar');
        if (meterBar) {
            meterBar.style.width = `${(q.confidence * 100).toFixed(1)}%`;
            meterBar.className = q.predicted_class === 'Pneumonia'
                ? 'h-full bg-rose-600 transition-all duration-500'
                : 'h-full bg-emerald-600 transition-all duration-500';
        }

        setText('quantum-expval', `${(q.pauli_z_expectation || 0).toFixed(4)}`);
        setText('quantum-latency', `${(q.inference_latency_ms || 0).toFixed(2)} ms`);
        
        if (q.circuit_summary) {
            setText('quantum-meta-gates', `${q.circuit_summary.total_gates} gates (${q.circuit_summary.cnot_gates} CNOT)`);
        }
    }

    if (data.classical_predictions) {
        renderClassicalPredictions(data.classical_predictions);
    }
}

// Render Agreement Banner
function renderAgreementBanner(data) {
    const agreeBadge = getEl('agreement-badge');
    if (!agreeBadge) return;

    if (data.mode === 'multi_disease') {
        const qDet = (data.quantum_prediction && data.quantum_prediction.detected_findings) || [];
        const clDet = (data.classical_predictions && data.classical_predictions.logistic_regression && data.classical_predictions.logistic_regression.detected_findings) || [];

        if (data.agreement) {
            agreeBadge.innerHTML = `<span class="text-emerald-700 font-semibold">✓ Multi-Model Consensus</span>: Quantum VQC and Classical models agree on detected pathologies: <strong>${qDet.length > 0 ? qDet.join(', ') : 'No acute findings'}</strong>.`;
            agreeBadge.className = 'p-3.5 rounded-lg bg-emerald-50 border border-emerald-200 text-xs text-emerald-800';
        } else {
            agreeBadge.innerHTML = `<span class="text-amber-700 font-semibold">⚠ Divergent Findings</span>: Quantum VQC flagged <strong>${qDet.join(', ') || 'None'}</strong>, while Classical Logistic Regression flagged <strong>${clDet.join(', ') || 'None'}</strong>.`;
            agreeBadge.className = 'p-3.5 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-800';
        }
    } else {
        const qClass = data.quantum_prediction ? data.quantum_prediction.predicted_class : 'Unknown';
        const svmClass = data.classical_predictions && data.classical_predictions.svm_rbf ? data.classical_predictions.svm_rbf.predicted_class : qClass;
        
        if (data.agreement) {
            agreeBadge.innerHTML = `<span class="text-emerald-700 font-semibold">✓ Consensual Output</span>: Quantum VQC and Classical SVM both classified this scan as <strong>${qClass}</strong>.`;
            agreeBadge.className = 'p-3.5 rounded-lg bg-emerald-50 border border-emerald-200 text-xs text-emerald-800';
        } else {
            agreeBadge.innerHTML = `<span class="text-amber-700 font-semibold">⚠ Divergent Output</span>: Quantum VQC classified <strong>${qClass}</strong> while Classical SVM classified <strong>${svmClass}</strong>.`;
            agreeBadge.className = 'p-3.5 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-800';
        }
    }
}

// Render PCA Chart
function renderPCAChart(angles, varRatios) {
    const ctx = getEl('pcaChart');
    if (!ctx) return;

    if (charts.pca) charts.pca.destroy();

    charts.pca = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: angles.map((_, i) => `Q${i} (Ry angle)`),
            datasets: [
                {
                    label: 'Encoded Angle (rad ∈ [0, π])',
                    data: angles,
                    backgroundColor: 'rgba(79, 70, 229, 0.75)',
                    borderColor: 'rgb(79, 70, 229)',
                    borderWidth: 1,
                    borderRadius: 4,
                    yAxisID: 'y'
                },
                {
                    label: 'Explained Variance Ratio',
                    data: varRatios,
                    backgroundColor: 'rgba(37, 99, 235, 0.4)',
                    borderColor: 'rgb(37, 99, 235)',
                    borderWidth: 1,
                    borderRadius: 4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Angle (rad)', color: '#475569', font: { size: 10 } },
                    grid: { color: 'rgba(226, 232, 240, 0.8)' },
                    ticks: { color: '#475569', font: { size: 10 } },
                    max: 3.2
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: 'Variance Ratio', color: '#2563eb', font: { size: 10 } },
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#2563eb', font: { size: 10 } },
                    min: 0,
                    max: 0.2
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#475569', font: { size: 10 } }
                }
            },
            plugins: {
                legend: { 
                    labels: { color: '#334155', font: { size: 11 } }
                }
            }
        }
    });
}

// Render Classical Baselines List
function renderClassicalPredictions(cl) {
    const container = getEl('classical-models-list');
    if (!container) return;

    const models = [
        { key: 'svm_rbf', name: 'Support Vector Machine (RBF)', desc: 'Gaussian kernel non-linear decision boundary' },
        { key: 'svm_linear', name: 'SVM (Linear)', desc: 'Linear maximum-margin hyperplane' },
        { key: 'logistic_regression', name: 'Logistic Regression', desc: 'Linear log-odds estimator' },
        { key: 'mlp', name: 'Multi-Layer Perceptron (MLP)', desc: 'Feedforward architecture (4-16-8-1)' }
    ];

    container.innerHTML = models.map(m => {
        const item = cl[m.key] || { predicted_class: 'N/A', confidence: 0.5 };
        const isPneumonia = item.predicted_class === 'Pneumonia';
        const badgeClass = isPneumonia ? 'badge-pneumonia' : 'badge-normal';
        return `
            <div class="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50/70 p-2.5">
                <div>
                    <h5 class="text-xs font-semibold text-slate-800">${m.name}</h5>
                    <p class="text-[10px] text-slate-500">${m.desc}</p>
                </div>
                <div class="text-right">
                    <span class="inline-block px-2 py-0.5 rounded text-xs font-bold ${badgeClass}">
                        ${item.predicted_class}
                    </span>
                    <p class="text-[10px] text-slate-500 mt-0.5">${(item.confidence * 100).toFixed(1)}% conf</p>
                </div>
            </div>
        `;
    }).join('');
}

// Load Circuit Specifications
async function loadCircuit() {
    try {
        const res = await fetch(`/api/circuit?mode=${currentMode}`);
        if (!res.ok) return;
        circuitData = await res.json();
        renderCircuitView(circuitData);
    } catch (err) {
        console.error("Failed to load circuit data:", err);
    }
}

function renderCircuitView(data) {
    setText('circuit-stat-qubits', data.n_qubits);
    setText('circuit-stat-layers', data.n_layers);
    setText('circuit-stat-gates', data.total_gates);
    setText('circuit-stat-params', data.param_count);

    const asciiEl = getEl('circuit-ascii');
    if (asciiEl) asciiEl.textContent = data.ascii_diagram || "Circuit diagram available.";

    const wiresContainer = getEl('circuit-wires-visual');
    if (!wiresContainer || !data.gates) return;

    let wiresHtml = '';
    for (let q = 0; q < data.n_qubits; q++) {
        const qGates = data.gates.filter(g => g.wire === q || g.control === q || g.target === q);
        wiresHtml += `
            <div class="circuit-wire">
                <div class="w-16 font-mono font-bold text-xs text-indigo-700 flex items-center">
                    |q<sub>${q}</sub>⟩ ─
                </div>
                <div class="flex-1 flex items-center space-x-1.5 overflow-x-auto py-1">
        `;

        qGates.forEach(g => {
            if (g.type === 'RY') {
                wiresHtml += `<span class="gate-box gate-ry" title="Ry gate (θ = ${g.angle.toFixed(3)} rad)">Ry</span>`;
            } else if (g.type === 'RZ') {
                wiresHtml += `<span class="gate-box gate-rz" title="Rz gate (φ = ${g.angle.toFixed(3)} rad)">Rz</span>`;
            } else if (g.type === 'CNOT') {
                if (g.control === q) {
                    wiresHtml += `<span class="gate-box gate-cnot" title="CNOT Control to Wire ${g.target}">●</span>`;
                } else {
                    wiresHtml += `<span class="gate-box gate-cnot" title="CNOT Target from Wire ${g.control}">⊕</span>`;
                }
            } else if (g.type === 'MEASURE' && g.wire === q) {
                wiresHtml += `<span class="gate-box gate-measure" title="Pauli-Z Expectation Measurement on Qubit ${q}">⟨Z<sub>${q}</sub>⟩</span>`;
            }
        });

        wiresHtml += `
                </div>
            </div>
        `;
    }
    wiresContainer.innerHTML = wiresHtml;
}

// Load Benchmark Matrix & Render Charts
async function loadBenchmarks() {
    try {
        const res = await fetch('/api/benchmarks');
        if (!res.ok) return;
        benchmarkData = await res.json();
        renderBenchmarkTable(benchmarkData);
    } catch (err) {
        console.error("Failed to load benchmarks:", err);
    }
}

function renderBenchmarkTable(data) {
    const tbody = getEl('benchmark-table-body');
    if (!tbody || !data.models) return;

    const rows = Object.entries(data.models).map(([key, m]) => {
        const met = m.metrics;
        const isQuantum = m.type === 'Quantum AI';
        const badgeClass = isQuantum ? 'badge-quantum' : 'badge-classical';
        const rowBg = isQuantum ? 'bg-indigo-50/50 font-medium' : '';
        return `
            <tr class="hover:bg-slate-50 transition-colors ${rowBg}">
                <td class="py-3 px-4 flex items-center space-x-2">
                    <span class="inline-block px-2 py-0.5 rounded text-[10px] ${badgeClass}">
                        ${m.type}
                    </span>
                    <span class="text-xs font-semibold text-slate-800">${m.name}</span>
                </td>
                <td class="py-3 px-4 text-center font-mono font-bold text-slate-900">${(met.accuracy * 100).toFixed(1)}%</td>
                <td class="py-3 px-4 text-center font-mono text-slate-700">${(met.f1_score * 100).toFixed(1)}%</td>
                <td class="py-3 px-4 text-center font-mono text-slate-700">${(met.recall * 100).toFixed(1)}%</td>
                <td class="py-3 px-4 text-center font-mono text-slate-700">${(met.precision * 100).toFixed(1)}%</td>
                <td class="py-3 px-4 text-center font-mono text-slate-700">${(met.roc_auc * 100).toFixed(1)}%</td>
                <td class="py-3 px-4 text-center font-mono text-blue-700">${m.inference_latency_ms.toFixed(2)} ms</td>
            </tr>
        `;
    }).join('');

    tbody.innerHTML = rows;

    if (data.models.quantum_vqc && data.models.svm_rbf) {
        renderConfusionMatrix('cm-quantum', data.models.quantum_vqc.metrics.confusion_matrix, 'Quantum VQC');
        renderConfusionMatrix('cm-svm', data.models.svm_rbf.metrics.confusion_matrix, 'Classical SVM');
    }
}

function renderConfusionMatrix(elementId, cm, title) {
    const el = getEl(elementId);
    if (!el || !cm) return;
    el.innerHTML = `
        <div class="grid grid-cols-2 gap-2 text-center text-xs font-mono">
            <div class="p-3 bg-emerald-50 border border-emerald-200 rounded-lg">
                <span class="text-slate-500 block text-[10px]">True Negative (TN)</span>
                <span class="text-lg font-bold text-emerald-800">${cm.tn}</span>
                <span class="text-[9px] text-slate-500 block">Normal correctly detected</span>
            </div>
            <div class="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                <span class="text-slate-500 block text-[10px]">False Positive (FP)</span>
                <span class="text-lg font-bold text-amber-800">${cm.fp}</span>
                <span class="text-[9px] text-slate-500 block">Normal misclassified</span>
            </div>
            <div class="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                <span class="text-slate-500 block text-[10px]">False Negative (FN)</span>
                <span class="text-lg font-bold text-amber-800">${cm.fn}</span>
                <span class="text-[9px] text-slate-500 block">Pneumonia missed</span>
            </div>
            <div class="p-3 bg-emerald-50 border border-emerald-200 rounded-lg">
                <span class="text-slate-500 block text-[10px]">True Positive (TP)</span>
                <span class="text-lg font-bold text-emerald-800">${cm.tp}</span>
                <span class="text-[9px] text-slate-500 block">Pneumonia confirmed</span>
            </div>
        </div>
    `;
}

function renderBenchmarkCharts(data) {
    if (!data || !data.models) return;

    // 1. ROC Curves Chart
    const rocCtx = getEl('rocChart');
    if (rocCtx && data.models.quantum_vqc && data.models.svm_rbf) {
        if (charts.roc) charts.roc.destroy();

        const qPoints = data.models.quantum_vqc.metrics.roc_curve;
        const svmPoints = data.models.svm_rbf.metrics.roc_curve;

        charts.roc = new Chart(rocCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: `Quantum VQC (AUC = ${(data.models.quantum_vqc.metrics.roc_auc*100).toFixed(1)}%)`,
                        data: qPoints.map(p => ({ x: p.fpr, y: p.tpr })),
                        borderColor: '#4f46e5',
                        backgroundColor: 'rgba(79, 70, 229, 0.05)',
                        borderWidth: 2,
                        tension: 0.15
                    },
                    {
                        label: `Classical SVM (AUC = ${(data.models.svm_rbf.metrics.roc_auc*100).toFixed(1)}%)`,
                        data: svmPoints.map(p => ({ x: p.fpr, y: p.tpr })),
                        borderColor: '#2563eb',
                        borderDash: [4, 4],
                        borderWidth: 2,
                        tension: 0.15
                    },
                    {
                        label: 'Random Chance (AUC = 50%)',
                        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                        borderColor: '#94a3b8',
                        borderDash: [2, 2],
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'False Positive Rate (1 - Specificity)', color: '#475569', font: { size: 10 } },
                        grid: { color: 'rgba(226, 232, 240, 0.8)' },
                        ticks: { color: '#475569', font: { size: 10 } },
                        min: 0, max: 1
                    },
                    y: {
                        type: 'linear',
                        title: { display: true, text: 'True Positive Rate (Sensitivity)', color: '#475569', font: { size: 10 } },
                        grid: { color: 'rgba(226, 232, 240, 0.8)' },
                        ticks: { color: '#475569', font: { size: 10 } },
                        min: 0, max: 1
                    }
                },
                plugins: {
                    legend: { labels: { color: '#334155', font: { size: 11 } } }
                }
            }
        });
    }

    // 2. Latency Comparison Chart
    const latCtx = getEl('latencyChart');
    if (latCtx) {
        if (charts.latency) charts.latency.destroy();

        const modelNames = Object.values(data.models).map(m => m.name.replace(' Baseline', ''));
        const latencies = Object.values(data.models).map(m => m.inference_latency_ms);
        const colors = Object.values(data.models).map(m => m.type === 'Quantum AI' ? 'rgba(79, 70, 229, 0.8)' : 'rgba(37, 99, 235, 0.7)');

        charts.latency = new Chart(latCtx, {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [{
                    label: 'Inference Latency (ms)',
                    data: latencies,
                    backgroundColor: colors,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: { display: true, text: 'Latency (milliseconds)', color: '#475569', font: { size: 10 } },
                        grid: { color: 'rgba(226, 232, 240, 0.8)' },
                        ticks: { color: '#475569', font: { size: 10 } }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#475569', font: { size: 10 } }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
}
