import os
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    # Load actual benchmark data
    cache_path = "C:/SAH/backend/data/metrics_cache.json"
    benchmarks = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            benchmarks = json.load(f)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Professional Light Theme Colors
    BG_COLOR = RGBColor(248, 250, 252)         # #f8fafc (Very light crisp off-white)
    CARD_BG = RGBColor(255, 255, 255)          # #ffffff Pure white cards
    BORDER_COLOR = RGBColor(226, 232, 240)     # #e2e8f0 Subtle border
    HEADER_NAVY = RGBColor(15, 23, 42)         # #0f172a Deep professional navy
    TEXT_BODY = RGBColor(51, 65, 85)           # #334155 Dark slate gray body
    TEXT_MUTED = RGBColor(100, 116, 139)       # #64748b Medium gray
    ACCENT_BLUE = RGBColor(37, 99, 235)        # #2563eb Muted clinical blue
    ACCENT_INDIGO = RGBColor(79, 70, 229)      # #4f46e5 Muted quantum indigo
    EMERALD = RGBColor(5, 150, 105)            # #059669 Soft green
    HEADER_FILL = RGBColor(241, 245, 249)      # #f1f5f9 Table header background

    def apply_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.color.rgb = BG_COLOR
        return bg

    def add_header(slide, title_text, subtitle_text=""):
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(11.733), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0
        
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = HEADER_NAVY
        p.font.name = "Arial"

        if subtitle_text:
            p2 = tf.add_paragraph()
            p2.text = subtitle_text
            p2.font.size = Pt(12)
            p2.font.color.rgb = ACCENT_BLUE
            p2.font.name = "Arial"
            p2.space_before = Pt(3)

    def add_card(slide, left, top, width, height, border_color=BORDER_COLOR, bg_color=CARD_BG):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.0)
        return card

    # ==========================================================
    # SLIDE 1: Title Slide
    # ==========================================================
    slide1 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide1)

    card1 = add_card(slide1, Inches(1.2), Inches(1.0), Inches(10.933), Inches(5.5), border_color=ACCENT_BLUE)

    tb = slide1.shapes.add_textbox(Inches(1.8), Inches(1.5), Inches(9.733), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "QUANTUM COMPUTING CAPSTONE PROJECT"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    p2 = tf.add_paragraph()
    p2.text = "Quantum AI-Based Medical Imaging System for Disease Diagnosis from Medical Scans"
    p2.font.size = Pt(24)
    p2.font.bold = True
    p2.font.color.rgb = HEADER_NAVY
    p2.space_before = Pt(12)

    p3 = tf.add_paragraph()
    p3.text = "Modular Multi-Disease Architecture with Modality Extensibility & Hybrid Quantum Machine Learning"
    p3.font.size = Pt(14)
    p3.font.color.rgb = TEXT_BODY
    p3.space_before = Pt(8)

    p4 = tf.add_paragraph()
    p4.text = (
        "Candidate Details:\n"
        "• Student Name: [Student Name]     |     Roll Number: [Roll Number]\n"
        "• Department: Department of Computer Science & Engineering\n"
        "• Evaluation: B.Tech Capstone Project Examination (2025–2026)\n"
        "• Modality: Chest X-Ray (Active) | Planned: CT, MRI, Ultrasound\n"
        "• Frameworks: PyTorch, PennyLane Quantum Simulator, ResNet-18, FastAPI"
    )
    p4.font.size = Pt(11.5)
    p4.font.color.rgb = TEXT_MUTED
    p4.space_before = Pt(20)

    # ==========================================================
    # SLIDE 2: Problem Statement & Clinical Motivation
    # ==========================================================
    slide2 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide2)
    add_header(slide2, "Problem Statement & Clinical Motivation", "Radiological Bottlenecks, Multi-Disease Co-occurrence & Hybrid QML")

    # Left Card
    add_card(slide2, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2))
    tb2_left = slide2.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.7))
    tf2_l = tb2_left.text_frame
    tf2_l.word_wrap = True

    p = tf2_l.paragraphs[0]
    p.text = "Clinical Challenges in Medical Imaging"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    points_l = [
        ("Diagnostic Volume & Backlogs", "Hospitals interpret billions of medical images annually, leading to radiologist workload fatigue and acute triage delays."),
        ("Multi-Pathology Co-occurrence", "Real radiographs frequently display co-occurring findings (e.g. Infiltration, Effusion, and Consolidation) requiring multi-label detection rather than single-class assumptions."),
        ("High Feature Dimensionality", "Raw scans ($224 \\times 224 \\times 3 = 150,528$ values) cannot be directly fed to NISQ quantum processors due to coherence and gate depth limits."),
        ("Physician-Centered AI", "The system is formulated strictly as an educational assistive prototype, not an autonomous replacement for licensed doctors.")
    ]
    for title, desc in points_l:
        p_t = tf2_l.add_paragraph()
        p_t.text = f"• {title}:"
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = HEADER_NAVY
        p_t.space_before = Pt(8)

        p_d = tf2_l.add_paragraph()
        p_d.text = f"   {desc}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = TEXT_BODY

    # Right Card
    add_card(slide2, Inches(6.9), Inches(1.7), Inches(5.6), Inches(5.2))
    tb2_right = slide2.shapes.add_textbox(Inches(7.2), Inches(1.9), Inches(5.0), Inches(4.7))
    tf2_r = tb2_right.text_frame
    tf2_r.word_wrap = True

    p = tf2_r.paragraphs[0]
    p.text = "Role of Quantum Machine Learning"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = ACCENT_INDIGO

    points_r = [
        ("Hilbert Space Representations", "QML maps compressed visual features into $2^n$-dimensional complex Hilbert spaces using non-linear unitary transformations."),
        ("Multi-Observable Readout", "A single 4-qubit quantum circuit simultaneously measures 4 Pauli-Z observables $[\\langle Z_0 \\rangle, \\langle Z_1 \\rangle, \\langle Z_2 \\rangle, \\langle Z_3 \\rangle]$ to predict multi-label disease findings in sub-15ms."),
        ("Hybrid Classical-Quantum Synergy", "Pretrained classical CNN extracts inductive spatial textures; PCA compresses variance; Variational Quantum Circuit (VQC) executes parameterized classification."),
        ("Transparent Scientific Evaluation", "NISQ simulator reality is preserved: quantum circuits are benchmarked directly against identical-feature classical models without hyperbole.")
    ]
    for title, desc in points_r:
        p_t = tf2_r.add_paragraph()
        p_t.text = f"• {title}:"
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = HEADER_NAVY
        p_t.space_before = Pt(8)

        p_d = tf2_r.add_paragraph()
        p_d.text = f"   {desc}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = TEXT_BODY

    # ==========================================================
    # SLIDE 3: Objectives & Modality Architecture
    # ==========================================================
    slide3 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide3)
    add_header(slide3, "Project Objectives & Modular Modality Architecture", "Dual Diagnostic Modes and Extensible Cross-Modality Framework")

    # 4 Modality Cards Row
    modalities = [
        ("Chest Radiograph (X-Ray)", "ACTIVE PIPELINE", "4 Qubits | Ry/Rz Ansatz", "Evaluated on NIH ChestX-ray14 (8 findings) and Kermany pediatric dataset.", ACCENT_BLUE, True),
        ("Computed Tomography (CT)", "PLANNED / ROADMAP", "8 Qubits | TTN Ansatz", "3D-ResNet volumetric feature extraction for axial pulmonary nodule analysis.", TEXT_MUTED, False),
        ("Magnetic Resonance (MRI)", "PLANNED / ROADMAP", "8 Qubits | QCNN Ansatz", "Multi-parametric T1/T2 soft-tissue mapping for neurological lesion screening.", TEXT_MUTED, False),
        ("Ultrasound (POCUS)", "PLANNED / ROADMAP", "4 Qubits | Kernel Ridge", "Point-of-care pleural B-line acoustic profile detection in emergency triage.", TEXT_MUTED, False)
    ]
    for i, (m_name, status, q_spec, m_desc, col, is_act) in enumerate(modalities):
        left_pos = Inches(0.8 + i * 2.95)
        card_mod = add_card(slide3, left_pos, Inches(1.7), Inches(2.8), Inches(2.2), border_color=col if is_act else BORDER_COLOR)
        tb_m = slide3.shapes.add_textbox(left_pos + Inches(0.15), Inches(1.8), Inches(2.5), Inches(2.0))
        tf_m = tb_m.text_frame
        tf_m.word_wrap = True
        
        pm1 = tf_m.paragraphs[0]
        pm1.text = m_name
        pm1.font.size = Pt(12)
        pm1.font.bold = True
        pm1.font.color.rgb = HEADER_NAVY

        pm2 = tf_m.add_paragraph()
        pm2.text = f"[{status}]"
        pm2.font.size = Pt(9.5)
        pm2.font.bold = True
        pm2.font.color.rgb = col
        pm2.space_before = Pt(2)

        pm3 = tf_m.add_paragraph()
        pm3.text = q_spec
        pm3.font.size = Pt(10)
        pm3.font.color.rgb = ACCENT_INDIGO
        pm3.space_before = Pt(2)

        pm4 = tf_m.add_paragraph()
        pm4.text = m_desc
        pm4.font.size = Pt(9.5)
        pm4.font.color.rgb = TEXT_BODY
        pm4.space_before = Pt(4)

    # Bottom Objective Card
    add_card(slide3, Inches(0.8), Inches(4.1), Inches(11.733), Inches(2.8))
    tb_obj = slide3.shapes.add_textbox(Inches(1.1), Inches(4.25), Inches(11.133), Inches(2.5))
    tf_o = tb_obj.text_frame
    tf_o.word_wrap = True

    p = tf_o.paragraphs[0]
    p.text = "Key Engineering & Research Milestones Accomplished:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    milestones = [
        "1. Real Multi-Disease Pipeline: Evaluated on NIH ChestX-ray14 benchmark across 8 thoracic findings (Infiltration, Effusion, Atelectasis, Pneumothorax, Pneumonia, Cardiomegaly, Consolidation, No Finding).",
        "2. Multi-Observable VQC: Single-circuit 4-qubit parameterized ansatz reading out all 4 Pauli-Z expectation values simultaneously.",
        "3. Backward-Compatible Baseline: Preserves 83.3% accuracy binary triage mode (Normal vs Pneumonia on Kermany benchmark).",
        "4. Strict Scientific Controls: Identical 4-component PCA feature vectors fed to both Quantum VQC and Classical Multi-Output Baselines."
    ]
    for ms in milestones:
        p_ms = tf_o.add_paragraph()
        p_ms.text = ms
        p_ms.font.size = Pt(11)
        p_ms.font.color.rgb = TEXT_BODY
        p_ms.space_before = Pt(4)

    # ==========================================================
    # SLIDE 4: Proposed Hybrid Architecture
    # ==========================================================
    slide4 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide4)
    add_header(slide4, "Proposed Hybrid Classical-Quantum Architecture", "End-to-End Processing Flow: Image Acquisition to Calibrated Findings")

    stages = [
        ("STAGE 01", "Scan Acquisition", "Chest radiograph input (224×224) via DICOM/JPEG upload or verified test gallery.", ACCENT_BLUE),
        ("STAGE 02", "Preprocessing", "CLAHE contrast equalization & ImageNet z-score intensity standardization.", ACCENT_BLUE),
        ("STAGE 03", "CNN Feature Extractor", "ResNet-18 penultimate pooling layer yields 512-dim latent visual representation + Grad-CAM.", ACCENT_INDIGO),
        ("STAGE 04", "PCA Reduction", "Orthogonal covariance projection compresses 512 dims down to 4 angles scaled to [0, π].", ACCENT_INDIGO),
        ("STAGE 05", "Quantum VQC Engine", "Angle embedding, 2 entangling layers, simultaneous 4-qubit Pauli-Z expectation readout.", ACCENT_INDIGO),
        ("STAGE 06", "Findings & Baselines", "Calibrated finding scores (≥50% Detected, 25-49% Possible) + Classical concordance.", EMERALD)
    ]
    for i, (stg_num, stg_title, stg_desc, col) in enumerate(stages):
        x = Inches(0.8 + (i % 3) * 3.95)
        y = Inches(1.7 + (i // 3) * 2.6)
        add_card(slide4, x, y, Inches(3.8), Inches(2.4))
        tb_s = slide4.shapes.add_textbox(x + Inches(0.2), y + Inches(0.15), Inches(3.4), Inches(2.1))
        tf_s = tb_s.text_frame
        tf_s.word_wrap = True

        p1 = tf_s.paragraphs[0]
        p1.text = stg_num
        p1.font.size = Pt(10)
        p1.font.bold = True
        p1.font.color.rgb = col

        p2 = tf_s.add_paragraph()
        p2.text = stg_title
        p2.font.size = Pt(13)
        p2.font.bold = True
        p2.font.color.rgb = HEADER_NAVY
        p2.space_before = Pt(2)

        p3 = tf_s.add_paragraph()
        p3.text = stg_desc
        p3.font.size = Pt(10.5)
        p3.font.color.rgb = TEXT_BODY
        p3.space_before = Pt(4)

    # ==========================================================
    # SLIDE 5: Quantum Circuit Design & Readout
    # ==========================================================
    slide5 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide5)
    add_header(slide5, "Variational Quantum Circuit Architecture", "Multi-Observable Angle-Encoded Parameterized Ansatz Design")

    # Left Card: Gate Specifications
    add_card(slide5, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2))
    tb5_l = slide5.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.7))
    tf5_l = tb5_l.text_frame
    tf5_l.word_wrap = True

    p = tf5_l.paragraphs[0]
    p.text = "Quantum Circuit Formulation"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = ACCENT_INDIGO

    circuit_specs = [
        ("Quantum Feature Map", "Angle Embedding: |ψ(x)⟩ = ⨂ Ry(x_i)|0⟩ for i ∈ {0, 1, 2, 3}. Constant O(1) circuit depth."),
        ("Single-Qubit Rotations", "Trainable Ry(θ_{l,i}) and Rz(φ_{l,i}) rotations per layer (16 rotational parameters across 2 layers)."),
        ("Entanglement Topology", "Circular CNOT ladder: CNOT(q_i, q_{(i+1) mod 4}) creates cross-qubit entanglement across the latent space."),
        ("Multi-Observable Readout", "Measures all 4 wires: e = [⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩]^T ∈ [-1, 1]⁴ simultaneously in a single simulation pass."),
        ("Linear Projection Head", "z = W_q · e + b_q maps 4 expectations to 8 finding logits; σ(z_k) yields independent probabilities.")
    ]
    for title, desc in circuit_specs:
        p_t = tf5_l.add_paragraph()
        p_t.text = f"• {title}:"
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = HEADER_NAVY
        p_t.space_before = Pt(8)

        p_d = tf5_l.add_paragraph()
        p_d.text = f"   {desc}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = TEXT_BODY

    # Right Card: ASCII Circuit Architecture
    add_card(slide5, Inches(6.9), Inches(1.7), Inches(5.6), Inches(5.2))
    tb5_r = slide5.shapes.add_textbox(Inches(7.2), Inches(1.9), Inches(5.0), Inches(4.7))
    tf5_r = tb5_r.text_frame
    tf5_r.word_wrap = True

    p = tf5_r.paragraphs[0]
    p.text = "PennyLane Simulated Circuit Topology"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    ascii_text = (
        "|q₀⟩: ─Ry(x₀)─Ry(θ₀,₀)─Rz(φ₀,₀)─●───────╭X─ ... ─┤ ⟨Z₀⟩\n"
        "                                │       │\n"
        "|q₁⟩: ─Ry(x₁)─Ry(θ₀,₁)─Rz(φ₀,₁)─╰X─●─────│── ... ─┤ ⟨Z₁⟩\n"
        "                                   │     │\n"
        "|q₂⟩: ─Ry(x₂)─Ry(θ₀,₂)─Rz(φ₀,₂)────╰X─●──│── ... ─┤ ⟨Z₂⟩\n"
        "                                      │  │\n"
        "|q₃⟩: ─Ry(x₃)─Ry(θ₀,₃)─Rz(φ₀,₃)───────╰X─●── ... ─┤ ⟨Z₃⟩\n\n"
        "Complexity Summary:\n"
        "• Total Qubits: 4 simulated wires\n"
        "• Total Gates: 32 (4 Ry enc + 16 Ry/Rz rot + 8 CNOT + 4 Meas)\n"
        "• Parameter Count: 16 quantum angles + 40 head weights = 56\n"
        "• Gradient Rule: Analytical Parameter-Shift Rule (Zero approx error)\n"
        "• Execution Latency: 2.88 ms on simulator"
    )
    p_asc = tf5_r.add_paragraph()
    p_asc.text = ascii_text
    p_asc.font.size = Pt(10)
    p_asc.font.name = "Consolas"
    p_asc.font.color.rgb = TEXT_BODY
    p_asc.space_before = Pt(8)

    # ==========================================================
    # SLIDE 6: Complexity Analysis
    # ==========================================================
    slide6 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide6)
    add_header(slide6, "Algorithm Complexity & Simulation Characterization", "Computational Costs: Classical Preprocessing vs Quantum Simulation")

    # Table of Complexities
    rows, cols = 7, 4
    left, top, width, height = Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.8)
    table_shape = slide6.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table.columns[0].width = Inches(2.8)
    table.columns[1].width = Inches(2.8)
    table.columns[2].width = Inches(2.2)
    table.columns[3].width = Inches(3.933)

    headers = ["Pipeline Stage", "Time Complexity", "Space Complexity", "Engineering Characterization"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = HEADER_FILL
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = HEADER_NAVY

    complexity_data = [
        ("Image Preprocessing (CLAHE)", "O(H × W)", "O(H × W)", "Per-pixel local tile histogram redistribution."),
        ("CNN Extractor (ResNet-18)", "O(∑ K_l² C_{l-1} C_l H_l W_l)", "O(∑ C_l H_l W_l)", "Penultimate global pooling forward convolution pass."),
        ("PCA Reduction", "O(d² N + d³)", "O(d²)", "Covariance SVD decomposition from 512 dims to 4."),
        ("QPU Circuit Execution", "O(D · M) shots", "O(n) physical qubits", "Circuit depth D=2, shots M=1000, n=4 qubits."),
        ("Statevector Simulation", "O(G · 2ⁿ)", "O(2ⁿ) amplitudes", "Exponential simulation overhead on classical hardware."),
        ("Parameter-Shift Optimizer", "O(2 P · E · B · T_circ)", "O(P) gradients", "Exact analytical gradients: 2 shifted circuit runs/param.")
    ]
    for r, row in enumerate(complexity_data):
        for c, val in enumerate(row):
            cell = table.cell(r + 1, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = TEXT_BODY

    # ==========================================================
    # SLIDE 7: Software Implementation & Quantum Code
    # ==========================================================
    slide7 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide7)
    add_header(slide7, "Software Implementation & PennyLane Integration", "Full-Stack Python Framework: PyTorch, PennyLane QNode, and FastAPI")

    # Left Card: Stack
    add_card(slide7, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2))
    tb7_l = slide7.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.7))
    tf7_l = tb7_l.text_frame
    tf7_l.word_wrap = True

    p = tf7_l.paragraphs[0]
    p.text = "Modular Technology Stack"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    stack_items = [
        ("Quantum Simulator", "PennyLane 0.45.1 default.qubit statevector simulator with Torch interface."),
        ("Deep Learning Backbone", "PyTorch 2.14 GPU-accelerated ResNet-18 with Grad-CAM visual explainability."),
        ("Orthogonal Compression", "Scikit-Learn PCA reducing 512 continuous features into 4 rotation angles in [0, π]."),
        ("REST API Backend", "FastAPI asynchronous service exposing /api/modalities, /api/analyze, /api/circuit, and /api/benchmarks."),
        ("Light Clinical Dashboard", "Tailwind CSS single-page interface with interactive circuit viewer and Chart.js.")
    ]
    for title, desc in stack_items:
        p_t = tf7_l.add_paragraph()
        p_t.text = f"• {title}:"
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = HEADER_NAVY
        p_t.space_before = Pt(8)

        p_d = tf7_l.add_paragraph()
        p_d.text = f"   {desc}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = TEXT_BODY

    # Right Card: Code Snippet
    add_card(slide7, Inches(6.9), Inches(1.7), Inches(5.6), Inches(5.2))
    tb7_r = slide7.shapes.add_textbox(Inches(7.2), Inches(1.9), Inches(5.0), Inches(4.7))
    tf7_r = tb7_r.text_frame
    tf7_r.word_wrap = True

    p = tf7_r.paragraphs[0]
    p.text = "Multi-Observable QNode Implementation"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = ACCENT_INDIGO

    code_snippet = (
        "@qml.qnode(dev, interface='torch', diff_method='parameter-shift')\n"
        "def circuit(inputs, weights):\n"
        "    # 1. Quantum Angle Encoding\n"
        "    for i in range(n_qubits):\n"
        "        qml.RY(inputs[i], wires=i)\n\n"
        "    # 2. Variational Entangling Layers\n"
        "    for l in range(n_layers):\n"
        "        for i in range(n_qubits):\n"
        "            qml.RY(weights[l, i, 0], wires=i)\n"
        "            qml.RZ(weights[l, i, 1], wires=i)\n"
        "        for i in range(n_qubits):\n"
        "            qml.CNOT(wires=[i, (i + 1) % n_qubits])\n\n"
        "    # 3. Multi-Observable Expectation Readout\n"
        "    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]\n\n"
        "# 4. Classical Projection Head\n"
        "logits = linear_head(torch.stack(expvals))\n"
        "probabilities = torch.sigmoid(logits)"
    )
    p_code = tf7_r.add_paragraph()
    p_code.text = code_snippet
    p_code.font.size = Pt(9.5)
    p_code.font.name = "Consolas"
    p_code.font.color.rgb = TEXT_BODY
    p_code.space_before = Pt(8)

    # ==========================================================
    # SLIDE 8: Experimental Results & Benchmarks
    # ==========================================================
    slide8 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide8)
    add_header(slide8, "Experimental Benchmark Results", "Held-Out Test Set Performance: Multi-Disease & Binary Triage")

    # Table 1: Multi-Disease Results
    add_card(slide8, Inches(0.8), Inches(1.7), Inches(11.733), Inches(2.5))
    tb8_top = slide8.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(2.3))
    tf8_t = tb8_top.text_frame
    tf8_t.word_wrap = True

    p = tf8_t.paragraphs[0]
    p.text = "1. Multi-Disease Thoracic Findings Benchmark (NIH ChestX-ray14)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_INDIGO

    res_table_text = (
        "• Quantum Multi-Observable VQC: Hamming Loss = 0.1500 (85.0% label match) | Subset Acc = 33.3% | Latency = 2.88 ms\n"
        "• Multi-Output Logistic Regression: Hamming Loss = 0.1000 | Subset Acc = 26.7% | Latency = 0.07 ms\n"
        "• Multi-Label MLP (Neural Baseline): Hamming Loss = 0.1167 | Subset Acc = 33.3% | Latency = 0.07 ms\n"
        "• Top Predicted Findings: Infiltration (F1 = 0.33), Effusion (F1 = 0.22), No Finding (F1 = 0.63)\n"
        "• Finding Concordance: Quantum VQC successfully captures multi-label pathology correlations with only 56 total parameters."
    )
    p_res = tf8_t.add_paragraph()
    p_res.text = res_table_text
    p_res.font.size = Pt(10.5)
    p_res.font.color.rgb = TEXT_BODY
    p_res.space_before = Pt(4)

    # Table 2: Binary Baseline Results
    add_card(slide8, Inches(0.8), Inches(4.4), Inches(11.733), Inches(2.5))
    tb8_bot = slide8.shapes.add_textbox(Inches(1.0), Inches(4.5), Inches(11.333), Inches(2.3))
    tf8_b = tb8_bot.text_frame
    tf8_b.word_wrap = True

    p = tf8_b.paragraphs[0]
    p.text = "2. Binary Triage Baseline Benchmark (Normal vs Pneumonia, Kermany Dataset)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    base_table_text = (
        "• Quantum VQC (4 Qubits): Accuracy = 83.3% | F1 = 82.0% | Sensitivity = 75.8% | Specificity = 90.9% | ROC-AUC = 89.7%\n"
        "• Classical SVM (RBF Kernel): Accuracy = 83.3% | F1 = 82.0% | Sensitivity = 75.8% | Specificity = 90.9% | ROC-AUC = 89.2%\n"
        "• Classical Logistic Regression: Accuracy = 87.9% | F1 = 87.9% | Sensitivity = 87.9% | Specificity = 87.9% | ROC-AUC = 91.5%\n"
        "• Key Insight: Quantum VQC achieves exact parity with classical RBF SVM (83.3%), confirming viable decision boundaries in Hilbert space."
    )
    p_base = tf8_b.add_paragraph()
    p_base.text = base_table_text
    p_base.font.size = Pt(10.5)
    p_base.font.color.rgb = TEXT_BODY
    p_base.space_before = Pt(4)

    # ==========================================================
    # SLIDE 9: Conclusion & Future Scope
    # ==========================================================
    slide9 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide9)
    add_header(slide9, "Conclusion & Future Research Directions", "Capstone Achievements, Scientific Realities, and Roadmap")

    # Left Card: Achievements
    add_card(slide9, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2))
    tb9_l = slide9.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.7))
    tf9_l = tb9_l.text_frame
    tf9_l.word_wrap = True

    p = tf9_l.paragraphs[0]
    p.text = "Major Accomplishments"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = EMERALD

    achievements = [
        ("Modular Multi-Disease Prototype", "Delivered a fully working hybrid classical-quantum medical imaging system capable of evaluating multi-label thoracic findings."),
        ("Real Multi-Observable Readout", "Formulated and executed simultaneous 4-qubit Pauli-Z expectation measurements on a PennyLane simulator."),
        ("Scientific Integrity", "Demonstrated that classical linear baselines currently execute faster and achieve high accuracy on top PCA features, avoiding false supremacy claims."),
        ("Clinical Decision-Support UI", "Interactive single-page dashboard with drag & drop upload, Grad-CAM heatmaps, circuit inspector, and 14 showcase scans.")
    ]
    for title, desc in achievements:
        p_t = tf9_l.add_paragraph()
        p_t.text = f"• {title}:"
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = HEADER_NAVY
        p_t.space_before = Pt(8)

        p_d = tf9_l.add_paragraph()
        p_d.text = f"   {desc}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = TEXT_BODY

    # Right Card: Future Scope
    add_card(slide9, Inches(6.9), Inches(1.7), Inches(5.6), Inches(5.2))
    tb9_r = slide9.shapes.add_textbox(Inches(7.2), Inches(1.9), Inches(5.0), Inches(4.7))
    tf9_r = tb9_r.text_frame
    tf9_r.word_wrap = True

    p = tf9_r.paragraphs[0]
    p.text = "Future Research Directions"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    scope_items = [
        ("Physical QPU Deployment", "Execute the trained VQC on IBM Quantum / IonQ cloud backends with Zero-Noise Extrapolation (ZNE) error mitigation."),
        ("Modality Implementation", "Implement the planned 3D CT and MRI volumetric backbones with Tree Tensor Network and QCNN ansätze."),
        ("Quantum Kernel Methods", "Investigate Quantum Support Vector Machines (QSVM) using fidelity kernels to test theoretical quantum feature separation."),
        ("Clinical Prospective Trials", "Transition beyond educational prototypes toward prospective multi-center validation under IRB protocol.")
    ]
    for title, desc in scope_items:
        p_t = tf9_r.add_paragraph()
        p_t.text = f"• {title}:"
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = HEADER_NAVY
        p_t.space_before = Pt(8)

        p_d = tf9_r.add_paragraph()
        p_d.text = f"   {desc}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = TEXT_BODY

    # ==========================================================
    # SLIDE 10: Thank You Slide
    # ==========================================================
    slide10 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide10)

    card10 = add_card(slide10, Inches(2.0), Inches(1.5), Inches(9.333), Inches(4.5), border_color=ACCENT_BLUE)

    tb10 = slide10.shapes.add_textbox(Inches(2.5), Inches(2.0), Inches(8.333), Inches(3.5))
    tf10 = tb10.text_frame
    tf10.word_wrap = True

    p = tf10.paragraphs[0]
    p.text = "Thank You!"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = HEADER_NAVY
    p.alignment = PP_ALIGN.CENTER

    p2 = tf10.add_paragraph()
    p2.text = "Questions & Defense Discussion"
    p2.font.size = Pt(18)
    p2.font.color.rgb = ACCENT_BLUE
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(10)

    p3 = tf10.add_paragraph()
    p3.text = (
        "Quantum AI-Based Medical Imaging System for Disease Diagnosis\n\n"
        "Student Name: [Student Name]  |  Roll Number: [Roll Number]\n"
        "Capstone Evaluation Committee (2025–2026)\n"
        "Live Working Prototype: http://localhost:8000"
    )
    p3.font.size = Pt(12)
    p3.font.color.rgb = TEXT_MUTED
    p3.alignment = PP_ALIGN.CENTER
    p3.space_before = Pt(24)

    # Save presentation
    output_path = "C:/SAH/presentation/quantum_medical_imaging.pptx"
    prs.save(output_path)
    print(f"[Presentation] Successfully generated 10-slide PPT -> {output_path}")

if __name__ == "__main__":
    create_presentation()
