import os
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class QuantumVQC(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # PennyLane quantum device (statevector simulator)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Define QNode
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            # 1. Quantum Feature Map (Angle Encoding via Ry rotations)
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            # 2. Parameterized Variational Ansatz
            for layer in range(self.n_layers):
                # Single-qubit rotations (Ry and Rz)
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entanglement: Circular CNOT ladder
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            # 3. Readout Measurement (Pauli-Z expectation on qubit 0)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        # Trainable circuit weights: shape (n_layers, n_qubits, 2)
        # Initialized with small random angles
        self.q_weights = nn.Parameter(
            0.05 * torch.randn(n_layers, n_qubits, 2, dtype=torch.float32)
        )

        # Classical post-processing calibration parameters
        self.scale = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, n_qubits) or (n_qubits,)
        Returns:
            prob: Tensor of shape (batch_size,) with probability of class 1 (Pneumonia)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        expvals = []
        for i in range(batch_size):
            ev = self.circuit(x[i], self.q_weights)
            expvals.append(ev)

        expvals = torch.stack(expvals).float()
        # Calibrate expectation value [-1, 1] to logit -> sigmoid
        logits = self.scale * expvals + self.bias
        probs = torch.sigmoid(logits)
        return probs

    def get_circuit_details(self, sample_input=None):
        """
        Generates circuit ASCII diagram and structured gate specifications for the frontend.
        """
        if sample_input is None:
            sample_input = torch.full((self.n_qubits,), float(np.pi / 4))
        elif isinstance(sample_input, np.ndarray):
            sample_input = torch.tensor(sample_input, dtype=torch.float32)

        # Draw ASCII circuit
        ascii_draw = qml.draw(self.circuit)(sample_input, self.q_weights)

        # Structured gate sequence for visual interactive diagram
        gates = []
        
        # 1. Encoding gates
        for i in range(self.n_qubits):
            gates.append({
                "type": "RY",
                "label": f"Ry({sample_input[i].item():.2f})",
                "wire": i,
                "stage": "encoding",
                "angle": float(sample_input[i].item()),
                "target": i
            })

        # 2. Variational layers
        w = self.q_weights.detach().cpu().numpy()
        for layer in range(self.n_layers):
            # Rotations
            for i in range(self.n_qubits):
                gates.append({
                    "type": "RY",
                    "label": f"Ry(θ_{layer},{i}={w[layer, i, 0]:.2f})",
                    "wire": i,
                    "stage": f"layer_{layer+1}",
                    "angle": float(w[layer, i, 0]),
                    "target": i
                })
                gates.append({
                    "type": "RZ",
                    "label": f"Rz(φ_{layer},{i}={w[layer, i, 1]:.2f})",
                    "wire": i,
                    "stage": f"layer_{layer+1}",
                    "angle": float(w[layer, i, 1]),
                    "target": i
                })
            # Entanglement
            for i in range(self.n_qubits):
                ctrl = i
                tgt = (i + 1) % self.n_qubits
                gates.append({
                    "type": "CNOT",
                    "label": "CNOT",
                    "control": ctrl,
                    "target": tgt,
                    "wire": tgt,
                    "stage": f"layer_{layer+1}_entangle"
                })

        # Readout
        gates.append({
            "type": "MEASURE",
            "label": "⟨Z₀⟩",
            "wire": 0,
            "stage": "readout",
            "target": 0
        })

        # Run circuit to get expectation value
        ev = float(self.circuit(sample_input, self.q_weights).item())
        prob = float(torch.sigmoid(self.scale * torch.tensor(ev) + self.bias).item())

        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "total_gates": len(gates),
            "cnot_count": self.n_layers * self.n_qubits,
            "param_count": int(self.q_weights.numel() + 2),
            "ascii_diagram": str(ascii_draw),
            "gates": gates,
            "sample_expval": ev,
            "sample_prob": prob,
            "weights": w.tolist()
        }

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'state_dict': self.state_dict()
        }, filepath)

    @classmethod
    def load(cls, filepath):
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
        model = cls(n_qubits=checkpoint['n_qubits'], n_layers=checkpoint['n_layers'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model


DEFAULT_FINDINGS = [
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Pneumothorax",
    "Pneumonia",
    "Cardiomegaly",
    "Consolidation",
    "No Finding"
]

class MultiDiseaseQuantumVQC(nn.Module):
    """
    Multi-observable Variational Quantum Classifier for multi-disease thoracic pathology finding detection.
    Reads out Pauli-Z expectations on all N qubits simultaneously and projects to multi-label logits.
    """
    def __init__(self, n_qubits=4, n_layers=2, n_findings=8, finding_names=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_findings = n_findings
        self.finding_names = finding_names or list(DEFAULT_FINDINGS)

        # PennyLane quantum simulator device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Multi-observable QNode
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            # 1. Feature Encoding (Angle Embedding via Ry rotations)
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            # 2. Parameterized Variational Ansatz
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            # 3. Readout: Multi-qubit Pauli-Z expectations
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

        # Trainable variational angles
        self.q_weights = nn.Parameter(
            0.05 * torch.randn(n_layers, n_qubits, 2, dtype=torch.float32)
        )

        # Trainable linear projection head from 4 observables to 8 findings
        self.head = nn.Linear(n_qubits, n_findings)

    def forward_logits(self, x):
        is_single = (x.ndim == 1)
        if is_single:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        expvals = []
        for i in range(batch_size):
            ev = self.circuit(x[i], self.q_weights)
            ev_tensor = torch.stack(ev).float() if isinstance(ev, (list, tuple)) else ev.float()
            expvals.append(ev_tensor)

        expvals = torch.stack(expvals).float()
        logits = self.head(expvals)

        if is_single:
            return logits[0], expvals[0]
        return logits, expvals

    def forward(self, x):
        logits, _ = self.forward_logits(x)
        probs = torch.sigmoid(logits)
        return probs

    def predict_findings(self, x, threshold=0.50, possible_threshold=0.25):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            logits, expvals = self.forward_logits(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            expvals_np = expvals.cpu().numpy()

        is_batch = (probs.ndim == 2)
        if not is_batch:
            probs = probs.reshape(1, -1)
            expvals_np = expvals_np.reshape(1, -1)

        batch_results = []
        for b in range(probs.shape[0]):
            sample_probs = probs[b]
            findings_list = []
            for idx, name in enumerate(self.finding_names):
                p = float(sample_probs[idx])
                if p >= threshold:
                    status = "Detected"
                elif p >= possible_threshold:
                    status = "Possible"
                else:
                    status = "Low"

                findings_list.append({
                    "finding": name,
                    "score": round(p * 100, 1),
                    "probability": round(p, 4),
                    "status": status,
                    "threshold": threshold
                })

            batch_results.append({
                "findings": findings_list,
                "probabilities": {name: round(float(sample_probs[i]), 4) for i, name in enumerate(self.finding_names)},
                "observable_expectations": [round(float(v), 4) for v in expvals_np[b]]
            })

        return batch_results[0] if not is_batch else batch_results

    def get_circuit_details(self, sample_input=None):
        if sample_input is None:
            sample_input = torch.full((self.n_qubits,), float(np.pi / 4))
        elif isinstance(sample_input, np.ndarray):
            sample_input = torch.tensor(sample_input, dtype=torch.float32)

        ascii_draw = qml.draw(self.circuit)(sample_input, self.q_weights)

        gates = []
        for i in range(self.n_qubits):
            gates.append({
                "type": "RY",
                "label": f"Ry({sample_input[i].item():.2f})",
                "wire": i,
                "stage": "encoding",
                "angle": float(sample_input[i].item()),
                "target": i
            })

        w = self.q_weights.detach().cpu().numpy()
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                gates.append({
                    "type": "RY",
                    "label": f"Ry(θ_{layer},{i}={w[layer, i, 0]:.2f})",
                    "wire": i,
                    "stage": f"layer_{layer+1}",
                    "angle": float(w[layer, i, 0]),
                    "target": i
                })
                gates.append({
                    "type": "RZ",
                    "label": f"Rz(φ_{layer},{i}={w[layer, i, 1]:.2f})",
                    "wire": i,
                    "stage": f"layer_{layer+1}",
                    "angle": float(w[layer, i, 1]),
                    "target": i
                })
            for i in range(self.n_qubits):
                ctrl = i
                tgt = (i + 1) % self.n_qubits
                gates.append({
                    "type": "CNOT",
                    "label": "CNOT",
                    "control": ctrl,
                    "target": tgt,
                    "wire": tgt,
                    "stage": f"layer_{layer+1}_entangle"
                })

        for i in range(self.n_qubits):
            gates.append({
                "type": "MEASURE",
                "label": f"⟨Z_{i}⟩",
                "wire": i,
                "stage": "readout",
                "target": i
            })

        ev = self.circuit(sample_input, self.q_weights)
        ev_list = [float(v.item()) for v in ev]

        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_findings": self.n_findings,
            "finding_names": self.finding_names,
            "total_gates": len(gates),
            "cnot_count": self.n_layers * self.n_qubits,
            "param_count": int(self.q_weights.numel() + self.head.weight.numel() + self.head.bias.numel()),
            "ascii_diagram": str(ascii_draw),
            "gates": gates,
            "sample_expvals": ev_list,
            "weights": w.tolist()
        }

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'n_findings': self.n_findings,
            'finding_names': self.finding_names,
            'state_dict': self.state_dict()
        }, filepath)

    @classmethod
    def load(cls, filepath):
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
        model = cls(
            n_qubits=checkpoint['n_qubits'],
            n_layers=checkpoint['n_layers'],
            n_findings=checkpoint['n_findings'],
            finding_names=checkpoint.get('finding_names')
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

