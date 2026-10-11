# Capstone Viva Voce Defense Guide: Quantum AI Medical Imaging System

*Prepare for examination questions with technically precise, academically grounded answers.*

---

### Q1: Why didn't you feed the chest X-ray image directly into the quantum circuit?
**Answer:**  
Feeding an uncompressed medical image directly into a quantum circuit is physically and computationally prohibitive under current Noisy Intermediate-Scale Quantum (NISQ) constraints:
1. **Qubit scaling:** An image of $224 \times 224$ pixels contains $50,176$ continuous pixel intensity values. In Angle Embedding, encoding this directly requires $50,176$ physical qubits. The largest physical QPUs today possess only a few hundred to a thousand noisy qubits.
2. **Circuit Depth in Amplitude Embedding:** While Amplitude Embedding can theoretically store $N$ features in $\lceil \log_2(N) \rceil = 16$ qubits ($2^{16} = 65,536$), state preparation for arbitrary classical data requires an exponential number of multi-controlled rotation and CNOT gates ($\mathcal{O}(2^n)$). On physical NISQ devices, such deep circuits exceed qubit coherence times ($T_1, T_2$), degenerating into pure decoherence noise before execution completes.
3. **Hybrid Solution:** Hence, the hybrid classical-quantum paradigm is the recognized academic state-of-the-art: a classical CNN handles spatial inductive biases, compressing pixel tensors into latent vectors, which PCA then reduces to a compact qubit budget.

---

### Q2: Why did you use PCA instead of an Autoencoder or training an end-to-end QNN?
**Answer:**
1. **Orthogonality and Variance Maximization:** PCA mathematically guarantees that the extracted principal components are completely orthogonal and uncorrelated ($\text{Cov}(z_i, z_j) = 0$ for $i \neq j$), which prevents redundant quantum rotation angles across different qubit wires.
2. **Determinism & Interpretability:** Unlike non-linear autoencoders which can suffer from arbitrary latent rotations, PCA is mathematically deterministic via Singular Value Decomposition (SVD). We can measure the exact explained variance captured by each qubit (e.g., Qubit 0 captures 12.9% of variance, Qubit 1 captures 8.86%).
3. **Avoids Double-Stochastic Optimization:** Training an autoencoder simultaneously with a variational quantum circuit introduces non-convex co-adaptation instabilities. PCA decouples the classical feature compression from the quantum Hilbert-space classification.

---

### Q3: Explain the difference between Angle Embedding and Amplitude Embedding.
**Answer:**
- **Angle Embedding:**
  - *Mechanism:* Each classical feature $x_i$ is mapped as the rotation angle of a single-qubit unitary gate: $| \psi \rangle = \bigotimes_{i=0}^{n-1} R_y(x_i) |0\rangle$.
  - *Qubit requirement:* Exactly $n$ qubits for $n$ features.
  - *Gate depth:* Constant depth $\mathcal{O}(1)$ parallel execution.
  - *Advantage:* Highly robust against noise, simple to implement on NISQ hardware and simulators, and has non-linear trigonometric feature transformation built into rotation angles ($\cos(x_i/2)|0\rangle + \sin(x_i/2)|1\rangle$).
- **Amplitude Embedding:**
  - *Mechanism:* Encodes normalized feature vector $\mathbf{x}$ directly into the probability amplitudes of an $n$-qubit state: $| \psi \rangle = \sum_{i=0}^{2^n - 1} x_i |i\rangle$.
  - *Qubit requirement:* Logarithmic $\lceil \log_2(N) \rceil$ qubits.
  - *Gate depth:* Exponential $\mathcal{O}(2^n)$ or $\mathcal{O}(N)$ CNOT entangling gates for state preparation.
  - *Disadvantage:* State preparation is severely impacted by quantum decoherence and gate infidelities.

---

### Q4: What is the Parameter-Shift Rule and why not use standard Finite Differences?
**Answer:**
- In classical deep learning, automatic differentiation uses backpropagation via the chain rule on intermediate activation graphs. In physical quantum circuits or black-box statevector execution, intermediate quantum statevectors cannot be non-destructively cloned or measured without collapsing the wavefunction (due to the **No-Cloning Theorem**).
- **Finite Differences** ($\frac{f(\theta + \epsilon) - f(\theta)}{\epsilon}$) approximates gradients numerically, but is susceptible to severe rounding errors and shot noise on quantum hardware.
- **The Parameter-Shift Rule** is an exact analytical formula derived from the generator $G = \frac{1}{2} Y$ of rotation gates with eigenvalues $\pm \frac{1}{2}$:
  $$\frac{\partial \langle \hat{O} \rangle}{\partial \theta_j} = \frac{1}{2} \left[ \langle \hat{O} \rangle_{\theta_j + \frac{\pi}{2}} - \langle \hat{O} \rangle_{\theta_j - \frac{\pi}{2}} \right]$$
- It measures the exact quantum gradient by executing the circuit twice with macroscopic shifts of $\pm \frac{\pi}{2}$, completely bypassing numerical approximation errors.

---

### Q5: What are Barren Plateaus and how did your project mitigate them?
**Answer:**
- **Definition:** Barren plateaus represent a phenomenon where the variance of the quantum cost function gradient decays exponentially with the number of qubits:
  $$\text{Var}_{\boldsymbol{\theta}} \left[ \frac{\partial \langle \hat{O} \rangle}{\partial \theta_j} \right] \in \mathcal{O}(2^{-n})$$
  When a barren plateau occurs, the gradient becomes indistinguishable from zero across almost the entire parameter landscape, causing classical gradient optimizers to stall.
- **Mitigation strategies in our architecture:**
  1. **Low Qubit Count ($n=4$):** Prevents exponential concentration of measure.
  2. **Shallow Variational Ansatz ($L=2$ layers):** Restricts the unitary group from forming a 2-design over the Haar measure.
  3. **Local Observables:** Measuring a local single-qubit operator $\langle Z_0 \rangle$ rather than a global observable $\langle Z_0 Z_1 Z_2 Z_3 \rangle$ guarantees polynomial gradient bounds (Cerezo et al., *Nature Communications 2021*).
  4. **Localized Parameter Initialization:** Circuit weights were initialized with small normal values ($\theta \sim \mathcal{N}(0, 0.05^2)$) near identity rather than uniformly random angles in $[0, 2\pi]$.

---

### Q6: Did this project achieve "Quantum Advantage"?
**Answer:**
- **No.** Claiming quantum advantage here would be scientifically dishonest.
- **Definition of Quantum Advantage:** Demonstrating that a quantum algorithm solves a practical problem provably faster (polynomial vs exponential speedup) or with superior accuracy than the best classical algorithms on identical or optimal representations.
- **Our findings:** The 4-qubit VQC achieved **83.3% test accuracy** and **89.7% ROC-AUC**, matching the classical non-linear RBF SVM (83.3%), while classical linear models achieved 87.9%.
- **Value of the project:** It proves feasibility and parameter efficiency—the VQC achieved comparable diagnostic accuracy with only **18 trainable parameters**, compared to 225 in the classical MLP and 158 support vectors in the SVM.

---

### Q7: Why did the classical Linear SVM and Logistic Regression score slightly higher (87.9%) than the VQC (83.3%)?
**Answer:**
- Because the features were extracted using **ResNet-18**, a deep convolutional network pre-trained on ImageNet and fine-tuned for transfer learning.
- Deep neural networks perform non-linear hierarchical representations in their convolutional layers, transforming complex non-linear image manifolds into linearly separable representations at the penultimate pooling layer.
- Consequently, the top PCA eigenvectors already contain a strongly linearly separable decision hyperplane. In such spaces, simple linear hyperplanes (Linear SVM) are optimal and avoid any overfitting, whereas variational quantum circuits and non-linear RBF kernels introduce additional curvature that yields identical 83.3% accuracy.

---

### Q8: What is the role of Entanglement in your circuit?
**Answer:**
- Entanglement is generated via a circular CNOT ladder ($CNOT(q_i, q_{(i+1)\%4})$) after the parameterized single-qubit rotations.
- Single-qubit rotations create superpositions on individual qubits, but cannot represent non-separable joint probability distributions.
- CNOT gates create quantum correlations across the different principal components, enabling the quantum classifier to model multi-variable non-linear interactions across the latent features that cannot be expressed as independent 1D functions.

---

### Q9: How can this system transition to real physical quantum hardware?
**Answer:**
- Currently, the prototype uses the PennyLane `default.qubit` statevector simulator.
- To execute on a physical superconducting quantum processor (e.g., IBM Quantum Eagle or Heron via `qiskit-ibm-runtime`):
  1. Replace `qml.device("default.qubit", wires=4)` with `qml.device("qiskit.remote", backend=service.backend("ibm_brisbane"), shots=1024)`.
  2. Implement **Readout Error Mitigation (M3)** and **Zero-Noise Extrapolation (ZNE)** to mitigate physical decoherence and gate crosstalk.
  3. Transpile the circular CNOT ladder to the physical coupling map of the QPU.

---

### Q10: What is the clinical and ethical disclaimer of this system?
**Answer:**
- This software is strictly an **educational and research prototype** created for academic evaluation.
- It has not received regulatory clearance (such as FDA 510(k) or CE marking) and must **never** be used for clinical decision-making or patient diagnosis.
- Any automated medical imaging AI must function strictly as an adjunct decision-support tool under certified physician supervision.
