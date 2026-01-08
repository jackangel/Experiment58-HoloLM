# HoloGPT: Holographic Memory Architecture

This repository implements **HoloGPT**, a language model that replaces standard Self-Attention with a **Holographic Associative Memory** mechanism. Unlike Transformers, which have $O(N^2)$ complexity relative to sequence length, HoloGPT utilizes a linear-recurrent "scan" with a fixed-size memory state.

## Core Architectural Concepts

### 1. The Holographic Memory State
Instead of storing all previous keys and values in a cache (KV Cache), HoloGPT maintains a global memory state represented as a square matrix $M \in \mathbb{R}^{d_{mat} \times d_{mat}}$. 

Information is stored via **outer products** (associative binding) and retrieved via **matrix-vector multiplication**.

### 2. The `holo_scan` Mechanism
The core of the model is a JIT-compiled recurrence relation that processes sequences in a single pass:

*   **Read Operation:** The model extracts information using a normalized query vector $q_t$.
    $$\text{readout}_t = M_{t-1} \cdot q_t$$
*   **Write Operation:** The model creates a new association between a value $v_t$ and a key $k_t$.
    $$\text{association}_t = v_t \otimes k_t^\top$$
*   **Gated Update:** The memory is updated using learned "forget" ($\gamma_f$) and "write" ($\gamma_w$) gates.
    $$M_t = (\gamma_{f,t} \odot M_{t-1}) + (\gamma_{w,t} \odot \text{association}_t)$$

### 3. FastHoloBlock Structure
Each layer in the network follows a modern "Pre-LN" residual architecture:

1.  **Input Normalization:** `LayerNorm`
2.  **Holographic Mixing:** The `holo_scan` kernel performs the associative memory retrieval and update.
3.  **Projection:** A linear layer maps the $d_{mat}$ readout back to the $d_{embed}$ space.
4.  **Residual Add:** $x = x + \text{Mixer}(x)$
5.  **Feed-Forward Network (FFN):** A standard Gated Expansion layer ($d_{embed} \to 4d_{embed} \to d_{embed}$) using GELU activation.
6.  **Residual Add:** $x = x + \text{FFN}(x)$

## Technical Specifications

| Component | Implementation Detail |
| :--- | :--- |
| **Memory Geometry** | Matrix-based associative storage ($64 \times 64$ by default) |
| **Recurrence** | Linear time complexity $O(N)$ |
| **Kernels** | TorchScript (JIT) optimized loops for the mathematical core |
| **Tokenization** | Byte-Pair Encoding (BPE) with a configurable vocabulary (default 4096) |
| **Normalization** | Dual-layer LayerNorm per block |
| **Gating** | Sigmoid-based learnable Forget/Write gates |

## Why Holographic Memory?
Traditional Attention focuses on specific past tokens. **Holographic Memory** compresses the entire sequence history into a static-sized matrix. This allows for:
1.  **Constant Inference Memory:** The memory state does not grow as you generate more tokens.
2.  **Associative Retrieval:** The model learns to "bind" concepts together (Key-Value pairs) directly within the weights of the hidden state matrix.
3.  **Hardware Efficiency:** By using matrix-vector operations instead of large attention maps, it reduces the VRAM footprint significantly.
