# HoloGPT - Holographic Memory (not really but it sounds fancy)

Welcome to **HoloGPT**, a project that uses a "Holographic" branding for what is actually a **Linear Recurrent Associative Memory** model. While it doesn't actually use lasers or interference patterns, it *does* use high-dimensional outer-product algebra to store and retrieve information, which is almost as cool.

## The Architecture (The "Fancy" Part)

Instead of the standard $O(N^2)$ Self-Attention mechanism found in Transformers, HoloGPT utilizes a fixed-size memory state $M \in \mathbb{R}^{d_{mat} \times d_{mat}}$. It operates as a sophisticated RNN where the hidden state is a matrix rather than a vector.

### 1. The "Holo" Scan (Recurrent Core)
At each timestep, the model updates its internal representation through a gated associative process:

*   **Binding (Write):** The model generates a Key ($k_t$) and a Value ($v_t$). It binds them using an outer product: $A_t = v_t \otimes k_t^\top$.
*   **Gated Update:** A learnable forget gate ($\gamma_f$) and write gate ($\gamma_w$) decide how much of the past to keep and how much of the new association to incorporate:
    $$M_t = (\gamma_{f,t} \cdot M_{t-1}) + (\gamma_{w,t} \cdot A_t)$$
*   **Retrieval (Read):** A Query ($q_t$) is used to probe the matrix: $y_t = M_t \cdot q_t$.

### 2. FastHoloBlock
Each layer wraps the memory scan in a modern residual framework:
- **Pre-LayerNorm:** Normalizes input before the mixer and the FFN.
- **FFN:** A standard Gated Expansion layer (GELU) for processing the retrieved information.
- **Residual Connections:** Added after both the mixer and the FFN to ensure gradient flow.

---

## Technical Reality Check (The "Not Really" Part)

While the architecture is efficient for generation, it comes with specific trade-offs that separate it from standard Transformers:

### Complexity Analysis
| Metric | Complexity | The Truth |
| :--- | :--- | :--- |
| **Inference Time** | $O(1)$ per token | Excellent for chat; generation speed doesn't slow down as context grows. |
| **Training Time** | $O(N \cdot d_{mat}^2)$ | **Bottleneck:** Because it is recurrent, it cannot be parallelized across the time dimension like a Transformer. |
| **Memory State** | $O(d_{mat}^2)$ | The "KV Cache" is a fixed-size matrix. It never grows, but it is a lossy compressor. |

### Limitations & Trade-offs
1.  **Lossy Compression:** Trying to fit an entire book into a $64 \times 64$ matrix is like trying to fit a symphony into a greeting card. High-level themes survive; exact word-for-word recall of distant text suffers.
2.  **Sequential Training:** Even with the JIT-compiled `holo_scan`, training is significantly slower than a Transformer on modern GPUs because we have to wait for step $t$ to finish before calculating $t+1$.
3.  **Recency Bias:** The forget gate naturally causes the model to "specialize" in more recent context, making long-range dependencies harder to maintain than in global attention.

## Heritage
This model sits in the family tree of:
*   **Fast Weight Programmers:** Directly inspired by the work of Jürgen Schmidhuber.
*   **Linear RNNs:** Similar in spirit to RWKV, Mamba, and S4.
*   **Linear Attention:** A variation of the idea that attention can be linearized into a hidden state.

## Summary
HoloGPT is an experiment in **associative memory scaling**. It trades the perfect recall and parallel training of Transformers for a fixed-memory footprint and linear time complexity during inference. It’s fancy, it’s experimental, and it definitely sounds better than "Outer-Product Recurrent Gated Matrix Model."
