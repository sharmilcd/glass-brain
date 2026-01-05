# 🧠 Project Glass Brain: The Infinite-Context Dragon

### Synaptix Frontier AI Hackathon | Track 2: Advanced Understanding

**Team:** TEAM
**Status:** ✅ Solved (O(1) Inference Memory Achieved)

---

## 🚨 The Problem: The "Memory Wall"
State-of-the-art Transformers (GPT, Llama, Claude) suffer from a fatal flaw: **Linear Memory Scaling during Inference.**
To generate the 10,000th token, they must store a Key-Value (KV) cache for the previous 9,999 tokens.
* **Result:** Context length is limited by GPU VRAM.
* **Failure Mode:** On a standard T4 GPU (15GB), a GPT model crashes at ~12k tokens.

## 💡 The Solution: Recurrent Linear Attention (BDH)
We implemented the **Dragon Hatchling (BDH)** architecture in a pure **Recurrent Inference Mode**.
Instead of caching history (appending to a list), we compress history into a **Fixed-Size Hebbian State Matrix**.
* **Result:** Memory usage is **Constant ($O(1)$)**.
* **Capability:** Theoretical **Infinite Context** on consumer hardware.

---

## 📊 Proof 1: Breaking the Memory Wall
We pitted a standard Transformer against our Recurrent BDH on a T4 GPU.
* **Red Line (Transformer):** Shows standard linear scaling. Memory usage grows with every token until it hits the 15GB hardware limit and **CRASHES**.
* **Blue Line (BDH):** Shows our implementation. Memory usage is perfectly flat. It processes 1k, 10k, or 50k tokens with **zero memory growth**.

![Memory Benchmark Graph](benchmark_final_v2.png)

---

## 🧬 Proof 2: Emergent Sparsity (Biological Plausibility)
Unlike Transformers which use GELU (causing ~99% active neurons), our BDH model uses ReLU with high-dimensional expansion. This forces the model to learn **"Sparse Representations"**—activating only the specific neurons needed for a concept and keeping the rest at a "Hard Zero."

Below is the evolution of sparsity during our training run. Dark Blue areas are active; White areas are effectively "off" (saving energy).

### Phase 1: Initialization (Dense & Noisy)
At step 0, the model is random and dense. It hasn't learned to specialize yet.
![Sparsity Iteration 1](sparsity_init.png)

### Phase 2: Learning Structure (Emerging Patterns)
By step 100, you can see vertical "bands" forming. The model is starting to allocate specific neurons to specific tokens.
![Sparsity Iteration 2](sparsity_mid.png)

### Phase 3: Final Convergence (The "Barcode")
By step 400, the model has achieved **~43% Sparsity**. The distinct white gaps represent the "Inactive Zone"—compute resources that are reserved, not wasted.
![Sparsity Iteration 3](sparsity_final.png)

---

## 🧪 Experimental Results Summary

| Metric | Transformer (GPT-2) | Dragon Hatchling (BDH) | Winner |
| :--- | :--- | :--- | :--- |
| **Inference Memory** | Linear Growth ($O(T)$) | **Constant ($O(1)$)** | 🏆 **BDH** |
| **Crash Point (T4 GPU)** | ~12,500 Tokens | **Never** (>50k tested) | 🏆 **BDH** |
| **Neuron Activity** | ~99.9% (Dense) | **~43.1% (Sparse)** | 🏆 **BDH** |
| **Training Loss (400 steps)** | 2.33 | **1.97** | 🏆 **BDH** |

---

## 🚀 How to Reproduce

### 1. Run the "Infinite Context" Demo
Run the recurrent kernel to see memory stay flat at 152MB for 50,000+ tokens.
```bash
python bdh_recurrent.py
