import sys
import os
import torch
import matplotlib.pyplot as plt
import time
import math
from torch.nn import functional as F

# --- 1. SETUP ---
sys.path.insert(0, os.path.abspath("bdh"))
sys.path.insert(0, os.path.abspath("nanoGPT"))

try:
    import bdh
    from model import GPT, GPTConfig
    print("Imports successful! ✅")
except ImportError:
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. RECURRENT STEP FUNCTION (From our previous success) ---
def bdh_recurrent_step(model, x_token, states, pos):
    """ Runs one single step of BDH inference using fixed state """
    C = model.config
    B, T = x_token.size()
    nh = C.n_head
    D = C.n_embd
    N = C.mlp_internal_dim_multiplier * D // nh

    # Embedding
    x = model.embed(x_token).unsqueeze(1)
    x = model.ln(x)

    for i in range(C.n_layer):
        # Sparse Proj
        x_sparse = F.relu(x @ model.encoder)

        # Hebbian Update (Key * Value)
        q = x_sparse
        k = x_sparse
        v = x.expand(-1, nh, -1, -1) # Broadcast V

        # Simple Linear Attention Update (ignoring RoPE for raw benchmark speed)
        # S_new = S_old + K.T @ V
        update = torch.matmul(k.transpose(-2, -1), v)
        states[i] = states[i] + update

        # Retrieval
        y = model.ln(torch.matmul(q, states[i]))
        y_sparse = F.relu(y @ model.encoder_v)

        # Output
        yMLP = (model.drop(x_sparse * y_sparse).transpose(1, 2).reshape(B, 1, 1, N*nh) @ model.decoder)
        x = model.ln(x + model.ln(yMLP))

    return x, states

# --- 3. INITIALIZE MODELS ---
dim = 256
layers = 4
heads = 4

print(f"Initializing models (Dim={dim})...")

# BDH (Recurrent Config)
bdh_conf = bdh.BDHConfig(n_layer=layers, n_head=heads, n_embd=dim, mlp_internal_dim_multiplier=2, vocab_size=65)
model_bdh = bdh.BDH(bdh_conf).to(device)
model_bdh.eval()

# GPT (Standard Config)
gpt_conf = GPTConfig(n_layer=layers, n_head=heads, n_embd=dim, block_size=8192, vocab_size=65)
model_gpt = GPT(gpt_conf).to(device)
model_gpt.eval()

# --- 4. RUN COMPARATIVE BENCHMARK ---
print("\n--- Running Infinite Context Benchmark ---")
# We test memory usage at specific context depths
checkpoints = [128, 512, 1024, 2048, 3072, 4096]

bdh_mem_log = []
gpt_mem_log = []

# --- A. TEST TRANSFORMER (Standard Forward) ---
print("Testing Transformer (Growing Memory)...")
for L in checkpoints:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, 65, (1, L)).to(device)
    try:
        with torch.no_grad():
            model_gpt(x)
        mem = torch.cuda.max_memory_allocated() / 1024**2
        gpt_mem_log.append(mem)
        print(f"   GPT @ {L}: {mem:.2f} MB")
    except RuntimeError:
        gpt_mem_log.append(None)
        print(f"   GPT @ {L}: OOM")

# --- B. TEST BDH (Recurrent Stream) ---
print("Testing BDH (Recurrent Mode)...")
# Initialize States ONCE
nh = bdh_conf.n_head
D = bdh_conf.n_embd
N = bdh_conf.mlp_internal_dim_multiplier * D // nh
states = [torch.zeros(1, nh, N, D).to(device) for _ in range(layers)]
current_token = torch.tensor([[0]]).to(device)

current_step = 0
bdh_results = {}

torch.cuda.reset_peak_memory_stats() # Reset before starting stream

# We stream continuously up to the max checkpoint
max_step = max(checkpoints)
for t in range(1, max_step + 1):
    with torch.no_grad():
        _, states = bdh_recurrent_step(model_bdh, current_token, states, t)

    # If we hit a checkpoint, record the CURRENT memory usage
    if t in checkpoints:
        mem = torch.cuda.max_memory_allocated() / 1024**2
        bdh_mem_log.append(mem)
        print(f"   BDH @ {t}: {mem:.2f} MB")

# --- 5. PLOT & SAVE ---
plt.figure(figsize=(10, 6), dpi=150)
plt.plot(checkpoints, gpt_mem_log, 'x-', color='red', linewidth=2.5, markersize=8, label='Transformer (O(T))')
plt.plot(checkpoints, bdh_mem_log, 'o-', color='blue', linewidth=2.5, markersize=8, label='BDH Recurrent (O(1))')

plt.title("The Memory Wall: Transformer vs Dragon Hatchling", fontsize=14, fontweight='bold')
plt.xlabel("Context Length (Tokens)", fontsize=12)
plt.ylabel("VRAM Usage (MB)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("benchmark_final_v2.png")
print("\n✅ Success! Saved 'benchmark_final_v2.png'")
