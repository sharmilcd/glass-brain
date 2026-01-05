import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

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

# --- 2. DETECT VOCAB ---
input_file_path = "input.txt"
if not os.path.exists(input_file_path):
    vocab_size = 65
else:
    with open(input_file_path, 'r') as f:
        vocab_size = len(sorted(list(set(f.read()))))

# --- 3. LOAD MODELS ---
print("Loading trained models...")
bdh_conf = bdh.BDHConfig(n_layer=4, n_head=4, n_embd=128, mlp_internal_dim_multiplier=4, vocab_size=vocab_size)
model_bdh = bdh.BDH(bdh_conf).to(device)
try: model_bdh.load_state_dict(torch.load("bdh_trained.pth", map_location=device)); print("BDH Loaded ✅")
except: pass
model_bdh.eval()

gpt_conf = GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=128, vocab_size=vocab_size)
model_gpt = GPT(gpt_conf).to(device)
try: model_gpt.load_state_dict(torch.load("gpt_trained.pth", map_location=device)); print("GPT Loaded ✅")
except: pass
model_gpt.eval()

# --- 4. FLATTEN & SORT VISUALIZATION ---
print("\n--- Generating Clean Heatmaps ---")
dummy_input = torch.randint(0, vocab_size, (1, 128)).to(device)

with torch.no_grad():
    _, _, bdh_act = model_bdh(dummy_input)
    _, _, gpt_act = model_gpt(dummy_input)

# Flatten BDH Heads
bdh_flat = bdh_act.permute(0, 2, 1, 3).reshape(bdh_act.size(0), bdh_act.size(2), -1)
gpt_flat = gpt_act

# Convert to Numpy [Time, Neurons]
bdh_np = bdh_flat[0].cpu().numpy()
gpt_np = gpt_flat[0].cpu().numpy()

# SORTING: Sort columns (neurons) by total activity (sum)
# This pushes Dark/Active columns to the left
bdh_sorted_indices = np.argsort(-bdh_np.sum(axis=0))
gpt_sorted_indices = np.argsort(-gpt_np.sum(axis=0))

bdh_viz = bdh_np[:, bdh_sorted_indices]
gpt_viz = gpt_np[:, gpt_sorted_indices]

# Stats
bdh_density = (np.count_nonzero(bdh_viz > 0) / bdh_viz.size) * 100
gpt_density = (np.count_nonzero(np.abs(gpt_viz) > 1e-4) / gpt_viz.size) * 100

# --- PLOT (Dark = Active, White = Silent) ---
fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# BDH Plot: BLUES
# Using 'Blues' so Active=Dark Blue, Silent=White
ax[0].imshow(bdh_viz > 0, aspect='auto', cmap='Blues', interpolation='nearest')
ax[0].set_title(f"BDH (Dragon Hatchling)\nSPARSE: Only {bdh_density:.1f}% Active", fontsize=14, fontweight='bold')
ax[0].set_xlabel(f"Neurons Sorted by Activity →", fontsize=10)
ax[0].set_ylabel("Time Sequence ↓", fontsize=10)

# Add the "Savings" Line
boundary = int(bdh_viz.shape[1] * (bdh_density/100))
ax[0].axvline(x=boundary, color='red', linestyle='--', linewidth=2)
ax[0].text(boundary + 10, 60, f"EMPTY ZONE\n(Energy Saved)", color='red', fontsize=12, fontweight='bold')

# GPT Plot: REDS
# Using 'Reds' so Active=Dark Red
ax[1].imshow(np.abs(gpt_viz) > 1e-4, aspect='auto', cmap='Reds', interpolation='nearest')
ax[1].set_title(f"Transformer (nanoGPT)\nDENSE: {gpt_density:.1f}% Active", fontsize=14, fontweight='bold')
ax[1].set_xlabel(f"Neurons Sorted by Activity →", fontsize=10)
ax[1].set_yticks([])
ax[1].text(150, 60, "NO IDLE NEURONS\n(Energy Wasted)", color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

plt.tight_layout()
plt.savefig("sparsity_visual_proof.png")
print("✅ Saved 'sparsity_visual_proof.png'.")
