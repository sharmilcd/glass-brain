import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import sys
import os

# --- 1. SETUP ---
sys.path.insert(0, os.path.abspath("nanoGPT"))
try:
    from model import GPT, GPTConfig
    print("✅ Transformer imported successfully.")
except ImportError:
    print("❌ Critical Error: Could not import 'nanoGPT'.")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_transformer_crash_test():
    print("\n💥 INITIALIZING TRANSFORMER CRASH TEST...")

    # 1. Load Standard Transformer (Same size as your BDH)
    # Scaled up to n_embd=1024 to force the crash faster
    config = GPTConfig(n_layer=4, n_head=8, n_embd=1024, block_size=100000, vocab_size=65)
    model = GPT(config).to(device)
    model.eval()

    print("🤖 Standard Transformer Initialized.")
    print("   Memory usage will grow QUADRATICALLY until crash.")
    print("-------------------------------------------------------")
    print(f"{'Token Step':<15} | {'VRAM Usage (MB)':<20} | {'Status':<10}")
    print("-------------------------------------------------------")

    # 2. The Growing Context
    x = torch.tensor([[0]], dtype=torch.long).to(device)

    try:
        for t in range(1, 50001):

            with torch.no_grad():
                # FIX: Handle the 3rd return value (activations) by adding another underscore
                logits, _, _ = model(x)

                # Pick next token (greedy)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)

                # APPEND TO HISTORY (The Memory Killer)
                x = torch.cat((x, next_token), dim=1)

            # REPORTING
            if t % 500 == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**2
                print(f"{t:<15} | {mem:<20.2f} | ⚠️ Growing...")

    except RuntimeError as e:
        print(f"\n❌ CRASHED at step {t}!")
        print(f"   Error: {e}")
        print("   (This proves the Memory Wall)")

if __name__ == "__main__":
    run_transformer_crash_test()
