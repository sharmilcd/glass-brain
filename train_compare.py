import sys
import os
# FIX PATHS
sys.path.insert(0, os.path.abspath("bdh"))
sys.path.insert(0, os.path.abspath("nanoGPT"))

import time
import requests
import numpy as np
import torch
import bdh
from model import GPT, GPTConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- CONFIGURATION ---
BLOCK_SIZE = 128
BATCH_SIZE = 32
MAX_ITERS = 500
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 100

# --- DATA PREP (With Vocab Calculation) ---
input_file_path = "input.txt"
if not os.path.exists(input_file_path):
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(input_file_path, "w") as f:
        f.write(requests.get(data_url).text)

# Calculate real vocab size
with open(input_file_path, 'r') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Detected Vocab Size: {vocab_size} (This fixes the loss mismatch!)")

# Simple tokenizer
stoi = { ch:i for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
data_tensor = torch.tensor(encode(text), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data_tensor) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_tensor[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_tensor[i+1:i+1+BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)

# --- INITIALIZE MODELS (Matched Vocab) ---
print("Initializing models...")

# BDH Setup
bdh_conf = bdh.BDHConfig(
    n_layer=4, n_head=4, n_embd=128,
    mlp_internal_dim_multiplier=4,
    vocab_size=vocab_size  # <--- CRITICAL FIX
)
model_bdh = bdh.BDH(bdh_conf).to(device)
opt_bdh = torch.optim.AdamW(model_bdh.parameters(), lr=LEARNING_RATE)

# GPT Setup
gpt_conf = GPTConfig(
    n_layer=4, n_head=4, n_embd=128,
    block_size=BLOCK_SIZE,
    vocab_size=vocab_size
)
model_gpt = GPT(gpt_conf).to(device)
opt_gpt = torch.optim.AdamW(model_gpt.parameters(), lr=LEARNING_RATE)

# --- TRAINING LOOP ---
print(f"Starting training for {MAX_ITERS} steps...")
start_time = time.time()

for step in range(MAX_ITERS):
    xb, yb = get_batch()

    # Train BDH
    logits_b, loss_b, _ = model_bdh(xb, yb)
    opt_bdh.zero_grad(set_to_none=True)
    loss_b.backward()
    opt_bdh.step()

    # Train GPT
    logits_g, loss_g, _ = model_gpt(xb, yb)
    opt_gpt.zero_grad(set_to_none=True)
    loss_g.backward()
    opt_gpt.step()

    if step % EVAL_INTERVAL == 0:
        # Losses should now start much closer (approx 4.17 for vocab 65)
        print(f"Step {step}: BDH Loss {loss_b.item():.4f} | GPT Loss {loss_g.item():.4f}")

print(f"Training finished in {time.time()-start_time:.2f} seconds.")

torch.save(model_bdh.state_dict(), "bdh_trained.pth")
torch.save(model_gpt.state_dict(), "gpt_trained.pth")
print("Models saved successfully!")
