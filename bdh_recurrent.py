import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import sys
import os

# --- 1. SETUP ---
sys.path.insert(0, os.path.abspath("bdh"))
try:
    import bdh
    print("✅ Model imported successfully.")
except ImportError:
    print("❌ Critical Error: Could not import 'bdh'.")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. THE RECURRENT KERNEL (Customized for BDH) ---
# This replicates the math of bdh.py but for a single token at a time
def bdh_recurrent_step(model, x_token, states, pos):
    """
    x_token: [Batch, 1] input token indices
    states: List of state matrices (one per layer depth)
    pos: Current integer position (for RoPE)
    """
    B, T = x_token.size() # Should be 1, 1
    C = model.config
    D = C.n_embd
    nh = C.n_head
    N = C.mlp_internal_dim_multiplier * D // nh

    # 1. Embedding & Pre-Norm
    x = model.embed(x_token).unsqueeze(1) # [B, 1, 1, D]
    x = model.ln(x)

    # 2. Iterate Depth (Layers)
    # BDH uses SHARED weights, so we loop over the same modules
    for i in range(C.n_layer):
        # --- A. Sparse Projection (The "Brain") ---
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent) # Q and K are both x_sparse

        # --- B. Linear Attention (Hebbian Update) ---
        # Get Q, K, V
        q_raw = x_sparse # [B, 1, 1, N]
        k_raw = x_sparse # [B, 1, 1, N]
        v_raw = x        # [B, 1, 1, D]

        # Apply RoPE (Rotary Positional Embedding) for current position
        # We need to grab the frequency for *just this position*
        # model.attn.freqs is [1, 1, 1, N]
        # We construct the phase manually for pos
        freqs = model.attn.freqs # [1, 1, 1, N]
        phase = pos * freqs

        # Apply Rotation
        # Note: model.attn.rope expects specific shapes, we simplify for inference
        q_rot = model.attn.rope(phase, q_raw)
        k_rot = q_rot # K is same as Q in this architecture

        # Reshape for Multi-Head Linear Attention
        # We need: [B, Heads, Head_Dim_N, Head_Dim_D]
        # N is huge, so we split N into heads
        head_dim_N = N // nh
        head_dim_D = D # V has dim D

        # Reshape Q, K, V for multi-head calc
        # Q: [B, nh, 1, head_dim_N]
        # V: [B, 1, 1, D] -> replicate across heads?
        # Actually BDH attn is specialized. Let's strictly follow the implementation:
        # scores = (QR @ KR.mT) -> This is [B, 1, 1, 1] in shape if T=1?
        # No, the provided BDH Attn is: Q=x_sparse, K=x_sparse, V=x.
        # x_sparse is [B, nh, T, N_head] (Wait, check forward dim)

        # RE-CHECKING DIMENSIONS FROM bdh.py:
        # x_sparse = F.relu(x_latent) -> [B, 1, T, N_total]
        # But attn expects [B, nh, T, N_head]??
        # The provided code says: x_sparse # B, nh, T, N  <-- Comment in code
        # So x_latent must have been reshaped?
        # x_latent = x @ encoder. encoder is [nh, D, N].
        # x is [B, 1, T, D].
        # x @ encoder -> [B, nh, T, N]. Yes.

        q = q_rot # [B, nh, 1, N]
        k = k_rot # [B, nh, 1, N]
        v = x     # [B, 1, 1, D] -> Needs broadcast to [B, nh, 1, D]?

        # STATE UPDATE: S = S + (K.T @ V)
        # K: [B, nh, 1, N] -> Transpose to [B, nh, N, 1]
        # V: [B, 1, 1, D] -> Broadcast to [B, nh, 1, D]
        v_expanded = v.expand(-1, nh, -1, -1)

        # Update State (In-Place Hebbian Learning)
        # S: [B, nh, N, D]
        update = torch.matmul(k.transpose(-2, -1), v_expanded)
        states[i] = states[i] + update

        # RETRIEVAL: O = Q @ S
        # Q: [B, nh, 1, N]
        # S: [B, nh, N, D]
        # Out: [B, nh, 1, D]
        yKV = torch.matmul(q, states[i])

        # --- C. Post-Attention Processing ---
        # Collapse heads?
        # In bdh.py: yKV = ln(yKV)
        # yKV is likely summed or reshaped?
        # bdh.py: y_latent = yKV @ encoder_v.
        # yKV needs to match encoder_v [nh, D, N].
        # So yKV is [B, nh, 1, D].

        yKV = model.ln(yKV) # Normalization on the attention output

        y_latent = yKV @ model.encoder_v # [B, nh, 1, N]
        y_sparse = F.relu(y_latent)

        # Gating
        xy_sparse = x_sparse * y_sparse

        # Decoder Projection
        # Reshape to combine heads
        # xy_sparse: [B, nh, 1, N] -> [B, 1, 1, N*nh]
        xy_flat = xy_sparse.transpose(1, 2).reshape(B, 1, 1, -1)

        yMLP = xy_flat @ model.decoder # [B, 1, 1, D]

        # Residuals
        y = model.ln(yMLP)
        x = model.ln(x + y)

    # 3. Final Prediction
    logits = x.view(B, 1, D) @ model.lm_head
    return logits, states

# --- 3. THE INFINITE LOOP DEMO ---
def run_infinite_context_demo():
    print("\n🚀 INITIALIZING INFINITE CONTEXT DEMO...")

    # 1. Load Model
    # Ensure config matches your trained model (vocab 65)
    # config = bdh.BDHConfig(n_layer=4, n_head=4, n_embd=128, mlp_internal_dim_multiplier=4, vocab_size=65)
    # We match the Transformer's new massive size
    config = bdh.BDHConfig(n_layer=4, n_head=8, n_embd=1024, mlp_internal_dim_multiplier=4, vocab_size=65)
    model = bdh.BDH(config).to(device)
    try:
        model.load_state_dict(torch.load("bdh_trained.pth", map_location=device))
        print("✅ Loaded trained weights.")
    except:
        print("⚠️ Using random weights (Architecture Test Only).")
    model.eval()

    # 2. Initialize "Brain State" (Fixed Size!)
    # Structure: [Batch, Heads, N, D]
    # N = config.mlp_internal_dim_multiplier * D // n_head
    # D = n_embd
    nh = config.n_head
    D = config.n_embd
    N = config.mlp_internal_dim_multiplier * D // nh

    # One state matrix per layer depth
    states = [torch.zeros(1, nh, N, D).to(device) for _ in range(config.n_layer)]

    print(f"🧠 Brain State Initialized. Matrix Size: [{nh}, {N}, {D}]")
    print(f"   This matrix will NOT grow, even if we process 1M tokens.")
    print("-------------------------------------------------------")
    print(f"{'Token Step':<15} | {'VRAM Usage (MB)':<20} | {'Status':<10}")
    print("-------------------------------------------------------")

    # 3. Run Forever
    current_token = torch.tensor([[0]]).to(device)

    try:
        start_time = time.time()
        # Simulate processing 50,000 tokens
        for t in range(1, 50001):
            with torch.no_grad():
                # Run the recurrent step
                logits, states = bdh_recurrent_step(model, current_token, states, pos=t)

                # Pick next token (greedy)
                current_token = torch.argmax(logits, dim=-1)

            # REPORTING
            if t % 2000 == 0:
                # Force garbage collection to prove stability
                torch.cuda.empty_cache()
                mem = torch.cuda.max_memory_allocated() / 1024**2
                print(f"{t:<15} | {mem:<20.2f} | ✅ Alive")

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except RuntimeError as e:
        print(f"\n❌ CRASHED at step {t}: {e}")

if __name__ == "__main__":
    run_infinite_context_demo()
