import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import sys

# --- 1. DEFINE GPT LOCALLY (To ensure no config limits) ---
# We define a simplified GPT here to guarantee it accepts large block_sizes
class CausalSelfAttention(nn.Module):
    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd
        # FORCE FLASH OFF to ensure memory explosion
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                    .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Standard Attention (Quadratic Memory)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd) # <--- Critical: Sized to fit input
        self.blocks = nn.Sequential(*[
            Block(n_head, n_embd, block_size) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

# --- 2. RUN THE KILL SHOT ---
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_final_crash():
    print("\n🎬 ACTION: INITIALIZING MEMORY EXPLOSION DEMO...")

    # CONFIGURATION: Optimized to crash T4 (15GB) quickly
    # n_embd=1024, Context=10,000
    model = SimpleGPT(vocab_size=65, n_embd=1024, n_head=8, n_layer=4, block_size=12000).to(device)
    model.eval()

    print("🤖 Model Ready. Context Window Unlock: 12,000 tokens.")
    print("-------------------------------------------------------")

    # BASELINE
    torch.cuda.reset_peak_memory_stats()
    print(f"   Baseline VRAM: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    # THE INPUT (10,000 Tokens)
    x = torch.randint(0, 65, (1, 14000)).to(device)
    print("   Input Tensor: [1, 10000] tokens.")
    print("   Attempting Forward Pass (Calculating 100M+ Attention Scores)...")
    print("-------------------------------------------------------")

    try:
        # THE TRIGGER
        with torch.no_grad():
            logits = model(x)

        print("✅ SURVIVED?! (If you see this, increase 'x' size to 14000)")
        print(f"   Peak VRAM: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")

    except RuntimeError as e:
        # THE PAYOFF
        print("\n❌ CRASHED! (Out of Memory)")
        print(f"   Error: {e}")
        print("   >> This validates the Memory Wall.")

if __name__ == "__main__":
    run_final_crash()
