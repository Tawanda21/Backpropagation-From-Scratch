# Phase 7 — PyTorch Verification

**Goal:** Prove our scratch implementation is mathematically correct by comparing against PyTorch

A bottom-up walkthrough of how we verify every component, what we compare, and why each test matters.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [PyTorch Model Port](#2-pytorch-model-port)
3. [Verification Suite](#3-verification-suite)
4. [Forward Pass Comparison](#4-forward-pass-comparison)
5. [Backward Pass Comparison](#5-backward-pass-comparison)
6. [Training Loop Comparison](#6-training-loop-comparison)
7. [Verification Results](#7-verification-results)
8. [Design Decisions at a Glance](#8-design-decisions-at-a-glance)
9. [Test Suite](#9-test-suite)

---

## 1. The Big Picture

We need to prove that our scratch implementation produces the same results as PyTorch. This is the gold standard for verification.

The verification process:

```
Same Random Seed
        |
        v
Scratch Implementation          PyTorch Implementation
        |                               |
        v                               v
    Forward Pass                   Forward Pass
        |                               |
        v                               v
    Compare Outputs (should match within 1e-5)
        |                               |
        v                               v
    Backward Pass                  Backward Pass
        |                               |
        v                               v
    Compare Gradients (should match within 1e-6)
        |                               |
        v                               v
    Training Step                   Training Step
        |                               |
        v                               v
    Compare Weights (should match within 1e-5)
```

**Key insight:** If all three comparisons pass, our implementation is mathematically equivalent to PyTorch.

---

## 2. PyTorch Model Port

**File:** `pytorch_migration/model.py`

### Complete PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 65
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 3
    max_seq_len: int = 512
    dropout: float = 0.2
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rrms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rrms * self.weight

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
    
    def forward(self, x, freqs_cis):
        B, T, _ = x.shape
        
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Repeat KV heads
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        
        # Transpose for attention
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Flash attention
        out = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2 * 4 * config.dim / 3)
        hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)
        
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        return self.w2(gate * up)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = MultiHeadAttention(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)
    
    def forward(self, x, freqs_cis):
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.output.weight = self.tok_embeddings.weight
        
        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len,
            config.rope_theta
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:T]
        
        for layer in self.layers:
            h = layer(h, freqs_cis)
        
        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        
        return logits, loss
```

### Helper Functions

```python
def precompute_freqs_cis(head_dim, max_seq_len, theta=10000.0):
    """Precompute RoPE frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len)
    freqs = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply RoPE to queries and keys."""
    # Reshape to complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Expand freqs
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    # Apply rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x, n_rep):
    """Repeat KV heads for GQA."""
    if n_rep == 1:
        return x
    B, T, n_kv_heads, head_dim = x.shape
    return x[:, :, :, None, :].expand(B, T, n_kv_heads, n_rep, head_dim).reshape(B, T, n_kv_heads * n_rep, head_dim)
```

---

## 3. Verification Suite

**File:** `pytorch_migration/verify.py`

### Setup Function

```python
import numpy as np
import torch
from scratch.model import GPT as ScratchGPT
from scratch.tensor import Tensor
from pytorch_migration.model import GPT as TorchGPT
from pytorch_migration.model import ModelConfig

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def compare_tensors(t1, t2, name="", rtol=1e-5, atol=1e-6):
    """Compare two tensors and report differences."""
    if isinstance(t1, Tensor):
        t1 = t1.data
    if isinstance(t2, torch.Tensor):
        t2 = t2.detach().cpu().numpy()
    elif isinstance(t2, Tensor):
        t2 = t2.data
    
    diff = np.abs(t1 - t2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    relative_diff = max_diff / (np.max(np.abs(t1)) + 1e-8)
    
    passed = max_diff < atol or relative_diff < rtol
    
    print(f"{name}:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Relative diff: {relative_diff:.6f}")
    print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    print()
    
    return passed
```

### Verification Class

```python
class VerificationSuite:
    """Complete verification suite comparing scratch and PyTorch."""
    
    def __init__(self, config=None):
        if config is None:
            config = ModelConfig(
                vocab_size=65,
                dim=64,
                n_layers=2,
                n_heads=2,
                n_kv_heads=1,
                max_seq_len=128
            )
        self.config = config
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Create models
        self.scratch_model = ScratchGPT(config)
        self.torch_model = TorchGPT(config)
        
        # Copy weights from scratch to PyTorch
        self._copy_weights()
        
        # Create test data
        self.B = 2
        self.T = 16
        self.x = np.random.randint(0, config.vocab_size, (self.B, self.T))
        self.y = np.random.randint(0, config.vocab_size, (self.B, self.T))
    
    def _copy_weights(self):
        """Copy weights from scratch model to PyTorch model."""
        scratch_params = self.scratch_model.parameters()
        torch_params = list(self.torch_model.parameters())
        
        for scratch_param, torch_param in zip(scratch_params, torch_params):
            torch_param.data = torch.tensor(scratch_param.data, dtype=torch.float32)
    
    def _copy_weights_back(self):
        """Copy weights from PyTorch to scratch model."""
        scratch_params = self.scratch_model.parameters()
        torch_params = list(self.torch_model.parameters())
        
        for scratch_param, torch_param in zip(scratch_params, torch_params):
            scratch_param.data = torch_param.detach().cpu().numpy()
    
    def verify_forward(self):
        """Verify forward pass outputs match."""
        print("=" * 60)
        print("VERIFYING FORWARD PASS")
        print("=" * 60)
        print()
        
        # Scratch forward
        scratch_x = Tensor(self.x)
        scratch_logits, scratch_loss = self.scratch_model(scratch_x, Tensor(self.y))
        
        # PyTorch forward
        torch_x = torch.tensor(self.x, dtype=torch.long)
        torch_y = torch.tensor(self.y, dtype=torch.long)
        
        with torch.no_grad():
            torch_logits, torch_loss = self.torch_model(torch_x, torch_y)
        
        # Compare logits
        passed_logits = compare_tensors(
            scratch_logits.data, 
            torch_logits, 
            "Logits comparison"
        )
        
        # Compare loss
        passed_loss = compare_tensors(
            scratch_loss.data, 
            torch_loss.item(),
            "Loss comparison"
        )
        
        return passed_logits and passed_loss
    
    def verify_backward(self):
        """Verify backward pass gradients match."""
        print("=" * 60)
        print("VERIFYING BACKWARD PASS")
        print("=" * 60)
        print()
        
        # Reset gradients
        for p in self.scratch_model.parameters():
            if p.grad is not None:
                p.grad = np.zeros_like(p.grad)
        self.torch_model.zero_grad()
        
        # Scratch backward
        scratch_x = Tensor(self.x)
        scratch_logits, scratch_loss = self.scratch_model(scratch_x, Tensor(self.y))
        scratch_loss.backward()
        
        # PyTorch backward
        torch_x = torch.tensor(self.x, dtype=torch.long)
        torch_y = torch.tensor(self.y, dtype=torch.long)
        torch_logits, torch_loss = self.torch_model(torch_x, torch_y)
        torch_loss.backward()
        
        # Compare gradients for each parameter
        scratch_params = list(self.scratch_model.parameters())
        torch_params = list(self.torch_model.parameters())
        
        all_passed = True
        for i, (sp, tp) in enumerate(zip(scratch_params, torch_params)):
            if sp.grad is not None:
                name = f"Parameter {i}"
                passed = compare_tensors(
                    sp.grad,
                    tp.grad,
                    name,
                    rtol=1e-5,
                    atol=1e-6
                )
                all_passed = all_passed and passed
        
        return all_passed
    
    def verify_training_step(self):
        """Verify a full training step produces the same weight updates."""
        print("=" * 60)
        print("VERIFYING TRAINING STEP")
        print("=" * 60)
        print()
        
        # Initialize weights identically
        set_seed(42)
        self.scratch_model = ScratchGPT(self.config)
        self.torch_model = TorchGPT(self.config)
        self._copy_weights()
        
        # Get initial weights
        scratch_init = [p.data.copy() for p in self.scratch_model.parameters()]
        torch_init = [p.data.clone() for p in self.torch_model.parameters()]
        
        # Scratch optimizer
        from scratch.optim import Adam
        scratch_opt = Adam(self.scratch_model.parameters(), lr=1e-3)
        
        # PyTorch optimizer
        torch_opt = torch.optim.Adam(self.torch_model.parameters(), lr=1e-3)
        
        # One training step on same data
        # Scratch
        scratch_x = Tensor(self.x)
        scratch_logits, scratch_loss = self.scratch_model(scratch_x, Tensor(self.y))
        scratch_opt.zero_grad()
        scratch_loss.backward()
        scratch_opt.step()
        
        # PyTorch
        torch_x = torch.tensor(self.x, dtype=torch.long)
        torch_y = torch.tensor(self.y, dtype=torch.long)
        torch_logits, torch_loss = self.torch_model(torch_x, torch_y)
        torch_opt.zero_grad()
        torch_loss.backward()
        torch_opt.step()
        
        # Compare updated weights
        scratch_params = self.scratch_model.parameters()
        torch_params = self.torch_model.parameters()
        
        all_passed = True
        for i, (sp, tp) in enumerate(zip(scratch_params, torch_params)):
            name = f"Weight {i} after update"
            passed = compare_tensors(
                sp.data,
                tp.data,
                name,
                rtol=1e-4,
                atol=1e-5
            )
            all_passed = all_passed and passed
        
        return all_passed
    
    def run_all(self):
        """Run all verification tests."""
        print("=" * 70)
        print("PYTORCH VERIFICATION SUITE")
        print("=" * 70)
        print()
        print(f"Config: dim={self.config.dim}, n_layers={self.config.n_layers}")
        print(f"Batch size: {self.B}, Sequence length: {self.T}")
        print()
        
        results = {
            'forward': self.verify_forward(),
            'backward': self.verify_backward(),
            'training_step': self.verify_training_step()
        }
        
        # Summary
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, passed in results.items():
            print(f"{name}: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        all_passed = all(results.values())
        print()
        print(f"{'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print("=" * 60)
        
        return all_passed
```

---

## 4. Forward Pass Comparison

**File:** `pytorch_migration/verify_forward.py`

### Detailed Forward Pass Verification

```python
def detailed_forward_verification():
    """Detailed verification of forward pass through each layer."""
    config = ModelConfig(dim=64, n_layers=2, n_heads=2, n_kv_heads=1)
    set_seed(42)
    
    # Create models
    scratch_model = ScratchGPT(config)
    torch_model = TorchGPT(config)
    
    # Copy weights
    for sp, tp in zip(scratch_model.parameters(), torch_model.parameters()):
        tp.data = torch.tensor(sp.data, dtype=torch.float32)
    
    # Create input
    B, T = 2, 16
    x = np.random.randint(0, config.vocab_size, (B, T))
    
    # Forward pass
    scratch_x = Tensor(x)
    scratch_logits, scratch_loss = scratch_model(scratch_x)
    
    torch_x = torch.tensor(x, dtype=torch.long)
    with torch.no_grad():
        torch_logits, torch_loss = torch_model(torch_x)
    
    # Compare layer by layer
    print("Layer-by-layer comparison:")
    print()
    
    # Embedding
    scratch_emb = scratch_model.tok_embeddings(Tensor(x)).data
    torch_emb = torch_model.tok_embeddings(torch_x).detach().numpy()
    compare_tensors(scratch_emb, torch_emb, "Embedding")
    
    # Each layer
    scratch_h = Tensor(x)
    torch_h = torch_x
    torch_h = torch_model.tok_embeddings(torch_h)
    
    for i, (scratch_layer, torch_layer) in enumerate(zip(
        scratch_model.layers, torch_model.layers
    )):
        # Scratch
        scratch_h = scratch_layer(scratch_h, scratch_model.freqs_cis[:T])
        
        # PyTorch
        torch_h = torch_layer(torch_h, torch_model.freqs_cis[:T])
        
        compare_tensors(
            scratch_h.data,
            torch_h.detach().numpy(),
            f"Layer {i}"
        )
    
    # Final norm and output
    scratch_h = scratch_model.norm(scratch_h)
    torch_h = torch_model.norm(torch_h)
    compare_tensors(scratch_h.data, torch_h.detach().numpy(), "Final norm")
    
    scratch_logits = scratch_model.output(scratch_h)
    torch_logits = torch_model.output(torch_h)
    compare_tensors(scratch_logits.data, torch_logits.detach().numpy(), "Output logits")
```

---

## 5. Backward Pass Comparison

**File:** `pytorch_migration/verify_backward.py`

### Gradient Comparison Details

```python
def verify_gradient_flow():
    """Verify gradients flow correctly through the entire model."""
    config = ModelConfig(dim=64, n_layers=2, n_heads=2, n_kv_heads=1)
    set_seed(42)
    
    # Create models
    scratch_model = ScratchGPT(config)
    torch_model = TorchGPT(config)
    
    # Copy weights
    for sp, tp in zip(scratch_model.parameters(), torch_model.parameters()):
        tp.data = torch.tensor(sp.data, dtype=torch.float32)
    
    # Create input with targets
    B, T = 2, 16
    x = np.random.randint(0, config.vocab_size, (B, T))
    y = np.random.randint(0, config.vocab_size, (B, T))
    
    # Scratch
    scratch_x = Tensor(x)
    scratch_logits, scratch_loss = scratch_model(scratch_x, Tensor(y))
    scratch_loss.backward()
    
    # PyTorch
    torch_x = torch.tensor(x, dtype=torch.long)
    torch_y = torch.tensor(y, dtype=torch.long)
    torch_logits, torch_loss = torch_model(torch_x, torch_y)
    torch_loss.backward()
    
    # Compare gradients with detailed stats
    print("Gradient comparison by layer:")
    print()
    
    scratch_params = list(scratch_model.parameters())
    torch_params = list(torch_model.parameters())
    
    for i, (sp, tp) in enumerate(zip(scratch_params, torch_params)):
        if sp.grad is not None:
            name = f"Parameter {i}"
            print(f"{name}:")
            print(f"  Scratch grad shape: {sp.grad.shape}")
            print(f"  Scratch grad mean: {sp.grad.mean():.6f}")
            print(f"  Scratch grad std: {sp.grad.std():.6f}")
            print(f"  Torch grad mean: {tp.grad.mean():.6f}")
            print(f"  Torch grad std: {tp.grad.std():.6f}")
            
            diff = np.abs(sp.grad - tp.grad.detach().numpy())
            print(f"  Max diff: {diff.max():.6f}")
            print(f"  Mean diff: {diff.mean():.6f}")
            print()
```

---

## 6. Training Loop Comparison

**File:** `pytorch_migration/verify_training.py`

### Full Training Loop Verification

```python
def verify_training_loop(num_steps=100):
    """Verify training loop produces same results over multiple steps."""
    config = ModelConfig(dim=64, n_layers=2, n_heads=2, n_kv_heads=1)
    set_seed(42)
    
    # Create models
    scratch_model = ScratchGPT(config)
    torch_model = TorchGPT(config)
    
    # Copy weights
    for sp, tp in zip(scratch_model.parameters(), torch_model.parameters()):
        tp.data = torch.tensor(sp.data, dtype=torch.float32)
    
    # Optimizers
    from scratch.optim import Adam
    scratch_opt = Adam(scratch_model.parameters(), lr=1e-3)
    torch_opt = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    
    # Track losses
    scratch_losses = []
    torch_losses = []
    
    for step in range(num_steps):
        # Generate same data
        B, T = 2, 16
        x = np.random.randint(0, config.vocab_size, (B, T))
        y = np.random.randint(0, config.vocab_size, (B, T))
        
        # Scratch
        scratch_x = Tensor(x)
        scratch_logits, scratch_loss = scratch_model(scratch_x, Tensor(y))
        scratch_opt.zero_grad()
        scratch_loss.backward()
        scratch_opt.step()
        scratch_losses.append(scratch_loss.data)
        
        # PyTorch
        torch_x = torch.tensor(x, dtype=torch.long)
        torch_y = torch.tensor(y, dtype=torch.long)
        torch_logits, torch_loss = torch_model(torch_x, torch_y)
        torch_opt.zero_grad()
        torch_loss.backward()
        torch_opt.step()
        torch_losses.append(torch_loss.item())
        
        # Compare weights every 10 steps
        if step % 10 == 0:
            print(f"Step {step}:")
            scratch_params = list(scratch_model.parameters())
            torch_params = list(torch_model.parameters())
            
            for i, (sp, tp) in enumerate(zip(scratch_params, torch_params)):
                compare_tensors(
                    sp.data,
                    tp.data,
                    f"  Parameter {i}",
                    rtol=1e-4,
                    atol=1e-5
                )
            print()
    
    # Compare loss curves
    print("Loss comparison:")
    scratch_losses = np.array(scratch_losses)
    torch_losses = np.array(torch_losses)
    compare_tensors(scratch_losses, torch_losses, "Loss curves", rtol=1e-3, atol=1e-4)
    
    # Plot loss curves
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(scratch_losses, label='Scratch', alpha=0.7)
    plt.plot(torch_losses, label='PyTorch', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Scratch vs PyTorch Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_comparison.png')
    plt.close()
    print("Loss curves saved to loss_comparison.png")
```

---

## 7. Verification Results

### Expected Output

```
======================================================================
PYTORCH VERIFICATION SUITE
======================================================================

Config: dim=64, n_layers=2
Batch size: 2, Sequence length: 16

============================================================
VERIFYING FORWARD PASS
============================================================

Logits comparison:
  Max diff: 0.000004
  Mean diff: 0.000001
  Relative diff: 0.000002
  ✓ PASSED

Loss comparison:
  Max diff: 0.000001
  Mean diff: 0.000001
  Relative diff: 0.000001
  ✓ PASSED

============================================================
VERIFYING BACKWARD PASS
============================================================

Parameter 0:
  Max diff: 0.000003
  Mean diff: 0.000001
  Relative diff: 0.000001
  ✓ PASSED

Parameter 1:
  Max diff: 0.000002
  Mean diff: 0.000001
  Relative diff: 0.000001
  ✓ PASSED

... (all parameters)

============================================================
VERIFYING TRAINING STEP
============================================================

Weight 0 after update:
  Max diff: 0.000012
  Mean diff: 0.000004
  Relative diff: 0.000008
  ✓ PASSED

... (all weights)

============================================================
SUMMARY
============================================================
forward: ✓ PASSED
backward: ✓ PASSED
training_step: ✓ PASSED

✓ ALL TESTS PASSED
============================================================
```

---

## 8. Design Decisions at a Glance

| Component | Our Choice | Alternative | Reason |
|---|---|---|---|
| Verification method | Direct comparison | Statistical tests | Direct is more rigorous |
| Tolerance | 1e-6 for gradients | 1e-4 or 1e-3 | Strict enough to catch bugs |
| Seed | 42 | Random seed | Reproducible results |
| Test config | Small model | Full model | Fast, still catches bugs |
| Number of tests | 3 core tests | Many small tests | Covers everything important |

---

## 9. Test Suite

**File:** `tests/test_verification.py`

| Test | What It Verifies |
|---|---|
| `test_forward_equivalence` | Forward outputs match PyTorch |
| `test_backward_equivalence` | Gradients match PyTorch |
| `test_training_equivalence` | Training updates match PyTorch |
| `test_rope_equivalence` | RoPE implementation matches |
| `test_attention_equivalence` | Attention outputs match |
| `test_ffn_equivalence` | FFN outputs match |
| `test_embeddings_equivalence` | Embedding layer matches |

---