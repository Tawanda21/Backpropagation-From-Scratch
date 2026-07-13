# Phase 3 — Transformer Architecture

**Goal:** Build the complete GPT model using our neural network building blocks

A bottom-up walkthrough of every component we built, what the math means, and why each design decision was made.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Multi-Head Attention](#2-multi-head-attention)
3. [Grouped Query Attention (GQA)](#3-grouped-query-attention-gqa)
4. [Rotary Position Embeddings (RoPE)](#4-rotary-position-embeddings-rope)
5. [FeedForward with SwiGLU](#5-feedforward-with-swiglu)
6. [Transformer Block](#6-transformer-block)
7. [The Complete GPT Model](#7-the-complete-gpt-model)
8. [Weight Tying](#8-weight-tying)
9. [Weight Initialization](#9-weight-initialization)
10. [Generation](#10-generation)
11. [Design Decisions at a Glance](#11-design-decisions-at-a-glance)
12. [Test Suite](#12-test-suite)

---

## 1. The Big Picture

A language model does one thing: **given a sequence of tokens, predict the next token**. We do this with a *decoder-only transformer* — the same family as GPT-2, LLaMA, and Mistral.

The data flows like this:

```
Token IDs (B, T)
    │
    ▼
Token Embedding          maps each integer id → a learned vector of size `dim`
    │
    ▼
Position Embedding        adds position information (RoPE in attention)
    │
    ▼
× N Transformer Blocks   each block refines every token's representation
    │  ┌──────────────────────────────────────────────────────┐
    │  │  RMSNorm → MultiHeadAttention (GQA, RoPE) → +      │
    │  │  RMSNorm → FeedForward (SwiGLU) → +                │
    │  └──────────────────────────────────────────────────────┘
    │
    ▼
Final RMSNorm
    │
    ▼
Linear Head              projects `dim` → `vocab_size` logits
    │
    ▼
Softmax → probability over every token in vocabulary
```

> **Decoder-only** means there is no encoder and no cross-attention. The model only sees its own past — it cannot peek at future tokens. This is enforced by the *causal mask* inside attention.

---

## 2. Multi-Head Attention

**File:** `scratch/transformer.py`  
**Paper:** [Vaswani et al., 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Intuition

Each token gets to ask a question ("what am I looking for?") and broadcast an answer ("what do I contain?"). The *query* encodes the question, the *key* encodes the answer label, and the *value* encodes the answer content.

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √head_dim ) · V
```

- `Q·Kᵀ` — how similar is each query to each key? (a score matrix)
- `/ √head_dim` — scale down to prevent softmax saturation
- `softmax(...)` — turn scores into a probability distribution (weights)
- `· V` — weighted average of all value vectors

### Implementation

```python
class MultiHeadAttention(Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        
        # Query projection
        self.wq = Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        
        # Key and Value projections (fewer heads for GQA)
        self.wk = Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        
        # Output projection
        self.wo = Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        
        self.dropout = Dropout(config.dropout) if config.dropout > 0 else None
    
    def forward(self, x, freqs_cis):
        # x: (B, T, dim)
        B, T, _ = x.data.shape
        
        # Project to Q, K, V
        xq = self.wq(x)  # (B, T, n_heads * head_dim)
        xk = self.wk(x)  # (B, T, n_kv_heads * head_dim)
        xv = self.wv(x)  # (B, T, n_kv_heads * head_dim)
        
        # Reshape for multi-head
        xq = xq.reshape(B, T, self.n_heads, self.head_dim)
        xk = xk.reshape(B, T, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Repeat K and V for GQA
        xk = repeat_kv(xk, self.n_rep)  # (B, T, n_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (B, T, n_heads, head_dim)
        
        # Transpose for attention: (B, n_heads, T, head_dim)
        xq = xq.transpose(0, 2, 1, 3)
        xk = xk.transpose(0, 2, 1, 3)
        xv = xv.transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention with causal mask
        # We'll implement this manually since we don't have Flash Attention
        scores = (xq @ xk.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Causal mask: prevent looking ahead
        mask = np.triu(np.ones((T, T)), k=1)
        scores.data = scores.data + (mask * -1e9)
        
        # Softmax
        attn = np.exp(scores.data - np.max(scores.data, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        
        if self.training and self.dropout is not None:
            attn = self.dropout(Tensor(attn, requires_grad=False))
            attn = attn.data
        
        # Apply attention to values
        out = Tensor(attn, requires_grad=False) @ xv  # (B, n_heads, T, head_dim)
        
        # Transpose back: (B, T, n_heads, head_dim)
        out = out.transpose(0, 2, 1, 3)
        
        # Reshape: (B, T, dim)
        out = out.reshape(B, T, -1)
        
        # Output projection
        out = self.wo(out)
        
        return out
```

### The Causal Mask

In a language model, token at position `t` must **not** see tokens at positions `> t`. We enforce this with a *causal mask*:

```
Scores (T=4):          After masking:
 q₀ → [s₀₀  s₀₁  s₀₂  s₀₃]     [s₀₀  -∞   -∞   -∞ ]
 q₁ → [s₁₀  s₁₁  s₁₂  s₁₃]     [s₁₀  s₁₁  -∞   -∞ ]
 q₂ → [s₂₀  s₂₁  s₂₂  s₂₃]     [s₂₀  s₂₁  s₂₂  -∞ ]
 q₃ → [s₃₀  s₃₁  s₃₂  s₃₃]     [s₃₀  s₃₁  s₃₂  s₃₃]
```

We set all scores above the diagonal to `-∞` before softmax, so they get probability 0.

**Why `-1e9`?** Large negative number that effectively becomes `-∞` in floating point. Adding this to scores ensures softmax treats them as zero.

---

## 3. Grouped Query Attention (GQA)

**Paper:** [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)

### The Problem

At inference time, we cache K and V for all previous tokens. With standard MHA and `n_heads=8`, this cache grows as:

```
KV cache size = 2 × n_layers × n_heads × head_dim × seq_len × 4 bytes
```

For a large model with long context, this becomes gigabytes.

### GQA: Share K/V Heads Across Q Heads

GQA uses fewer K/V heads than Q heads. `n_kv_heads` K/V projections serve `n_heads` Q projections:

```
n_heads    = 8   (8 separate Q projections)
n_kv_heads = 4   (4 shared KV projections)
n_rep      = 2   (each KV head serves 2 Q heads)
```

At inference, the KV cache shrinks by `n_rep×` with minimal quality loss.

### Implementation: `repeat_kv`

Since the attention matmul expects matching head counts, we expand K and V:

```python
def repeat_kv(x, n_rep):
    """
    Repeat key/value heads to match query heads.
    
    Input:  (B, T, n_kv_heads, head_dim)
    Output: (B, T, n_heads, head_dim)
    """
    B, T, n_kv_heads, head_dim = x.data.shape
    
    if n_rep == 1:
        return x
    
    # Expand and reshape
    # (B, T, n_kv_heads, 1, head_dim) → (B, T, n_kv_heads, n_rep, head_dim)
    out = x.data[:, :, :, None, :].repeat(n_rep, axis=-2)
    # (B, T, n_kv_heads * n_rep, head_dim)
    out = out.reshape(B, T, n_kv_heads * n_rep, head_dim)
    
    return Tensor(out, (x,), 'repeat_kv')
```

### The Backward Pass

The gradient through repeat_kv needs to sum across the repeated dimensions:

```python
def repeat_kv_backward(grad, n_rep):
    """
    Sum gradients across repeated heads.
    
    Input:  (B, T, n_heads, head_dim)
    Output: (B, T, n_kv_heads, head_dim)
    """
    B, T, n_heads, head_dim = grad.shape
    n_kv_heads = n_heads // n_rep
    
    # Reshape and sum
    grad = grad.reshape(B, T, n_kv_heads, n_rep, head_dim)
    grad = np.sum(grad, axis=-2)
    
    return grad
```

---

## 4. Rotary Position Embeddings (RoPE)

**Paper:** [Su et al., 2021 — RoFormer](https://arxiv.org/abs/2104.09864)

### Why Position Encodings?

Self-attention is *permutation-equivariant* by default: if you shuffle the tokens, the attention scores just shuffle too. The model has no idea which token came first. Position encodings break this symmetry.

### How RoPE Works

RoPE encodes position by *rotating* the query and key vectors in 2D before the dot-product:

```
q' = Rotate(q, position_q)
k' = Rotate(k, position_k)
score = q' · k' = q · k * cos(position_q - position_k)
```

The score now depends on the **relative position**, not absolute positions.

### Implementation

```python
def precompute_freqs_cis(head_dim, max_seq_len, theta=10000.0):
    """
    Precompute the rotation frequencies.
    
    Returns:
        freqs_cis: (max_seq_len, head_dim//2) complex numbers
    """
    # Frequencies: θ_i = 10000^(-2i/head_dim)
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))
    
    # Positions: m = 0, 1, 2, ...
    positions = np.arange(max_seq_len)
    
    # Outer product: (max_seq_len, head_dim//2)
    freqs = np.outer(positions, freqs)
    
    # Convert to complex: e^(j·freq)
    return np.exp(1j * freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply rotary embeddings to query and key tensors.
    
    Args:
        xq: (B, T, n_heads, head_dim)
        xk: (B, T, n_kv_heads, head_dim)
        freqs_cis: (T, head_dim//2) complex
    
    Returns:
        rotated xq, rotated xk
    """
    # Reshape to complex: (..., head_dim//2, 2) → (..., head_dim//2) complex
    xq_complex = np.view_as_complex(xq.data.reshape(*xq.data.shape[:-1], -1, 2))
    xk_complex = np.view_as_complex(xk.data.reshape(*xk.data.shape[:-1], -1, 2))
    
    # Expand freqs to match shapes
    # (T, head_dim//2) → (1, T, 1, head_dim//2)
    freqs_cis = freqs_cis[None, :, None, :]
    
    # Multiply: e^(jθ) * (a + jb)
    xq_rotated = xq_complex * freqs_cis
    xk_rotated = xk_complex * freqs_cis
    
    # Convert back to real: (..., head_dim)
    xq_out = np.stack([xq_rotated.real, xq_rotated.imag], axis=-1)
    xk_out = np.stack([xk_rotated.real, xk_rotated.imag], axis=-1)
    xq_out = xq_out.reshape(*xq.data.shape)
    xk_out = xk_out.reshape(*xk.data.shape)
    
    return Tensor(xq_out, (xq,), 'rope'), Tensor(xk_out, (xk,), 'rope')
```

**The key insight:** Multiplying by `e^(j·m·θ_i)` is a 2D rotation. The dot product `q'·k'` depends on the difference in positions, giving us relative position encoding.

---

## 5. FeedForward with SwiGLU

**File:** `scratch/transformer.py`  
**Paper:** [Shazeer, 2020](https://arxiv.org/abs/2002.05202)

### What FFN Does

After attention lets tokens communicate, the FFN processes each token *independently*. Think of attention as "communication" and FFN as "thinking."

### From ReLU to SwiGLU

The original FFN used:
```
FFN(x) = ReLU(x·W1) · W2
```

SwiGLU replaces this with a *gated* version:
```
FFN(x) = ( SiLU(x·W1) ⊙ x·W3 ) · W2
```

where `⊙` is element-wise multiplication.

**Three weight matrices:**
- `W1` (gate): passes through SiLU activation
- `W3` (up): the "content" projection — no activation
- `W2` (down): projects back to `dim`

### Implementation

```python
class FeedForward(Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        
        # Hidden dimension: (8/3) * dim, rounded to nearest multiple_of
        hidden_dim = int(2 * 4 * config.dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        
        self.w1 = Linear(config.dim, hidden_dim, bias=False)  # Gate
        self.w2 = Linear(hidden_dim, config.dim, bias=False)  # Down
        self.w3 = Linear(config.dim, hidden_dim, bias=False)  # Up
    
    def forward(self, x):
        # Gate: SiLU(x @ w1)
        gate = gelu(self.w1(x))  # Using GELU as approximation to SiLU
        
        # Up: x @ w3
        up = self.w3(x)
        
        # Multiply and project back
        hidden = gate * up
        return self.w2(hidden)
```

### SiLU

SiLU (Sigmoid Linear Unit): `SiLU(x) = x · σ(x)`

Unlike ReLU, SiLU is smooth and has a small negative lobe, giving gradient information even for negative inputs.

### Hidden Dimension Sizing

Standard FFN: `hidden = 4 × dim`

We use LLaMA's formula: `(8/3) × dim`, rounded up to nearest `multiple_of`.

```python
hidden_dim = int(2 * 4 * config.dim / 3)  # ≈ (8/3) × dim
hidden_dim = multiple_of * ceil(hidden_dim / multiple_of)
```

This gives slightly fewer parameters than `4×` while performing better.

---

## 6. Transformer Block

**File:** `scratch/transformer.py`

### Pre-Norm vs Post-Norm

**Original transformer (post-norm):**
```
x → SubLayer → x + SubLayer(x) → LayerNorm
```

**Modern (pre-norm, what we use):**
```
x → LayerNorm → SubLayer → x + SubLayer(LayerNorm(x))
```

In pre-norm, the residual path `x` flows through unchanged. Gradients can flow directly back to early layers without passing through any normalizer, preventing vanishing gradients.

### Implementation

```python
class TransformerBlock(Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-norm for attention
        self.attention_norm = LayerNorm(config.dim, eps=config.norm_eps)
        self.attention = MultiHeadAttention(config)
        
        # Pre-norm for feedforward
        self.ffn_norm = LayerNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)
        
        # Dropout
        self.dropout = Dropout(config.dropout) if config.dropout > 0 else None
    
    def forward(self, x, freqs_cis):
        # Attention block with residual connection
        h = self.attention_norm(x)
        h = self.attention(h, freqs_cis)
        if self.training and self.dropout is not None:
            h = self.dropout(h)
        x = x + h
        
        # Feedforward block with residual connection
        h = self.ffn_norm(x)
        h = self.feed_forward(h)
        if self.training and self.dropout is not None:
            h = self.dropout(h)
        x = x + h
        
        return x
```

**Why two residual connections?** Each block adds two contributions (attention + FFN). The residual stream accumulates information from all layers.

---

## 7. The Complete GPT Model

**File:** `scratch/transformer.py`

### Assembly

```python
class GPT(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.tok_embeddings = Embedding(config.vocab_size, config.dim)
        
        # Transformer blocks
        self.layers = []
        for i in range(config.n_layers):
            self.layers.append(TransformerBlock(config, i))
        
        # Final normalization
        self.norm = LayerNorm(config.dim, eps=config.norm_eps)
        
        # Output head (tied with embedding)
        self.output = Linear(config.dim, config.vocab_size, bias=False)
        
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
    
    def forward(self, tokens, targets=None):
        """
        Args:
            tokens: (B, T) integer token IDs
            targets: (B, T) integer target token IDs (optional)
        
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar Tensor (if targets provided)
        """
        B, T = tokens.data.shape
        
        # Get embeddings
        h = self.tok_embeddings(tokens)  # (B, T, dim)
        
        # Get RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:T]
        
        # Pass through layers
        for layer in self.layers:
            h = layer(h, freqs_cis)
        
        # Final norm
        h = self.norm(h)
        
        # Output logits
        logits = self.output(h)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = cross_entropy(logits, targets)
        
        return logits, loss
```

### The Full Forward Pass

```
Input: (B, T) token IDs
    │
    ▼
Token Embedding: (B, T) → (B, T, dim)
    │
    ▼
For each layer (1...n_layers):
    │
    ├─► LayerNorm → Attention → + (residual)
    │
    └─► LayerNorm → FeedForward → + (residual)
    │
    ▼
Final LayerNorm
    │
    ▼
Output Projection: (B, T, dim) → (B, T, vocab_size)
    │
    ▼
Cross-Entropy Loss (if targets provided)
```

---

## 8. Weight Tying

**Paper:** [Press & Wolf, 2017](https://arxiv.org/abs/1611.01462)

### The Idea

The input embedding and the output head share the **exact same weight matrix**:

```python
self.tok_embeddings.weight = self.output.weight
```

This means token embeddings and output logit projections are learned jointly.

### Why This Works

- **Parameter savings:** For `vocab_size=32000, dim=512`, this saves 16.4M parameters
- **Improved perplexity:** The embedding space is forced to be consistent with what the model predicts
- **No quality loss:** Tied weights often perform better than separate weights

### Implementation

```python
class GPT(Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embeddings = Embedding(config.vocab_size, config.dim)
        
        # ... other layers ...
        
        self.output = Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.output.weight = self.tok_embeddings.weight
```

**Important:** This means `tok_embeddings.weight` and `output.weight` are the same Tensor object. Gradients from the output head accumulate directly in the embedding matrix.

---

## 9. Weight Initialization

### GPT-2 Style Init

All weights initialized with:
```
N(μ=0, σ=0.02)
```

This is the GPT-2 scheme: a narrow normal distribution that keeps activations near zero at the start.

```python
def _init_weights(self, module):
    if isinstance(module, Linear):
        # GPT-2 init
        if module.weight is not None:
            module.weight.data = np.random.randn(*module.weight.data.shape) * 0.02
        
        if module.bias is not None:
            module.bias.data = np.zeros_like(module.bias.data)
    
    elif isinstance(module, Embedding):
        module.weight.data = np.random.randn(*module.weight.data.shape) * 0.02
```

### Residual Scaling

The output projections of each sub-layer (`wo` in attention, `w2` in FFN) are scaled down:

```python
def _init_weights(self, module):
    # ... normal init ...
    
    # Residual scaling
    if isinstance(module, Linear) and (module is self.layers[-1].attention.wo or module is self.layers[-1].feed_forward.w2):
        module.weight.data *= 0.02 / np.sqrt(2 * self.config.n_layers)
```

**Why?** Each block adds two residual contributions. With `n_layers` blocks, the total residual variance grows with `n_layers`. Scaling by `1/√(2×n_layers)` cancels this growth.

---

## 10. Generation

**File:** `scratch/generation.py`

### Autoregressive Decoding

The model generates one token at a time:

```
[BOS] → predicts token 1
[BOS, t1] → predicts token 2
[BOS, t1, t2] → predicts token 3
...
```

At each step:
1. Run forward pass on current sequence
2. Take only the **last** position's logits
3. Sample from the probability distribution
4. Append the sampled token and repeat

### Implementation

```python
def generate(model, prompt, max_new_tokens=100, temperature=1.0, top_k=None):
    """
    Generate text from a prompt.
    """
    model.eval()
    
    # Convert prompt to tokens
    if isinstance(prompt, str):
        tokens = tokenizer.encode(prompt)
    else:
        tokens = prompt
    
    # Generate
    for _ in range(max_new_tokens):
        # Get logits for last position
        logits, _ = model(tokens)
        logits = logits[:, -1, :] / temperature
        
        # Top-k sampling
        if top_k is not None:
            # Keep only top_k tokens
            top_k_vals, top_k_indices = np.partition(logits, -top_k)[-top_k:]
            mask = logits < top_k_vals[:, [0]]
            logits[mask] = -np.inf
        
        # Sample from distribution
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        next_token = np.random.choice(probs.shape[-1], p=probs[0])
        
        # Append
        tokens = Tensor(np.concatenate([tokens.data, [[next_token]]], axis=-1))
        
        # Stop if end token
        if next_token == tokenizer.eos_token_id:
            break
    
    # Convert back to string
    return tokenizer.decode(tokens.data[0])
```

### Temperature

```python
logits = logits / temperature
```

- `temperature < 1` → sharper distribution (more deterministic)
- `temperature > 1` → flatter distribution (more random)
- `temperature = 1` → unmodified distribution

### Top-K Sampling

```python
top_k_vals, _ = np.partition(logits, -top_k)[-top_k:]
logits[logits < top_k_vals[:, [-1]]] = -np.inf
```

Restrict to only the `k` highest-logit tokens. This prevents the model from picking very unlikely tokens while still allowing diversity.

---

## 11. Design Decisions at a Glance

| Component | Our Choice | Alternative | Reason |
|---|---|---|---|
| Attention | Multi-head with GQA | Full MHA | Reduces KV cache at inference |
| Position encoding | RoPE | Sinusoidal, ALiBi | Better length generalization |
| Normalization | RMSNorm (pre-norm) | LayerNorm (post-norm) | Faster, more stable gradients |
| FFN activation | SwiGLU | ReLU, GELU | Better empirical performance |
| Output projection | Tied to embedding | Separate | Saves parameters, improves perplexity |
| Bias terms | None | Bias in all linears | Consistent with LLaMA |
| Weight init | GPT-2 (σ=0.02) + residual scale | Glorot, orthogonal | Prevents depth-related variance growth |

---

## 12. Test Suite

**File:** `tests/test_transformer.py`

| Test | What It Verifies |
|---|---|
| `test_attention_forward` | Attention outputs correct shape |
| `test_attention_causal` | Causal mask prevents looking ahead |
| `test_gqa_repeat` | repeat_kv expands heads correctly |
| `test_gqa_backward` | Gradients sum correctly through repeat_kv |
| `test_rope_precompute` | Frequencies have correct shape and magnitude |
| `test_rope_apply` | RoPE preserves norm of vectors |
| `test_ffn_forward` | FFN outputs correct shape |
| `test_ffn_swiglu` | SwiGLU uses three weight matrices |
| `test_transformer_block` | Block outputs correct shape |
| `test_transformer_block_residual` | Zero weights → identity output |
| `test_gpt_forward` | GPT outputs correct shape |
| `test_gpt_loss` | Cross-entropy loss is computed |
| `test_gpt_tied_weights` | Embedding and output share weights |
| `test_gpt_generate` | Generation produces correct shape |
| `test_gpt_temperature` | Temperature affects entropy |
| `test_gpt_top_k` | Top-k restricts vocabulary |

### Test Example: Causal Mask

```python
def test_attention_causal():
    config = ModelConfig(dim=64, n_heads=4, n_kv_heads=2, max_seq_len=10)
    attn = MultiHeadAttention(config)
    
    x = Tensor(np.random.randn(1, 5, 64))
    freqs_cis = precompute_freqs_cis(16, 5)
    
    out = attn(x, freqs_cis)
    
    # Check that position 0 only attends to position 0
    # By checking attention weights through gradient
    # (We'll test this by ensuring position 1 can't see position 2)
    ...
```

---