# Phase 2 — Neural Network Building Blocks

**Goal:** Build the standard neural network layers using our Tensor operations

A bottom-up walkthrough of every component we built, what the math means, and why each design decision was made.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [The Module Base Class](#2-the-module-base-class)
3. [Linear Layer](#3-linear-layer)
4. [Dropout](#4-dropout)
5. [Layer Normalization](#5-layer-normalization)
6. [Embedding Layer](#6-embedding-layer)
7. [Loss Functions](#7-loss-functions)
8. [Optimizers](#8-optimizers)
9. [Parameter Management](#9-parameter-management)
10. [Design Decisions at a Glance](#10-design-decisions-at-a-glance)
11. [Test Suite](#11-test-suite)

---

## 1. The Big Picture

With the Tensor foundation complete, we need the standard building blocks of neural networks. These layers encapsulate the parameters and forward/backward logic needed for deep learning.

The data flows like this:

```
Input Tensor (B, features)
    │
    ▼
Linear Layer (W @ x + b)
    │
    ▼
Activation (ReLU/GELU/Tanh)
    │
    ▼
Dropout (during training only)
    │
    ▼
LayerNorm (stabilizes training)
    │
    ▼
Output Tensor
```

**Key insight:** Every layer is a Module that:
1. Stores its parameters (weights, biases, etc.) as Tensors
2. Defines a `forward()` method that computes the output
3. Optionally defines a `training` flag that changes behavior (Dropout)
4. Manages parameter updates through the optimizer

---

## 2. The Module Base Class

**File:** `scratch/nn.py`

### Why a Base Class?

Every neural network layer needs:
- A way to store parameters
- A way to collect all parameters for the optimizer
- A `forward()` method that computes the output
- Optional training/eval mode for layers like Dropout

```python
class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def parameters(self):
        """Collect all parameters from this module and its children."""
        params = []
        for param in self._parameters.values():
            if param is not None:
                params.append(param)
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def zero_grad(self):
        """Set gradients of all parameters to None/zero."""
        for param in self.parameters():
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)
    
    def train(self):
        """Set module to training mode."""
        self.training = True
        for module in self._modules.values():
            module.train()
    
    def eval(self):
        """Set module to evaluation mode."""
        self.training = False
        for module in self._modules.values():
            module.eval()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
```

**Why `_parameters` and `_modules` dictionaries?** This makes it easy to recursively collect all parameters. When we call `model.parameters()`, it traverses the entire module tree.

**Why `training` flag?** Dropout behaves differently during training vs inference. LayerNorm is the same regardless, but we keep the flag for consistency.

**Why `__call__` calls `forward`?** This is the PyTorch pattern. It lets us intercept calls (for hooks, etc.) but for now it's just a convenience.

### Registering Parameters

```python
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self._parameters['weight'] = Tensor(
            np.random.randn(in_features, out_features) * 0.01
        )
        if bias:
            self._parameters['bias'] = Tensor(np.zeros(out_features))
```

Parameters are stored in `_parameters` so `parameters()` can collect them.

---

## 3. Linear Layer

**File:** `scratch/nn.py`

### What It Does

The linear (dense/fully-connected) layer computes:
```
y = x @ W + b
```

where:
- `x` is shape `(batch_size, in_features)`
- `W` is shape `(in_features, out_features)`
- `b` is shape `(out_features,)`
- `y` is shape `(batch_size, out_features)`

### Implementation

```python
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        # Initialize weights with small random values
        # He initialization: σ = sqrt(2 / in_features)
        scale = np.sqrt(2.0 / in_features)
        weight_data = np.random.randn(in_features, out_features) * scale
        
        self._parameters['weight'] = Tensor(weight_data, requires_grad=True)
        
        if bias:
            self._parameters['bias'] = Tensor(np.zeros(out_features), requires_grad=True)
        else:
            self._parameters['bias'] = None
    
    def forward(self, x):
        # x: (..., in_features)
        # weight: (in_features, out_features)
        # bias: (out_features,)
        
        y = x @ self._parameters['weight']
        
        if self._parameters['bias'] is not None:
            y = y + self._parameters['bias']
        
        return y
```

**Why He initialization?** For ReLU activations, He init keeps variance stable:
```
Var(y) = Var(x) * Var(W) * in_features
```
If `Var(W) = 2 / in_features`, then `Var(y) = 2 * Var(x)`. This prevents exploding or vanishing gradients.

**Why bias initialized to zero?** Bias doesn't need careful initialization. Zero is fine because the gradient will quickly adjust it.

### Weight Initialization Options

```python
def init_weights(weight, method='he'):
    if method == 'he':
        scale = np.sqrt(2.0 / weight.data.shape[0])
    elif method == 'xavier':
        scale = np.sqrt(1.0 / weight.data.shape[0])
    elif method == 'gpt2':
        scale = 0.02
    else:
        scale = 1.0
    
    weight.data = np.random.randn(*weight.data.shape) * scale
    return weight
```

For transformers, we'll use GPT-2 style init: `σ = 0.02` with residual scaling.

---

## 4. Dropout

**File:** `scratch/nn.py`  
**Paper:** [Srivastava et al., 2014](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

### Why Dropout?

Dropout prevents overfitting by randomly "dropping out" (setting to zero) a fraction of neurons during training. This forces the network to learn redundant representations.

### The Math

During training:
```
mask = Bernoulli(p)  # probability p of keeping each neuron
y = x * mask / p     # scale to keep expected value constant
```

During inference:
```
y = x  # no dropout, no scaling
```

The scaling by `1/p` during training keeps the expected value the same:
```
E[y_train] = E[x * mask / p] = E[x] * p / p = E[x] = y_test
```

### Implementation

```python
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p  # probability of keeping a neuron
    
    def forward(self, x):
        if not self.training or self.p == 1.0:
            return x
        
        # Generate mask: keep with probability p
        mask = np.random.rand(*x.data.shape) < self.p
        
        # Scale to keep expected value constant
        scaled_x = x.data * mask / self.p
        
        out = Tensor(scaled_x, (x,), 'dropout')
        
        def _backward():
            if x.requires_grad:
                # Gradient is scaled by the same mask
                x.grad += out.grad * mask / self.p
        
        out._backward = _backward
        return out
```

**Why the backward pass?** The mask is applied during forward, so the gradient needs to flow through the same mask.

**Important:** The mask changes every forward pass during training. This is what makes Dropout work — different subsets of neurons are dropped each batch.

### Testing Dropout

```python
def test_dropout_training():
    # During training, some values should be zero
    x = Tensor(np.ones((10, 10)))
    dropout = Dropout(p=0.5)
    dropout.train()
    y = dropout(x)
    assert np.sum(y.data == 0) > 0  # Some zeros
    assert np.sum(y.data == 0) < 100  # Not all zeros
    assert np.mean(y.data) ≈ 1.0  # Expected value preserved

def test_dropout_eval():
    # During evaluation, no dropout
    x = Tensor(np.ones((10, 10)))
    dropout = Dropout(p=0.5)
    dropout.eval()
    y = dropout(x)
    assert np.all(y.data == 1.0)  # Identity
```

---

## 5. Layer Normalization

**File:** `scratch/nn.py`  
**Paper:** [Ba, Kiros, & Hinton, 2016](https://arxiv.org/abs/1607.06450)

### Why Normalize?

Without normalization, activations can grow or shrink exponentially as they pass through many layers. LayerNorm keeps them in a consistent range, which stabilizes training.

### The Math

LayerNorm normalizes across the feature dimension (last axis):
```
μ = mean(x, axis=-1, keepdims=True)
σ² = var(x, axis=-1, keepdims=True)
x_norm = (x - μ) / sqrt(σ² + ε)
y = γ * x_norm + β
```

where:
- `γ` (gamma) is a learnable scale parameter
- `β` (beta) is a learnable shift parameter
- `ε` is a small constant for numerical stability

### Implementation

```python
class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self._parameters['gamma'] = Tensor(np.ones(normalized_shape))
        self._parameters['beta'] = Tensor(np.zeros(normalized_shape))
    
    def forward(self, x):
        gamma = self._parameters['gamma']
        beta = self._parameters['beta']
        
        # Mean and variance across the last dimension
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        out_data = gamma.data * x_norm + beta.data
        
        # Build computational graph
        out = Tensor(out_data, (x, gamma, beta), 'layernorm')
        
        def _backward():
            if any(t.requires_grad for t in [x, gamma, beta]):
                # We need to compute gradients through mean and var
                N = x.data.shape[-1]
                
                # Gradient of x_norm
                grad_x_norm = out.grad * gamma.data
                
                # Gradient through variance
                grad_var = np.sum(
                    grad_x_norm * (x.data - mean) * -0.5 * (var + self.eps)**-1.5,
                    axis=-1, keepdims=True
                )
                
                # Gradient through mean
                grad_mean = np.sum(
                    grad_x_norm * -1 / np.sqrt(var + self.eps),
                    axis=-1, keepdims=True
                )
                grad_mean += grad_var * np.mean(-2 * (x.data - mean), axis=-1, keepdims=True)
                
                # Gradient through x
                if x.requires_grad:
                    grad_x = (
                        grad_x_norm / np.sqrt(var + self.eps)
                        + grad_var * 2 * (x.data - mean) / N
                        + grad_mean / N
                    )
                    x.grad += grad_x
                
                # Gradient through gamma
                if gamma.requires_grad:
                    gamma.grad += np.sum(out.grad * x_norm, axis=-1, keepdims=False)
                
                # Gradient through beta
                if beta.requires_grad:
                    beta.grad += np.sum(out.grad, axis=-1, keepdims=False)
        
        out._backward = _backward
        return out
```

**Why this is tricky:** LayerNorm's gradient requires propagating through mean and variance, which depend on all elements in the feature dimension. The chain rule gives us three terms:
1. Gradient through normalized values
2. Gradient through variance
3. Gradient through mean

Each term requires careful summation across the feature dimension.

### Pre-LayerNorm vs Post-LayerNorm

In transformers, we use **Pre-LayerNorm**:
```
x → LayerNorm → Attention → + x → LayerNorm → FFN → + x
```

Instead of Post-LayerNorm:
```
x → Attention → + x → LayerNorm → FFN → + x → LayerNorm
```

**Why Pre-LayerNorm?** The residual path `x` stays clean (no normalization), so gradients flow directly back to early layers. This makes training more stable.

---

## 6. Embedding Layer

**File:** `scratch/nn.py`

### What It Does

The embedding layer maps discrete token IDs to continuous vectors:
```
embedding = lookup_table[token_ids]
```

Where:
- `lookup_table` is shape `(vocab_size, embedding_dim)`
- `token_ids` is shape `(batch_size, seq_len)`
- `embedding` is shape `(batch_size, seq_len, embedding_dim)`

### Implementation

```python
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize with small random values
        scale = np.sqrt(1.0 / num_embeddings)
        weight_data = np.random.randn(num_embeddings, embedding_dim) * scale
        
        self._parameters['weight'] = Tensor(weight_data, requires_grad=True)
    
    def forward(self, indices):
        # indices: (...,) integer tensor
        # weight: (num_embeddings, embedding_dim)
        
        # Convert indices to numpy if they're a Tensor
        if isinstance(indices, Tensor):
            idx_data = indices.data.astype(np.int64)
        else:
            idx_data = np.array(indices, dtype=np.int64)
        
        # Check bounds
        if np.any(idx_data >= self.num_embeddings) or np.any(idx_data < 0):
            raise ValueError(f"Index out of bounds. Max: {self.num_embeddings-1}")
        
        # Lookup
        out_data = self._parameters['weight'].data[idx_data]
        
        out = Tensor(out_data, (self._parameters['weight'],), 'embedding')
        
        def _backward():
            if self._parameters['weight'].requires_grad:
                # Gradient is accumulated at the indices used
                grad = np.zeros_like(self._parameters['weight'].data)
                np.add.at(grad, idx_data, out.grad)
                self._parameters['weight'].grad += grad
        
        out._backward = _backward
        return out
```

**Why `np.add.at`?** Some indices may appear multiple times in the batch. `np.add.at` accumulates gradients at those positions.

**Why no backward for indices?** Indices are discrete inputs; they don't need gradients. We don't track them in the computational graph.

---

## 7. Loss Functions

**File:** `scratch/loss.py`

### Cross-Entropy Loss

**The most important loss function for classification.**

#### The Math

For a model predicting class probabilities, cross-entropy is:
```
loss = -log(p_correct)
```

where `p_correct` is the predicted probability of the correct class.

With softmax (we assume logits are raw scores before softmax):
```
p_i = exp(logits_i) / sum_j exp(logits_j)
loss = -log(p_target)
```

#### Numerically Stable Implementation

Naively computing softmax can overflow:
```
exp(1000) / (exp(1000) + exp(999))  # exp(1000) is inf
```

We avoid this by subtracting the maximum logit:
```
logits_stable = logits - max(logits)
p_i = exp(logits_stable_i) / sum_j exp(logits_stable_j)
```

This keeps values in a safe range.

#### Implementation

```python
def cross_entropy(logits, targets, reduction='mean'):
    """
    Cross-entropy loss with built-in softmax.
    
    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        targets: Tensor of shape (batch_size,) with integer labels
        reduction: 'mean' or 'sum'
    """
    # Forward pass with stable softmax
    max_logits = np.max(logits.data, axis=-1, keepdims=True)
    exp_logits = np.exp(logits.data - max_logits)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
    softmax = exp_logits / sum_exp
    
    # Cross-entropy: -log(softmax[target])
    batch_size = logits.data.shape[0]
    targets_data = targets.data.astype(np.int64)
    
    # Negative log probability of correct class
    loss_values = -np.log(softmax[np.arange(batch_size), targets_data] + 1e-8)
    
    if reduction == 'mean':
        loss_data = np.mean(loss_values)
    else:  # 'sum'
        loss_data = np.sum(loss_values)
    
    out = Tensor(loss_data, (logits,), 'cross_entropy')
    
    def _backward():
        if logits.requires_grad:
            # Gradient: softmax - one_hot
            grad = softmax.copy()
            grad[np.arange(batch_size), targets_data] -= 1.0
            
            if reduction == 'mean':
                grad = grad / batch_size
            
            logits.grad += out.grad * grad
    
    out._backward = _backward
    return out
```

**Why the gradient is `softmax - one_hot`:** This is the key insight. The derivative of cross-entropy with softmax simplifies beautifully:
```
dL/dlogits = softmax - one_hot
```

This is why we combine softmax and cross-entropy into one function — the gradient is simpler and more stable.

### Mean Squared Error (MSE) Loss

```python
def mse_loss(pred, target, reduction='mean'):
    """
    Mean squared error loss.
    """
    diff = pred.data - target.data
    squared = diff ** 2
    
    if reduction == 'mean':
        loss_data = np.mean(squared)
    else:  # 'sum'
        loss_data = np.sum(squared)
    
    out = Tensor(loss_data, (pred,), 'mse')
    
    def _backward():
        if pred.requires_grad:
            # dL/dpred = 2 * (pred - target) / N
            grad = 2 * diff
            if reduction == 'mean':
                grad = grad / pred.data.size
            pred.grad += out.grad * grad
    
    out._backward = _backward
    return out
```

---

## 8. Optimizers

**File:** `scratch/optim.py`

### SGD (Stochastic Gradient Descent)

```python
class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.velocities = [np.zeros_like(p.data) for p in parameters]
    
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad = np.zeros_like(p.grad)
    
    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            grad = p.grad.copy()
            
            # Weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad += self.weight_decay * p.data
            
            # Momentum
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
                p.data += self.velocities[i]
            else:
                p.data -= self.lr * grad
```

**Why momentum?** Momentum helps SGD escape local minima and speeds up convergence in shallow directions.

### Adam

**Paper:** [Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)

Adam (Adaptive Moment Estimation) combines momentum with adaptive learning rates.

```python
class Adam:
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [np.zeros_like(p.data) for p in parameters]  # First moment
        self.v = [np.zeros_like(p.data) for p in parameters]  # Second moment
        self.t = 0  # Timestep
    
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad = np.zeros_like(p.grad)
    
    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            grad = p.grad.copy()
            
            # Weight decay
            if self.weight_decay > 0:
                grad += self.weight_decay * p.data
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

**Why Adam works:**
- **Momentum (first moment):** Smooths gradients and accelerates convergence
- **Adaptive learning rates (second moment):** Per-parameter learning rates adapt to gradient magnitude
- **Bias correction:** Corrects for initialization bias in early steps

**Why the two moments?** The first moment tracks the average gradient (direction), the second tracks variance (step size). This gives Adam its adaptive behavior.

### Gradient Clipping

```python
def clip_grad_norm_(parameters, max_norm=1.0):
    """
    Clip gradients to prevent exploding gradients.
    
    Returns:
        total_norm: The norm of all gradients before clipping
    """
    total_norm = 0.0
    
    # Compute total norm
    for p in parameters:
        if p.grad is not None:
            total_norm += np.sum(p.grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    # Clip
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad *= clip_coef
    
    return total_norm
```

**Why clip gradients?** Gradients can explode in deep networks (especially transformers), causing instability. Clipping keeps them in a safe range.

---

## 9. Parameter Management

### Parameter Collection

```python
def get_parameters(model):
    """Collect all parameters from a model."""
    if hasattr(model, 'parameters'):
        return model.parameters()
    else:
        return []
```

### Parameter Count

```python
def count_parameters(model):
    """Count total number of trainable parameters."""
    total = 0
    for p in model.parameters():
        total += p.data.size
    return total
```

### Parameter Saving/Loading

```python
def save_checkpoint(model, optimizer, step, loss, path):
    """Save model parameters and optimizer state."""
    checkpoint = {
        'step': step,
        'loss': loss,
        'model_params': [(p.data.copy(), p.grad.copy()) for p in model.parameters()],
        'optimizer_state': {
            'm': [m.copy() for m in optimizer.m],
            'v': [v.copy() for v in optimizer.v],
            't': optimizer.t
        }
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(model, optimizer, path):
    """Load model parameters and optimizer state."""
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    for p, (data, grad) in zip(model.parameters(), checkpoint['model_params']):
        p.data = data
        p.grad = grad
    
    optimizer.m = checkpoint['optimizer_state']['m']
    optimizer.v = checkpoint['optimizer_state']['v']
    optimizer.t = checkpoint['optimizer_state']['t']
    
    return checkpoint['step'], checkpoint['loss']
```

---

## 10. Design Decisions at a Glance

| Component | Our Choice | Alternative | Reason |
|---|---|---|---|
| Parameter storage | Dictionary on Module | List, separate container | Easy to recursively collect all params |
| LayerNorm position | Pre-norm | Post-norm | More stable gradients |
| Dropout scaling | Inverted dropout | Naive dropout | Keeps expected value constant |
| Embedding init | Uniform(0, 1/vocab) | Random normal | Prevents initial overconfidence |
| Cross-entropy | Logits + softmax combined | Separate softmax then CE | Numerically stable, simpler gradient |
| Loss reduction | Mean | Sum | Independent of batch size |
| Optimizer | Adam | SGD | More robust, works with fewer hyperparameters |
| Gradient clipping | Always applied | Only when needed | Prevents explosions, little downside |

---

## 11. Test Suite

**File:** `tests/test_nn.py`

| Test | What It Verifies |
|---|---|
| `test_linear_forward` | Linear layer outputs correct shape and values |
| `test_linear_backward` | Gradients through linear layer are correct |
| `test_linear_parameters` | Linear layer has correct number of parameters |
| `test_dropout_training` | Dropout zeros some values during training |
| `test_dropout_eval` | Dropout is identity during eval |
| `test_dropout_backward` | Gradients through dropout are scaled correctly |
| `test_layernorm_forward` | LayerNorm outputs zero mean, unit variance |
| `test_layernorm_backward` | Gradients through LayerNorm are correct |
| `test_layernorm_parameters` | Gamma and beta are learned |
| `test_embedding_forward` | Embedding lookup works correctly |
| `test_embedding_backward` | Gradients accumulate at correct indices |
| `test_cross_entropy_forward` | Loss matches expected value |
| `test_cross_entropy_backward` | Gradient is softmax - one_hot |
| `test_mse_forward` | MSE matches expected value |
| `test_mse_backward` | Gradient is 2*(pred - target)/N |
| `test_sgd_step` | SGD updates weights correctly |
| `test_adam_step` | Adam updates weights correctly |
| `test_gradient_clipping` | Gradients are scaled when norm exceeds threshold |
| `test_parameter_collection` | All parameters are collected correctly |
| `test_save_load_checkpoint` | Checkpoint saves and loads correctly |

### Test Example: Cross-Entropy Gradient

```python
def test_cross_entropy_gradient():
    # Create logits with a clear correct answer
    logits = Tensor(np.array([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0]
    ]))
    targets = Tensor(np.array([2, 0]))
    
    # Forward pass
    loss = cross_entropy(logits, targets)
    
    # Backward pass
    loss.backward()
    
    # Expected gradient: softmax - one_hot
    softmax = np.exp(logits.data) / np.sum(np.exp(logits.data), axis=-1, keepdims=True)
    one_hot = np.eye(3)[targets.data]
    expected = (softmax - one_hot) / 2  # mean reduction
    
    assert np.allclose(logits.grad, expected, atol=1e-6)
```

---
