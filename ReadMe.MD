
---

# Building Backpropagation from Scratch: My Journey from Zero to a Working GPT

This project started with a simple question: *"What actually happens when I call `.backward()` in PyTorch?"*

I'd been using deep learning frameworks for years, but the internals remained a black box. So I decided to build everything from scratch—the tensor operations, the computational graph, the chain rule propagation, and eventually a complete transformer model. No PyTorch, no TensorFlow. Just Python, NumPy, and a lot of debugging.

What followed was 200+ hours of wrestling with matrix calculus, debugging gradient shapes, and having multiple "aha" moments that completely changed how I think about deep learning.

---

## The Core Insight That Changed Everything

After implementing the autograd engine, I finally understood something fundamental:

**Backpropagation isn't an algorithm for updating weights. It's an algorithm for efficiently reusing intermediate calculations.**

Here's what I mean: If you have 50 million parameters and you want to update them using gradient descent, you need the gradient of the loss with respect to each parameter. Without backprop, you'd need to run the entire neural network forward 50 million times—once for each parameter—to compute each gradient separately.

With backprop, you run the network forward **once**, cache all the intermediate activations, and then run backward **once**, reusing those cached values to compute all 50 million gradients simultaneously. This is the efficiency hack that makes modern deep learning possible.

---

## What This Project Actually Does

I built a complete automatic differentiation (autograd) engine from scratch, then used it to train a character-level GPT model on Shakespeare's works. The entire pipeline—from tensor operations to the transformer architecture—is implemented without any deep learning frameworks.

After completing the scratch implementation, I ported the exact same model to PyTorch to verify mathematical correctness and understand the performance benefits of industrial-grade autograd.

---

## Project Structure

Here's how I organized everything:

```
autodiff-gpt/
│
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── LICENSE                            # MIT License
│
├── scratch/                           # From-scratch implementation
│   ├── __init__.py
│   ├── tensor.py                      # Tensor class with autograd
│   ├── operations.py                  # All tensor operations (matmul, add, etc.)
│   ├── nn.py                          # Neural network layers (Linear, LayerNorm, Dropout)
│   ├── optim.py                       # SGD, Adam optimizers from scratch
│   ├── loss.py                        # Cross-entropy, MSE loss functions
│   ├── transformer.py                 # GPT model architecture
│   └── train.py                       # Training loop with data loading
│
├── pytorch_migration/                 # PyTorch equivalent for verification
│   ├── __init__.py
│   ├── model.py                       # Same model implemented in PyTorch
│   └── verify.py                      # Verification suite comparing gradients and losses
│
├── debugging/                         # Gradient debugging tools I built
│   ├── __init__.py
│   ├── gradient_checking.py          # Numerical gradient checking against analytical
│   ├── gradient_clipping.py          # Clip gradient norms to prevent exploding
│   ├── saliency_maps.py              # Input attribution using backprop
│   ├── gradient_flow.py              # Track layer-by-layer gradient magnitudes
│   └── visualization.py              # Plot gradient flows and loss curves
│
├── data/                              # Dataset preparation
│   ├── shakespeare.txt               # Training data (complete works)
│   └── prepare_data.py               # Data preprocessing and tokenization
│
├── experiments/                       # All experiment results
│   ├── losses/
│   │   ├── scratch_losses.npy        # Loss values from my implementation
│   │   └── pytorch_losses.npy        # Loss values from PyTorch for comparison
│   ├── gradients/
│   │   └── gradient_flow_plots/      # Visualizations of gradient flow
│   └── benchmarks/
│       └── performance_metrics.json  # Speed and memory benchmarks
│
└── notebooks/                         # Jupyter notebooks for exploration
    ├── 01_tensor_basics.ipynb        # Understanding tensor operations
    ├── 02_backprop_visualized.ipynb  # Visualizing gradient flow through the graph
    └── 03_model_comparison.ipynb     # Scratch vs PyTorch comparison
```

---

## When I Actually Use Backpropagation

Throughout this project, I developed a clear mental model of when backpropagation is needed:

**I use backpropagation during:**
- **Every training step** - After the forward pass computes the loss, backprop calculates gradients for weight updates
- **Gradient clipping** - To inspect gradient magnitudes and scale them if they're exploding
- **Debugging** - To check if gradients are vanishing (near zero) or exploding (NaN/Inf values)
- **Interpretability work** - To generate saliency maps showing which input features most influence the output
- **Gradient checking** - To validate that my custom gradient implementations are correct

**I never use backpropagation during:**
- **Inference or prediction** - When I'm just feeding data forward to get outputs
- **Model evaluation** - Computing validation metrics doesn't require gradients

The rule is simple: **If I need gradients of a loss with respect to parameters, I use backpropagation. If I'm just doing forward computation, I don't.**

---

## The Architecture I Built

### The Autograd Engine

At the heart of everything is a `Tensor` class that tracks operations in a computational graph:

```python
class Tensor:
    def __init__(self, data, children=(), op='', requires_grad=True):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._prev = set(children)  # Nodes this tensor depends on
        self._op = op              # Operation that created this tensor
        self._backward = lambda: None  # Local gradient function
        self.requires_grad = requires_grad
    
    def backward(self):
        # Topologically sort the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # Initialize gradient at output (dy/dy = 1)
        self.grad = np.ones_like(self.data)
        
        # Apply chain rule in reverse order
        for node in reversed(topo):
            node._backward()
```

Every mathematical operation—addition, multiplication, matrix multiplication, exponentiation, etc.—implements its own `_backward` function that computes local gradients using the chain rule.

### The GPT Model

I built a transformer-based language model with:
- **~10 million parameters** (configurable)
- **12 transformer layers** with multi-head self-attention
- **8 attention heads** per layer
- **Context window** of 512 tokens
- **Embedding dimension** of 384
- **Dropout** and **layer normalization** for regularization

The model takes a sequence of characters, predicts the next character at each position, and is trained on Shakespeare's complete works.

---

## My Implementation Journey

### Phase 1: Building the Tensor Foundation

I started with the simplest operations—addition and multiplication—and built their backward passes. This taught me the fundamental pattern:

```python
def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
    out = Tensor(self.data + other.data, (self, other), '+')
    
    def _backward():
        # Gradient of addition is just passed through
        if self.requires_grad:
            self.grad += out.grad * 1.0
        if other.requires_grad:
            other.grad += out.grad * 1.0
    out._backward = _backward
    return out
```

The tricky part came with matrix multiplication:

```python
def __matmul__(self, other):
    # Gradient of X @ W is:
    # dL/dX = dL/dY @ W.T
    # dL/dW = X.T @ dL/dY
    def _backward():
        if self.requires_grad:
            self.grad += out.grad @ other.data.T
        if other.requires_grad:
            other.grad += self.data.T @ out.grad
```

Getting the transpose order wrong caused my gradients to explode for two days before I caught it.

### Phase 2: Building Neural Network Layers

With the tensor operations working, I built standard neural network components:

- **Linear layers** - Matrix multiplication plus bias
- **Layer normalization** - With trainable gamma and beta parameters
- **Dropout** - With scaling during training
- **Activation functions** - ReLU, GELU, tanh

Layer normalization's backward pass was particularly interesting because it required computing gradients through the mean and variance calculations:

```python
def layer_norm_backward(grad_output, x, gamma, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_centered = x - mean
    x_norm = x_centered / np.sqrt(var + eps)
    
    # Gradient through normalization
    grad_norm = grad_output * gamma
    grad_var = np.sum(grad_norm * x_centered * -0.5 * (var + eps)**-1.5, axis=-1, keepdims=True)
    grad_mean = np.sum(grad_norm * -1 / np.sqrt(var + eps), axis=-1, keepdims=True)
    grad_mean += grad_var * np.mean(-2 * x_centered, axis=-1, keepdims=True)
    
    grad_x = grad_norm / np.sqrt(var + eps) + grad_var * 2 * x_centered / x.shape[-1] + grad_mean / x.shape[-1]
    return grad_x
```

This took several hours to derive correctly on paper before implementing.

### Phase 3: Building the Transformer

The most complex part was implementing multi-head self-attention with proper gradient flow:

```python
class MultiHeadAttention:
    def __init__(self, n_embd, n_head, dropout=0.0):
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head
        
        # Projections
        self.W_q = Tensor(np.random.randn(n_embd, n_embd) * 0.02)
        self.W_k = Tensor(np.random.randn(n_embd, n_embd) * 0.02)
        self.W_v = Tensor(np.random.randn(n_embd, n_embd) * 0.02)
        self.W_o = Tensor(np.random.randn(n_embd, n_embd) * 0.02)
    
    def forward(self, x):
        # Q, K, V projections
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Split into heads and compute attention
        Q = Q.reshape(..., self.n_head, self.head_size)
        K = K.reshape(..., self.n_head, self.head_size)
        V = V.reshape(..., self.n_head, self.head_size)
        
        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.head_size)
        # Mask future positions
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        scores = scores + (mask * -1e9)
        attn = softmax(scores, axis=-1)
        
        # Apply attention to values
        out = attn @ V
        out = out.reshape(..., self.n_embd)
        return out @ self.W_o
```

The backward pass for attention required careful handling of the mask and softmax gradients.

### Phase 4: Training the Model

Training the full model was the moment of truth. I implemented:

- **Cross-entropy loss** with built-in softmax
- **Adam optimizer** from scratch
- **Gradient clipping** to prevent exploding gradients
- **Learning rate scheduling** with cosine decay and warmup

The training loop was straightforward:

```python
for step in range(num_steps):
    # Sample a batch
    x, y = get_batch(data, batch_size, block_size)
    
    # Forward pass
    logits = model(x)
    loss = cross_entropy(logits, y)
    
    # Backward pass (THIS IS BACKPROPAGATION)
    optimizer.zero_grad()
    loss.backward()
    
    # Optional: clip gradients
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update weights
    optimizer.step()
```

The first time I saw the loss decrease steadily, I knew everything was working.

### Phase 5: Verification with PyTorch

To prove mathematical correctness, I ported the exact same model to PyTorch and compared:

- **Loss values** - Should match within 1e-4
- **Gradient values** - Should match within 1e-6 for each parameter
- **Model outputs** - Should be identical for the same input

The verification process caught several subtle bugs:

1. **Broadcasting in bias addition** - The gradient for bias needed to sum across the batch dimension
2. **Softmax cross-entropy gradient** - It's `softmax - one_hot`, not `softmax * (1 - one_hot)`
3. **Layer norm gradient** - Forgot to normalize the gradient by the batch size

Once these were fixed, the loss curves overlapped perfectly.

---

## What I Learned

### 1. Matrix Calculus is the Real Challenge

The math behind backpropagation is just the chain rule, but applying it to matrices requires careful attention to dimensions. I spent hours deriving gradients on paper before writing code. The most common mistakes were:

- Forgetting to transpose matrices in matmul backward
- Mishandling broadcasting when adding biases
- Incorrect normalization of gradients for mean and variance operations

### 2. Gradient Checking is Essential

Every new operation I added was verified with numerical gradient checking:

```python
def check_gradient(param, loss_fn, eps=1e-5):
    # Analytical gradient
    loss_fn()
    analytical = param.grad.copy()
    
    # Numerical gradient
    param.zero_grad()
    original = param.data.copy()
    
    param.data = original + eps
    loss_plus = loss_fn()
    
    param.data = original - eps
    loss_minus = loss_fn()
    
    numerical = (loss_plus - loss_minus) / (2 * eps)
    
    # Should be within 1e-6
    return np.abs(analytical - numerical)
```

This caught bugs early and saved countless hours of debugging.

### 3. Memory Management is a Trade-off

The naive approach stores every intermediate tensor for the backward pass. For a 10 million parameter model, this meant storing:

- All activations from the forward pass
- All gradients from the backward pass
- About 2.1 GB of total memory

I could have traded memory for recomputation (recomputing activations during backward), but that would have been slower. There's always a trade-off between speed and memory.

### 4. Understanding Gradients is Powerful

Once I understood how gradients flow through the network, I could:

- Diagnose vanishing gradients by inspecting layer-by-layer gradient norms
- Identify which parts of the model were learning the most
- Debug training instability by checking for gradient spikes
- Generate saliency maps to understand what the model was paying attention to

---

## Performance Comparison: Scratch vs PyTorch

| Operation | My Implementation | PyTorch (CUDA) | Speedup |
|-----------|------------------|----------------|---------|
| Forward Pass | 45 ms | 2.1 ms | 21.4x |
| Backward Pass | 89 ms | 3.8 ms | 23.4x |
| Full Training Step | 134 ms | 5.9 ms | 22.7x |

PyTorch's C++ backend and CUDA optimizations make it significantly faster, but the scratch implementation proved mathematical correctness. The difference isn't in the algorithm—it's in the engineering.

---

## How to Use This Project

### Installation

```bash
git clone https://github.com/Tawanda21/autodiff-gpt.git
cd autodiff-gpt
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Training the Model

```bash
python scratch/train.py \
    --dataset data/shakespeare.txt \
    --block_size 256 \
    --n_embd 384 \
    --n_head 6 \
    --n_layer 6 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --num_steps 100000
```

### Verifying Against PyTorch

```bash
python pytorch_migration/verify.py \
    --check_gradients \
    --compare_losses \
    --visualize
```

### Debugging Gradients

```bash
# Check if your gradients are correct numerically
python debugging/gradient_checking.py --model_path checkpoints/model.pkl

# Generate saliency maps for interpretability
python debugging/saliency_maps.py --input "ROMEO:" --target_token 42

# Visualize gradient flow through layers
python debugging/gradient_flow.py --model_path checkpoints/model.pkl
```

### Running Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

---

## What I'd Do Differently

Looking back, there are a few things I'd change:

1. **Start with a smaller model** - I jumped into transformers too quickly. I should have started with a simple MLP on MNIST to validate the autograd engine first.

2. **Better test coverage** - I wrote tests after the fact. Next time, I'd write them before implementing each operation.

3. **Documentation for each operation** - I understood the math when I wrote it, but three weeks later, I had to re-derive some gradients. I should have documented the derivations.

4. **Profile earlier** - I waited until the end to optimize performance. Profiling early would have identified bottlenecks sooner.

---

## Future Plans

I'm continuing to build on this project:

- **Mixed precision training** - Implementing FP16 support for the autograd engine
- **JAX-style forward-mode autodiff** - Computing Jacobian-vector products
- **Graph optimizations** - Removing unnecessary nodes from the computational graph
- **Sparse gradients** - Only computing gradients for non-zero parameters
- **Hessian-vector products** - For second-order optimization

---

## Closing Thoughts

This project was the single best learning experience of my ML journey. I finally understand:

- **What's actually happening** when I call `.backward()` in PyTorch
- **Why gradient clipping is necessary** for transformers
- **How to debug training issues** by inspecting gradients
- **When to use backpropagation** and when it's unnecessary

If you're serious about deep learning, I can't recommend building this enough. The theory only goes so far—implementing it forces you to confront every assumption you've made.

---

## Acknowledgments

- **Andrej Karpathy** - His nanoGPT project inspired this work and showed me what's possible with a clean implementation
- **CS231n** - The autograd lecture finally made the chain rule click for me
- **The PyTorch team** - For building the gold standard that I used as a reference

---

## Let's Connect

I'm always happy to discuss this project or answer questions about the implementation. Reach out if you want to:

- Ask about specific gradient derivations
- Discuss architectural decisions
- Share your own autograd implementation
- Collaborate on extending this work

**GitHub**: [@Tawanda21](https://github.com/Tawanda21)  
**LinkedIn**: [Tawanda Mudonhi](https://www.linkedin.com/in/tawanda-mudonhi-5005b5203)

---

## License

MIT License - feel free to use, modify, and distribute.

---

**If you found this helpful, please consider starring the repository. It helps others find it and gives me motivation to continue improving it.**