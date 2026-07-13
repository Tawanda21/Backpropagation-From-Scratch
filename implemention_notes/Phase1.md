
---

# Phase 1 — The Tensor Foundation

**Goal:** Build the core Tensor class with automatic differentiation

A bottom-up walkthrough of every component we built, what the math means, and why each design decision was made.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [The Tensor Class](#2-the-tensor-class)
3. [Basic Operations (Addition & Multiplication)](#3-basic-operations-addition--multiplication)
4. [The Backward Pass & Topological Sort](#4-the-backward-pass--topological-sort)
5. [Matrix Multiplication](#5-matrix-multiplication)
6. [Reduction Operations (Sum & Mean)](#6-reduction-operations-sum--mean)
7. [Activation Functions](#7-activation-functions)
8. [Reshape & Transpose](#8-reshape--transpose)
9. [Broadcasting](#9-broadcasting)
10. [Gradient Checking](#10-gradient-checking)
11. [Design Decisions at a Glance](#11-design-decisions-at-a-glance)
12. [Test Suite](#12-test-suite)

---

## 1. The Big Picture

Automatic differentiation (autograd) does one thing: **given a computational graph, compute gradients of a scalar output with respect to every input that requires gradients.**

The data flows like this:

```
Forward Pass
    │
    ▼
Build Computational Graph
    │
    ▼
Loss (scalar Tensor)
    │
    ▼
Backward Pass (loss.backward())
    │
    ▼
Topological Sort → Reverse Order
    │
    ▼
Apply Chain Rule at Each Node
    │
    ▼
Gradients Accumulate in Every Tensor's .grad
```

> **Key insight:** The forward pass builds the graph. The backward pass traverses it in reverse. Each node stores a `_backward` function that computes local gradients using the chain rule. This is the *entire* algorithm — everything else is implementation detail.

### The Chain Rule in Code

For a composition `y = f(g(x))`:

```
dy/dx = dy/dg * dg/dx
```

In our graph, each node's `_backward` function receives `dy/dout` (gradient of loss with respect to this node's output) and computes `dy/din` (gradient with respect to its inputs) using:

```
dy/din = dy/dout * dout/din
```

The `_backward` function then adds (accumulates) this gradient to each input's `.grad` attribute.

---

## 2. The Tensor Class

**File:** `scratch/tensor.py`

### The Core Data Structure

```python
class Tensor:
    def __init__(self, data, children=(), op='', requires_grad=True):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._prev = set(children)
        self._op = op
        self._backward = lambda: None
        self.requires_grad = requires_grad
```

| Attribute | Type | Purpose |
|---|---|---|
| `data` | `np.ndarray` | The actual numbers (forward pass values) |
| `grad` | `np.ndarray` | Gradients of the loss w.r.t this tensor (backward pass) |
| `_prev` | `set[Tensor]` | Parents in the computational graph (what created this) |
| `_op` | `str` | Operation that created this (for debugging) |
| `_backward` | `Callable` | Local gradient function — computes gradients for parents |
| `requires_grad` | `bool` | Whether we need gradients for this tensor |

**Why NumPy?** We're building from scratch. PyTorch would defeat the point. NumPy gives us fast array operations without doing autograd for us. We'll handle the gradients ourselves.

**Why `_prev` as a set?** A tensor can be used in multiple operations (e.g., `x` used in both `y = x + 1` and `z = x * 2`). A set prevents duplicate parents.

**Why gradients initialized to zeros?** Gradients accumulate. If a tensor is used multiple times, we need to add all contributions. Starting from zeros lets us accumulate safely.

**Why `_backward = lambda: None`?** Every tensor needs a `_backward` function. Leaf tensors (created by the user, not by an operation) have no-op backward functions. This is cleaner than conditional checks.

### The Forward-Only Path

If a tensor doesn't need gradients, we can skip all tracking:

```python
def __init__(self, data, requires_grad=True):
    ...
    if not requires_grad:
        self.grad = None
        self._prev = set()
        self._backward = lambda: None
```

This is important for efficiency. Operations like dropout or activations might not need gradients in inference mode.

---

## 3. Basic Operations (Addition & Multiplication)

**File:** `scratch/operations.py`

### Addition

```python
def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
    out = Tensor(self.data + other.data, (self, other), '+')
    
    def _backward():
        # Gradient of addition: dy/dx = 1, dy/dother = 1
        if self.requires_grad:
            self.grad += out.grad * 1.0
        if other.requires_grad:
            other.grad += out.grad * 1.0
    
    out._backward = _backward
    return out
```

**Math:**
```
y = x + z
dy/dx = 1
dy/dz = 1
```

The gradient of addition simply passes through unchanged.

**Why `+=` instead of `=`?** A tensor can appear in multiple operations:
```
y = x + 1
z = x * 2
loss = y + z
```
When we backprop through `loss`, `x` receives gradients from both `y` and `z`. We need to add them. If we used `=`, we'd overwrite.

**Why convert `other` to Tensor?** This allows `Tensor(5) + 3` to work. The scalar `3` becomes a Tensor with `requires_grad=False`.

### Multiplication

```python
def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
    out = Tensor(self.data * other.data, (self, other), '*')
    
    def _backward():
        if self.requires_grad:
            self.grad += out.grad * other.data
        if other.requires_grad:
            other.grad += out.grad * self.data
    
    out._backward = _backward
    return out
```

**Math:**
```
y = x * z
dy/dx = z
dy/dz = x
```

The gradient with respect to one input is the other input's value.

**Important:** The gradient is `out.grad * other.data`, not `out.grad * other.grad`. We're multiplying by the forward value, not the gradient. This is a common source of bugs.

### Subtraction & Negation

```python
def __neg__(self):
    return self * -1

def __sub__(self, other):
    return self + (-other)

def __rsub__(self, other):
    return other + (-self)
```

This reuses addition and multiplication, reducing code duplication.

### Power

```python
def __pow__(self, exponent):
    out = Tensor(self.data ** exponent, (self,), f'**{exponent}')
    
    def _backward():
        if self.requires_grad:
            self.grad += out.grad * exponent * (self.data ** (exponent - 1))
    
    out._backward = _backward
    return out
```

**Math:**
```
y = x^n
dy/dx = n * x^(n-1)
```

**Limitation:** This only works for constant exponents. For `x**x`, we'd need a more complex approach (not needed for our transformer).

---

## 4. The Backward Pass & Topological Sort

**File:** `scratch/tensor.py`

### Topological Sort

Before we can apply the chain rule, we need to order the nodes so that every node comes after its parents (forward order) or before its children (reverse order).

```python
def _topological_sort(self):
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    
    build_topo(self)
    return topo
```

**Why this works:** DFS visits children (parents in the graph) first, then appends the node. This means parents appear before children in `topo`. When reversed, children appear before parents — exactly what we need for backprop.

**Visual example:**
```
Graph:    a → b → c (where a and b are Tensors, c = b * a)
Forward:  a → b → c
Topo:     [a, b, c]  (parents before children)
Reversed: [c, b, a]  (children before parents)
```

### The Backward Method

```python
def backward(self):
    if self.data.size != 1:
        raise RuntimeError("backward can only be called on scalar tensors")
    
    topo = self._topological_sort()
    
    # Initialize gradient at output: dy/dy = 1
    self.grad = np.ones_like(self.data)
    
    # Apply chain rule in reverse order
    for node in reversed(topo):
        node._backward()
```

**Why only scalars?** The chain rule is defined for scalar outputs. For vector-valued functions, you'd compute Jacobians, which gets complex. In practice, we always have a scalar loss.

**Why set `self.grad = 1`?** The loss's gradient with respect to itself is 1 (dy/dy = 1). This seeds the backward pass.

**Why `reversed(topo)`?** `topo` gives parents before children. Reversed gives children before parents, so each node's parents have already had their gradients computed.

**Important:** We're modifying the graph in-place. This is intentional — gradients accumulate as tensors are reused.

---

## 5. Matrix Multiplication

**File:** `scratch/operations.py`

### The Forward Pass

```python
def __matmul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
    out = Tensor(self.data @ other.data, (self, other), 'matmul')
    
    def _backward():
        # dL/dself = dL/dout @ other.T
        if self.requires_grad:
            self.grad += out.grad @ other.data.T
        
        # dL/dother = self.T @ dL/dout
        if other.requires_grad:
            other.grad += self.data.T @ out.grad
    
    out._backward = _backward
    return out
```

### The Math

For matrix multiplication `Y = X @ W` where:
- `X` is shape `(B, M, K)`
- `W` is shape `(K, N)`
- `Y` is shape `(B, M, N)`

The gradients are:
```
dL/dX = dL/dY @ W.T     # shape: (B, M, K)
dL/dW = X.T @ dL/dY     # shape: (K, N)
```

**Intuition:** The gradient with respect to the first input is the output gradient multiplied by the transpose of the second input. The transpose "un-does" the multiplication.

**Why the transpose?** Matrix multiplication is: `Y[i,j] = sum_k X[i,k] * W[k,j]`. When computing `dL/dX[i,k]`, the gradient flows through all `j` where `X[i,k]` contributes to `Y[i,j]`. This sum is exactly `dL/dY @ W.T`.

### Common Mistake: Transpose Direction

```
❌ Wrong:  self.grad += out.grad @ other.data
✅ Right:  self.grad += out.grad @ other.data.T
```

If you get the transpose wrong, gradients will have the wrong shape and the model won't learn. This is the most common source of bugs in manual autograd.

### Shape Checking

```python
def __matmul__(self, other):
    # Check shapes match
    assert self.data.shape[-1] == other.data.shape[-2], \
        f"Shape mismatch: {self.data.shape} and {other.data.shape}"
    ...
```

This catches errors early and makes debugging easier.

---

## 6. Reduction Operations (Sum & Mean)

**File:** `scratch/operations.py`

### Sum

```python
def sum(self, axis=None, keepdims=False):
    out = Tensor(
        np.sum(self.data, axis=axis, keepdims=keepdims),
        (self,),
        f'sum(axis={axis})'
    )
    
    def _backward():
        if self.requires_grad:
            # Expand gradient back to original shape
            grad_expanded = np.expand_dims(out.grad, axis=axis)
            if keepdims:
                self.grad += grad_expanded
            else:
                # Need to expand to original shape
                self.grad += np.broadcast_to(grad_expanded, self.data.shape)
    
    out._backward = _backward
    return out
```

**Why expand?** Sum reduces dimensions. The gradient needs to have the same shape as the input, so we expand the gradient back to the original shape.

**Example:**
```
x = [[1, 2], [3, 4]]  # shape (2, 2)
y = sum(x, axis=1)    # shape (2,)
dy/dx = [[1, 1], [1, 1]]  # shape (2, 2)
```

Each output element gets gradient 1 for all input elements that contributed to it.

### Mean

```python
def mean(self, axis=None, keepdims=False):
    out = Tensor(
        np.mean(self.data, axis=axis, keepdims=keepdims),
        (self,),
        f'mean(axis={axis})'
    )
    
    def _backward():
        if self.requires_grad:
            # Gradient of mean is 1/N
            N = self.data.shape[axis] if axis is not None else self.data.size
            grad_expanded = np.expand_dims(out.grad / N, axis=axis)
            if keepdims:
                self.grad += grad_expanded
            else:
                self.grad += np.broadcast_to(grad_expanded, self.data.shape)
    
    out._backward = _backward
    return out
```

**Math:**
```
y = mean(x) = (1/N) * sum(x)
dy/dx = 1/N
```

The gradient is uniformly `1/N` for all inputs.

**Why this matters:** Cross-entropy loss uses mean for batch averaging. The `1/N` scaling ensures gradients don't grow with batch size.

---

## 7. Activation Functions

**File:** `scratch/operations.py`

### ReLU

```python
def relu(self):
    out = Tensor(np.maximum(0, self.data), (self,), 'relu')
    
    def _backward():
        if self.requires_grad:
            self.grad += out.grad * (self.data > 0)
    
    out._backward = _backward
    return out
```

**Math:**
```
y = max(0, x)
dy/dx = 1 if x > 0 else 0
```

**Why `self.data > 0`?** The gradient is 1 for positive inputs, 0 for negative ones. No gradient flows through dead ReLUs.

### GELU

```python
def gelu(self):
    # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x = self.data
    c = 0.044715
    tanh_arg = np.sqrt(2 / np.pi) * (x + c * x**3)
    out_data = 0.5 * x * (1 + np.tanh(tanh_arg))
    out = Tensor(out_data, (self,), 'gelu')
    
    def _backward():
        if self.requires_grad:
            # Derivative: 0.5 * (1 + tanh(tanh_arg)) + 0.5 * x * (1 - tanh(tanh_arg)^2) * derivative_of_tanh_arg
            tanh_val = np.tanh(tanh_arg)
            derivative_tanh_arg = np.sqrt(2 / np.pi) * (1 + 3 * c * x**2)
            grad = 0.5 * (1 + tanh_val) + 0.5 * x * (1 - tanh_val**2) * derivative_tanh_arg
            self.grad += out.grad * grad
    
    out._backward = _backward
    return out
```

**Why GELU?** It's smooth and differentiable everywhere (unlike ReLU). Used in transformers because it empirically works better.

**The derivative:** We need the chain rule:
```
d/dx [0.5 * x * (1 + tanh(f(x)))] 
= 0.5 * (1 + tanh(f(x))) + 0.5 * x * (1 - tanh(f(x))^2) * f'(x)
```

where `f(x) = sqrt(2/pi) * (x + 0.044715 * x^3)`.

### Tanh

```python
def tanh(self):
    out = Tensor(np.tanh(self.data), (self,), 'tanh')
    
    def _backward():
        if self.requires_grad:
            self.grad += out.grad * (1 - out.data**2)
    
    out._backward = _backward
    return out
```

**Math:**
```
y = tanh(x)
dy/dx = 1 - tanh^2(x) = 1 - y^2
```

We use `out.data` (the forward value) to compute the derivative. This is more efficient than recomputing `tanh(self.data)`.

---

## 8. Reshape & Transpose

**File:** `scratch/operations.py`

### Reshape

```python
def reshape(self, *shape):
    out = Tensor(self.data.reshape(*shape), (self,), f'reshape{shape}')
    
    def _backward():
        if self.requires_grad:
            # Gradient just flows through the reshape operation
            self.grad += out.grad.reshape(self.data.shape)
    
    out._backward = _backward
    return out
```

**Why this works:** Reshaping doesn't change the data, only the view. The gradient just needs to be reshaped back to the original shape.

### Transpose

```python
def transpose(self, axes=None):
    out = Tensor(np.transpose(self.data, axes), (self,), f'transpose{axes}')
    
    def _backward():
        if self.requires_grad:
            # Transpose gradient back to original order
            # If we transposed with axes, we need to invert the permutation
            if axes is None:
                inv_axes = tuple(range(len(self.data.shape)-1, -1, -1))
            else:
                inv_axes = [0] * len(axes)
                for i, a in enumerate(axes):
                    inv_axes[a] = i
            self.grad += np.transpose(out.grad, inv_axes)
    
    out._backward = _backward
    return out
```

**Why invert axes?** If we transposed once, the gradient needs to be transposed back to the original orientation.

---

## 9. Broadcasting

**File:** `scratch/operations.py`

### The Problem

NumPy broadcasting allows operations between tensors of different shapes:
```
x: (3, 1)  +  y: (1, 4)  →  out: (3, 4)
```

The gradient needs to be reduced (summed) along the broadcasted dimensions.

### Solution: Sum Across Broadcasted Dimensions

```python
def _broadcast_backward(grad, shape):
    """Reduce grad to match the shape."""
    # Calculate which dimensions to sum over
    grad_shape = list(grad.shape)
    shape = list(shape)
    
    # Pad shape to match grad_shape
    while len(shape) < len(grad_shape):
        shape.insert(0, 1)
    
    # Sum over dimensions where shape=1 but grad_shape>1
    sum_dims = [i for i in range(len(grad_shape)) if grad_shape[i] != shape[i]]
    
    if sum_dims:
        grad = np.sum(grad, axis=tuple(sum_dims), keepdims=False)
    
    # Remove leading dimensions of size 1
    while len(grad.shape) > len(shape) and grad.shape[0] == 1:
        grad = np.squeeze(grad, axis=0)
    
    return grad
```

**Example:**
```
x: (3, 1)  →  x: (3, 4)  (broadcasted along dimension 1)
grad from output: (3, 4)
grad for x: sum along dimension 1 → (3,)
```

We need to sum the gradient along the dimensions that were broadcasted.

### Using It

```python
def __add__(self, other):
    # ... forward pass ...
    
    def _backward():
        if self.requires_grad:
            self_grad = _broadcast_backward(out.grad, self.data.shape)
            self.grad += self_grad
        if other.requires_grad:
            other_grad = _broadcast_backward(out.grad, other.data.shape)
            other.grad += other_grad
    
    out._backward = _backward
    return out
```

This handles all broadcasting cases automatically.

---

## 10. Gradient Checking

**File:** `debugging/gradient_checking.py`

### The Method

Numerical gradient checking is the gold standard for verifying autograd implementations:

```python
def gradient_check(tensor, func, eps=1e-5, rtol=1e-5, atol=1e-6):
    """
    Check gradients of func(tensor) using numerical differentiation.
    
    Returns:
        relative_error: max relative error between analytical and numerical gradients
    """
    # Forward pass
    out = func(tensor)
    out.backward()
    analytical_grad = tensor.grad.copy()
    
    # Numerical gradient
    tensor.grad = None
    original_data = tensor.data.copy()
    
    numerical_grad = np.zeros_like(original_data)
    
    for i in range(original_data.size):
        # f(x + eps)
        tensor.data = original_data.copy()
        tensor.data.flat[i] += eps
        out_plus = func(tensor)
        
        # f(x - eps)
        tensor.data = original_data.copy()
        tensor.data.flat[i] -= eps
        out_minus = func(tensor)
        
        # (f(x+eps) - f(x-eps)) / (2*eps)
        numerical_grad.flat[i] = (out_plus.data - out_minus.data) / (2 * eps)
    
    # Restore
    tensor.data = original_data
    
    # Compare
    diff = np.abs(analytical_grad - numerical_grad)
    max_diff = np.max(diff)
    max_val = np.max(np.abs(analytical_grad) + np.abs(numerical_grad))
    relative_error = max_diff / max_val if max_val > 0 else 0
    
    assert relative_error < rtol, f"Gradient check failed: relative error {relative_error}"
    return relative_error
```

**Why `2*eps`?** Central difference is more accurate than forward or backward difference. The error is `O(eps^2)` instead of `O(eps)`.

**What's a good relative error?** For floating point, `1e-6` or less is excellent. `1e-4` is okay. `1e-2` means there's a bug.

### Testing a New Operation

```python
def test_matmul_gradient():
    # Create tensors
    x = Tensor(np.random.randn(3, 4))
    w = Tensor(np.random.randn(4, 5))
    
    # Check gradients
    def func(x):
        return (x @ w).sum()
    
    grad_check(x, func, eps=1e-5)
    grad_check(w, func, eps=1e-5)
```

Run this for every new operation. If gradient checking passes, your backprop is correct.

---

## 11. Design Decisions at a Glance

| Component | Our Choice | Alternative | Reason |
|---|---|---|---|
| Backend | NumPy | Pure Python, C++ | NumPy is fast and already vectorized |
| Gradient storage | `.grad` on Tensor | Separate graph | Simple, no extra data structures |
| Graph building | Dynamic (eager) | Static (compiled) | Easier to debug; we don't need performance yet |
| Gradient accumulation | `+=` | `=` | Handles shared tensors correctly |
| Backward function | Stored on Tensor | Stored in graph | Each node knows its own backward |
| Topological sort | DFS | Kahn's algorithm | Simple, works for DAGs |
| Broadcasting | NumPy automatic | Manual | NumPy handles it, we just reduce |
| Gradient checking | Central difference | Forward difference | More accurate |

---

## 12. Test Suite

**File:** `tests/test_tensor.py`

| Test | What It Verifies |
|---|---|
| `test_tensor_creation` | Tensor stores data correctly, grad initialized to zeros |
| `test_add_forward` | Addition outputs correct values |
| `test_add_backward` | Gradients for addition are correct |
| `test_mul_forward` | Multiplication outputs correct values |
| `test_mul_backward` | Gradients for multiplication are correct |
| `test_matmul_forward` | Matrix multiplication outputs correct values |
| `test_matmul_backward` | Gradients for matmul are correct (both inputs) |
| `test_sum_forward` | Sum outputs correct values |
| `test_sum_backward` | Gradients for sum are correctly broadcasted |
| `test_mean_forward` | Mean outputs correct values |
| `test_mean_backward` | Gradients for mean are scaled by 1/N |
| `test_relu_forward` | ReLU outputs correct values |
| `test_relu_backward` | Gradients for ReLU (dead neurons get 0 gradient) |
| `test_gelu_forward` | GELU outputs correct values |
| `test_gelu_backward` | GELU gradient matches numerical derivative |
| `test_reshape_forward` | Reshape changes shape correctly |
| `test_reshape_backward` | Gradients flow through reshape correctly |
| `test_transpose_forward` | Transpose changes shape correctly |
| `test_transpose_backward` | Gradients flow through transpose correctly |
| `test_broadcasting` | Broadcasting works with addition and gradients |
| `test_shared_tensor` | Gradients accumulate when tensor is used twice |
| `test_gradient_check` | Numerical gradient checking passes for all operations |

**The most important test:** `test_shared_tensor` verifies gradient accumulation. This is the key insight that makes autograd work.

---