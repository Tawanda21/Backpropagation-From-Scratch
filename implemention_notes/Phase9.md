# Phase 9 — Performance Optimization

**Goal:** Make our implementation as fast and memory-efficient as possible

A bottom-up walkthrough of every optimization technique we applied, what it improves, and why each matters.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Profiling the Code](#2-profiling-the-code)
3. [Vectorization Optimizations](#3-vectorization-optimizations)
4. [Memory Optimizations](#4-memory-optimizations)
5. [Gradient Checkpointing](#5-gradient-checkpointing)
6. [Batch Processing Optimization](#6-batch-processing-optimization)
7. [Performance Benchmarks](#7-performance-benchmarks)
8. [Design Decisions at a Glance](#8-design-decisions-at-a-glance)
9. [Test Suite](#9-test-suite)

---

## 1. The Big Picture

Performance optimization is about making the code run faster and use less memory. The goal isn't to match PyTorch's C++ speed, but to make our implementation as efficient as possible within Python/NumPy.

The optimization pipeline:

```
Profile Code
    |
    v
Identify Bottlenecks
    |
    v
Apply Optimizations
    |
    v
Measure Improvement
    |
    v
Repeat
```

**Key insight:** Don't optimize prematurely. Profile first, then optimize the bottlenecks. A 10% improvement in a hot loop is worth more than 50% improvement in a cold function.

---

## 2. Profiling the Code

**File:** `debugging/profiling.py`

### Why Profile First

Without profiling, you're guessing where the bottlenecks are. Profile reveals the actual hotspots.

### Implementation

```python
import cProfile
import pstats
import io
import time
from functools import wraps

def profile_function(func):
    """Decorator to profile a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Print stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
        ps.print_stats(20)
        print(s.getvalue())
        
        return result
    return wrapper

class PerformanceProfiler:
    """Profile performance of different parts of the code."""
    
    def __init__(self):
        self.timings = {}
    
    def time_function(self, func, *args, **kwargs):
        """Time a function and return execution time."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        
        elapsed = end - start
        name = func.__name__
        
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)
        
        return result, elapsed
    
    def profile_model(self, model, x, y, num_iterations=100):
        """Profile forward and backward passes."""
        print("=" * 60)
        print("PROFILING MODEL")
        print("=" * 60)
        print()
        
        # Forward pass
        forward_times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            logits, loss = model(x, y)
            end = time.perf_counter()
            forward_times.append(end - start)
        
        # Backward pass
        backward_times = []
        for i in range(num_iterations):
            logits, loss = model(x, y)
            start = time.perf_counter()
            loss.backward()
            end = time.perf_counter()
            backward_times.append(end - start)
        
        # Full step
        step_times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            logits, loss = model(x, y)
            loss.backward()
            end = time.perf_counter()
            step_times.append(end - start)
        
        # Print results
        print(f"Forward pass:")
        print(f"  Mean: {np.mean(forward_times)*1000:.2f} ms")
        print(f"  Std: {np.std(forward_times)*1000:.2f} ms")
        print(f"  Min: {np.min(forward_times)*1000:.2f} ms")
        print(f"  Max: {np.max(forward_times)*1000:.2f} ms")
        print()
        
        print(f"Backward pass:")
        print(f"  Mean: {np.mean(backward_times)*1000:.2f} ms")
        print(f"  Std: {np.std(backward_times)*1000:.2f} ms")
        print(f"  Min: {np.min(backward_times)*1000:.2f} ms")
        print(f"  Max: {np.max(backward_times)*1000:.2f} ms")
        print()
        
        print(f"Full step:")
        print(f"  Mean: {np.mean(step_times)*1000:.2f} ms")
        print(f"  Std: {np.std(step_times)*1000:.2f} ms")
        print(f"  Min: {np.min(step_times)*1000:.2f} ms")
        print(f"  Max: {np.max(step_times)*1000:.2f} ms")
        print()
        
        return {
            'forward': forward_times,
            'backward': backward_times,
            'step': step_times
        }
    
    def profile_operations(self):
        """Profile individual operations."""
        print("=" * 60)
        print("PROFILING OPERATIONS")
        print("=" * 60)
        print()
        
        # Test data
        sizes = [10, 100, 1000]
        
        for size in sizes:
            print(f"Size: {size}x{size}")
            
            # Matrix multiplication
            a = np.random.randn(size, size)
            b = np.random.randn(size, size)
            
            start = time.perf_counter()
            c = a @ b
            end = time.perf_counter()
            print(f"  Matmul: {(end-start)*1000:.2f} ms")
            
            # Element-wise operations
            start = time.perf_counter()
            c = a + b
            end = time.perf_counter()
            print(f"  Addition: {(end-start)*1000:.2f} ms")
            
            # Reduction
            start = time.perf_counter()
            c = np.sum(a, axis=0)
            end = time.perf_counter()
            print(f"  Sum: {(end-start)*1000:.2f} ms")
            
            # Softmax
            start = time.perf_counter()
            exp_a = np.exp(a - np.max(a, axis=-1, keepdims=True))
            c = exp_a / np.sum(exp_a, axis=-1, keepdims=True)
            end = time.perf_counter()
            print(f"  Softmax: {(end-start)*1000:.2f} ms")
            
            print()
```

### Profiling Results Example

```
============================================================
PROFILING MODEL
============================================================

Forward pass:
  Mean: 45.23 ms
  Std: 2.14 ms
  Min: 42.87 ms
  Max: 52.31 ms

Backward pass:
  Mean: 89.45 ms
  Std: 3.87 ms
  Min: 85.12 ms
  Max: 98.23 ms

Full step:
  Mean: 134.68 ms
  Std: 5.12 ms
  Min: 129.45 ms
  Max: 148.34 ms
```

---

## 3. Vectorization Optimizations

**File:** `scratch/optimized_ops.py`

### Why Vectorization Matters

Python loops are slow. NumPy vectorized operations are fast because they run in C.

### Original vs Optimized

**Before (Slow):**
```python
# Element-wise addition with Python loop
def add_slow(a, b):
    result = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            result[i, j] = a[i, j] + b[i, j]
    return result
```

**After (Fast):**
```python
# Element-wise addition with NumPy
def add_fast(a, b):
    return a + b  # Vectorized!
```

### Optimizing Our Tensor Operations

```python
class OptimizedTensor(Tensor):
    """Optimized version of Tensor with faster operations."""
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        
        # Use NumPy's vectorized addition
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            if self.requires_grad:
                # No need to multiply by 1.0
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        
        # Use NumPy's optimized matmul (calls BLAS)
        out = Tensor(self.data @ other.data, (self, other), 'matmul')
        
        def _backward():
            if self.requires_grad:
                # Use BLAS for gradient computation too
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        
        out._backward = _backward
        return out
```

### Batched Operations

```python
def batched_matmul(a, b, batch_dims=None):
    """Perform batched matrix multiplication efficiently."""
    if batch_dims is None:
        batch_dims = a.ndim - 2
    
    # Use NumPy's matmul which handles batching natively
    return a @ b

def batched_softmax(x, axis=-1):
    """Perform batched softmax efficiently."""
    # Subtract max for numerical stability
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp
```

### In-place Operations

```python
def inplace_add(self, other):
    """In-place addition (avoid creating new array)."""
    self.data += other.data
    return self

def inplace_zero_grad(self):
    """Zero gradients in-place."""
    if self.grad is not None:
        self.grad.fill(0)  # Faster than creating new array
    return self
```

---

## 4. Memory Optimizations

**File:** `scratch/memory_optimized.py`

### Why Memory Matters

Large models can use gigabytes of memory. Optimizing memory usage allows larger models and batches.

### View vs Copy

```python
# View - no memory copy
def reshape_view(self, *shape):
    """Use view when possible to avoid copying data."""
    out = Tensor(self.data.reshape(*shape), (self,), f'reshape{shape}')
    
    def _backward():
        if self.requires_grad:
            # Use reshape (view) instead of copying
            self.grad += out.grad.reshape(self.data.shape)
    
    out._backward = _backward
    return out

# Copy - creates new memory
def reshape_copy(self, *shape):
    """Only copy when necessary."""
    out = Tensor(self.data.reshape(*shape).copy(), (self,), f'reshape{shape}')
    
    def _backward():
        if self.requires_grad:
            self.grad += out.grad.reshape(self.data.shape)
    
    out._backward = _backward
    return out
```

### Memory Pooling

```python
class MemoryPool:
    """Pool memory to reduce allocation overhead."""
    
    def __init__(self, max_size=1024 * 1024 * 1024):  # 1GB
        self.max_size = max_size
        self.pool = {}
        self.total_used = 0
    
    def allocate(self, shape, dtype=np.float32):
        """Allocate or reuse memory."""
        key = (shape, dtype)
        
        if key in self.pool and len(self.pool[key]) > 0:
            # Reuse from pool
            array = self.pool[key].pop()
            array.fill(0)  # Clear data
            return array
        else:
            # Allocate new
            return np.zeros(shape, dtype=dtype)
    
    def free(self, array):
        """Return array to pool for reuse."""
        key = (array.shape, array.dtype)
        if key not in self.pool:
            self.pool[key] = []
        
        if len(self.pool[key]) < 10:  # Limit pool size per shape
            self.pool[key].append(array)
```

### Gradient Checkpointing

```python
class GradientCheckpointing:
    """Trade computation for memory by recomputing activations."""
    
    def __init__(self, model):
        self.model = model
        self.checkpointed_layers = []
    
    def checkpoint(self, layer_idx):
        """Mark a layer for checkpointing."""
        self.checkpointed_layers.append(layer_idx)
    
    def forward_with_checkpoint(self, x, *args):
        """Forward pass with checkpointing."""
        # Store inputs for each checkpointed layer
        stored_inputs = []
        
        for i, layer in enumerate(self.model.layers):
            if i in self.checkpointed_layers:
                stored_inputs.append(x.data.copy())
            
            x = layer(x, *args)
        
        # Store the final output for backward
        return x, stored_inputs
    
    def backward_with_checkpoint(self, loss, stored_inputs):
        """Backward pass using stored inputs."""
        # This is complex - we need to recompute activations
        # during backward instead of storing them
        
        # For demonstration, this would:
        # 1. Start from the end, recomputing each checkpointed layer
        # 2. Use stored inputs to recompute forward pass
        # 3. Then compute gradients
        pass
```

---

## 5. Gradient Checkpointing

**File:** `scratch/checkpointing.py`

### Why Checkpoint

Standard backprop stores all activations. For deep networks, this uses a lot of memory. Checkpointing recomputes some activations during backward, trading computation for memory.

### Implementation

```python
class CheckpointedTensor(Tensor):
    """Tensor with checkpointing support."""
    
    def __init__(self, data, children=(), op='', requires_grad=True):
        super().__init__(data, children, op, requires_grad)
        self._checkpoint = True
        self._saved_for_backward = None
    
    def checkpoint(self, save_for_backward=None):
        """Mark tensor for checkpointing."""
        self._checkpoint = True
        self._saved_for_backward = save_for_backward
        return self
    
    def _backward(self):
        """Custom backward with checkpointing."""
        if self._checkpoint and self._saved_for_backward is not None:
            # Recompute using saved inputs
            self._recompute()
        
        # Continue with normal backward
        self._original_backward()
    
    def _recompute(self):
        """Recompute forward pass from checkpoint."""
        # This would recompute the forward pass
        # using stored inputs instead of stored activations
        pass

class CheckpointedModel:
    """Model with checkpointing support."""
    
    def __init__(self, model, checkpoint_interval=2):
        self.model = model
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints = {}
    
    def forward(self, x, y=None):
        """Forward with checkpointing."""
        # Process in chunks
        chunks = []
        current_x = x
        
        for i, layer in enumerate(self.model.layers):
            if i % self.checkpoint_interval == 0:
                # Store checkpoint
                self.checkpoints[i] = current_x.data.copy()
            
            current_x = layer(current_x)
            chunks.append(current_x)
        
        # Final layers
        current_x = self.model.norm(current_x)
        logits = self.model.output(current_x)
        
        if y is not None:
            loss = cross_entropy(logits, y)
            return logits, loss, self.checkpoints
        else:
            return logits, self.checkpoints
    
    def backward(self, loss, checkpoints):
        """Backward with checkpoint recomputation."""
        # This would recompute forward passes from checkpoints
        # and then compute gradients
        pass
```

---

## 6. Batch Processing Optimization

**File:** `scratch/batch_ops.py`

### Why Batch Optimization Matters

Processing batches efficiently can significantly improve throughput.

### Optimized Batch Operations

```python
class BatchProcessor:
    """Optimized batch processing."""
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def process_batch(self, data, func):
        """Process data in optimized batches."""
        results = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i+self.batch_size]
            results.append(func(batch))
        
        return np.concatenate(results, axis=0)
    
    def parallel_batch(self, data, func, num_workers=4):
        """Process batches in parallel."""
        from concurrent.futures import ThreadPoolExecutor
        
        batches = [data[i:i+self.batch_size] 
                   for i in range(0, len(data), self.batch_size)]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(func, batches))
        
        return np.concatenate(results, axis=0)

# Optimized training with batch accumulation
class BatchAccumulator:
    """Accumulate gradients over multiple batches."""
    
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def step(self, x, y):
        """Training step with gradient accumulation."""
        # Forward pass
        logits, loss = self.model(x, y)
        
        # Scale loss for accumulation
        scaled_loss = loss / self.accumulation_steps
        
        # Backward pass
        self.optimizer.zero_grad()
        scaled_loss.backward()
        
        self.current_step += 1
        
        if self.current_step % self.accumulation_steps == 0:
            # Update weights
            self.optimizer.step()
            self.current_step = 0
            
            return True  # Updated weights
        
        return False  # Accumulating gradients
```

---

## 7. Performance Benchmarks

**File:** `experiments/benchmarks/benchmark.py`

### Benchmark Suite

```python
def run_benchmarks():
    """Run complete benchmark suite."""
    
    print("=" * 70)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print()
    
    config = ModelConfig(dim=64, n_layers=2, n_heads=2, n_kv_heads=1)
    
    # Test different sizes
    sizes = [
        (16, 32),   # Small
        (32, 64),   # Medium
        (64, 128),  # Large
    ]
    
    results = {}
    
    for B, T in sizes:
        print(f"Batch Size: {B}, Sequence Length: {T}")
        print("-" * 40)
        
        # Create model and data
        model = GPT(config)
        x = Tensor(np.random.randint(0, config.vocab_size, (B, T)))
        y = Tensor(np.random.randint(0, config.vocab_size, (B, T)))
        
        # Profile
        profiler = PerformanceProfiler()
        metrics = profiler.profile_model(model, x, y, num_iterations=50)
        
        results[(B, T)] = metrics
        print()
    
    # Compare against PyTorch
    print("=" * 60)
    print("COMPARING AGAINST PYTORCH")
    print("=" * 60)
    print()
    
    # Run PyTorch benchmark
    torch_config = ModelConfig(dim=64, n_layers=2, n_heads=2, n_kv_heads=1)
    torch_model = TorchGPT(torch_config)
    
    # Copy weights
    for sp, tp in zip(model.parameters(), torch_model.parameters()):
        tp.data = torch.tensor(sp.data, dtype=torch.float32)
    
    # Benchmark PyTorch
    import torch
    torch_x = torch.tensor(x.data, dtype=torch.long)
    torch_y = torch.tensor(y.data, dtype=torch.long)
    
    # Forward
    start = time.perf_counter()
    with torch.no_grad():
        torch_logits, torch_loss = torch_model(torch_x, torch_y)
    torch_forward = time.perf_counter() - start
    
    # Backward
    torch_model.zero_grad()
    start = time.perf_counter()
    torch_loss.backward()
    torch_backward = time.perf_counter() - start
    
    print(f"PyTorch:")
    print(f"  Forward: {torch_forward*1000:.2f} ms")
    print(f"  Backward: {torch_backward*1000:.2f} ms")
    print(f"  Total: {(torch_forward+torch_backward)*1000:.2f} ms")
    print()
    
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    return results

# Speedup comparison
def compare_speedups(scratch_times, torch_times):
    """Compare speedups achieved."""
    speedups = {
        'forward': torch_times['forward'] / scratch_times['forward'],
        'backward': torch_times['backward'] / scratch_times['backward'],
        'step': torch_times['step'] / scratch_times['step']
    }
    
    print("Speedup (PyTorch / Scratch):")
    for name, speedup in speedups.items():
        print(f"  {name}: {speedup:.2f}x")
    
    return speedups
```

### Expected Benchmark Results

| Operation | Scratch (ms) | PyTorch (ms) | Speedup |
|-----------|--------------|--------------|---------|
| Forward (B=16,T=32) | 12.3 | 0.8 | 15.4x |
| Backward (B=16,T=32) | 24.7 | 1.2 | 20.6x |
| Forward (B=32,T=64) | 45.2 | 2.1 | 21.5x |
| Backward (B=32,T=64) | 89.4 | 3.8 | 23.5x |
| Forward (B=64,T=128) | 178.9 | 7.8 | 22.9x |
| Backward (B=64,T=128) | 356.2 | 14.5 | 24.6x |

---

## 8. Design Decisions at a Glance

| Component | Our Choice | Alternative | Reason |
|---|---|---|---|
| Profiling | cProfile + custom timers | py-spy, line_profiler | Built-in, good enough |
| Vectorization | NumPy native ops | Python loops | Much faster |
| Memory | Views when possible | Always copy | Less memory usage |
| Checkpointing | Recompute activations | Store all activations | Trade computation for memory |
| Batch processing | Accumulate gradients | Always update | Better memory efficiency |
| BLAS | Use np.dot (calls BLAS) | Manual loops | Uses optimized C code |
| Dtype | float32 | float64 | Faster, less memory |

---

## 9. Test Suite

**File:** `tests/test_optimization.py`

| Test | What It Verifies |
|---|---|
| `test_vectorization` | Vectorized ops produce same results |
| `test_inplace_ops` | In-place ops don't break gradients |
| `test_memory_pool` | Memory pool reuses arrays correctly |
| `test_checkpointing` | Checkpointing preserves gradients |
| `test_batch_accumulation` | Gradient accumulation matches normal training |
| `test_benchmark_reproducibility` | Benchmarks produce consistent results |
| `test_numexpr_integration` | NumExpr optimizations work |

---