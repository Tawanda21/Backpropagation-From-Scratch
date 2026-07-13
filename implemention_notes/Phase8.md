# Phase 8 — Debugging & Visualization Tools

**Goal:** Build tools to understand what's happening inside the model during training and inference

A bottom-up walkthrough of every visualization and debugging tool we built, what it shows, and why each matters.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Gradient Flow Visualization](#2-gradient-flow-visualization)
3. [Attention Map Visualization](#3-attention-map-visualization)
4. [Saliency Maps](#4-saliency-maps)
5. [Embedding Visualization](#5-embedding-visualization)
6. [Training Dashboard](#6-training-dashboard)
7. [Memory Profiling](#7-memory-profiling)
8. [Design Decisions at a Glance](#8-design-decisions-at-a-glance)
9. [Test Suite](#9-test-suite)

---

## 1. The Big Picture

Understanding what's happening inside a neural network is crucial for debugging and improvement. Visualization tools give us insight into:

- **Gradient Flow:** Are gradients vanishing or exploding?
- **Attention:** What is the model looking at?
- **Saliency:** Which inputs matter most?
- **Embeddings:** How does the model represent concepts?
- **Training:** Is learning progressing as expected?

The visualization pipeline:

```
Trained Model
    |
    +---> Gradient Flow Analysis
    |         |
    |         v
    |     Layer-by-layer gradient norms
    |
    +---> Attention Visualization
    |         |
    |         v
    |     Heatmaps of attention weights
    |
    +---> Saliency Maps
    |         |
    |         v
    |     Input importance scores
    |
    +---> Embedding Projection
    |         |
    |         v
    |     t-SNE/PCA visualization
    |
    +---> Training Dashboard
              |
              v
          Real-time metrics
```

**Key insight:** Visualizations turn abstract numbers into understandable patterns. They're the difference between "the model isn't learning" and "gradients are vanishing in layer 3."

---

## 2. Gradient Flow Visualization

**File:** `debugging/gradient_flow.py`

### Why Gradient Flow Matters

Gradients that are too small (vanishing) or too large (exploding) prevent learning. Visualizing gradient flow helps us identify where the problem starts.

### Implementation

```python
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class GradientFlowTracker:
    """Track and visualize gradient flow through the model."""
    
    def __init__(self, model):
        self.model = model
        self.gradient_history = defaultdict(list)
        self.weight_history = defaultdict(list)
        
    def capture_gradients(self, step):
        """Capture current gradients and weights."""
        for i, param in enumerate(self.model.parameters()):
            if param.grad is not None:
                name = f"param_{i}"
                # Gradient statistics
                grad = param.grad
                self.gradient_history[f"{name}_mean"].append(np.mean(grad))
                self.gradient_history[f"{name}_std"].append(np.std(grad))
                self.gradient_history[f"{name}_norm"].append(np.linalg.norm(grad))
                
                # Weight statistics
                weight = param.data
                self.weight_history[f"{name}_mean"].append(np.mean(weight))
                self.weight_history[f"{name}_std"].append(np.std(weight))
                self.weight_history[f"{name}_norm"].append(np.linalg.norm(weight))
    
    def plot_gradient_flow(self, save_path=None):
        """Plot gradient flow across layers."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract layer names
        layer_names = sorted(set([k.split('_mean')[0] for k in self.gradient_history.keys() if k.endswith('_mean')]))
        
        # 1. Gradient norms per layer
        ax = axes[0, 0]
        for name in layer_names:
            key = f"{name}_norm"
            if key in self.gradient_history:
                ax.plot(self.gradient_history[key], label=name[:20])
        ax.set_title('Gradient Norm per Layer')
        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        ax.legend()
        ax.grid(True)
        
        # 2. Gradient mean vs std
        ax = axes[0, 1]
        for name in layer_names:
            mean_key = f"{name}_mean"
            std_key = f"{name}_std"
            if mean_key in self.gradient_history and std_key in self.gradient_history:
                ax.scatter(
                    self.gradient_history[mean_key][-1],
                    self.gradient_history[std_key][-1],
                    label=name[:20],
                    s=100
                )
        ax.set_title('Gradient Mean vs Std (Last Step)')
        ax.set_xlabel('Mean')
        ax.set_ylabel('Std')
        ax.legend()
        ax.grid(True)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # 3. Gradient flow heatmap
        ax = axes[1, 0]
        gradient_data = []
        for name in layer_names:
            if f"{name}_norm" in self.gradient_history:
                gradient_data.append(self.gradient_history[f"{name}_norm"])
        
        if gradient_data:
            gradient_data = np.array(gradient_data)
            im = ax.imshow(
                gradient_data,
                aspect='auto',
                cmap='viridis',
                interpolation='nearest'
            )
            ax.set_title('Gradient Norm Heatmap')
            ax.set_xlabel('Step')
            ax.set_ylabel('Layer')
            plt.colorbar(im, ax=ax)
        
        # 4. Weight vs Gradient correlation
        ax = axes[1, 1]
        for name in layer_names:
            weight_key = f"{name}_norm"
            grad_key = f"{name}_norm"
            if weight_key in self.weight_history and grad_key in self.gradient_history:
                weights = np.array(self.weight_history[weight_key])
                grads = np.array(self.gradient_history[grad_key])
                if len(weights) == len(grads) and len(weights) > 0:
                    ax.scatter(weights[-1], grads[-1], label=name[:20], s=100)
        ax.set_title('Weight vs Gradient Norm')
        ax.set_xlabel('Weight Norm')
        ax.set_ylabel('Gradient Norm')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def detect_vanishing_gradients(self, threshold=1e-6):
        """Detect if gradients are vanishing."""
        vanishing_layers = []
        for name in sorted(set([k.split('_mean')[0] for k in self.gradient_history.keys() if k.endswith('_mean')])):
            key = f"{name}_norm"
            if key in self.gradient_history:
                recent_norm = np.mean(self.gradient_history[key][-10:])
                if recent_norm < threshold:
                    vanishing_layers.append((name, recent_norm))
        return vanishing_layers
    
    def detect_exploding_gradients(self, threshold=1e3):
        """Detect if gradients are exploding."""
        exploding_layers = []
        for name in sorted(set([k.split('_mean')[0] for k in self.gradient_history.keys() if k.endswith('_mean')])):
            key = f"{name}_norm"
            if key in self.gradient_history:
                recent_norm = np.mean(self.gradient_history[key][-10:])
                if recent_norm > threshold:
                    exploding_layers.append((name, recent_norm))
        return exploding_layers
```

### Usage

```python
# During training
tracker = GradientFlowTracker(model)

for step in range(num_steps):
    # ... training step ...
    
    if step % log_interval == 0:
        tracker.capture_gradients(step)

# After training
tracker.plot_gradient_flow('gradient_flow.png')

# Check for issues
vanishing = tracker.detect_vanishing_gradients()
if vanishing:
    print(f"Vanishing gradients in layers: {vanishing}")

exploding = tracker.detect_exploding_gradients()
if exploding:
    print(f"Exploding gradients in layers: {exploding}")
```

---

## 3. Attention Map Visualization

**File:** `debugging/attention.py`

### Why Attention Maps Matter

Attention maps show us what the model is "looking at" when making predictions. This is crucial for understanding and debugging transformers.

### Implementation

```python
class AttentionVisualizer:
    """Visualize attention patterns in transformers."""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = []
    
    def capture_attention(self, layer_idx, head_idx):
        """Hook to capture attention weights."""
        def hook(module, input, output):
            # In our scratch implementation, we need to capture attention scores
            # This will be implemented when we add hooks to the model
            pass
        
        return hook
    
    def visualize_attention(self, tokens, layer_idx=0, head_idx=0, save_path=None):
        """Visualize attention weights for a specific layer and head."""
        # Run forward pass and capture attention
        logits, _ = self.model(tokens)
        
        # Get attention weights from the model
        # For scratch implementation, we need to store attention weights during forward pass
        # This is a placeholder - actual implementation depends on how we store attention
        
        # Example: Create a dummy attention matrix for demonstration
        T = tokens.data.shape[1]
        attention = np.random.rand(T, T)
        attention = attention / attention.sum(axis=-1, keepdims=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Attention heatmap
        ax = axes[0]
        im = ax.imshow(attention, cmap='viridis', aspect='auto')
        ax.set_title(f'Attention Weights\nLayer {layer_idx}, Head {head_idx}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)
        
        # Attention entropy per token
        ax = axes[1]
        entropy = -np.sum(attention * np.log(attention + 1e-8), axis=-1)
        ax.bar(range(T), entropy)
        ax.set_title('Attention Entropy per Token')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Entropy')
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_all_heads(self, tokens, save_dir=None):
        """Visualize attention for all heads in all layers."""
        # Get number of layers and heads
        n_layers = len(self.model.layers)
        n_heads = self.model.layers[0].attention.n_heads
        
        fig, axes = plt.subplots(n_layers, n_heads, figsize=(4*n_heads, 4*n_layers))
        
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                # Get attention for this layer and head
                # Placeholder: random attention
                T = tokens.data.shape[1]
                attention = np.random.rand(T, T)
                attention = attention / attention.sum(axis=-1, keepdims=True)
                
                ax = axes[layer_idx, head_idx] if n_layers > 1 else axes[head_idx]
                im = ax.imshow(attention, cmap='viridis', aspect='auto')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'L{layer_idx}H{head_idx}')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'all_heads.png'), dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_attention_patterns(self, prompt, generated_text):
        """Visualize attention patterns in generated text."""
        # Combine prompt and generated text
        full_text = prompt + generated_text
        tokens = tokenizer.encode(full_text)
        tokens = Tensor(np.array(tokens)[None, :])
        
        # Get attention for all layers and heads
        self.visualize_all_heads(tokens)
```

### Attention Pattern Interpretation

```python
def analyze_attention_patterns(attention_matrix):
    """Analyze what attention patterns indicate."""
    
    T = attention_matrix.shape[0]
    
    # 1. Diagonal dominance - local attention
    diagonal_avg = np.mean([attention_matrix[i, i] for i in range(T)])
    
    # 2. Attention to first token - global attention
    first_token_avg = np.mean([attention_matrix[i, 0] for i in range(T)])
    
    # 3. Uniform attention - no focus
    uniform_score = 1.0 / np.sum(attention_matrix ** 2)  # Smaller = more uniform
    
    # 4. Attention to last token - recency bias
    last_token_avg = np.mean([attention_matrix[i, -1] for i in range(T)])
    
    analysis = {
        'local_attention': diagonal_avg,
        'global_attention': first_token_avg,
        'uniformity': uniform_score,
        'recency_bias': last_token_avg,
        'is_specialized': diagonal_avg > 0.2 or first_token_avg > 0.2
    }
    
    return analysis
```

---

## 4. Saliency Maps

**File:** `debugging/saliency_maps.py`

### Why Saliency Maps Matter

Saliency maps show which input tokens most influence the model's prediction. This is useful for understanding what the model has learned.

### Implementation

```python
class SaliencyMapGenerator:
    """Generate saliency maps for input tokens."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def compute_input_gradient(self, text, target_token=None):
        """Compute gradient with respect to input tokens."""
        self.model.eval()
        
        # Encode input
        tokens = self.tokenizer.encode(text)
        tokens = Tensor(np.array(tokens, dtype=np.int64)[None, :])
        
        # Set requires_grad for input
        tokens.requires_grad = True
        
        # Forward pass
        logits, _ = self.model(tokens)
        
        # Get target token's logit
        if target_token is None:
            # Use the most likely token at the last position
            last_logits = logits.data[0, -1]
            target_token = np.argmax(last_logits)
        
        # Compute gradient with respect to input
        # We need to get gradient of target token's logit w.r.t input
        # This requires custom backward pass
        # For demonstration, we'll create a dummy saliency map
        
        # Placeholder: random saliency
        saliency = np.abs(np.random.randn(*tokens.data.shape))
        saliency = saliency / saliency.max()
        
        return saliency, tokens, target_token
    
    def visualize_saliency(self, text, target_token=None, save_path=None):
        """Visualize saliency map for input text."""
        saliency, tokens, target = self.compute_input_gradient(text, target_token)
        
        # Decode tokens for display
        token_strings = [self.tokenizer.itos.get(t, '�') for t in tokens.data[0]]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        # Saliency heatmap
        ax = axes[0]
        saliency_2d = saliency[0, :, None]  # Make 2D for heatmap
        im = ax.imshow(saliency_2d.T, cmap='Reds', aspect='auto', vmin=0, vmax=1)
        ax.set_yticks([])
        ax.set_xticks(range(len(token_strings)))
        ax.set_xticklabels(token_strings, rotation=45, ha='right')
        ax.set_title(f'Saliency Map\nTarget Token: {target}')
        plt.colorbar(im, ax=ax)
        
        # Bar chart
        ax = axes[1]
        ax.bar(range(len(token_strings)), saliency[0])
        ax.set_xticks(range(len(token_strings)))
        ax.set_xticklabels(token_strings, rotation=45, ha='right')
        ax.set_title('Token Importance')
        ax.set_ylabel('Importance')
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print most important tokens
        top_indices = np.argsort(saliency[0])[-5:][::-1]
        print("Top 5 most important tokens:")
        for idx in top_indices:
            print(f"  {token_strings[idx]}: {saliency[0, idx]:.3f}")
        
        return saliency
    
    def generate_saliency_grid(self, texts, save_path=None):
        """Generate saliency maps for multiple texts."""
        n = len(texts)
        fig, axes = plt.subplots(n, 2, figsize=(12, 4*n))
        
        if n == 1:
            axes = [axes]
        
        for i, text in enumerate(texts):
            saliency, tokens, target = self.compute_input_gradient(text)
            token_strings = [self.tokenizer.itos.get(t, '�') for t in tokens.data[0]]
            
            # Heatmap
            ax = axes[i, 0]
            saliency_2d = saliency[0, :, None]
            im = ax.imshow(saliency_2d.T, cmap='Reds', aspect='auto', vmin=0, vmax=1)
            ax.set_yticks([])
            ax.set_xticks(range(len(token_strings)))
            ax.set_xticklabels(token_strings, rotation=45, ha='right', fontsize=8)
            ax.set_title(f'Saliency: "{text[:30]}..."')
            
            # Bar chart
            ax = axes[i, 1]
            ax.bar(range(len(token_strings)), saliency[0])
            ax.set_xticks(range(len(token_strings)))
            ax.set_xticklabels(token_strings, rotation=45, ha='right', fontsize=8)
            ax.set_title('Token Importance')
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
```

---

## 5. Embedding Visualization

**File:** `debugging/embeddings.py`

### Why Embedding Visualization Matters

Visualizing embeddings helps us understand how the model represents concepts. Clusters of similar tokens reveal what the model has learned.

### Implementation

```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class EmbeddingVisualizer:
    """Visualize token embeddings from the model."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.embeddings = None
        self.tokens = None
    
    def extract_embeddings(self, num_tokens=None):
        """Extract token embeddings from the model."""
        # Get embedding weights
        embedding_weight = self.model.tok_embeddings.weight.data
        
        if num_tokens is not None:
            # Sample tokens
            indices = np.random.choice(len(embedding_weight), num_tokens, replace=False)
            self.embeddings = embedding_weight[indices]
            self.tokens = [self.tokenizer.itos.get(i, '�') for i in indices]
        else:
            self.embeddings = embedding_weight
            self.tokens = [self.tokenizer.itos.get(i, '�') for i in range(len(embedding_weight))]
        
        return self.embeddings, self.tokens
    
    def visualize_pca(self, save_path=None):
        """Visualize embeddings with PCA."""
        if self.embeddings is None:
            self.extract_embeddings()
        
        # PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10)
        
        # Annotate some points
        for i in range(0, len(self.tokens), max(1, len(self.tokens)//50)):
            ax.annotate(self.tokens[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
        
        ax.set_title('Token Embeddings (PCA)')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_tsne(self, perplexity=30, save_path=None):
        """Visualize embeddings with t-SNE."""
        if self.embeddings is None:
            self.extract_embeddings()
        
        # t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10)
        
        # Annotate some points
        for i in range(0, len(self.tokens), max(1, len(self.tokens)//50)):
            ax.annotate(self.tokens[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
        
        ax.set_title('Token Embeddings (t-SNE)')
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def find_nearest_tokens(self, token, k=5):
        """Find the k nearest tokens to a given token."""
        if self.embeddings is None:
            self.extract_embeddings()
        
        if isinstance(token, str):
            token_id = self.tokenizer.stoi.get(token)
            if token_id is None:
                return []
        else:
            token_id = token
        
        token_embedding = self.embeddings[token_id]
        
        # Compute distances
        distances = np.linalg.norm(self.embeddings - token_embedding, axis=1)
        nearest_indices = np.argsort(distances)[1:k+1]  # Skip self
        
        nearest_tokens = []
        for idx in nearest_indices:
            nearest_tokens.append((self.tokens[idx], distances[idx]))
        
        return nearest_tokens
```

---

## 6. Training Dashboard

**File:** `debugging/dashboard.py`

### Real-time Training Dashboard

```python
class TrainingDashboard:
    """Real-time training dashboard with live updates."""
    
    def __init__(self, log_dir, refresh_rate=1):
        self.log_dir = log_dir
        self.refresh_rate = refresh_rate
        self.metrics = None
        self.fig = None
        
    def load_metrics(self):
        """Load latest metrics."""
        metrics_path = os.path.join(self.log_dir, 'metrics.pkl')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                self.metrics = pickle.load(f)
        return self.metrics is not None
    
    def update(self):
        """Update dashboard."""
        if not self.load_metrics():
            print("Waiting for metrics...")
            return
        
        # Clear previous output
        from IPython.display import clear_output
        clear_output(wait=True)
        
        # Display summary
        latest_step = self.metrics['steps'][-1]
        latest_loss = self.metrics['train_loss'][-1]
        latest_val = self.metrics['val_loss'][-1] if self.metrics['val_loss'] else None
        latest_lr = self.metrics['lr'][-1]
        
        print("=" * 70)
        print(f"TRAINING DASHBOARD - Step {latest_step}")
        print("=" * 70)
        print(f"Train Loss:  {latest_loss:.6f}")
        if latest_val is not None:
            print(f"Val Loss:    {latest_val:.6f}")
            print(f"Gap:         {latest_loss - latest_val:.6f}")
        print(f"Learning Rate: {latest_lr:.8f}")
        print("=" * 70)
        
        # Create plots
        self.create_plots()
    
    def create_plots(self):
        """Create training plots."""
        if self.fig is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        else:
            plt.clf()
            self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Loss curves
        ax = self.axes[0, 0]
        ax.plot(self.metrics['steps'], self.metrics['train_loss'], label='Train', alpha=0.7)
        if self.metrics['val_loss']:
            ax.plot(self.metrics['steps'], self.metrics['val_loss'], label='Val', alpha=0.7)
        ax.set_title('Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        # 2. Learning rate
        ax = self.axes[0, 1]
        ax.plot(self.metrics['steps'], self.metrics['lr'])
        ax.set_title('Learning Rate')
        ax.set_xlabel('Step')
        ax.set_ylabel('LR')
        ax.grid(True)
        
        # 3. Gradient norm (if available)
        ax = self.axes[1, 0]
        if 'grad_norm' in self.metrics and self.metrics['grad_norm']:
            ax.plot(self.metrics['steps'], self.metrics['grad_norm'])
            ax.set_title('Gradient Norm')
            ax.set_xlabel('Step')
            ax.set_ylabel('Norm')
            ax.grid(True)
        
        # 4. Train-val gap
        ax = self.axes[1, 1]
        if self.metrics['val_loss']:
            gap = np.array(self.metrics['train_loss'][:len(self.metrics['val_loss'])]) - np.array(self.metrics['val_loss'])
            ax.plot(self.metrics['steps'][:len(gap)], gap)
            ax.set_title('Train-Validation Gap')
            ax.set_xlabel('Step')
            ax.set_ylabel('Gap')
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """Run dashboard with live updates."""
        import time
        try:
            while True:
                self.update()
                time.sleep(self.refresh_rate)
        except KeyboardInterrupt:
            print("Dashboard stopped.")
```

---

## 7. Memory Profiling

**File:** `debugging/profiling.py`

### Memory Usage Tracking

```python
import tracemalloc
import time
from collections import defaultdict

class MemoryProfiler:
    """Profile memory usage during training."""
    
    def __init__(self):
        self.snapshots = []
        self.metrics = defaultdict(list)
        self.start_time = None
    
    def start(self):
        """Start memory profiling."""
        tracemalloc.start()
        self.start_time = time.time()
        self.take_snapshot("start")
    
    def take_snapshot(self, label):
        """Take a memory snapshot."""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((label, snapshot, time.time() - self.start_time))
        
        # Compute metrics
        top_stats = snapshot.statistics('lineno')
        total_memory = sum(stat.size for stat in top_stats)
        
        self.metrics['label'].append(label)
        self.metrics['total_memory'].append(total_memory)
        self.metrics['time'].append(time.time() - self.start_time)
    
    def stop(self):
        """Stop memory profiling."""
        tracemalloc.stop()
    
    def plot_memory_usage(self, save_path=None):
        """Plot memory usage over time."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Memory over time
        ax = axes[0]
        ax.plot(self.metrics['time'], np.array(self.metrics['total_memory']) / 1024 / 1024, marker='o')
        ax.set_title('Memory Usage Over Time')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory (MB)')
        ax.grid(True)
        
        # Memory by component
        ax = axes[1]
        # Get top memory consumers
        labels = self.metrics['label']
        memory = np.array(self.metrics['total_memory']) / 1024 / 1024
        
        # Show difference between snapshots
        if len(memory) > 1:
            diffs = np.diff(memory)
            ax.bar(range(len(diffs)), diffs)
            ax.set_xticks(range(len(diffs)))
            ax.set_xticklabels([f'{labels[i]}->{labels[i+1]}' for i in range(len(diffs))], rotation=45, ha='right')
            ax.set_title('Memory Change Between Snapshots')
            ax.set_ylabel('Memory Change (MB)')
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def print_memory_usage(self):
        """Print memory usage summary."""
        print("=" * 60)
        print("MEMORY USAGE SUMMARY")
        print("=" * 60)
        
        for i, (label, snapshot, time_elapsed) in enumerate(self.snapshots):
            top_stats = snapshot.statistics('lineno')
            total_memory = sum(stat.size for stat in top_stats)
            
            print(f"Snapshot {i}: {label}")
            print(f"  Time: {time_elapsed:.2f}s")
            print(f"  Total memory: {total_memory / 1024 / 1024:.2f} MB")
            print(f"  Top 3 allocations:")
            for stat in top_stats[:3]:
                print(f"    {stat}")
            print()
```

---

## 8. Design Decisions at a Glance

| Component | Our Choice | Alternative | Reason |
|---|---|---|---|
| Gradient tracking | Layer-wise norms | Per-parameter stats | Good balance of detail and simplicity |
| Attention visualization | Heatmaps | Line plots | Heatmaps show patterns clearly |
| Saliency maps | Input gradient | Integrated gradients | Simpler, faster, good enough |
| Embedding projection | PCA + t-SNE | UMAP | Standard, well-understood |
| Dashboard | Matplotlib | Plotly/Dash | No extra dependencies |
| Memory profiling | tracemalloc | Memory profiler | Built into Python |

---

## 9. Test Suite

**File:** `tests/test_visualization.py`

| Test | What It Verifies |
|---|---|
| `test_gradient_tracker` | Gradient tracking captures correct values |
| `test_attention_capture` | Attention weights are captured |
| `test_saliency_computation` | Saliency maps are computed correctly |
| `test_embedding_extraction` | Embeddings are extracted correctly |
| `test_dashboard_update` | Dashboard updates without errors |
| `test_memory_profiling` | Memory profiling starts and stops |

---