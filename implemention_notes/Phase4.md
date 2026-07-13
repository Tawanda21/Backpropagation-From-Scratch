# Phase 4 — Training Pipeline

**Goal:** Build the complete training infrastructure to train our GPT model

A bottom-up walkthrough of every component we built, what it does, and why each design decision was made.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Data Loading & Batching](#2-data-loading--batching)
3. [Learning Rate Scheduling](#3-learning-rate-scheduling)
4. [The Training Loop](#4-the-training-loop)
5. [Checkpointing](#5-checkpointing)
6. [Logging & Monitoring](#6-logging--monitoring)
7. [Configuration Management](#7-configuration-management)
8. [Training Script](#8-training-script)
9. [Design Decisions at a Glance](#9-design-decisions-at-a-glance)
10. [Test Suite](#10-test-suite)

---

## 1. The Big Picture

Training a language model involves repeatedly:
1. Sampling a batch of text from the dataset
2. Running the model forward to get predictions
3. Computing the loss
4. Running backpropagation to get gradients
5. Updating the model weights
6. Monitoring progress

The data flows like this:

```text
Raw Text (Shakespeare)
    │
    ▼
Tokenization (character-level)
    │
    ▼
Training Data (array of integers)
    │
    ▼
Batch Sampling (random contiguous chunks)
    │
    ▼
┌─────────────────────────────────────┐
│         Training Loop                │
│  ┌─────────────────────────────────┐ │
│  │  For each step:                  │ │
│  │    1. Get batch (x, y)           │ │
│  │    2. Forward pass → logits, loss│ │
│  │    3. Backward pass → gradients  │ │
│  │    4. Clip gradients             │ │
│  │    5. Update weights (Adam)      │ │
│  │    6. Update learning rate       │ │
│  │    7. Log metrics                │ │
│  │    8. Save checkpoint            │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
    │
    ▼
Trained Model → Generate Text
```

**Key insight:** Training is just gradient descent on a loss function. Everything else (batching, scheduling, logging) is infrastructure to make this work efficiently.

---

## 2. Data Loading & Batching

**File:** `scratch/data.py`

### The Dataset

We use Shakespeare's complete works — about 1 million characters. At character-level tokenization this is ~1M tokens.

```python
def load_shakespeare_data():
    """Load Shakespeare dataset and create train/val split."""
    with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Character-level tokenization
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode text
    data = np.array([stoi[ch] for ch in text], dtype=np.int64)
    
    # Split into train and validation (90/10)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, stoi, itos, vocab_size
```

### Why Character-Level?

Character-level tokenization is simpler than BPE and great for learning:
- **Pros:** Simple to implement, no external tokenizer needed
- **Cons:** Longer sequences, less semantic meaning per token
- **For learning:** Perfect — you focus on the model, not tokenization

### Batch Sampling

We sample random contiguous chunks from the dataset:

```python
def get_batch(data, block_size, batch_size):
    """
    Sample a batch of sequences from the dataset.
    
    Args:
        data: (N,) array of token IDs
        block_size: length of each sequence (context window)
        batch_size: number of sequences in the batch
    
    Returns:
        x: (batch_size, block_size) input tokens
        y: (batch_size, block_size) target tokens (shifted by 1)
    """
    # Random starting positions
    idx = np.random.randint(0, len(data) - block_size, (batch_size,))
    
    # Build input sequences
    x = np.array([data[i:i+block_size] for i in idx])
    
    # Build target sequences (shifted by 1)
    y = np.array([data[i+1:i+block_size+1] for i in idx])
    
    return Tensor(x), Tensor(y)
```

**Why shift by 1?** The model predicts the next token. For each position `t`, input is `tokens[t]` and target is `tokens[t+1]`. This is autoregressive training.

### Efficient DataLoader

For larger datasets, we want to iterate without loading everything into memory:

```python
class DataLoader:
    def __init__(self, data, block_size, batch_size, shuffle=True):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Pre-compute all possible starting positions
        self.num_batches = len(data) // (block_size * batch_size)
        self.indices = np.arange(len(data) - block_size)
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx >= len(self.indices):
            raise StopIteration
        
        # Get batch of starting positions
        batch_indices = self.indices[self.idx:self.idx + self.batch_size]
        self.idx += self.batch_size
        
        # Build batch
        x = np.array([self.data[i:i+self.block_size] for i in batch_indices])
        y = np.array([self.data[i+1:i+self.block_size+1] for i in batch_indices])
        
        return Tensor(x), Tensor(y)
```

---

## 3. Learning Rate Scheduling

**File:** `scratch/scheduler.py`

### Why Schedule Learning Rate?

The learning rate is the most important hyperparameter. A good schedule:
- **Warmup:** Gradually increases LR at the start (prevents early instability)
- **Decay:** Decreases LR later (fine-tunes the model)

### Cosine Decay with Warmup

This is the most common schedule for transformers:

```python
class CosineDecayWithWarmup:
    """
    Learning rate schedule with warmup and cosine decay.
    
    LR schedule:
    - Linear warmup: LR increases from 0 to lr_max over warmup_steps
    - Cosine decay: LR decays from lr_max to lr_min over remaining steps
    """
    
    def __init__(self, lr_max, warmup_steps, total_steps, lr_min=0.0):
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_min = lr_min
        self.current_step = 0
    
    def get_lr(self, step=None):
        if step is None:
            step = self.current_step
        
        # Warmup phase
        if step < self.warmup_steps:
            # Linear warmup: 0 → lr_max
            return self.lr_max * step / self.warmup_steps
        
        # Decay phase
        elif step < self.total_steps:
            # Cosine decay: lr_max → lr_min
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return self.lr_min + (self.lr_max - self.lr_min) * cosine_decay
        
        else:
            # Minimum LR after total steps
            return self.lr_min
    
    def step(self):
        self.current_step += 1
        return self.get_lr()
```

### Visualizing the Schedule

```text
LR
 ^
 |
 |     /\
 |    /  \___  Cosine Decay
 |   /      \
 |  /        \___
 | /             
 |/______________
 +------> Steps
    Warmup    Total
```

**Why warmup?** At initialization, gradients can be large and noisy. Warmup gives the model time to find a stable direction before taking large steps.

**Why cosine decay?** Smoothly reduces the LR, allowing the model to settle into a minimum.

### Linear Warmup Alternative

For simpler cases, linear decay works too:

```python
class LinearDecay:
    def __init__(self, lr_max, total_steps):
        self.lr_max = lr_max
        self.total_steps = total_steps
    
    def get_lr(self, step):
        return self.lr_max * (1 - step / self.total_steps)
```

---

## 4. The Training Loop

**File:** `scratch/train.py`

### The Core Loop

```python
def train(model, config):
    """Main training loop."""
    
    # Load data
    train_data, val_data, stoi, itos, vocab_size = load_shakespeare_data()
    
    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.lr_max,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1
    )
    
    # Create scheduler
    scheduler = CosineDecayWithWarmup(
        lr_max=config.lr_max,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
        lr_min=config.lr_min
    )
    
    # Training loop
    for step in range(config.max_steps + 1):
        # Get batch
        x, y = get_batch(train_data, config.block_size, config.batch_size)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            grad_norm = clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Step optimizer
        optimizer.step()
        
        # Update learning rate
        lr = scheduler.step()
        optimizer.lr = lr
        
        # Logging
        if step % config.log_interval == 0:
            val_loss = evaluate(model, val_data, config)
            print(f"Step {step}: loss={loss.data:.4f}, val_loss={val_loss:.4f}, lr={lr:.6f}")
            
            # Save metrics
            save_metrics(step, loss.data, val_loss, lr)
        
        # Checkpointing
        if step % config.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scheduler, step, loss.data)
        
        # Generate sample
        if step % config.sample_interval == 0:
            sample = generate_text(model, stoi, itos)
            save_sample(step, sample)
```

### Validation Loop

```python
def evaluate(model, data, config):
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    for _ in range(config.val_batches):
        x, y = get_batch(data, config.block_size, config.batch_size)
        _, loss = model(x, y)
        total_loss += loss.data
        num_batches += 1
    
    model.train()
    return total_loss / num_batches
```

### Training Configuration

```python
@dataclass
class TrainConfig:
    # Data
    block_size: int = 256
    batch_size: int = 64
    
    # Training
    max_steps: int = 100000
    warmup_steps: int = 2000
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    grad_clip: float = 1.0
    
    # Logging
    log_interval: int = 100
    checkpoint_interval: int = 5000
    sample_interval: int = 1000
    val_batches: int = 20
    
    # Model
    model_config: ModelConfig = None
```

---

## 5. Checkpointing

**File:** `scratch/checkpoint.py`

### Why Checkpoint?

Training can take hours or days. Checkpoints let us:
- Resume training if interrupted
- Save the best model
- Experiment with different configurations

### Implementation

```python
def save_checkpoint(model, optimizer, scheduler, step, loss, path):
    """Save model and optimizer state."""
    checkpoint = {
        'step': step,
        'loss': loss,
        'model_state': [p.data.copy() for p in model.parameters()],
        'optimizer_state': {
            'm': [m.copy() for m in optimizer.m],
            'v': [v.copy() for v in optimizer.v],
            't': optimizer.t
        },
        'scheduler_state': {
            'current_step': scheduler.current_step
        },
        'config': {
            'model': model.config,
            'training': train_config
        }
    }
    
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    # Keep only the last N checkpoints
    cleanup_old_checkpoints(path, keep=5)

def load_checkpoint(model, optimizer, scheduler, path):
    """Load model and optimizer state."""
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Load model parameters
    for p, data in zip(model.parameters(), checkpoint['model_state']):
        p.data = data
    
    # Load optimizer state
    optimizer.m = checkpoint['optimizer_state']['m']
    optimizer.v = checkpoint['optimizer_state']['v']
    optimizer.t = checkpoint['optimizer_state']['t']
    
    # Load scheduler state
    scheduler.current_step = checkpoint['scheduler_state']['current_step']
    
    return checkpoint['step'], checkpoint['loss']
```

### Saving Best Model

```python
def save_best_model(model, val_loss, best_val_loss, path):
    """Save model if validation loss improves."""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        # Save best model
        best_path = os.path.join(os.path.dirname(path), 'best_model.pkl')
        save_checkpoint(model, None, None, 0, val_loss, best_path)
        
        return best_val_loss
    return best_val_loss
```

---

## 6. Logging & Monitoring

**File:** `scratch/logging.py`

### Metrics Tracking

```python
class MetricsLogger:
    """Track and save training metrics."""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = {
            'steps': [],
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'grad_norm': []
        }
    
    def log(self, step, train_loss=None, val_loss=None, lr=None, grad_norm=None):
        """Log metrics at a specific step."""
        self.metrics['steps'].append(step)
        
        if train_loss is not None:
            self.metrics['train_loss'].append(train_loss)
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        if lr is not None:
            self.metrics['lr'].append(lr)
        if grad_norm is not None:
            self.metrics['grad_norm'].append(grad_norm)
    
    def save(self):
        """Save metrics to file."""
        with open(os.path.join(self.log_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(self.metrics, f)
    
    def plot(self):
        """Plot metrics."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Train loss
        axes[0, 0].plot(self.metrics['steps'], self.metrics['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        
        # Validation loss
        axes[0, 1].plot(self.metrics['steps'], self.metrics['val_loss'])
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        
        # Learning rate
        axes[1, 0].plot(self.metrics['steps'], self.metrics['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('LR')
        
        # Gradient norm
        axes[1, 1].plot(self.metrics['steps'], self.metrics['grad_norm'])
        axes[1, 1].set_title('Gradient Norm')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Norm')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'metrics.png'))
        plt.close()
```

### Progress Bar

```python
from tqdm import tqdm

def train_with_progress(model, config):
    """Training loop with progress bar."""
    
    # Setup
    train_data, val_data, _, _, _ = load_shakespeare_data()
    optimizer = Adam(model.parameters(), lr=config.lr_max)
    scheduler = CosineDecayWithWarmup(config.lr_max, config.warmup_steps, config.max_steps)
    
    # Progress bar
    pbar = tqdm(range(config.max_steps + 1), desc="Training")
    
    for step in pbar:
        # ... training steps ...
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.data:.4f}',
            'val': f'{val_loss:.4f}',
            'lr': f'{lr:.6f}'
        })
```

---

## 7. Configuration Management

**File:** `config.py`

### Unified Config

```python
@dataclass
class Config:
    """Complete configuration for training and model."""
    
    # Model
    vocab_size: int = 65
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    dropout: float = 0.2
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    multiple_of: int = 256
    
    # Training
    block_size: int = 256
    batch_size: int = 64
    max_steps: int = 100000
    warmup_steps: int = 2000
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Logging
    log_interval: int = 100
    checkpoint_interval: int = 5000
    sample_interval: int = 1000
    val_batches: int = 20
    log_dir: str = 'experiments'
    
    @classmethod
    def from_yaml(cls, path):
        """Load config from YAML file."""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path):
        """Save config to YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)
```

### Presets

```python
def small_config():
    """Small model for fast training."""
    return Config(
        dim=384,
        n_layers=6,
        n_heads=6,
        n_kv_heads=3,
        block_size=256,
        batch_size=64,
        max_steps=50000
    )

def medium_config():
    """Medium model for better quality."""
    return Config(
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        block_size=512,
        batch_size=32,
        max_steps=100000
    )

def large_config():
    """Large model (if you have GPU memory)."""
    return Config(
        dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=6,
        block_size=1024,
        batch_size=16,
        max_steps=200000
    )
```

---

## 8. Training Script

**File:** `scripts/train.py`

### Main Entry Point

```python
#!/usr/bin/env python3
"""
Train a GPT model from scratch.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scratch.model import GPT, ModelConfig
from scratch.train import train, TrainConfig
from scratch.data import load_shakespeare_data
from scratch.checkpoint import load_checkpoint

def main():
    parser = argparse.ArgumentParser(description='Train GPT from scratch')
    
    # Model args
    parser.add_argument('--dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--n_kv_heads', type=int, default=4, help='Number of KV heads')
    
    # Training args
    parser.add_argument('--block_size', type=int, default=256, help='Context length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_steps', type=int, default=100000, help='Training steps')
    parser.add_argument('--lr_max', type=float, default=3e-4, help='Max learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Misc
    parser.add_argument('--log_dir', type=str, default='experiments', help='Log directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    
    args = parser.parse_args()
    
    # Build configs
    model_config = ModelConfig(
        vocab_size=65,  # Shakespeare character set
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.block_size
    )
    
    train_config = TrainConfig(
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr_max=args.lr_max,
        grad_clip=args.grad_clip,
        log_dir=args.log_dir
    )
    
    # Create model
    model = GPT(model_config)
    
    # Resume if requested
    if args.resume:
        load_checkpoint(model, None, None, args.resume)
        print(f"Loaded checkpoint from {args.resume}")
    
    # Run training
    if not args.eval_only:
        train(model, train_config)
    else:
        # Evaluation only
        from scratch.train import evaluate
        _, val_data, _, _, _ = load_shakespeare_data()
        val_loss = evaluate(model, val_data, train_config)
        print(f"Validation loss: {val_loss:.4f}")

if __name__ == '__main__':
    main()
```

### Running the Script

```bash
# Small model (fast training)
python scripts/train.py --dim 384 --n_layers 6 --max_steps 50000

# Medium model (better quality)
python scripts/train.py --dim 512 --n_layers 8 --max_steps 100000

# Large model (requires GPU)
python scripts/train.py --dim 768 --n_layers 12 --batch_size 16 --max_steps 200000

# Resume training
python scripts/train.py --resume experiments/checkpoint_50000.pkl

# Just evaluate
python scripts/train.py --eval_only
```

---

## 9. Design Decisions at a Glance

| Component | Our Choice | Alternative | Reason |
|---|---|---|---|
| Tokenization | Character-level | BPE, WordPiece | Simpler for learning; good enough for Shakespeare |
| Batch sampling | Random contiguous chunks | Sequential | Random gives more diverse gradients |
| LR schedule | Cosine decay with warmup | Constant, Step decay | Standard for transformers; works well |
| Gradient clipping | Always applied | Only when needed | Prevents explosions; little downside |
| Checkpointing | Pickle | PyTorch's state_dict | Simple; works with our custom classes |
| Logging | Custom | TensorBoard, WandB | No external dependencies; enough for learning |
| Validation | Random batches | Full evaluation | Fast; good enough for monitoring |

---

## 10. Test Suite

**File:** `tests/test_training.py`

| Test | What It Verifies |
|---|---|
| `test_data_loading` | Data loads correctly, train/val split works |
| `test_get_batch` | Batch has correct shape, targets are shifted |
| `test_dataloader` | DataLoader iterates correctly, shuffles |
| `test_schedule_warmup` | Warmup increases LR from 0 to lr_max |
| `test_schedule_decay` | Cosine decay decreases LR correctly |
| `test_schedule_step` | Step updates LR correctly |
| `test_checkpoint_save_load` | Checkpoint saves and loads correctly |
| `test_checkpoint_resume` | Training resumes from checkpoint |
| `test_training_step` | Single training step updates weights |
| `test_loss_decreases` | Loss decreases over training |
| `test_gradient_clipping` | Gradients are clipped when exceeding norm |

### Test Example: Training Step

```python
def test_training_step():
    config = ModelConfig(dim=64, n_layers=2, n_heads=2, n_kv_heads=1)
    model = GPT(config)
    
    # Get initial parameters
    initial_params = [p.data.copy() for p in model.parameters()]
    
    # One training step
    train_data, _, _, _, _ = load_shakespeare_data()
    x, y = get_batch(train_data, block_size=64, batch_size=16)
    logits, loss = model(x, y)
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Parameters should have changed
    for p, p_init in zip(model.parameters(), initial_params):
        assert not np.allclose(p.data, p_init, atol=1e-5)
```
