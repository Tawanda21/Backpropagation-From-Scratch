# Phase 6 — Training the GPT

**Goal:** Actually train our GPT model on Shakespeare and watch it learn to generate text

A bottom-up walkthrough of the complete training process, what to expect, and how to debug issues.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Training Setup](#2-training-setup)
3. [The Training Run](#3-the-training-run)
4. [Monitoring Progress](#4-monitoring-progress)
5. [Expected Results](#5-expected-results)
6. [Debugging Common Issues](#6-debugging-common-issues)
7. [Generating Text](#7-generating-text)
8. [Full Training Script](#8-full-training-script)
9. [What to Expect at Each Stage](#9-what-to-expect-at-each-stage)
10. [Design Decisions at a Glance](#10-design-decisions-at-a-glance)

---

## 1. The Big Picture

Training a language model involves iteratively updating weights to minimize the cross-entropy loss. The process looks like:

```
Raw Text -> Tokenizer -> Token IDs -> Model -> Logits -> Loss -> Gradients -> Update
     |                                                                         |
     +-------------------------------------------------------------------------+
                              (Repeat for each step)
```

**Key insight:** Training is just repeated gradient descent on a loss function. The model slowly learns the statistical patterns in the text.

---

## 2. Training Setup

**File:** `scratch/train_config.py`

### Complete Configuration

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model architecture
    vocab_size: int = 65  # Shakespeare character set
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 3
    max_seq_len: int = 512
    dropout: float = 0.2
    
    # Training hyperparameters
    block_size: int = 256
    batch_size: int = 64
    max_steps: int = 50000
    warmup_steps: int = 1000
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Data
    data_dir: str = 'data/processed'
    
    # Logging
    log_interval: int = 50
    eval_interval: int = 500
    checkpoint_interval: int = 5000
    sample_interval: int = 1000
    save_dir: str = 'experiments'
    
    # Generation
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 200
```

### Building Everything

```python
def setup_training(config):
    """Set up all components for training."""
    
    # 1. Load data
    print("Loading data...")
    train_tokens, val_tokens, tokenizer, metadata = load_dataset(config.data_dir)
    config.vocab_size = metadata['vocab_size']
    
    # 2. Create model
    print("Creating model...")
    model_config = ModelConfig(
        vocab_size=config.vocab_size,
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        max_seq_len=config.block_size,
        dropout=config.dropout
    )
    model = GPT(model_config)
    
    print(f"Model has {count_parameters(model):,} parameters")
    
    # 3. Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.lr_max,
        weight_decay=config.weight_decay
    )
    
    # 4. Create scheduler
    scheduler = CosineDecayWithWarmup(
        lr_max=config.lr_max,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
        lr_min=config.lr_min
    )
    
    # 5. Create logger
    logger = MetricsLogger(config.save_dir)
    
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'tokenizer': tokenizer,
        'train_tokens': train_tokens,
        'val_tokens': val_tokens,
        'logger': logger,
        'config': config
    }
```

---

## 3. The Training Run

**File:** `scratch/train.py`

### Main Training Loop

```python
def train():
    """Main training function."""
    
    # Setup
    config = TrainingConfig()
    components = setup_training(config)
    
    model = components['model']
    optimizer = components['optimizer']
    scheduler = components['scheduler']
    tokenizer = components['tokenizer']
    train_tokens = components['train_tokens']
    val_tokens = components['val_tokens']
    logger = components['logger']
    
    print(f"Starting training for {config.max_steps} steps...")
    print(f"Training tokens: {len(train_tokens):,}")
    print(f"Validation tokens: {len(val_tokens):,}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for step in range(config.max_steps + 1):
        # --- Forward pass ---
        x, y = get_batch(train_tokens, config.block_size, config.batch_size)
        logits, loss = model(x, y)
        
        # --- Backward pass ---
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            grad_norm = clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate
        lr = scheduler.step()
        optimizer.lr = lr
        
        # --- Logging ---
        if step % config.log_interval == 0:
            # Get validation loss
            val_loss = evaluate(val_tokens, model, config)
            
            # Log metrics
            logger.log(
                step=step,
                train_loss=loss.data,
                val_loss=val_loss,
                lr=lr,
                grad_norm=grad_norm if config.grad_clip > 0 else None
            )
            
            # Print progress
            print(f"Step {step:6d} | "
                  f"Loss: {loss.data:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"Grad: {grad_norm:.4f}" if config.grad_clip > 0 else "")
        
        # --- Checkpointing ---
        if step % config.checkpoint_interval == 0 and step > 0:
            save_checkpoint(
                model, optimizer, scheduler,
                step, loss.data,
                os.path.join(config.save_dir, f'checkpoint_{step:06d}.pkl')
            )
            logger.save()
        
        # --- Generate samples ---
        if step % config.sample_interval == 0 and step > 0:
            sample_text = generate_sample(
                model, tokenizer, config,
                prompt="ROMEO:\n"
            )
            save_sample(step, sample_text, config.save_dir)
    
    print("Training complete!")
```

### Validation Function

```python
def evaluate(tokens, model, config):
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0
    num_batches = min(20, len(tokens) // (config.block_size * config.batch_size))
    
    for _ in range(num_batches):
        x, y = get_batch(tokens, config.block_size, config.batch_size)
        _, loss = model(x, y)
        total_loss += loss.data
    
    model.train()
    return total_loss / num_batches
```

---

## 4. Monitoring Progress

**File:** `experiments/monitor.py`

### Real-time Monitoring

```python
def monitor_training(log_dir):
    """Monitor training progress in real-time."""
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    
    # Load metrics
    with open(os.path.join(log_dir, 'metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(metrics['steps'], metrics['train_loss'], label='Train')
    axes[0, 0].plot(metrics['steps'], metrics['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Learning rate
    axes[0, 1].plot(metrics['steps'], metrics['lr'])
    axes[0, 1].set_title('Learning Rate')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('LR')
    axes[0, 1].grid(True)
    
    # Gradient norm
    if 'grad_norm' in metrics and metrics['grad_norm']:
        axes[1, 0].plot(metrics['steps'], metrics['grad_norm'])
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Norm')
        axes[1, 0].grid(True)
    
    # Loss gap
    gap = np.array(metrics['train_loss']) - np.array(metrics['val_loss'])
    axes[1, 1].plot(metrics['steps'][:len(gap)], gap)
    axes[1, 1].set_title('Train-Val Gap')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Gap')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
```

### Training Dashboard

```python
class TrainingDashboard:
    """Interactive training dashboard."""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics = None
        self.load_metrics()
    
    def load_metrics(self):
        """Load latest metrics."""
        metrics_path = os.path.join(self.log_dir, 'metrics.pkl')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                self.metrics = pickle.load(f)
    
    def update(self):
        """Update and display dashboard."""
        self.load_metrics()
        
        if self.metrics is None:
            print("No metrics found")
            return
        
        clear_output(wait=True)
        
        # Summary stats
        latest_step = self.metrics['steps'][-1]
        latest_loss = self.metrics['train_loss'][-1]
        latest_val = self.metrics['val_loss'][-1]
        latest_lr = self.metrics['lr'][-1]
        
        print("=" * 60)
        print(f"Training Dashboard - Step {latest_step}")
        print("=" * 60)
        print(f"Train Loss: {latest_loss:.4f}")
        print(f"Val Loss:   {latest_val:.4f}")
        print(f"Learning Rate: {latest_lr:.6f}")
        print(f"Train-Valid Gap: {latest_loss - latest_val:.4f}")
        print("=" * 60)
        
        # Plot
        self.plot_metrics()
    
    def plot_metrics(self):
        """Plot training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(self.metrics['steps'], self.metrics['train_loss'], label='Train')
        axes[0, 0].plot(self.metrics['steps'], self.metrics['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.metrics['steps'], self.metrics['lr'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].grid(True)
        
        # Gradient norm
        if 'grad_norm' in self.metrics and self.metrics['grad_norm']:
            axes[1, 0].plot(self.metrics['steps'], self.metrics['grad_norm'])
            axes[1, 0].set_title('Gradient Norm')
            axes[1, 0].grid(True)
        
        # Loss gap
        gap = np.array(self.metrics['train_loss']) - np.array(self.metrics['val_loss'])
        axes[1, 1].plot(self.metrics['steps'][:len(gap)], gap)
        axes[1, 1].set_title('Train-Val Gap')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
```

---

## 5. Expected Results

### Loss Curves

Here's what you should expect to see:

| Steps | Train Loss | Val Loss | What's Happening |
|-------|------------|----------|------------------|
| 0 | ~10.4 | ~10.4 | Random initialization (log(vocab_size)) |
| 100 | ~8.0 | ~8.1 | Learning starts, loss drops rapidly |
| 500 | ~5.0 | ~5.2 | Model learns common patterns |
| 1000 | ~3.5 | ~3.8 | Learning grammar and basic structure |
| 5000 | ~2.0 | ~2.3 | Learning character-level patterns |
| 10000 | ~1.5 | ~1.8 | Learning words and phrases |
| 50000 | ~1.0 | ~1.3 | Learning sentence structure |

### Sample Text Progression

**Step 0 (Random):**
```
ROMEO:
fghzklmowpqyjfghzklmowpqyjfghzklmowpqyj...
```

**Step 5000:**
```
ROMEO:
the the the the the the the the the the...
```

**Step 20000:**
```
ROMEO:
I am a man that I have not seen the world,
And I have not seen the world,
And I have not seen the world...
```

**Step 50000 (Final):**
```
ROMEO:
I do not think that I have seen the day
When I have found a friend that loved me so.
But come, the night is dark and we must go
To seek the duke, who is our enemy.
```

---

## 6. Debugging Common Issues

### Issue 1: Loss Not Decreasing

**Symptoms:** Loss stays around 10.4 (log(vocab_size))

**Possible Causes:**
- Learning rate too low
- Learning rate too high (gradients exploding)
- Weight initialization wrong
- Data not loaded correctly

**Fixes:**
```python
# Check learning rate
print(f"Current LR: {optimizer.lr}")

# Check gradients
for name, param in zip(model.parameter_names(), model.parameters()):
    if param.grad is not None:
        print(f"{name}: mean={param.grad.mean():.6f}, std={param.grad.std():.6f}")

# Check data
x, y = get_batch(train_tokens, block_size, batch_size)
print(f"X shape: {x.data.shape}")
print(f"Y shape: {y.data.shape}")
print(f"Unique in X: {len(np.unique(x.data))}")
```

### Issue 2: Loss Exploding (NaN or Inf)

**Symptoms:** Loss becomes NaN or Inf after a few steps

**Possible Causes:**
- Learning rate too high
- Gradient clipping not applied
- Numerical instability in softmax
- Weight initialization too large

**Fixes:**
```python
# Reduce learning rate
config.lr_max = 1e-4  # Try lower

# Apply gradient clipping
clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check for NaN in gradients
for param in model.parameters():
    if param.grad is not None and np.any(np.isnan(param.grad)):
        print("NaN in gradients!")

# Add epsilon in softmax
def stable_softmax(x, eps=1e-8):
    max_x = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + eps)
```

### Issue 3: Overfitting (Low Train Loss, High Val Loss)

**Symptoms:** Train loss keeps decreasing, val loss increases

**Possible Causes:**
- Model too large for dataset
- Too many training steps
- No regularization (dropout, weight decay)

**Fixes:**
```python
# Increase dropout
config.dropout = 0.3

# Add weight decay
optimizer = Adam(model.parameters(), lr=config.lr_max, weight_decay=0.1)

# Early stopping
if val_loss > best_val_loss * 1.1:
    print("Early stopping!")
    break

# Reduce model size
config.dim = 256
config.n_layers = 4
```

### Issue 4: Vanishing Gradients

**Symptoms:** Loss plateaus, gradients near 0 in early layers

**Possible Causes:**
- Too many layers
- Post-norm instead of pre-norm
- Wrong initialization

**Fixes:**
```python
# Check gradient norm per layer
for i, (name, param) in enumerate(zip(model.parameter_names(), model.parameters())):
    if param.grad is not None:
        norm = np.linalg.norm(param.grad)
        print(f"Layer {i}: {norm:.6f}")

# Use pre-norm (we already are)
# Check weight initialization
def init_weights(module):
    if isinstance(module, Linear):
        module.weight.data *= 0.02 / np.sqrt(2 * config.n_layers)
```

---

## 7. Generating Text

**File:** `scripts/generate.py`

### Generation Script

```python
def generate_text(model, tokenizer, config, prompt="ROMEO:\n"):
    """Generate text from a trained model."""
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = np.array(tokens, dtype=np.int64)[None, :]  # Add batch dimension
    
    generated = tokens.copy()
    
    for _ in range(config.max_new_tokens):
        # Only use last block_size tokens
        context = generated[:, -config.block_size:]
        
        # Forward pass
        logits, _ = model(Tensor(context))
        logits = logits.data
        
        # Get last token's logits
        next_logits = logits[:, -1, :] / config.temperature
        
        # Apply top-k
        if config.top_k is not None:
            idx = np.argpartition(next_logits, -config.top_k)[:, -config.top_k:]
            values = np.take_along_axis(next_logits, idx, axis=-1)
            threshold = np.min(values, axis=-1, keepdims=True)
            next_logits = np.where(next_logits >= threshold, next_logits, -np.inf)
        
        # Sample
        probs = np.exp(next_logits - np.max(next_logits, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        next_token = np.random.choice(probs.shape[-1], p=probs[0])
        
        # Append
        generated = np.concatenate([generated, [[next_token]]], axis=-1)
        
        # Stop on EOS
        if hasattr(tokenizer, 'eos_token_id') and next_token == tokenizer.eos_token_id:
            break
    
    # Decode
    return tokenizer.decode(generated[0])

def interactive_generate(model_path, tokenizer_path, config_path):
    """Interactive text generation."""
    # Load model
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    config = load_config(config_path)
    
    print("Ready to generate. Enter 'quit' to exit.")
    
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() == 'quit':
            break
        
        text = generate_text(model, tokenizer, config, prompt)
        print("\nGenerated:")
        print(text)
        print("-" * 60)
```

---

## 8. Full Training Script

**File:** `scripts/train_full.py`

```python
#!/usr/bin/env python3
"""
Complete training script for GPT on Shakespeare.
"""

import os
import sys
import argparse
import pickle
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scratch.model import GPT, ModelConfig
from scratch.tensor import Tensor
from scratch.nn import Module
from scratch.optim import Adam
from scratch.loss import cross_entropy
from scratch.transformer import GPT
from scratch.scheduler import CosineDecayWithWarmup
from scratch.checkpoint import save_checkpoint, load_checkpoint
from scratch.data import get_batch, load_dataset
from scratch.logging import MetricsLogger
from scratch.tokenizer import CharTokenizer

@dataclass
class TrainingConfig:
    # Model
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 3
    dropout: float = 0.2
    
    # Training
    block_size: int = 256
    batch_size: int = 64
    max_steps: int = 50000
    warmup_steps: int = 1000
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Data
    data_dir: str = 'data/processed'
    
    # Logging
    log_interval: int = 50
    eval_interval: int = 500
    checkpoint_interval: int = 5000
    sample_interval: int = 1000
    save_dir: str = 'experiments'
    
    # Generation
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 200

def main():
    parser = argparse.ArgumentParser(description='Train GPT on Shakespeare')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Setup
    print("=" * 70)
    print("Training GPT on Shakespeare")
    print("=" * 70)
    
    # Load data
    print("Loading data...")
    train_tokens, val_tokens, tokenizer, metadata = load_dataset(config.data_dir)
    vocab_size = metadata['vocab_size']
    
    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Create model
    print("Creating model...")
    model_config = ModelConfig(
        vocab_size=vocab_size,
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        max_seq_len=config.block_size,
        dropout=config.dropout
    )
    model = GPT(model_config)
    
    # Count parameters
    def count_params():
        total = 0
        for p in model.parameters():
            total += p.data.size
        return total
    
    print(f"Model parameters: {count_params():,}")
    
    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.lr_max,
        weight_decay=config.weight_decay
    )
    
    # Create scheduler
    scheduler = CosineDecayWithWarmup(
        lr_max=config.lr_max,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
        lr_min=config.lr_min
    )
    
    # Create logger
    logger = MetricsLogger(config.save_dir)
    
    # Resume if requested
    start_step = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_step, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, args.resume
        )
    
    # Evaluate only
    if args.eval_only:
        val_loss = evaluate(val_tokens, model, config)
        print(f"Validation loss: {val_loss:.4f}")
        return
    
    # Training loop
    print("\nStarting training...")
    print("-" * 70)
    
    for step in range(start_step, config.max_steps + 1):
        # Forward pass
        x, y = get_batch(train_tokens, config.block_size, config.batch_size)
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = 0
        if config.grad_clip > 0:
            grad_norm = clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate
        lr = scheduler.step()
        optimizer.lr = lr
        
        # Logging
        if step % config.log_interval == 0:
            # Validation
            if step % config.eval_interval == 0:
                val_loss = evaluate(val_tokens, model, config)
                
                # Track best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, scheduler,
                        step, val_loss,
                        os.path.join(config.save_dir, 'best_model.pkl')
                    )
            else:
                val_loss = None
            
            # Log
            logger.log(
                step=step,
                train_loss=loss.data,
                val_loss=val_loss,
                lr=lr,
                grad_norm=grad_norm
            )
            
            # Print
            print(f"Step {step:6d} | "
                  f"Loss: {loss.data:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"Grad: {grad_norm:.4f}")
        
        # Generate samples
        if step % config.sample_interval == 0 and step > 0:
            sample = generate_sample(
                model, tokenizer, config,
                prompt="ROMEO:\n"
            )
            sample_path = os.path.join(config.save_dir, f'sample_{step:06d}.txt')
            with open(sample_path, 'w') as f:
                f.write(sample)
            
            print("\n" + "=" * 40)
            print(f"Sample at step {step}:")
            print("=" * 40)
            print(sample)
            print("=" * 40 + "\n")
        
        # Checkpoint
        if step % config.checkpoint_interval == 0 and step > 0:
            save_checkpoint(
                model, optimizer, scheduler,
                step, loss.data,
                os.path.join(config.save_dir, f'checkpoint_{step:06d}.pkl')
            )
            logger.save()
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 70)

if __name__ == '__main__':
    main()
```

---

## 9. What to Expect at Each Stage

### Stage 1: Random Initialization (Step 0)
- Loss: ~log(vocab_size) ≈ 4.18 for 65 characters
- Text: Random gibberish
- Gradients: Small, random

### Stage 2: Learning the Alphabet (Steps 1-1000)
- Loss: Rapidly drops to ~3.0
- Text: Starts recognizing characters
- Gradients: Large, model is learning fast

### Stage 3: Learning Words (Steps 1000-10000)
- Loss: Slowly drops to ~2.0
- Text: Repeats common words, some structure
- Gradients: Stabilizing

### Stage 4: Learning Grammar (Steps 10000-30000)
- Loss: Gradual decline to ~1.5
- Text: Coherent sentences, some grammar
- Gradients: Smaller, fine-tuning

### Stage 5: Refinement (Steps 30000-50000)
- Loss: Slowly approaches ~1.0
- Text: Genuinely coherent, starts to sound like Shakespeare
- Gradients: Very small, model is converging

---

## 10. Design Decisions at a Glance

| Component | Our Choice | Alternative | Reason |
|---|---|---|---|
| Training steps | 50,000 | 10,000 or 100,000 | Good balance for Shakespeare |
| Batch size | 64 | 32 or 128 | Fits in memory, good gradient estimates |
| Learning rate | 3e-4 | 1e-3 or 1e-5 | Standard for Adam on transformers |
| Warmup steps | 1,000 | 0 or 5,000 | Stabilizes early training |
| Gradient clipping | 1.0 | 0.1 or 10.0 | Prevents explosions, standard value |
| Weight decay | 0.1 | 0.0 or 0.01 | Regularization for transformer |
| Evaluation frequency | Every 500 steps | Every 100 or 1000 | Good balance |

---
