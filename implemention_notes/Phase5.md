# Phase 5 — Tokenization & Data Pipeline

**Goal:** Build a robust character-level tokenizer and data preparation pipeline for our GPT model

A bottom-up walkthrough of every component we built, what it does, and why each design decision was made.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Character-Level Tokenizer](#2-character-level-tokenizer)
3. [Special Tokens](#3-special-tokens)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Data Pipeline](#5-data-pipeline)
6. [Streaming DataLoader](#6-streaming-dataloader)
7. [Text Generation Utilities](#7-text-generation-utilities)
8. [Design Decisions at a Glance](#8-design-decisions-at-a-glance)
9. [Test Suite](#9-test-suite)

---

## 1. The Big Picture

Tokenization converts raw text into numbers that the model can process. The data flows like this:

```
Raw Text (Shakespeare)
    │
    ▼
Character-Level Tokenizer
    │
    ▼
Token IDs (integers)
    │
    ▼
Dataset Preparation
    │
    ▼
Training/Validation Split
    │
    ▼
DataLoader (streaming batches)
    │
    ▼
Model Input (B, T) token IDs
```

**Key insight:** Tokenization is the bridge between human language and machine learning. A good tokenizer is simple, complete, and handles edge cases gracefully.

---

## 2. Character-Level Tokenizer

**File:** `scratch/tokenizer.py`

### Why Character-Level?

Character-level tokenization treats each character as a token:
- **Simple:** No complex algorithms, no external dependencies
- **Complete:** Every character in the text is represented
- **Small vocabulary:** Typically 50-100 characters (vs 30,000+ for BPE)
- **Perfect for learning:** You understand exactly what's happening

### Implementation

```python
class CharTokenizer:
    """Character-level tokenizer for text."""
    
    def __init__(self, text=None):
        """
        Initialize tokenizer from text or pre-existing vocab.
        
        Args:
            text: Optional text to build vocab from
        """
        if text is not None:
            self.build_vocab(text)
        else:
            self.stoi = {}
            self.itos = {}
            self.vocab_size = 0
    
    def build_vocab(self, text):
        """Build vocabulary from text."""
        # Get all unique characters
        chars = sorted(list(set(text)))
        
        # Create mappings
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        print(f"Built vocabulary with {self.vocab_size} characters")
    
    def encode(self, text):
        """
        Convert text to token IDs.
        
        Args:
            text: String to encode
        
        Returns:
            List of integers (token IDs)
        """
        # Handle unknown characters
        unknown_token = self.stoi.get('�', None)
        if unknown_token is None:
            # If no unknown token, use 0 (first char) as fallback
            unknown_token = 0
        
        tokens = []
        for ch in text:
            if ch in self.stoi:
                tokens.append(self.stoi[ch])
            else:
                tokens.append(unknown_token)
                print(f"Warning: Unknown character '{ch}' replaced with unknown token")
        
        return tokens
    
    def decode(self, tokens):
        """
        Convert token IDs back to text.
        
        Args:
            tokens: List of integers (token IDs)
        
        Returns:
            String
        """
        chars = []
        for token in tokens:
            if token in self.itos:
                chars.append(self.itos[token])
            else:
                chars.append('�')  # Replacement character
        return ''.join(chars)
    
    def encode_tensor(self, text):
        """Encode text and return as Tensor."""
        tokens = self.encode(text)
        return Tensor(np.array(tokens, dtype=np.int64))
    
    def decode_tensor(self, tensor):
        """Decode Tensor to text."""
        tokens = tensor.data.astype(np.int64).flatten()
        return self.decode(tokens)
    
    def save(self, path):
        """Save tokenizer to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'stoi': self.stoi,
                'itos': self.itos,
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, path):
        """Load tokenizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.stoi = data['stoi']
        self.itos = data['itos']
        self.vocab_size = data['vocab_size']
        return self
```

### Example Usage

```python
# Create tokenizer from text
text = "Hello, world!"
tokenizer = CharTokenizer(text)

# Encode
tokens = tokenizer.encode("Hello")
print(tokens)  # [44, 65, 75, 75, 82]

# Decode
text = tokenizer.decode(tokens)
print(text)  # "Hello"

# Encode with unknown characters
tokens = tokenizer.encode("Hello 👋")
# Warning: Unknown character '👋' replaced with unknown token
print(tokens)  # [44, 65, 75, 75, 82, 0]
```

### Handling Unknown Characters

For real-world text, characters outside the training set will appear. We handle this with:

1. **Unknown token:** Map all unknown characters to a special token
2. **Warning:** Print warnings so you know when it happens
3. **Robust decoding:** Handle tokens that don't map back

```python
def get_vocab_stats(self, text):
    """Get statistics about vocabulary coverage."""
    unique_chars = set(text)
    known_chars = set(self.stoi.keys())
    unknown_chars = unique_chars - known_chars
    
    coverage = len(known_chars) / len(unique_chars)
    
    return {
        'unique_chars': len(unique_chars),
        'known_chars': len(known_chars),
        'unknown_chars': len(unknown_chars),
        'coverage': coverage
    }
```

---

## 3. Special Tokens

**File:** `scratch/tokenizer.py`

### Why Special Tokens?

Special tokens give the model control signals:
- **BOS:** Beginning of sequence — signals the start
- **EOS:** End of sequence — signals the end
- **PAD:** Padding — for batching variable-length sequences
- **UNK:** Unknown — for out-of-vocabulary characters

### Implementation

```python
class SpecialTokenizer(CharTokenizer):
    """Tokenizer with special tokens."""
    
    SPECIAL_TOKENS = {
        'PAD': 0,   # Padding token
        'UNK': 1,   # Unknown token
        'BOS': 2,   # Beginning of sequence
        'EOS': 3,   # End of sequence
    }
    
    def __init__(self, text=None):
        super().__init__(text)
        self.special_tokens = self.SPECIAL_TOKENS.copy()
        
        if text is not None:
            self.build_vocab(text)
    
    def build_vocab(self, text):
        """Build vocabulary with special tokens."""
        # Get all unique characters
        chars = sorted(list(set(text)))
        
        # Start with special tokens
        self.stoi = {ch: i + len(self.SPECIAL_TOKENS) 
                     for i, ch in enumerate(chars)}
        
        # Add special token mappings
        for token_name, token_id in self.SPECIAL_TOKENS.items():
            # Each special token is represented by a special character
            # We'll use special strings like '<PAD>', '<UNK>', etc.
            special_char = f'<{token_name}>'
            self.stoi[special_char] = token_id
        
        # Reverse mapping
        self.itos = {i: ch for ch, i in self.stoi.items()}
        
        # Vocabulary size includes special tokens
        self.vocab_size = len(self.stoi)
        
        # Store token IDs for easy access
        self.pad_token_id = self.SPECIAL_TOKENS['PAD']
        self.unk_token_id = self.SPECIAL_TOKENS['UNK']
        self.bos_token_id = self.SPECIAL_TOKENS['BOS']
        self.eos_token_id = self.SPECIAL_TOKENS['EOS']
    
    def encode(self, text, add_bos=False, add_eos=False):
        """Encode with optional special tokens."""
        tokens = []
        
        if add_bos:
            tokens.append(self.bos_token_id)
        
        tokens.extend(super().encode(text))
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, tokens, skip_special=False):
        """Decode with optional special tokens."""
        chars = []
        for token in tokens:
            if skip_special and token in self.SPECIAL_TOKENS.values():
                continue
            
            if token in self.itos:
                char = self.itos[token]
                # Don't output special token strings
                if skip_special and char.startswith('<') and char.endswith('>'):
                    continue
                chars.append(char)
            else:
                chars.append('�')
        
        return ''.join(chars)
```

### Example Usage

```python
# Create tokenizer with special tokens
tokenizer = SpecialTokenizer("Hello, world!")

# Encode with special tokens
tokens = tokenizer.encode("Hello", add_bos=True, add_eos=True)
print(tokens)  # [2, 44, 65, 75, 75, 82, 3]

# Decode skipping special tokens
text = tokenizer.decode(tokens, skip_special=True)
print(text)  # "Hello"

# Decode including special tokens
text = tokenizer.decode(tokens, skip_special=False)
print(text)  # "<BOS>Hello<EOS>"
```

---

## 4. Dataset Preparation

**File:** `data/prepare_data.py`

### The Complete Pipeline

```python
def prepare_dataset(text_path, save_dir, tokenizer_class=CharTokenizer):
    """
    Prepare dataset for training.
    
    Args:
        text_path: Path to raw text file
        save_dir: Directory to save processed data
        tokenizer_class: Tokenizer class to use
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load text
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Loaded {len(text)} characters")
    
    # Create tokenizer
    tokenizer = tokenizer_class(text)
    
    # Encode all text
    tokens = tokenizer.encode(text)
    print(f"Encoded to {len(tokens)} tokens")
    
    # Split into train and validation
    split_point = int(0.9 * len(tokens))
    train_tokens = tokens[:split_point]
    val_tokens = tokens[split_point:]
    
    print(f"Train tokens: {len(train_tokens)}")
    print(f"Validation tokens: {len(val_tokens)}")
    
    # Save data
    with open(os.path.join(save_dir, 'train.bin'), 'wb') as f:
        np.array(train_tokens, dtype=np.int64).tofile(f)
    
    with open(os.path.join(save_dir, 'val.bin'), 'wb') as f:
        np.array(val_tokens, dtype=np.int64).tofile(f)
    
    # Save tokenizer
    tokenizer.save(os.path.join(save_dir, 'tokenizer.pkl'))
    
    # Save metadata
    metadata = {
        'vocab_size': tokenizer.vocab_size,
        'train_size': len(train_tokens),
        'val_size': len(val_tokens),
        'vocab': tokenizer.stoi,
        'text_chars': len(text)
    }
    
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return tokenizer, train_tokens, val_tokens

def load_dataset(data_dir):
    """Load prepared dataset."""
    # Load tokens
    with open(os.path.join(data_dir, 'train.bin'), 'rb') as f:
        train_tokens = np.fromfile(f, dtype=np.int64)
    
    with open(os.path.join(data_dir, 'val.bin'), 'rb') as f:
        val_tokens = np.fromfile(f, dtype=np.int64)
    
    # Load tokenizer
    tokenizer = CharTokenizer()
    tokenizer.load(os.path.join(data_dir, 'tokenizer.pkl'))
    
    # Load metadata
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    return train_tokens, val_tokens, tokenizer, metadata
```

### Dataset Statistics

```python
def analyze_dataset(tokens, tokenizer):
    """Analyze dataset statistics."""
    
    stats = {
        'total_tokens': len(tokens),
        'unique_tokens': len(set(tokens)),
        'vocab_size': tokenizer.vocab_size,
        'coverage': len(set(tokens)) / tokenizer.vocab_size
    }
    
    # Token frequency
    frequencies = np.bincount(tokens)
    freq_stats = {
        'min': np.min(frequencies[frequencies > 0]),
        'max': np.max(frequencies),
        'mean': np.mean(frequencies[frequencies > 0]),
        'std': np.std(frequencies[frequencies > 0])
    }
    
    stats.update(freq_stats)
    
    # Most common tokens
    top_tokens = np.argsort(frequencies)[-10:][::-1]
    stats['top_tokens'] = [
        {
            'token': tokenizer.itos[t],
            'id': t,
            'freq': frequencies[t]
        }
        for t in top_tokens
    ]
    
    return stats
```

---

## 5. Data Pipeline

**File:** `scratch/data.py`

### Complete Data Module

```python
class DataModule:
    """Complete data management for training."""
    
    def __init__(self, data_dir, block_size, batch_size):
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        
        # Load dataset
        self.load()
    
    def load(self):
        """Load dataset and tokenizer."""
        self.train_tokens, self.val_tokens, self.tokenizer, self.metadata = load_dataset(self.data_dir)
        self.vocab_size = self.metadata['vocab_size']
    
    def get_batch(self, split='train'):
        """Get a batch from train or validation set."""
        data = self.train_tokens if split == 'train' else self.val_tokens
        
        # Random starting positions
        idx = np.random.randint(0, len(data) - self.block_size, (self.batch_size,))
        
        # Build input and target sequences
        x = np.stack([data[i:i+self.block_size] for i in idx])
        y = np.stack([data[i+1:i+self.block_size+1] for i in idx])
        
        return Tensor(x), Tensor(y)
    
    def get_batch_tensor(self, split='train'):
        """Get batch as Tensors."""
        x, y = self.get_batch(split)
        return Tensor(x), Tensor(y)
    
    def get_data_loader(self, split='train', shuffle=True):
        """Get a DataLoader for the dataset."""
        data = self.train_tokens if split == 'train' else self.val_tokens
        return DataLoader(data, self.block_size, self.batch_size, shuffle=shuffle)
    
    def sample_text(self, num_chars=100, split='train'):
        """Sample a random piece of text from the dataset."""
        data = self.train_tokens if split == 'train' else self.val_tokens
        
        # Random starting position
        start = np.random.randint(0, len(data) - num_chars)
        tokens = data[start:start+num_chars]
        
        # Decode
        return self.tokenizer.decode(tokens)
    
    def stats(self):
        """Get dataset statistics."""
        return {
            'vocab_size': self.vocab_size,
            'train_tokens': len(self.train_tokens),
            'val_tokens': len(self.val_tokens),
            'block_size': self.block_size,
            'batch_size': self.batch_size,
            'batches_per_epoch': len(self.train_tokens) // (self.block_size * self.batch_size)
        }
```

### Full Training Example

```python
# Prepare dataset
prepare_dataset('data/shakespeare.txt', 'data/processed')

# Create data module
data = DataModule('data/processed', block_size=256, batch_size=64)

# Get batch
x, y = data.get_batch('train')
print(f"Input shape: {x.data.shape}")  # (64, 256)
print(f"Target shape: {y.data.shape}")  # (64, 256)

# Sample text
text = data.sample_text(200)
print(f"Sample: {text}")
```

---

## 6. Streaming DataLoader

**File:** `scratch/data.py`

### Memory-Efficient Loading

For large datasets, loading everything into memory isn't possible. We use memory mapping:

```python
class StreamingDataLoader:
    """Memory-efficient data loader for large datasets."""
    
    def __init__(self, data_path, block_size, batch_size, shuffle=True):
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Memory map the data file
        self.data = np.memmap(data_path, dtype=np.int64, mode='r')
        self.length = len(self.data)
        
        # Precompute valid starting positions
        self.valid_starts = np.arange(self.length - block_size)
        
        # Shuffle indices
        self.indices = self.valid_starts.copy()
        if shuffle:
            np.random.shuffle(self.indices)
        
        self.position = 0
    
    def __iter__(self):
        self.position = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.position >= len(self.indices):
            raise StopIteration
        
        # Get batch of starting positions
        batch_indices = self.indices[self.position:self.position + self.batch_size]
        self.position += self.batch_size
        
        # Build batch
        x = np.zeros((len(batch_indices), self.block_size), dtype=np.int64)
        y = np.zeros((len(batch_indices), self.block_size), dtype=np.int64)
        
        for i, start in enumerate(batch_indices):
            x[i] = self.data[start:start + self.block_size]
            y[i] = self.data[start + 1:start + self.block_size + 1]
        
        return Tensor(x), Tensor(y)
    
    def reset(self):
        """Reset the data loader."""
        self.position = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
```

### Usage

```python
# Create streaming loader
loader = StreamingDataLoader(
    'data/processed/train.bin',
    block_size=256,
    batch_size=64
)

# Iterate through batches
for step, (x, y) in enumerate(loader):
    # Train on batch
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    
    if step > 1000:
        break
```

---

## 7. Text Generation Utilities

**File:** `scratch/generation.py`

### Complete Generation Pipeline

```python
class TextGenerator:
    """Text generation utilities."""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Cache for faster generation
        self.cache = {}
    
    def generate(
        self,
        prompt,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        top_p=None,
        stop_on_eos=True
    ):
        """
        Generate text from prompt.
        
        Args:
            prompt: String prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = default)
            top_k: Top-k sampling (None = no restriction)
            top_p: Top-p (nucleus) sampling (None = no restriction)
            stop_on_eos: Stop generation on EOS token
        
        Returns:
            Generated text
        """
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        tokens = np.array(tokens, dtype=np.int64)[None, :]  # Add batch dimension
        
        # Generate
        generated = tokens.copy()
        eos_token = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
        
        for i in range(max_new_tokens):
            # Forward pass
            logits, _ = self.model(Tensor(generated))
            logits = logits.data
            
            # Get last token's logits
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                next_logits = self._top_k_filter(next_logits, top_k)
            
            # Top-p sampling
            if top_p is not None:
                next_logits = self._top_p_filter(next_logits, top_p)
            
            # Sample
            probs = np.exp(next_logits) / np.sum(np.exp(next_logits), axis=-1, keepdims=True)
            next_token = np.random.choice(probs.shape[-1], p=probs[0])
            
            # Append
            generated = np.concatenate([generated, [[next_token]]], axis=-1)
            
            # Stop on EOS
            if stop_on_eos and eos_token is not None and next_token == eos_token:
                break
        
        # Decode
        return self.tokenizer.decode(generated[0])
    
    def _top_k_filter(self, logits, k):
        """Filter to top-k logits."""
        # Get top-k values and indices
        idx = np.argpartition(logits, -k)[:, -k:]
        values = np.take_along_axis(logits, idx, axis=-1)
        
        # Get the k-th largest value
        threshold = np.min(values, axis=-1, keepdims=True)
        
        # Set all below threshold to -inf
        logits_filtered = np.where(logits >= threshold, logits, -np.inf)
        
        return logits_filtered
    
    def _top_p_filter(self, logits, p):
        """Filter to top-p (nucleus) sampling."""
        # Sort logits
        sorted_idx = np.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = np.take_along_axis(logits, sorted_idx, axis=-1)
        
        # Convert to probabilities
        probs = np.exp(sorted_logits - np.max(sorted_logits, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        
        # Cumulative sum
        cumsum = np.cumsum(probs, axis=-1)
        
        # Find cutoff
        mask = np.ones_like(cumsum, dtype=bool)
        for i in range(cumsum.shape[0]):
            cutoff_idx = np.argmax(cumsum[i] > p)
            mask[i, cutoff_idx+1:] = False
        
        # Create mask for logits
        mask_filtered = np.zeros_like(logits, dtype=bool)
        for i in range(mask.shape[0]):
            mask_filtered[i, sorted_idx[i][mask[i]]] = True
        
        # Set all below threshold to -inf
        logits_filtered = np.where(mask_filtered, logits, -np.inf)
        
        return logits_filtered
    
    def generate_samples(self, prompts, num_samples=3, **kwargs):
        """Generate multiple samples from prompts."""
        results = []
        for prompt in prompts:
            for _ in range(num_samples):
                text = self.generate(prompt, **kwargs)
                results.append({
                    'prompt': prompt,
                    'generated': text,
                    'full': prompt + text
                })
        return results
```

---

## 8. Design Decisions at a Glance

| Component | Our Choice | Alternative | Reason |
|---|---|---|---|
| Tokenization | Character-level | BPE, WordPiece | Simpler, perfect for learning |
| Special tokens | PAD, UNK, BOS, EOS | Only EOS | Provides control signals |
| Unknown handling | Replace with UNK | Raise error | Robust for real-world text |
| Data storage | Binary files | Text files, HDF5 | Fast loading, memory-efficient |
| Memory mapping | Yes, for large files | Load all to RAM | Scalable to huge datasets |
| Special token implementation | Strings in vocab | Separate token IDs | Simpler for the model |

---

## 9. Test Suite

**File:** `tests/test_tokenizer.py`

| Test | What It Verifies |
|---|---|
| `test_tokenizer_build` | Tokenizer builds vocab correctly |
| `test_tokenizer_encode` | Encoding works with all characters |
| `test_tokenizer_decode` | Decoding recovers original text |
| `test_tokenizer_unknown` | Unknown characters are handled |
| `test_special_tokens` | Special tokens are correctly added |
| `test_special_encode_decode` | Encoder/decoder handle special tokens |
| `test_dataset_prepare` | Dataset preparation works |
| `test_dataset_load` | Dataset loads correctly |
| `test_streaming_loader` | Streaming loader works with large files |
| `test_text_generation` | Generation produces reasonable text |

### Test Example: Round Trip

```python
def test_tokenizer_roundtrip():
    text = "Hello, world! This is a test."
    tokenizer = CharTokenizer(text)
    
    # Encode
    tokens = tokenizer.encode(text)
    print(f"Encoded to {len(tokens)} tokens")
    
    # Decode
    decoded = tokenizer.decode(tokens)
    
    # Should match exactly
    assert text == decoded
```

---