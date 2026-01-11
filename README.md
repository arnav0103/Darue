# Narrative Consistency Analysis System

> **Using biologically-inspired neural networks to detect narrative contradictions in literature.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.1+ee4c2c.svg)](https://pytorch.org/)
[![Pathway](https://img.shields.io/badge/Pathway-0.27+-green.svg)](https://pathway.com/)

This project implements a narrative consistency detection system using **TextPath**, a custom language model based on the [Dragon Hatchling (BDH)](https://arxiv.org/abs/2509.26507) architecture. It leverages entity threading and perplexity-based reasoning to identify subtle contradictions in character backstories.

---

## 📚 Project Overview

The system analyzes backstories for characters in classic novels to determine if they are consistent with the established narrative. It achieves this by:

1. **Pretraining novel-specific models** to understand the source material.
2. **Extracting character threads** to isolate and learn specific character arcs.
3. **Calculating Perplexity Delta** to measure how much a backstory aligns with the novel's text.
4. **Calibrating predictions** using a logistic regression classifier on the extracted consistent metrics.

### The Dataset

- **Two classic novels**: *The Count of Monte Cristo* (Alexandre Dumas) and *In Search of the Castaways* (Jules Verne)
- **80 training examples**: Character backstories labeled as consistent or contradictory
- **Novel statistics**:
  - *The Count of Monte Cristo*: 61,676 lines, 13 main characters
  - *In Search of the Castaways*: 18,728 lines, 12 main characters
- **Challenge**: Minimal training data, massive context (entire novels), subtle contradictions

---

## 🧠 The Architecture: Why BDH?

Traditional transformers struggle with this task because they need massive training data. We chose the **Dragon Hatchling (BDH)** architecture for its unique biological properties:

### Core BDH Properties

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **Hebbian Learning** | "Neurons that fire together, wire together" | Naturally learns character relationships and narrative patterns from sequential text |
| **Sparse Activations** | Only ~5% of neurons fire per input | Creates interpretable, monosemantic representations where each concept has distinct neural signatures |
| **Scale-Free Networks** | Power-law connectivity like biological brains | Efficient information routing with fewer parameters |
| **Dynamic Synapses** | Edge weights update during inference | Builds context-specific working memory across long narratives |

### TextPath: BDH for Text

```
┌────────────────────────────────────────────────────────────┐
│                    TextPath Architecture                   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input Tokens [batch, seq_len]                             │
│      ↓                                                     │
│  Token Embedding (16K vocab → 256D)                        │
│      ↓                                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ BDH Layer × 4                                        │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ 1. Project to neurons: v → x (ReLU + Sparse)  │  │  │
│  │  │ 2. Multi-head attention: x × x → a           │  │  │
│  │  │ 3. Hebbian update: y = (a · Dy) ⊙ x          │  │  │
│  │  │ 4. Residual: v ← v + y · E                   │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│      ↓                                                     │
│  Language Model Head → Next Token Predictions              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Parameters**: ~8M (compared to 100M+ for equivalent transformers)
│  │   └─ RoPE Positional Encoding (max_seq_len=4096)               │  │
│  └────────────────────────────────────────────────────────────────┘  │
│           ↓                                                          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Classification Mode (when enabled):                            │  │
│  │   LayerNorm → Dropout → Linear(256→128) → GELU                 │  │
│  │   → Dropout → Linear(128→2) → [Contradict, Consistent]         │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Output: Logits [batch_size, 2] for classification                   │
│          OR Logits [batch_size, seq_len, vocab_size] for LM          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### BDH Biological Properties

The Dragon Hatchling architecture provides three key advantages for narrative understanding:

#### 1. Hebbian Learning
> *"Neurons that fire together, wire together"*

Pre-training on sequential novel passages builds causal circuits encoding:
- Character relationships (Dantès → Mercédès, Fernand → betrayal)
- Plot events (imprisonment → escape → revenge)
- Narrative logic (foreshadowing → resolution)

---

## 🎯 The Key Innovation: Perplexity Delta Scoring

Instead of using a traditional classification head, we use a **generative reasoning** approach:

### The Intuition

If a backstory is **consistent** with a novel, then conditioning the language model on that backstory should **reduce** perplexity when predicting novel passages.

$$\Delta = \text{Loss}(\text{novel} \mid \emptyset) - \text{Loss}(\text{novel} \mid \text{backstory})$$

- **Positive Δ**: Backstory helps predict novel → **CONSISTENT**
- **Negative Δ**: Backstory hurts prediction → **CONTRADICTORY**
- **Near-zero Δ**: Backstory is neutral → **AMBIGUOUS**

### Why This Works

1. **Information-theoretic**: Perplexity measures how "surprised" the model is
2. **Calibrated**: We train a logistic regression on `[Δ, cosine_similarity]` features
3. **Interpretable**: Delta scores show *how much* the model believes the backstory

### Entity Threading: Learning Character Arcs

Standard chunk-based pretraining breaks narrative continuity. We solve this with **entity threading**:

```python
# Extract all paragraphs mentioning "Dantès" from Monte Cristo
thread_dantès = extract_character_paragraphs(novel, ["Dantès", "Edmond"])
# → Creates continuous narrative: sailor → betrayal → prison → escape → count
```

This forces the BDH model to learn:
- Long-range character development
- Relationship dynamics  
- Plot-critical events
- Narrative causality

**Result**: The model develops a coherent "memory" of each character's story arc.

---

## 📂 Project Structure

```
KDSH/
├── run_pipeline.py                    #  Main CLI entry point (187 lines)
├── requirements.txt                   # Dependencies (293 packages)
├── results.csv                        # Final predictions for submission
├── LICENSE                            # MIT License
│
├── Dataset/
│   ├── train.csv                      # 80 labeled training pairs
│   ├── test.csv                       # Unlabeled test set
│   ├── Books/
│   │   ├── The Count of Monte Cristo.txt  (61,676 lines)
│   │   └── In search of the castaways.txt (18,728 lines)
│   └── entity_threads/                # Character-specific narrative threads
│       ├── The Count of Monte Cristo/
│       │   ├── thread_dantès.txt      # Dantès' complete arc
│       │   ├── thread_villefort.txt
│       │   ├── thread_fernand.txt
│       │   └── ... (13 characters)
│       └── In search of the castaways/
│           ├── thread_paganel.txt
│           ├── thread_glenarvan.txt
│           └── ... (12 characters)
│
├── models/                            # Trained model checkpoints
│   ├── custom_tokenizer.json          # 16,384 vocab BPE tokenizer
│   ├── textpath_the_count_of_monte_cristo.pt    # Monte Cristo BDH (~50MB)
│   ├── textpath_in_search_of_the_castaways.pt   # Castaways BDH (~50MB)
│   └── calibration_model.pkl          # Logistic regression calibrator (~100KB)
│
├── src/                               # Source code modules
│   ├── __init__.py                    # Package exports
│   ├── config.py                      # PipelineConfig dataclass (125 lines)
│   │
│   ├── data_processing/               # Data and RAG
│   │   ├── __init__.py
│   │   ├── retrieval.py               # PathwayNovelRetriever (186 lines)
│   │   ├── classification_dataset.py  # PyTorch Dataset
│   │   ├── build_retrievers.py        # Retriever factory
│   │   └── entity_threading.py        # Character thread extraction (321 lines)
│   │
│   ├── models/                        # Neural network modules
│   │   ├── __init__.py
│   │   ├── textpath.py                # TextPath/BDH core (381 lines)
│   │   └── pretrain_bdh_native.py     # Hebbian pretraining
│   │
│   ├── training/                      # Training infrastructure
│   │   ├── __init__.py
│   │   ├── calibration.py             # Logistic regression calibration (353 lines)
│   │   └── pretraining.py             # Pretraining runner
│   │
│   ├── evaluation/                    # Evaluation and prediction
│   │   ├── __init__.py
│   │   └── evaluate.py                # Metrics, prediction (431 lines)
│   │
│   ├── analysis/                      # Scoring modules
│   │   ├── __init__.py
│   │   └── consistency_scorer.py      # Perplexity delta scorer (321 lines)
│   │
│   ├── visualization/                 # Analysis and plots
│   │   ├── __init__.py
│   │   └── visualize.py               # All visualization functions (681 lines)
│   │
│   └── utils/                         # Helper functions
│       └── __init__.py
│
├── visualizations/                    # Generated plots
│   ├── consistency_embedding_space.png
│   ├── prediction_confidence.png
│   ├── accuracy_by_character.png
│   └── accuracy_by_book.png
│
├── repos/                             # External dependencies
│   └── bdh_educational/               # Educational BDH implementation
│       ├── bdh.py                     # Core BDH module (380 lines)
│       └── __pycache__/
│
├── outputs/                           # Training artifacts
│   └── optimal_config.json            # Best hyperparameters
│
└── logs/                              # Training logs
```

## 🔧 Installation

### Prerequisites
- Python 3.11+
- conda (recommended) or pip
- ~8GB RAM (for embedding models)

### Setup

```bash
# Clone the repository
git clone https://github.com/kabyik-kayal/kdsh.git
cd kdsh

# Create conda environment
conda create -n kds python=3.11
conda activate kds

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

| Package | Purpose |
|---------|----------|
| `torch` | Deep learning framework |
| `pathway` | RAG document indexing |
| `sentence-transformers` | Embedding model for retrieval |
| `tokenizers` | BPE tokenizer |
| `scikit-learn` | Metrics and evaluation |
| `pandas` | Data manipulation |
| `matplotlib` | Visualization |
| `tqdm` | Progress bars |

---

## 🚀 Usage

### Quick Start: Complete Pipeline

```bash
#This pretrains the BDH models on both novels with entity threading (50 epochs, ~2 hours on GPU)
python run_pipeline.py --mode pretrain 

#This will run the entire pipeline from classification model training to prediction (EXCLUDING Pretraining)
python run_pipeline.py --mode full 
```

This will:
1. **Pretrain** BDH models on both novels with entity threading (50 epochs, ~2 hours on GPU)
2. **Train** calibration model using perplexity delta features (~5 minutes)
3. **Evaluate** on validation split and print metrics
4. **Predict** on test set and save to `results.csv`
5. **Visualize** results and save plots to `visualizations/`

### Pipeline Modes

The pipeline has 5 modes for different stages:

| Mode | What It Does | When to Use |
|------|--------------|-------------|
| **`pretrain`** | Train BDH language models on novels + entity threads | First run, or to retrain models |
| **`train`** | Train logistic calibration model on delta features | After pretraining, or with new hyperparameters |
| **`evaluate`** | Compute metrics on validation split | Check model performance |
| **`predict`** | Generate predictions for test.csv | Create submission file |
| **`visualize`** | Create all analysis plots | Analyze model behavior |
| **`full`** | Run all modes in sequence | Complete end-to-end run |

### Individual Mode Examples

```bash
# 1. Pretrain BDH models (do this first!)
python run_pipeline.py --mode pretrain --pretrain-epochs 50

# 2. Train calibration model
python run_pipeline.py --mode train

# 3. Evaluate on validation set
python run_pipeline.py --mode evaluate

# 4. Generate test predictions
python run_pipeline.py --mode predict

# 5. Create visualizations
python run_pipeline.py --mode visualize
```

### Command-Line Options

```bash
python run_pipeline.py --help

Usage: run_pipeline.py [OPTIONS]

Options:
  --mode {pretrain,train,predict,evaluate,visualize,full}
                        Pipeline mode (default: full)
  --pretrain-epochs INT
                        BDH pretraining epochs (default: 50)
  --epochs INT          Calibration training epochs (default: 15)
  --batch-size INT      Training batch size (default: 4)
  --lr FLOAT            Learning rate (default: 1e-4)
  --device {cuda,mps,cpu}
                        Device to use (default: auto-detect)
```

### Advanced Examples

```bash
# Longer pretraining for better narrative understanding
python run_pipeline.py --mode pretrain --pretrain-epochs 100

# Faster training with larger batches (requires more RAM)
python run_pipeline.py --mode train --batch-size 16 --lr 2e-4

# Force CPU usage (if GPU has issues)
python run_pipeline.py --mode full --device cpu

# Quick evaluation without retraining
python run_pipeline.py --mode evaluate
```

### What Gets Generated

After running, you'll find:

```
KDSH/
├── models/
│   ├── textpath_the_count_of_monte_cristo.pt  # Pretrained BDH (50MB)
│   ├── textpath_in_search_of_the_castaways.pt # Pretrained BDH (50MB)
│   └── calibration_model.pkl                   # Logistic regression (~100KB)
│
├── results.csv                                 # Test predictions (ready for submission)
│
├── visualizations/                             # Analysis plots
│   ├── delta_distribution.png
│   ├── confusion_matrix.png
│   ├── calibration_curve.png
│   ├── feature_importance.png
│   └── evaluation_dashboard.png
│
└── outputs/
    └── optimal_config.json                     # Best hyperparameters found
```

---

## ⚙️ Configuration

All settings live in [src/config.py](src/config.py) as the `PipelineConfig` dataclass. You can modify defaults directly in the file or override via command-line arguments.

### Key Configuration Groups

#### 📁 Paths Configuration

```python
novels_dir = ROOT / 'Dataset' / 'Books'          # Novel .txt files
train_csv = ROOT / 'Dataset' / 'train.csv'       # Training labels
test_csv = ROOT / 'Dataset' / 'test.csv'         # Test set (no labels)
tokenizer_path = ROOT / 'models' / 'custom_tokenizer.json'
models_dir = ROOT / 'models'                     # Checkpoint directory
output_predictions = ROOT / 'results.csv'        # Final predictions
```

#### 🎛️ Model Architecture

```python
# TextPath/BDH configuration
vocab_size = 16384          # BPE tokenizer vocabulary
max_seq_len = 512           # Maximum tokens per sequence
n_heads = 8                 # Multi-head attention
n_neurons = 2048            # BDH neurons (scale-free graph)
d_model = 256               # Embedding dimension
n_layers = 4                # Number of BDH layers
sparsity_target = 0.05      # 5% activation rate
```

#### 🏋️ Training Hyperparameters

```python
batch_size = 4              # Training batch size
epochs = 15                 # Calibration training epochs
learning_rate = 1e-4        # Initial learning rate
weight_decay = 0.01         # AdamW regularization
pretrain_epochs = 50        # BDH pretraining epochs
```

#### 🔍 RAG (Retrieval) Settings

```python
chunk_size = 200            # Words per chunk (~250 tokens)
overlap = 50                # Overlapping words between chunks
top_k_retrieval = 2         # Number of passages to retrieve per query
```

#### 🎯 Device Selection

```python
# Auto-detected in order: CUDA → MPS (Apple Silicon) → CPU
device = 'cuda' if torch.cuda.is_available() 
         else 'mps' if torch.backends.mps.is_available()
         else 'cpu'
```

### Modifying Configuration

**Option 1: Edit directly**

```python
# In src/config.py
@dataclass
class PipelineConfig:
    pretrain_epochs: int = 100  # Changed from 50
    batch_size: int = 8         # Changed from 4
```

**Option 2: Command-line override**

```bash
python run_pipeline.py --pretrain-epochs 100 --batch-size 8
```

**Option 3: Programmatic override**

```python
from src.config import get_config

config = get_config()
config.pretrain_epochs = 100
config.batch_size = 8
```

---

## 📊 How It Works: The Complete Pipeline

### Stage 1: Entity Threading (Pretraining Prep)

Extract character-specific narratives from novels:

```python
# From entity_threading.py
def create_character_threads(novel_path, character_list):
    """
    Extract all paragraphs mentioning each character.
    Creates continuous narrative threads preserving character arcs.
    """
    paragraphs = split_into_paragraphs(novel_text)
    
    for character, aliases in character_list:
        # Find all paragraphs mentioning this character
        thread = [p for p in paragraphs if any(alias in p for alias in aliases)]
        # Save as continuous text
        save_thread(f"thread_{character}.txt", thread)
```

**Output**: 25 character threads (13 for Monte Cristo, 12 for Castaways)

### Stage 2: BDH Pretraining

Train novel-specific language models on mixed data:

```python
# From pretrain_bdh_native.py
def pretrain_bdh_novel(novel_name):
    """
    Pretrain BDH on:
    - 70% raw novel text (narrative structure)
    - 30% entity threads (character-specific sequences)
    """
    model = TextPath(config)  # BDH architecture
    
    for epoch in range(50):
        for batch in mixed_data_loader:
            logits = model(input_ids)
            loss = cross_entropy(logits, targets)  # Next-token prediction
            loss.backward()
            optimizer.step()
```

**Output**: Two pretrained models (~50MB each)
- `textpath_the_count_of_monte_cristo.pt`
- `textpath_in_search_of_the_castaways.pt`

### Stage 3: Perplexity Delta Computation

For each training sample, compute consistency score:

```python
# From consistency_scorer.py
class ConsistencyScorer:
    def compute_delta(self, backstory, novel_chunk):
        """
        Delta = Loss(chunk | empty) - Loss(chunk | backstory)
        """
        # Baseline: predict chunk without context
        loss_baseline = model.compute_loss(novel_chunk)
        
        # Conditioned: predict chunk given backstory
        loss_conditioned = model.compute_loss(
            context=backstory, 
            target=novel_chunk
        )
        
        delta = loss_baseline - loss_conditioned
        # Positive delta = backstory helps = CONSISTENT
        # Negative delta = backstory hurts = CONTRADICTORY
        
        return delta, cosine_similarity(backstory_emb, chunk_emb)
```

### Stage 4: Calibration Training

Train logistic regression on delta features:

```python
# From calibration.py
def train_calibration_model(train_csv):
    """
    Train simple classifier on extracted features:
    - Feature 1: Perplexity delta
    - Feature 2: Cosine similarity (retrieval score)
    """
    features = []
    labels = []
    
    for sample in train_csv:
        # Retrieve relevant passages
        chunks = retriever.search(sample.content, k=2)
        
        # Compute delta
        delta, cos_sim = scorer.compute_delta(sample.content, chunks)
        features.append([delta, cos_sim])
        labels.append(sample.label)
    
    # Train calibrated classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    pipeline.fit(features, labels)
    return pipeline
```

**Output**: `calibration_model.pkl` (~100KB)

### Stage 5: Prediction

Generate predictions for test set:

```python
# From evaluate.py
def run_prediction(test_csv, calibration_model):
    predictions = []
    
    for sample in test_csv:
        # 1. Retrieve relevant passages
        chunks = retriever.search(sample.content, k=2)
        
        # 2. Compute delta features
        delta, cos_sim = scorer.compute_delta(sample.content, chunks)
        
        # 3. Predict using calibration model
        prob = calibration_model.predict_proba([[delta, cos_sim]])
        label = 1 if prob[1] > 0.5 else 0  # 1=consistent, 0=contradict
        
        predictions.append({
            'id': sample.id,
            'label': label
        })
    
    save_predictions('results.csv', predictions)
```

**Output**: `results.csv` (ready for submission)

---

## 🔬 Technical Deep Dive

### Why Perplexity Delta Works

Perplexity measures how "surprised" a language model is by a sequence. Lower perplexity = better prediction.

**Intuition**: If a backstory is consistent with a novel, it should *reduce* the model's surprise when reading novel passages.

$$\text{Perplexity}(x) = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(x_i | x_{<i})\right)$$

$$\Delta = \text{PPL}(x | \emptyset) - \text{PPL}(x | \text{backstory})$$

**Example**:

```
Novel passage: "Dantès was arrested on his wedding day and imprisoned in Château d'If."

Consistent backstory: "Dantès was betrayed by jealous rivals who framed him for treason."
→ Model thinks: "Oh, that makes sense! Betrayal → arrest → prison"
→ Perplexity decreases → Positive Δ → CONSISTENT

Contradictory backstory: "Dantès was a wealthy count who lived in Paris all his life."
→ Model thinks: "Wait, that doesn't match. How was he arrested at a wedding?"
→ Perplexity increases → Negative Δ → CONTRADICTORY
```

### BDH Architecture Details

The Dragon Hatchling uses **Hebbian learning principles**:

```python
# From bdh.py (simplified)
class BDH(nn.Module):
    def forward(self, input_ids):
        v_ast = self.emb(input_ids)  # Token embeddings
        
        for layer in range(self.L):
            # 1. Sparse neuron activation (ReLU → ~5% active)
            x = F.relu(v_ast @ self.Dx)  # Project to neurons
            
            # 2. Hebbian attention (neurons communicate)
            a_ast = self.linear_attn(x, x, v_ast)  # x × x^T × v
            
            # 3. Synaptic strengthening (co-firing → stronger weights)
            y = F.relu(a_ast @ self.Dy) * x  # Hadamard product
            
            # 4. Residual update (gradual memory accumulation)
            v_ast = v_ast + y @ self.E
        
        return v_ast @ self.readout  # Next-token logits
```

**Key differences from Transformers**:

| Feature | Transformer | BDH |
|---------|-------------|-----|
| **Connectivity** | Dense (all-to-all) | Sparse (scale-free graph) |
| **Activation** | ~100% neurons | ~5% neurons (sparse) |
| **Learning** | Gradient descent only | Hebbian + gradient descent |
| **Parameters** | O(d²) | O(d·√N) (low-rank) |
| **Interpretability** | Polysemantic | Monosemantic neurons |

### Pathway RAG Integration

Satisfies Track B requirement for Pathway framework:

```python
# From retrieval.py
class PathwayNovelRetriever:
    def __init__(self, novel_path):
        # Create Pathway table (required for Track B)
        self.chunks_table = pw.debug.table_from_rows(
            schema=pw.schema_from_dict({"text": str}),
            rows=[(chunk,) for chunk in self.chunks]
        )
        
        # Use Pathway's embedding framework
        from pathway.xpacks import llm
        self.embedder = llm.embedders.SentenceTransformerEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Build vector index
        self.embeddings = self.chunks_table.select(
            embedding=self.embedder(pw.this.text)
        )
    
    def search(self, query, k=2):
        """Retrieve top-k most relevant chunks"""
        query_emb = self.embedder.encode([query])[0]
        scores = cosine_similarity(query_emb, self.embeddings)
        top_k_idx = np.argsort(scores)[-k:]
        return [self.chunks[i] for i in top_k_idx]
```

### Model Training Details

**Pretraining** (Unsupervised Language Modeling):
- **Objective**: Next-token prediction
- **Data**: 70% raw novel + 30% entity threads
- **Batch size**: 16 sequences × 512 tokens
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.01)
- **Duration**: 50 epochs (~60 mins on  H200)
- **Loss**: Cross-entropy on vocabulary

**Calibration** (Supervised Classification):
- **Objective**: Binary classification on delta features
- **Data**: 80 training samples → [delta, cos_sim] features
- **Model**: Logistic Regression with StandardScaler
- **Cross-validation**: 5-fold CV for robustness
- **Duration**: <1 minute on CPU
- **Metrics**: Accuracy, F1, Precision, Recall

## 📊 Visualizations & Analysis

The pipeline generates comprehensive analysis plots:

### Delta Distribution Plot
Shows the separation between consistent and contradictory samples based on perplexity delta:

```python
# From visualize.py
plot_delta_distribution(
    deltas=[...],          # Delta values for all samples
    labels=[...],          # True labels (0=contradict, 1=consistent)
    novel_names=[...]      # Novel for each sample
)
```

**What to look for**: Clear separation between red (contradict) and green (consistent) distributions

### Confusion Matrix
Standard evaluation visualization:

```python
plot_confusion_matrix(y_true, y_pred)
```

Shows: True Positives, False Positives, True Negatives, False Negatives

### Calibration Curve
Checks if predicted probabilities match actual frequencies:

```python
plot_calibration_curve(y_true, y_probs)
```

**Interpretation**: Diagonal line = perfectly calibrated model

### Feature Importance
Shows relative importance of delta vs. cosine similarity:

```python
plot_feature_importance(calibration_model.coef_)
```

### Generate All Visualizations

```bash
python run_pipeline.py --mode visualize
```

Saves plots to `visualizations/` directory.

---

## 🎓 Key Insights & Lessons Learned

### What Works

✅ **Entity Threading**: Massive improvement over random chunking. Character-specific sequences help the model learn narrative arcs.

✅ **Perplexity Delta**: More principled than classification head. Information-theoretic measure naturally captures consistency.

✅ **Novel-Specific Models**: Each novel has unique style, vocabulary, and themes. Separate models perform better than one shared model.

✅ **BDH Sparse Activations**: ~5% activation rate creates interpretable representations. Makes debugging easier.

✅ **Pathway Integration**: Clean API for document indexing. Sentence-transformers embedder works well out of the box.