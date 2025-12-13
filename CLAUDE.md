# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HAFE-ABSA: A Fair Heterogeneous Graph Neural Network system for Aspect-Based Sentiment Analysis. This project implements fairness-aware methods using Information Theory-based heterogeneous edge detection combined with Type-Aware Graph Convolutional Networks for sentiment analysis on restaurant reviews.

**Key Innovation**: The system addresses fairness issues in ABSA where low-frequency aspects (rare terms like "ambiance") receive worse predictions than high-frequency ones (common terms like "food"). It uses:
1. **FairPHM Module** (src/fairPHM.py): Information theory-based heterogeneous neighbor detection
2. **Type-Aware GCN** (src/type_aware_gcn.py): Edge-type-aware graph convolutions with 4 edge types (OPINION, SYNTAX_CORE, COREF, OTHER)
3. **F3 Module**: Feature fairness enhancement through unsupervised learning

## Build and Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- PyTorch 2.0.1 with PyTorch Geometric 2.3.1
- Transformers 4.30.0 (BERT embeddings)
- Stanza 1.5.0 (dependency parsing)

### Training

**Basic Training**:
```bash
# Train HAFE model on SemEval-2014
python train.py --dataset semeval2014 --model hafe --epochs 50

# Train baseline model
python train.py --dataset semeval2014 --model baseline --epochs 50

# Train with Type-Aware GCN (recommended)
python train.py --dataset semeval2014 --model hafe --use_type_aware --epochs 50
```

**Training Parameters**:
- `--dataset`: semeval2014 | semeval2016
- `--model`: hafe | baseline
- `--use_type_aware`: Enable Type-Aware GCN (uses different weights for different edge types)
- `--hidden_dim`: Hidden dimension (default 128)
- `--lr`: Learning rate (default 0.001)
- `--epochs`: Training epochs (default 50)
- `--eval_every`: Evaluation frequency (default 5)

### Running All Experiments
```bash
# Linux/Mac
./run_all_experiments.sh

# Windows
powershell -File run_all_experiments.ps1
```

### Comparing Results
```bash
python compare_results.py
```

## Architecture Overview

### Core Pipeline Flow

1. **Data Loading** (src/data_loader.py):
   - Parses SemEval XML files containing sentences with aspect terms and sentiment labels
   - Each sample: `{text, aspects: [{term, polarity, from, to}]}`
   - Computes aspect frequency distribution for fairness evaluation

2. **Graph Construction** (src/graph_builder.py):
   - **Dependency Parsing**: Uses Stanza to create syntactic structure
   - **BERT Features**: Extracts 768-dim embeddings for each word
   - **Edge Creation**:
     - Syntactic edges from dependency parse
     - Aspect co-reference edges (between multiple aspects in same sentence)
   - **Edge Type Classification**: Assigns 4 types to each edge:
     - Type 0 (OPINION): Aspect→Opinion word edges (most important)
     - Type 1 (SYNTAX_CORE): Core syntactic relations (nsubj, dobj, amod)
     - Type 2 (COREF): Aspect-aspect co-reference
     - Type 3 (OTHER): Function words, connectors
   - Output: `{features: [N,768], edge_index: [2,E], edge_types: [E], aspect_indices: [A], labels: [A]}`

3. **Model Forward Pass** (src/hafe_absa.py):
   - **F3 Preprocessing** (one-time):
     - Merges all training graphs
     - Detects heterogeneous neighbors using information theory
     - Trains fairness estimator (500 epochs)
   - **Feature Enhancement**: F3 module adjusts features to reduce bias
   - **Type-Aware GCN**: Two layers with edge-type-specific weight matrices
   - **Classification**: Linear classifier on aspect node embeddings → 3-class sentiment (pos/neg/neu)

4. **Evaluation** (src/evaluator.py):
   - Standard metrics: Accuracy, Macro-F1, Micro-F1
   - **Fairness metrics**:
     - Variance, Min-Max Gap, Gini coefficient of per-aspect F1 scores
     - DP-Aspect: Performance gap between high-freq (top 25%) vs low-freq (bottom 25%) aspects

### Module Details

**src/fairPHM.py** (64KB, most complex):
- `InformationTheoryProcessor`: Fast vectorized mutual information computation
- `InfoTheoryHeteroDetector`: Identifies heterogeneous edges using MI thresholds
- `EnhancedUnsupervisedF3Module`: Core fairness module
  - Detects biased neighbors via information theory
  - Trains autoencoder-style estimator to enhance features
  - Caches results by dataset hash for efficiency

**src/type_aware_gcn.py**:
- `TypeAwareGCNConv`: Custom MessagePassing layer
  - 4 separate weight matrices (one per edge type)
  - Learnable edge importance weights (initialized: OPINION=2.0, SYNTAX_CORE=1.5, COREF=1.0, OTHER=0.5)
  - Degree-normalized message aggregation

**src/hafe_absa.py**:
- `HAFE_ABSA_Model`: Full pipeline with F3 + Type-Aware GCN
- `BaselineASGCN`: Standard GCN baseline (can also use type-aware version)

## Important Implementation Details

### F3 Preprocessing
The F3 module MUST be preprocessed before training:
```python
model.preprocess_f3(train_graphs)  # Called once in train.py
```
This merges all training graphs and trains the fairness estimator. Results are cached in `checkpoints/` by dataset hash.

### Edge Types in Forward Pass
When using `--use_type_aware`, the forward pass signature is:
```python
logits = model(features, edge_index, aspect_indices, edge_types)
```
Without type-aware, `edge_types` can be None and standard GCN is used.

### Graph Data Structure
Each graph dict contains:
- `features`: [num_nodes, 768] BERT embeddings
- `edge_index`: [2, num_edges] COO format edge list
- `edge_types`: [num_edges] integer edge types (0-3)
- `aspect_indices`: [num_aspects] indices of aspect nodes in graph
- `labels`: [num_aspects] sentiment labels (0=pos, 1=neg, 2=neu)
- `text`, `words`, `aspect_words`: metadata for debugging

### Device Handling
The code auto-detects device in this order:
1. MPS (Apple Silicon)
2. CUDA (NVIDIA GPU)
3. CPU

FairPHM module defaults to 'cuda' device string but will work on any PyTorch device.

### Checkpoint Naming
Saved models follow pattern: `{dataset}_{model}_best.pt` or `{dataset}_{model}_typeaware_best.pt`

## Common Workflows

### Adding a New Dataset
1. Create XML parser in `src/data_loader.py` (see `_load_semeval2016_xml()` for reference)
2. Add dataset choice to `train.py` argparse
3. Ensure XML format includes: text, aspect terms, polarities, character offsets

### Modifying Edge Types
1. Update `EdgeType` enum in `src/graph_builder.py`
2. Adjust `_identify_edge_types()` logic
3. Update `num_edge_types` parameter in TypeAwareGCNConv (currently 4)
4. Retrain from scratch (cached F3 results are dataset-specific)

### Debugging Graph Construction
Graphs may fail to build if:
- Stanza parsing fails (empty `words` list)
- Aspect term not found in tokenized words (uses fallback to index 0)
- BERT tokenization mismatch

Check train.py output for "警告" (warning) messages showing skipped samples.

## Results Organization

- `checkpoints/`: Saved model weights (.pt files)
- `results/`: Experiment logs (.log files) and outputs
- Per-aspect F1 scores are printed for top-10 most frequent aspects
- Fairness metrics (Gini, Gap, DP-Aspect) measure disparity across aspect frequencies

