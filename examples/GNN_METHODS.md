# GNN Methodology — Neuroprobe Competition Submission

## Overview

This document describes the Graph Neural Network (GNN) approach submitted to the
[Neuroprobe benchmark](https://neuroprobe.dev) for evaluating intracranial EEG (iEEG)
foundation models on naturalistic stimuli decoding tasks.

---

## Paper Methods Section (Draft)

**Graph Neural Network for iEEG Population Decoding**

We model multi-electrode intracranial EEG as a graph where nodes correspond to
recording electrodes and edges encode spatial proximity derived from electrode
coordinates. For each subject, a k-nearest-neighbor graph (k=8) is constructed
from MNI electrode coordinates, with edges weighted by symmetric degree
normalization (D^{−1/2} A D^{−1/2}).

Neural signals are preprocessed via Laplacian rereferencing followed by
short-time Fourier transform (STFT; Hann window, 512 samples, 75% overlap),
yielding a spectrogram of shape (E × T × F) per trial, where E is the number
of electrodes, T the number of time bins (~16 bins per second), and F the
number of frequency bins retained (0–150 Hz, ~65 bins at 2048 Hz sampling rate).

Our spatio-temporal GNN (ST-GCN) first applies shared 1D temporal convolutions
independently over each electrode's frequency-time representation, producing a
compact per-electrode feature vector of dimension H. Two graph convolutional
layers then perform spatial message passing over the electrode graph, allowing
each node to aggregate evidence from anatomically proximate electrodes. Node
features are globally averaged and passed through a two-layer MLP classification
head. All convolutions use batch normalization and ReLU activations; dropout
(p=0.4) is applied after each major block.

For the graph attention variant (GAT), the fixed GCN aggregation is replaced by
multi-head attention (4 heads) over concatenated node feature pairs, using the
kNN adjacency as an attention mask. This allows the model to learn task-relevant
electrode interactions beyond physical proximity.

Models are trained with Adam (lr=5×10^{−4}, weight decay=0) using cross-entropy
loss. Training runs for up to 150 epochs with early stopping on held-out
validation macro ROC-AUC (patience=20 epochs). A random 20% of training
samples is held out for validation.

All architectures are evaluated under the within-session, cross-session, and
cross-subject train/test splits defined by the Neuroprobe benchmark protocol,
using the Lite electrode subset. The target metric is macro-averaged ROC-AUC
over binary classification tasks.

---

## Architecture Details

### Variant 0: `gnn_v0_bugfix` — GCN + LSTM (Baseline)

The original architecture with axis bugs corrected. Included as a baseline to
isolate the effect of the axis fix from architectural improvements.

**Pipeline:**
```
(B, E, T, F)
  → for each time step t: GCN layers over electrodes → mean pool over E
  → stack → (B, T, H)
  → Bidirectional LSTM
  → mean pool over T → (B, 2H)
  → FC(2H→256) → ReLU → Dropout → FC(256→C)
```

**Known limitation:** Per-timestep loop (T iterations of GCN per sample) is
slow and prevents scaling up hidden dimensions on larger datasets.

---

### Variant 1: `gnn_v1_stgcn` — Efficient Spatio-Temporal GCN *(Default)*

Replaces the per-timestep loop with a fully batched Conv1d over the temporal
dimension, then applies GCN for spatial aggregation.

**Pipeline:**
```
(B, E, T, F)
  → reshape to (B·E, F, T)
  → Conv1d(F→H, k=5) + BN + ReLU
  → Conv1d(H→H, k=3) + BN + ReLU
  → AdaptiveAvgPool1d(1) → squeeze → (B·E, H)
  → reshape to (B, E, H)
  → GCN(H→H) + BN + ReLU  [×2 layers]
  → mean pool over E → (B, H)
  → FC(H→256) → ReLU → Dropout → FC(256→C)
```

**Advantages over v0:**
- No per-timestep loop → ~8× faster forward pass
- Temporal conv captures multi-scale frequency-time patterns per electrode
- BatchNorm after GCN stabilises training with larger hidden sizes

**Hyperparameters:** H=128 (first layer), 256 (second layer), k_neighbors=8,
dropout=0.4, lr=5×10^{−4}, max_iter=150, patience=20.

---

### Variant 2: `gnn_v2_gat` — Spatio-Temporal Graph Attention Network

Same temporal conv front-end as v1, but replaces GCN aggregation with
multi-head graph attention (GAT).

**Pipeline:**
```
(B, E, T, F)
  → [same temporal conv as v1] → (B, E, H)
  → GAT(H→H, 4 heads) + BN + ELU  [×2 layers, masked to kNN edges]
  → mean pool over E → (B, H)
  → FC(H→256) → ReLU → Dropout → FC(256→C)
```

**Attention mechanism (per layer, per head):**
```
e_ij = LeakyReLU( a^T [W h_i || W h_j] )   (masked: e_ij = -∞ if no edge)
α_ij = softmax_j( e_ij )
h_i' = ELU( Σ_j α_ij · W h_j )
```
Outputs from all heads are concatenated and projected.

**When to prefer GAT over ST-GCN:** Tasks where electrode importance varies
(e.g., language tasks like `gpt2_surprisal` vs. visual tasks like `global_flow`).
GAT learns task-specific attention weights; GCN uses fixed graph topology.

---

## Graph Construction

For each subject, the electrode graph is built once at training time:

1. Compute pairwise Euclidean distances between MNI electrode coordinates
2. For each electrode, connect to its k=8 nearest neighbours
3. Make symmetric: `A = (A + A^T) / 2`, then binarise
4. Add self-loops: `A = A + I`
5. Symmetric normalisation: `A_norm = D^{-1/2} A D^{-1/2}`

The normalised adjacency is reused across all forward passes. For cross-subject
splits, a fresh graph is built for each subject independently.

---

## Preprocessing

| Step | Details |
|------|---------|
| Rereferencing | Laplacian (each electrode minus mean of neighbours) |
| STFT | Window: Hann, 512 samples, 75% overlap, centred |
| Frequency range | 0–150 Hz (~65 bins at 2048 Hz) |
| Time range | 0–1 s post word onset (Neuroprobe Lite, `--only_1second`) |
| Normalisation | StandardScaler (fit on train, applied to test) |
| Electrode subset | Neuroprobe Lite subset per subject |

---

## Reproducing Results

```bash
# Quick benchmark (3 tasks, 1 subject/trial, ~30 min)
cd examples/
bash run_gnn_benchmark.sh

# Full competition run with best variant (all 15 tasks, all subjects)
# Replace run_eval_population.sh template with:
python eval_population.py \
  --classifier_type gnn \
  --gnn_variant gnn_v1_stgcn \
  --subject_id <ID> --trial_id <ID> \
  --eval_name onset,speech,volume,delta_volume,pitch,word_index,word_gap,\
gpt2_surprisal,word_head_pos,word_part_speech,word_length,\
global_flow,local_flow,frame_brightness,face_num \
  --split_type WithinSession \
  --only_1second --full --verbose \
  --save_dir eval_results/gnn_final
```

---

## File Reference

| File | Role |
|------|------|
| `examples/eval_utils.py` | `GNNClassifier` class with all 3 variants |
| `examples/eval_population.py` | Main runner; add `--gnn_variant` flag |
| `examples/run_gnn_benchmark.sh` | Quick comparison script for all variants |
| `examples/GNN_METHODS.md` | This file |
