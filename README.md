# Integrating Gene Regulatory Network Structure into Machine Learning Models
**Predicting steady-state mRNA levels using GRN-constrained Graph Neural Networks**

| | |
|---|---|
| **Author** | Hrithik Chandra |
| **Supervisor** | Dr Jan T. Kim |
| **Institution** | University of Hertfordshire — Department of Computer Science |
| **Module** | 6COM2017 – Artificial Intelligence Project (BSc Hons Computer Science) |
| **Submission** | March 2026 |

---

## Abstract

Machine-learning models for gene expression typically treat genes as independent inputs, ignoring the regulatory structure that governs the system. This project investigates whether incorporating gene regulatory network (GRN) structure as an explicit architectural component in a graph neural network (GNN) improves prediction of steady-state mRNA levels, compared with tabular machine-learning baselines that use only per-gene features.

A Transsys-inspired ODE simulator was implemented to generate synthetic expression profiles from **100 random regulatory networks** with fully known ground-truth graphs and kinetic parameters, enabling controlled ablation experiments free from network-inference errors. Two families of models were trained on log-transformed mRNA targets using a **network-level 80/20 train/test split**:

- **Tabular baselines** — Ridge Regression and Random Forest — evaluated with and without hand-crafted GRN topology features
- **Four-layer edge-weighted GNN** — evaluated with and without regulatory edges

Results show a clear three-tier performance hierarchy:

| Model | R² (test) | Notes |
|---|---|---|
| No GRN information (params only) | ≈ 0.04 – 0.06 | Ridge / RF params-only |
| GRN topology features (tabular) | ≈ 0.32 – 0.37 | Ridge+GRN / RF+GRN |
| GRN-aware GNN (message passing) | ≈ **0.59** | 4-layer EdgeWeightedConv |

The GRN-aware GNN achieves an absolute R² gain of **+0.55** over its edge-free ablation, demonstrating that explicit message-passing along signed regulatory edges captures multi-hop dependencies that flat tabular representations cannot replicate.

---

## Research Question

> *Does giving a machine-learning model explicit access to gene regulatory network structure — through graph-based message-passing along regulatory edges — improve its ability to predict steady-state mRNA levels, compared with tabular baselines that use only per-gene kinetic features?*

---

## Background

Gene expression is governed by a **Gene Regulatory Network (GRN)** — a directed, signed graph in which nodes are genes and edges encode transcription factor relationships (activation or repression) with associated strengths. Predicting expression patterns is central to computational biology and medicine, impacting research into cancer, neurodegeneration and drug response.

Two paradigms exist:
- **Mechanistic models** (ODE/Boolean): biologically interpretable but require precise kinetic parameters and scale poorly
- **Machine-learning models**: flexible and powerful but typically treat genes as independent inputs, ignoring regulatory structure

This project bridges that gap. Rather than using real RNA-seq datasets — where the ground-truth GRN is unknown or only partially inferred — all experiments use **synthetic data from a known Transsys-inspired ODE simulator**, enabling fully controlled ablations with no ambiguity about regulatory topology.

---

## Project Aims and Objectives

1. Implement a synthetic data-generation pipeline using the Transsys ODE-based GRN simulator to produce controlled gene expression datasets with fully known ground-truth graphs across 100 networks and 50 conditions each.
2. Design and extract GRN topology features (in/out-degree, PageRank, betweenness centrality, signed regulatory strengths) to augment tabular baselines, enabling a fair information-controlled comparison.
3. Train and evaluate tabular baselines — Ridge Regression and Random Forest — under two conditions: (a) kinetic parameters only, and (b) parameters augmented with GRN topology features.
4. Design, implement and tune a Graph Convolutional Network using signed regulatory edges for message-passing, predicting log-transformed steady-state mRNA levels from per-gene kinetic parameters.
5. Compare model performance using R², MSE and MAE on 20 held-out unseen networks and assess whether explicit graph message-passing provides a measurable advantage over the best tabular baseline given equivalent GRN information.
6. Interpret results in terms of when and how GRN structure contributes predictive signal for gene expression.

---

## Methods Overview

### Synthetic Data Generator (Transsys-inspired)
- 100 random GRNs generated with 5–30 genes each, edge probability 0.2–0.35, and activation ratio 0.5–0.7
- Each network simulated to steady state from 50 random initial conditions → 5,000 expression profiles total
- Gene kinetic parameters: basal expression rate, mRNA decay rate, protein decay rate
- Regulatory edges: signed strength (+ activator, − repressor)
- Outputs: `{net_id}_genes.csv`, `{net_id}_edges.csv`, `expression_profiles.csv`, `network_metadata.csv`

### Model Families

**Tabular Baselines (`train_mrna_baseline.py`)**
- Inputs: per-gene kinetic parameters only (3 features), or parameters + 9 GRN topology features
- GRN features: in-degree, out-degree, n-activators, n-repressors, total activation/repression, net regulation, PageRank, betweenness centrality
- Models: Ridge Regression, Random Forest
- Target: log1p(mRNA)

**GRN-aware GNN (`train_gnn.py`)**
- Architecture: 4-layer `EdgeWeightedConv` (custom SAGEConv-style message-passing), hidden dim 64, dropout 0.4
- Inputs: per-gene kinetic parameters (same 3 features as tabular baselines)
- Edges: signed regulatory edges from ground-truth GRN (activator = +strength, repressor = −strength)
- Ablation: expression-only GNN (same architecture, empty edge set)
- Optimiser: Adam (lr=5×10⁻⁴), ReduceLROnPlateau, early stopping (patience=40), gradient clipping
- Loss: MSE on log1p(mRNA)

### Evaluation
- Network-level 80/20 train/test split: 80 training networks, 20 unseen test networks
- Metrics: R², MSE, MAE
- No expression data is used as model input — GRN structure and kinetic parameters only

---

## Key Results

The results confirm a clear hierarchy across three information regimes:

1. **No GRN** — models see only kinetic parameters (basal expression, mRNA decay, protein decay). Performance is near-zero (R² ≈ 0.04–0.06), confirming that parameters alone are insufficient to predict expression.
2. **Flat GRN features** — adding hand-crafted topology features (degree, PageRank, betweenness) to the tabular models raises R² to 0.32–0.37. Structure helps, but only local-neighbourhood summaries are captured.
3. **GRN message-passing** — the 4-layer GNN with signed edges achieves R² ≈ 0.59, capturing multi-hop cascading regulatory effects across up to 4 regulatory hops that flat representations cannot encode.

The GNN's advantage arises specifically from message-passing: its expression-only ablation (same architecture, no edges) scores R² ≈ 0.04, confirming the improvement is structural rather than architectural.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── train_gnn.py                        ← Main GNN training script
├── train_mrna_baseline.py              ← Tabular mRNA baseline training
├── plt_mrna_baselines.py               ← Baseline result plots
│
├── data/
│   ├── synthetic_transsys/             ← Primary dataset (100 networks × 50 seeds)
│   │   ├── expression_profiles.csv
│   │   ├── generation_config.json
│   │   ├── network_metadata.csv
│   │   └── grn_edges/
│   │       ├── net_000_genes.csv       ← Kinetic parameters per gene
│   │       ├── net_000_edges.csv       ← Signed regulatory edges
│   │       └── ...                     ← net_001 … net_099
│   ├── synthetic_transsys_backup_50/   ← Backup dataset (50 networks × 50 seeds)
│   ├── ml_ready/
│   │   └── pergene_dataset.csv         ← Early per-gene protein pilot dataset
│   └── dream4/                         ← DREAM4 reference data (not used in final eval)
│
├── experiments/                        ← Early prototypes and helper scripts
│   ├── generate_synthetic_dataset.py   ← Dataset generator (calls transsys_simulator.py)
│   ├── build_pergene_dataset.py        ← Per-gene protein dataset builder (pilot)
│   ├── train_pergene_baselines.py      ← Protein-prediction pilot baselines
│   ├── analyze_network_expression.py   ← GNN qualitative visualisation per network
│   ├── plotting.py                     ← Centralised plotting utilities
│   ├── diff-eq-v1.py                   ← ODE prototype v1
│   ├── diff-eq-v2.py                   ← ODE prototype v2
│   └── transyss-prototype.py           ← Early Transsys prototype
│
├── simulators/
│   ├── transsys_simulator.py           ← Core ODE-based GRN simulator
│   └── config_examples/
│       ├── demo-3-gene-loop.py         ← 3-gene feedback loop validation demo
│       └── demo-random-10-gene-network.py ← Random 10-gene network demo
│
├── results/
│   ├── gnn/
│   │   ├── gnn_comparison.png
│   │   ├── gnn_expr_only_best.pt       ← Best expression-only GNN checkpoint
│   │   ├── gnn_grn_aware_best.pt       ← Best GRN-aware GNN checkpoint
│   │   └── gnn_results.json
│   ├── mrna_baselines/
│   │   ├── mrna_comparison.png
│   │   ├── mrna_baselines_results.csv
│   │   ├── mrna_baselines_results.json
│   │   ├── mrna_baselines_results_long.csv
│   │   ├── mrna_feature_importance.csv
│   │   └── mrna_rf_grn_predictions.csv
│   ├── tables/
│   │   ├── pergene_baseline_results.csv
│   │   ├── pergene_elasticnet_coefficients.csv
│   │   ├── pergene_feature_importance.csv
│   │   └── pergene_predictions.csv
│   └── figures/
│
├── features/                           ← Feature extraction utilities
├── models/                             ← Archived model implementations
├── notebooks/                          ← Jupyter notebooks for exploration
├── docs/                               ← Project documents
└── report/
    └── drafts/
        └── 01_baseline_results_writeup.md
```

---

## Pipeline: How to Run End-to-End

Follow these steps in order. Each step produces outputs consumed by the next.

### Step 1 — Validate the simulator (optional but recommended)
```bash
python simulators/config_examples/demo-3-gene-loop.py
python simulators/config_examples/demo-random-10-gene-network.py
```
These scripts run small demo GRNs to confirm the simulator is working before generating the full dataset.

### Step 2 — Generate the synthetic dataset
```bash
python experiments/generate_synthetic_dataset.py
```
Outputs (written to `data/synthetic_transsys/`):
- `grn_edges/net_XXX_genes.csv` — kinetic parameters per gene per network
- `grn_edges/net_XXX_edges.csv` — signed regulatory edges per network
- `expression_profiles.csv` — steady-state mRNA/protein per condition per network
- `network_metadata.csv` — summary info per network
- `generation_config.json` — exact config used for reproducibility

> ⏱ Takes approximately 10–30 minutes depending on hardware.

### Step 3 — Run the tabular mRNA baselines
```bash
python train_mrna_baseline.py
```
Outputs (written to `results/mrna_baselines/`):
- `mrna_baselines_results.json` / `.csv` — R², MSE, MAE for all four model × feature-set combinations
- Prediction scatter plots, residual plots, feature importance plots

### Step 4 — Plot baseline comparison charts (optional)
```bash
python plt_mrna_baselines.py
```
Generates bar charts comparing parameters-only vs. parameters+GRN for Ridge and Random Forest.

### Step 5 — Train the GRN-aware GNN
```bash
python train_gnn.py
```
Outputs (written to `results/gnn/`):
- `gnn_grn_aware_best.pt` — best GRN-aware model checkpoint
- `gnn_expr_only_best.pt` — best expression-only ablation checkpoint
- `gnn_results.json` — final R², MSE, MAE for both variants
- Training curves, predicted vs actual plots, R² comparison chart

> ⏱ Up to 400 epochs with early stopping. On CPU, expect 30–90 minutes.

### Step 6 — Qualitative per-network analysis (optional)
```bash
python experiments/analyze_network_expression.py
```
Loads the trained GRN-aware GNN, applies it to a single network, and visualises the regulatory graph with node colours proportional to predicted expression.

---

## Architecture Notes

### EdgeWeightedConv (custom message-passing layer)
The core model uses a custom `MessagePassing` layer that mirrors the ODE simulator's regulatory logic:

```
message(j → i) = edge_weight_ij × W_neigh × x_j
update(i)      = W_self × x_i + Σⱼ message(j → i)
```

- Activating edges (positive weight) push the hidden representation positively
- Repressing edges (negative weight) push it negatively
- This directly encodes the regulatory sum in the simulator's ODE derivatives

### Why 4 layers?
A 4-layer GNN has a receptive field spanning up to 4 regulatory hops, which is sufficient to cover most regulatory cascades in 5–30 gene networks while avoiding excessive over-smoothing on smaller graphs.

### Why signed SAGE-style over GAT/GCN?
The custom edge-weighted layer encodes the exact inductive bias from the simulator: expression is driven by a weighted sum of upstream regulator proteins. Standard GCN normalises by degree (loses signed information); GAT adds attention parameters that would obscure whether gains come from structure or capacity. The SAGE-style design keeps the comparison clean.

---

## Reproducibility

All experiments use `RANDOM_SEED = 42` for NumPy and PyTorch. The dataset generator writes `generation_config.json` to record all hyperparameters. The network-level train/test split is deterministic (`all_networks[:n_train]` with sorted network IDs). All results can be regenerated from scratch using the steps above.

---

## Limitations and Future Work

- **Synthetic data only**: experiments use controlled ODE-generated data. Extension to real benchmarks (GeneNetWeaver, GTEx, DREAM5) is the most important next step.
- **Scale**: networks are 5–30 genes. Scaling to genome-wide networks requires sparse GNN implementations and sub-graph sampling.
- **Steady state only**: time-series dynamics are not modelled. Temporal GNNs (e.g., T-GCN) could extend the approach to trajectory prediction.
- **Architecture search**: GAT, GIN or transformer-based message-passing may improve performance and are natural next experiments once the baseline message-passing benefit is established.
- **Attention mechanism**: adding attention to edge aggregation would allow the model to learn context-dependent weighting of regulatory inputs.

---

## Citation

If you use this codebase or build on this work, please cite:

```
Chandra, H. (2026). Integrating Gene Regulatory Information into Machine Learning Models.
BSc (Hons) Computer Science Final Project Report.
University of Hertfordshire, Hatfield, UK.
Supervised by Dr Jan T. Kim.
```

---

## Environment Setup — Step-by-Step Guide

This section explains how to create an identical virtual environment from scratch, covering Windows, macOS and Linux.

### Prerequisites

Before starting, ensure the following are installed on your system:

| Tool | Minimum version | Check command |
|---|---|---|
| Python | 3.9 or higher | `python --version` |
| pip | 21.0 or higher | `pip --version` |
| Git | Any recent version | `git --version` |

> **Note on PyTorch Geometric**: `torch-geometric` requires a compatible version of PyTorch and a matching CUDA version (for GPU) or CPU build. The steps below install CPU-only PyTorch, which works on all machines without a GPU. If you have an NVIDIA GPU, see the GPU section at the end.

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

If you are working from a local copy (not yet on GitHub), navigate to the project root:

```bash
cd path/to/your/project-folder
```

---

### Step 2 — Create a virtual environment

A virtual environment isolates this project's packages from your system Python installation.

**Windows (Command Prompt or PowerShell):**
```powershell
python -m venv .venv
```

**macOS / Linux:**
```bash
python3 -m venv .venv
```

This creates a `.venv/` folder inside your project root.

---

### Step 3 — Activate the virtual environment

You must activate the environment before installing or running anything. You will need to re-activate it every time you open a new terminal session.

**Windows — PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

> If you see a permissions error on Windows, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**Windows — Command Prompt (cmd.exe):**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

Once activated, your terminal prompt will change to show `(.venv)` at the start, confirming the environment is active.

---

### Step 4 — Upgrade pip

Always upgrade pip before installing packages to avoid resolver errors:

```bash
pip install --upgrade pip
```

---

### Step 5 — Install core scientific libraries

Install the main dependencies from `requirements.txt`:

```bash
pip install numpy matplotlib scikit-learn scipy networkx pandas seaborn
```

Or equivalently:

```bash
pip install -r requirements.txt
```

> **Note**: The `requirements.txt` in this repo pins the package names but not versions, which means pip will install the latest compatible versions. See Step 7 for pinning exact versions for strict reproducibility.

---

### Step 6 — Install PyTorch (CPU version)

`torch-geometric` depends on a specific PyTorch version, so install PyTorch first:

**CPU-only (works on all machines — recommended for most users):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Verify the installation:**
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

You should see output like: `PyTorch version: 2.x.x+cpu`

---

### Step 7 — Install PyTorch Geometric

PyTorch Geometric (`torch-geometric`) requires the PyTorch version and platform to be specified. Run the following, replacing `${TORCH}` with the version returned in Step 6 (e.g. `2.3.0`):

**Automated (recommended — detects your PyTorch version automatically):**
```bash
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cpu.html
```

**Manual (if the above fails — replace `2.3.0` with your actual PyTorch version):**
```bash
pip install torch-geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

**Verify the installation:**
```bash
python -c "import torch_geometric; print('PyG version:', torch_geometric.__version__)"
```

---

### Step 8 — Verify the full environment

Run this one-liner to confirm every required package imports correctly:

```bash
python -c "
import numpy; print('numpy', numpy.__version__)
import pandas; print('pandas', pandas.__version__)
import scipy; print('scipy', scipy.__version__)
import sklearn; print('scikit-learn', sklearn.__version__)
import networkx; print('networkx', networkx.__version__)
import matplotlib; print('matplotlib', matplotlib.__version__)
import seaborn; print('seaborn', seaborn.__version__)
import torch; print('torch', torch.__version__)
import torch_geometric; print('torch_geometric', torch_geometric.__version__)
print('All packages imported successfully.')
"
```

You should see version numbers printed for each package followed by `All packages imported successfully.`

---

### Step 9 — Run the simulator demo to confirm everything works

```bash
python simulators/config_examples/demo-3-gene-loop.py
```

If a time-course plot appears (or is saved to `results/`) and no errors are printed, the environment is fully operational.

---

### Deactivating and Reactivating

To leave the virtual environment:
```bash
deactivate
```

To reactivate it next time (from the project root):

**Windows PowerShell:** `.\.venv\Scripts\Activate.ps1`  
**macOS / Linux:** `source .venv/bin/activate`

---

### Freezing Exact Versions (for strict reproducibility)

Once you have a working environment, save an exact snapshot of all package versions:

```bash
pip freeze > requirements_frozen.txt
```

Anyone can then recreate an identical environment:

```bash
python -m venv .venv_exact
source .venv_exact/bin/activate       # or .venv_exact\Scripts\Activate.ps1 on Windows
pip install -r requirements_frozen.txt
```

---

### GPU Setup (optional — NVIDIA GPUs only)

If you have an NVIDIA GPU, you can accelerate GNN training significantly. First identify your CUDA version:

```bash
nvidia-smi
```

Then install the matching PyTorch + CUDA build. For example, for **CUDA 12.1**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For **CUDA 11.8**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then reinstall PyTorch Geometric with the matching CUDA suffix (e.g. `cu121` instead of `cpu`):
```bash
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-<version>+cu121.html
```

The training script (`train_gnn.py`) automatically detects GPU availability:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

No code changes are required — it will use the GPU automatically.

---

### Common Issues and Fixes

| Problem | Likely cause | Fix |
|---|---|---|
| `Activate.ps1 cannot be loaded` | PowerShell execution policy blocked | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `torch_scatter` not found | PyG wheel URL mismatch | Check PyTorch version with `python -c "import torch; print(torch.__version__)"` and use matching wheel URL |
| `ModuleNotFoundError: torch_geometric` | PyG not installed | Re-run Step 7 with correct PyTorch version |
| `No module named 'networkx'` | Wrong environment active | Check `which python` / `where python` — ensure `.venv` is active |
| ODE solver warnings during simulation | Stiff networks | Expected; the generator logs and skips networks that fail to converge |
| CUDA out of memory | GPU batch size too large | Reduce `BATCH_SIZE` in `train_gnn.py` (default is 64) |

---

*For questions about the project, contact: Hrithik Chandra — University of Hertfordshire, BSc Computer Science, 2026.*
