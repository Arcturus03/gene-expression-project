"""
GRN-aware Graph Neural Network for gene expression prediction.
Uses SAGEConv with signed edge weights via custom message passing.
Compares Expression-Only GNN vs GRN-Aware GNN (same architecture, different edges).
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data", "synthetic_transsys_backup_50")
#DATA_DIR  = os.path.join(BASE_DIR, "data", "synthetic_transsys")  # Use the new dataset generated with 100 networks and 50 seeds each
GRN_DIR   = os.path.join(DATA_DIR, "grn_edges")    # folder with net_XXX_edges/genes CSVs
EXPR_PATH       = os.path.join(DATA_DIR, "expression_profiles.csv")
META_PATH       = os.path.join(DATA_DIR, "network_metadata.csv")
RESULTS_DIR     = os.path.join(BASE_DIR, "results", "gnn")
RANDOM_SEED     = 42
HIDDEN_DIM      = 64   # hidden dimension for GNN layers, increased from 64 to 128 to give the model more capacity to learn complex patterns in the data, especially since the task of predicting gene expression from GRN structure can be quite challenging and may require a richer representation.
NUM_LAYERS      = 4     # number of GNN layers increased from 2 to 3 to allow for better information propagation across the graph, especially since some networks may have longer paths between regulators and targets. This can help the model capture more complex dependencies in the GRN.
DROPOUT         = 0.4   # dropout rate for regularization, helps prevent overfitting by randomly dropping units during training. A value of 0.4 means that 40% of the units will be dropped, which is a common choice for regularization in neural networks.
LR              = 5e-4  # was 1e-3, lowered learning rate so that 
WEIGHT_DECAY    = 1e-4
BATCH_SIZE      = 64
MAX_EPOCHS      = 400
PATIENCE        = 40        # early stopping patience is increased to 30 epochs to allow more time for convergence, especially since the dataset is synthetic and may require more epochs to learn effectively. This means that if the validation loss does not improve for 30 consecutive epochs, the training will stop early to prevent overfitting and save time.
TRAIN_RATIO     = 0.8   # 80% of networks for training

os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# ── 1. LOAD METADATA ────────────────────────────────────────────────────────
meta = pd.read_csv(META_PATH)
all_networks = sorted(meta["network_id"].unique())
n_train = int(len(all_networks) * TRAIN_RATIO)
train_nets = set(all_networks[:n_train])
test_nets  = set(all_networks[n_train:])
print(f"Networks — Train: {len(train_nets)}, Test: {len(test_nets)}")

# ── 2. LOAD EXPRESSION PROFILES ─────────────────────────────────────────────
expr_df = pd.read_csv(EXPR_PATH)

# ── 3. LOAD ALL EDGE FILES ───────────────────────────────────────────────────
def load_edges(network_id):
    """Returns (edge_index [2,E], signed_strength [E]) tensors, gene→int mapping."""
    path = os.path.join(GRN_DIR, f"{network_id}_edges.csv")
    
    # Guard: skip networks with empty edge files
    if os.path.getsize(path) == 0:
        return None, None, None
    
    df   = pd.read_csv(path)
    if df.empty or len(df) == 0:
        return None, None, None
    
    
    # gene name → integer index
    all_genes = sorted(set(df["factor"]) | set(df["target"]),
                        key=lambda g: int(g[1:]))
    gene2idx = {g: i for i, g in enumerate(all_genes)}
    src = torch.tensor([gene2idx[f] for f in df["factor"]], dtype=torch.long)
    tgt = torch.tensor([gene2idx[t] for t in df["target"]], dtype=torch.long)
    edge_index = torch.stack([src, tgt], dim=0)
    edge_weight = torch.tensor(df["signed_strength"].values, dtype=torch.float)
    return edge_index, edge_weight, gene2idx


# Data Sanity
# Use the dynamic GRN_DIR defined in config instead of hardcoding
edges = glob.glob(f"{GRN_DIR}/net_*_edges.csv")
genes = glob.glob(f"{GRN_DIR}/net_*_genes.csv")

print(f"Edge files found: {len(edges)}")  
print(f"Gene files found: {len(genes)}")   
print(f"Example: {edges[0]}")


# ── 4. BUILD PyG DATASET ─────────────────────────────────────────────────────
def load_gene_params(network_id):
    """Returns dict: gene_name → {basal_expression, mrna_decay, protein_decay}"""
    path = os.path.join(GRN_DIR, f"{network_id}_genes.csv")
    df   = pd.read_csv(path).set_index("gene")
    return df[["basal_expression", "mrna_decay", "protein_decay"]].to_dict(orient="index")


def build_dataset(network_ids, use_grn_edges=True):
    """
    Task: predict steady-state mRNA from intrinsic gene parameters only.
    GRN-aware model aggregates structural info from regulatory neighbours.
    This tests: does knowing GRN topology predict expression
    when no expression data is given as input?
    """
    dataset = []
    for net_id in network_ids:
        edge_index, edge_weight, gene2idx = load_edges(net_id)
        
        # Skip networks with no edges
        if gene2idx is None:
            print(f"  Skipping {net_id} — empty edge file")
            continue
        
        gene_params = load_gene_params(net_id)
        genes_sorted = sorted(gene2idx.keys(), key=lambda g: int(g[1:]))

        net_rows = expr_df[expr_df["network_id"] == net_id]  
        
        for _, row in net_rows.iterrows():
            node_features = []
            mrna_targets  = []
            valid = True

            for g in genes_sorted:
                m_col = f"{g}_mRNA"
                if m_col not in row.index or pd.isna(row[m_col]):
                    valid = False
                    break
                p = gene_params[g]
                # Node features: ONLY intrinsic gene parameters — NO expression data
                # Forces the GRN-aware model to use regulatory edges for prediction
                node_features.append([
                    p["basal_expression"],   # how much this gene transcribes by default
                    p["mrna_decay"],         # how fast mRNA degrades
                    p["protein_decay"],      # how fast protein degrades
                ])
                mrna_targets.append(np.log1p(float(row[m_col])))

            if not valid:
                continue

            x = torch.tensor(node_features, dtype=torch.float)       # [N, 3]
            y = torch.tensor(mrna_targets,  dtype=torch.float).unsqueeze(1)

            if use_grn_edges:
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
            else:
                empty_ei = torch.zeros((2, 0), dtype=torch.long)
                empty_ea = torch.zeros(0, dtype=torch.float)
                data = Data(x=x, edge_index=empty_ei, edge_attr=empty_ea, y=y)

            dataset.append(data)
    return dataset



print("Building GRN-aware datasets...")
train_grn  = build_dataset(list(train_nets), use_grn_edges=True)
test_grn   = build_dataset(list(test_nets),  use_grn_edges=True)
train_expr = build_dataset(list(train_nets), use_grn_edges=False)
test_expr  = build_dataset(list(test_nets),  use_grn_edges=False)
print(f"Train graphs: {len(train_grn)}, Test graphs: {len(test_grn)}")




# ── 5. GNN MODEL ─────────────────────────────────────────────────────────────

class EdgeWeightedConv(MessagePassing):
    """
    Custom SAGEConv-style layer that incorporates signed edge weights.
    Message from neighbour j to node i = edge_weight_ij * W * x_j
    Aggregation: sum (then combined with self-connection in forward).
    """
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr="add")
        self.lin_neigh = nn.Linear(in_ch, out_ch, bias=False)
        self.lin_self  = nn.Linear(in_ch, out_ch)

    def forward(self, x, edge_index, edge_weight):
        # Self part
        out = self.lin_self(x)
        # Neighbour aggregation with edge weights
        out = out + self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out

    def message(self, x_j, edge_weight):
        # x_j: [E, in_ch], edge_weight: [E]
        return self.lin_neigh(x_j) * edge_weight.unsqueeze(-1)


class GRNPredictor(nn.Module):
    def __init__(self, in_ch= 3, hidden=HIDDEN_DIM, n_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(EdgeWeightedConv(in_ch, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))
        for _ in range(n_layers - 1):
            self.convs.append(EdgeWeightedConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


# ── 6. TRAINING LOOP ─────────────────────────────────────────────────────────
def train_model(train_data, test_data, model_name):
    loader_tr = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    loader_te = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    model = GRNPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0
        for batch in loader_tr:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        train_loss = total_loss / len(train_data)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in loader_te:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr)
                val_loss += F.mse_loss(pred, batch.y).item() * batch.num_graphs
        val_loss /= len(test_data)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{model_name}_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [{model_name}] Early stop at epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(f"  [{model_name}] Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    return model, train_losses, val_losses


def evaluate(model, test_data):
    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            all_pred.append(pred.cpu().numpy())
            all_true.append(batch.y.cpu().numpy())
    y_pred = np.concatenate(all_pred).flatten()
    y_true = np.concatenate(all_true).flatten()
    return {
        "R2":  r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
    }, y_pred, y_true


# ── 7. RUN BOTH MODELS ───────────────────────────────────────────────────────
print("\n=== Training Expression-Only GNN (no edges) ===")
model_expr, tl_expr, vl_expr = train_model(train_expr, test_expr, "gnn_expr_only")
model_expr.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "gnn_expr_only_best.pt")))
metrics_expr, pred_expr, true_expr = evaluate(model_expr, test_expr)
print(f"Expression-Only GNN -> R²={metrics_expr['R2']:.4f}, MSE={metrics_expr['MSE']:.4f}")

print("\n=== Training GRN-Aware GNN (regulatory edges + signed weights) ===")
model_grn, tl_grn, vl_grn = train_model(train_grn, test_grn, "gnn_grn_aware")
model_grn.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "gnn_grn_aware_best.pt")))
metrics_grn, pred_grn, true_grn = evaluate(model_grn, test_grn)
print(f"GRN-Aware GNN       -> R²={metrics_grn['R2']:.4f}, MSE={metrics_grn['MSE']:.4f}")

# ── 8. SAVE RESULTS ──────────────────────────────────────────────────────────
results = {
    "GNN_ExprOnly": metrics_expr,
    "GNN_GRNAware": metrics_grn,
    "improvement_R2": metrics_grn["R2"] - metrics_expr["R2"],
}
with open(os.path.join(RESULTS_DIR, "gnn_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\n── Results ────────────────────────")
for k, v in results.items():
    print(f"  {k}: {v}")

# ── 9. PLOTS ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Training curves
ax = axes[0]
ax.plot(tl_expr, label="Expr-Only Train", color="steelblue",  alpha=0.7)
ax.plot(vl_expr, label="Expr-Only Val",   color="steelblue",  linestyle="--")
ax.plot(tl_grn,  label="GRN-Aware Train", color="darkorange", alpha=0.7)
ax.plot(vl_grn,  label="GRN-Aware Val",   color="darkorange", linestyle="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.set_title("Training Curves")
ax.legend(); ax.set_yscale("log")

# Predicted vs Actual
ax = axes[1]
sample = np.random.choice(len(true_grn), size=min(2000, len(true_grn)), replace=False)
ax.scatter(true_grn[sample], pred_grn[sample], alpha=0.3, s=5, color="darkorange")
lims = [min(true_grn.min(), pred_grn.min()), max(true_grn.max(), pred_grn.max())]
ax.plot(lims, lims, "k--", linewidth=1)
ax.set_xlabel("Actual log1p(protein)"); ax.set_ylabel("Predicted")
ax.set_title(f"GRN-Aware GNN (R²={metrics_grn['R2']:.4f})")

# R² comparison bar chart
ax = axes[2]
labels = ["GNN\nExpr-Only", "GNN\nGRN-Aware"]
vals   = [metrics_expr["R2"], metrics_grn["R2"]]
colors = ["steelblue", "darkorange"]
bars = ax.bar(labels, vals, color=colors, edgecolor="black", width=0.4)
ax.set_ylim(max(0, min(vals) - 0.1), min(1.05, max(vals) + 0.1))
ax.set_ylabel("R² Score"); ax.set_title("GNN: Expr-Only vs GRN-Aware")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gnn_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved → {RESULTS_DIR}/gnn_comparison.png")
