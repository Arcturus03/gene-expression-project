"""
Network Expression Analyzer & Visualizer (Upgraded)

INSTRUCTIONS TO RUN THIS SCRIPT:
--------------------------------
Open the terminal, ensure you are in the 'gene-expression-project' folder, 
and run one of the following commands based on what you want:

1. To get the DEFAULT output (automatically analyzes 'net_040'):
    python experiments/analyze_network_expression.py

2. To get a SPECIFIC network output (for example, 'net_012'):
    python experiments/analyze_network_expression.py net_012

3. To get ALL available networks processed at once:
    python experiments/analyze_network_expression.py --all

This script loads the trained GRN-aware GNN, runs inference on one or more
networks, and produces:
    - Console summary of top 5 "ON" and bottom 5 "OFF" genes (by mRNA level)
    - A NetworkX visualisation where:
      * Node colour = predicted steady-state mRNA (original scale, after expm1)
      * Edge colour = activator (green) vs repressor (red)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 1. PATHS & HYPERPARAMS (must match train_gnn.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "synthetic_transsys")
GRN_DIR = os.path.join(DATA_DIR, "grn_edges")
MODEL_PATH = os.path.join(BASE_DIR, "results", "gnn", "gnn_grn_aware_best.pt")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "gnn")

HIDDEN_DIM = 64
NUM_LAYERS = 4
DROPOUT = 0.4


# 2. GNN ARCHITECTURE (copied from train_gnn.py)
class EdgeWeightedConv(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr="add")
        self.lin_neigh = nn.Linear(in_ch, out_ch, bias=False)
        self.lin_self = nn.Linear(in_ch, out_ch)

    def forward(self, x, edge_index, edge_weight):
        out = self.lin_self(x)
        out = out + self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out

    def message(self, x_j, edge_weight):
        return self.lin_neigh(x_j) * edge_weight.unsqueeze(-1)


class GRNPredictor(nn.Module):
    def __init__(self, in_ch=3, hidden=HIDDEN_DIM, n_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

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
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
        return self.head(x)


def load_model(device="cpu"):
    model = GRNPredictor().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# 3. CORE ANALYSIS FOR A SINGLE NETWORK
def analyze_specific_network(network_id: str, model, device="cpu"):
    # Load edges and gene parameters
    edge_path = os.path.join(GRN_DIR, f"{network_id}_edges.csv")
    gene_path = os.path.join(GRN_DIR, f"{network_id}_genes.csv")

    if not (os.path.exists(edge_path) and os.path.exists(gene_path)):
        print(f"[WARN] Missing files for {network_id}, skipping.")
        return

    edge_df = pd.read_csv(edge_path)
    gene_df = pd.read_csv(gene_path).set_index("gene")

    all_genes = sorted(
        set(edge_df["factor"]) | set(edge_df["target"]),
        key=lambda g: int(g[1:])
    )
    gene2idx = {g: i for i, g in enumerate(all_genes)}

    src = torch.tensor([gene2idx[f] for f in edge_df["factor"]], dtype=torch.long)
    tgt = torch.tensor([gene2idx[t] for t in edge_df["target"]], dtype=torch.long)
    edge_index = torch.stack([src, tgt], dim=0)
    edge_weight = torch.tensor(edge_df["signed_strength"].values, dtype=torch.float)

    features = []
    for g in all_genes:
        features.append([
            gene_df.loc[g, "basal_expression"],
            gene_df.loc[g, "mrna_decay"],
            gene_df.loc[g, "protein_decay"],
        ])
    x = torch.tensor(features, dtype=torch.float).to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # Predict log1p(mRNA), then convert back with expm1
    with torch.no_grad():
        preds_log = model(x, edge_index, edge_weight).squeeze().cpu().numpy()
    predictions = {g: np.expm1(p) for g, p in zip(all_genes, preds_log)}

    # Sort genes for ON/OFF lists
    sorted_desc = sorted(predictions.items(), key=lambda kv: kv[1], reverse=True)
    sorted_asc = sorted(predictions.items(), key=lambda kv: kv[1])

    print(f"\n--- GNN PREDICTED EXPRESSION FOR {network_id} ---")
    print("Top 5 HIGHLY Expressed Genes (Switched ON):")
    for g, val in sorted_desc[:5]:
        print(f"  {g}: {val:.4f} mRNA")

    print("\nBottom 5 REPRESSED Genes (Switched OFF):")
    for g, val in sorted_asc[:5]:
        print(f"  {g}: {val:.4f} mRNA")

    # Build NetworkX graph
    G = nx.DiGraph()
    for _, row in edge_df.iterrows():
        G.add_edge(row["factor"], row["target"], weight=row["signed_strength"])

    node_colors = [predictions[node] for node in G.nodes()]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=1.5, seed=42)

    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=700,
        node_color=node_colors,
        cmap=plt.cm.viridis,
    )
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold")

    # Edge colours: green = activator, red = repressor
    edges = G.edges(data=True)
    green_edges = [(u, v) for u, v, d in edges if d["weight"] > 0]
    red_edges = [(u, v) for u, v, d in edges if d["weight"] < 0]

    nx.draw_networkx_edges(G, pos, edgelist=green_edges,
                        edge_color="mediumseagreen", arrowsize=20)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges,
                        edge_color="crimson", arrowsize=20)

    plt.colorbar(nodes, label="Predicted steady-state mRNA level (expm1 scale)")
    plt.title(
        f"Predicted Gene Expression Map ({network_id})\n"
        f"(Yellow = High Expression, Purple = Low Expression)"
    )
    plt.axis("off")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(RESULTS_DIR, f"{network_id}_expression_map.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved expression map visualization to {plot_path}")


# 4. CLI ENTRY POINT
if __name__ == "__main__":
    device = torch.device("cpu")
    model = load_model(device)

    # ---------------------------------------------------------
    # HOW THE COMMAND LINE CHOICES WORK:
    # 1. DEFAULT: If you don't provide an argument, it falls back to 'net_040'
    # 2. ALL: If you type '--all' or '-a', it searches for and runs every network file
    # 3. SPECIFIC: If you type a specific name like 'net_012', it runs only that one
    # ---------------------------------------------------------
    
    if len(sys.argv) == 1:
        target_ids = ["net_040"]
    elif sys.argv[1] in ("--all", "-a"):
        # Infer all network IDs from *_edges.csv files
        ids = []
        for fname in os.listdir(GRN_DIR):
            if fname.endswith("_edges.csv"):
                nid = fname.replace("_edges.csv", "")
                ids.append(nid)
        target_ids = sorted(ids)
    else:
        target_ids = [sys.argv[1]]

    for nid in target_ids:
        analyze_specific_network(nid, model, device=device)