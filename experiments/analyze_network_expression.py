import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 1. SETUP PATHS & HYPERPARAMS (Must match train_gnn.py)
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data", "synthetic_transsys_backup_50")
GRN_DIR   = os.path.join(DATA_DIR, "grn_edges")
MODEL_PATH = os.path.join(BASE_DIR, "results", "gnn", "gnn_grn_aware_best.pt")
RESULTS_DIR= os.path.join(BASE_DIR, "results", "gnn")

HIDDEN_DIM = 64
NUM_LAYERS = 4
DROPOUT    = 0.4

# 2. REDEFINE THE GNN ARCHITECTURE TO LOAD WEIGHTS
class EdgeWeightedConv(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr="add")
        self.lin_neigh = nn.Linear(in_ch, out_ch, bias=False)
        self.lin_self  = nn.Linear(in_ch, out_ch)
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
        return self.head(x)

def analyze_specific_network(network_id):
    # Load Model
    device = torch.device("cpu")
    model = GRNPredictor().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load Edges & Parameters
    edge_df = pd.read_csv(os.path.join(GRN_DIR, f"{network_id}_edges.csv"))
    gene_df = pd.read_csv(os.path.join(GRN_DIR, f"{network_id}_genes.csv")).set_index("gene")
    
    all_genes = sorted(set(edge_df["factor"]) | set(edge_df["target"]), key=lambda g: int(g[1:]))
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
            gene_df.loc[g, "protein_decay"]
        ])
    x = torch.tensor(features, dtype=torch.float)

    # Predict Expressions
    with torch.no_grad():
        preds = model(x, edge_index, edge_weight).squeeze().numpy()

    # Map back to gene names and convert out of log-space via expm1
    predictions = {g: np.expm1(p) for g, p in zip(all_genes, preds)}
    
    # Sort genes by predicted expression (Highest to Lowest)
    sorted_genes = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    
    print(f"\n--- GNN PREDICTED EXPRESSION FOR {network_id} ---")
    print("Top 5 HIGHLY Expressed Genes (Switched ON):")
    for g, val in sorted_genes[:5]:
        print(f"  {g}: {val:.4f} mRNA")
        
    print("\nBottom 5 REPRESSED Genes (Switched OFF):")
    for g, val in sorted_genes[-5:]:
        print(f"  {g}: {val:.4f} mRNA")

    # Generate Visualization
    G = nx.DiGraph()
    for _, row in edge_df.iterrows():
        G.add_edge(row["factor"], row["target"], weight=row["signed_strength"])
    
    # Color nodes based on expression level
    node_colors = [predictions[node] for node in G.nodes()]
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=1.5, seed=42)
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=700, 
                                   node_color=node_colors, cmap=plt.cm.viridis)
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold")
    
    # Draw edges: Red for repressor, Green for activator
    edges = G.edges(data=True)
    green_edges = [(u, v) for u, v, d in edges if d['weight'] > 0]
    red_edges = [(u, v) for u, v, d in edges if d['weight'] < 0]
    
    nx.draw_networkx_edges(G, pos, edgelist=green_edges, edge_color="mediumseagreen", arrowsize=20)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color="crimson", arrowsize=20)
    
    plt.colorbar(nodes, label="Predicted steady-state mRNA level")
    plt.title(f"Predicted Gene Expression Map ({network_id})\n(Yellow = High Expression, Purple = Low Expression)")
    plt.axis("off")
    
    plot_path = os.path.join(RESULTS_DIR, f"{network_id}_expression_map.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved expression map visualization to {plot_path}")

if __name__ == "__main__":
    # Test it on network 040 (which should be part of the test set)
    analyze_specific_network("net_040")
