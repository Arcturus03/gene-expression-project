
# ── EXPERIMENT 2: mRNA prediction from gene params + GRN features ──────────
# Mirrors the GNN task exactly for fair comparison
# Input: gene structural params + GRN topology features (NO expression data)
# Target: log1p(mRNA)

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Build per-gene rows with only structural features
# For each gene in each network condition:
#   features = [basal_expression, mrna_decay, protein_decay, 
#               grn_in_degree, grn_out_degree, grn_n_activators,
#               grn_n_repressors, grn_pagerank, grn_betweenness]
#   target   = log1p(mRNA)

# Run two comparisons:
# A) Params only (no GRN) vs B) Params + GRN features
# Same train/test network split as GNN (80/20)

"""
train_mrna_baselines.py — Tabular baselines for mRNA prediction task.

PURPOSE:
    Provides a fair comparison against the GNN experiment by running
    Ridge and Random Forest models on the SAME task:
    - Input:  gene structural parameters + GRN topology features (NO expression data)
    - Target: log1p(steady-state mRNA)
    - Same 80/20 network split as train_gnn.py

COMPARISONS:
    A) Params Only    — basal_expression, mrna_decay, protein_decay
    B) Params + GRN   — above + in_degree, out_degree, n_activators,
                        n_repressors, total_activation, total_repression,
                        net_regulation, pagerank, betweenness

This directly answers: does flat GRN structural information help tabular
models, and how does that compare to GNN message-passing?
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── CONFIG — must match train_gnn.py exactly ────────────────────────────────
# Use absolute path based on the script location to allow running from any directory
import os
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data", "synthetic_transsys_backup_50")
GRN_DIR     = os.path.join(DATA_DIR, "grn_edges")
EXPR_PATH   = os.path.join(DATA_DIR, "expression_profiles.csv")
META_PATH   = os.path.join(DATA_DIR, "network_metadata.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "mrna_baselines")
RANDOM_SEED = 42
TRAIN_RATIO = 0.8

os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

# ── 1. LOAD METADATA AND DEFINE TRAIN/TEST SPLIT ────────────────────────────
meta = pd.read_csv(META_PATH)
all_networks = sorted(meta["network_id"].unique())
n_train      = int(len(all_networks) * TRAIN_RATIO)
train_nets   = set(all_networks[:n_train])
test_nets    = set(all_networks[n_train:])
print(f"Networks — Train: {len(train_nets)}, Test: {len(test_nets)}")

# ── 2. LOAD EXPRESSION PROFILES ─────────────────────────────────────────────
expr_df = pd.read_csv(EXPR_PATH)

# ── 3. COMPUTE GRN FEATURES PER GENE PER NETWORK ────────────────────────────
def compute_grn_features(network_id):
    """
    Returns dict: gene_name → {structural params + GRN topology features}
    GRN features computed from the regulatory edge list using networkx.
    """
    edge_path = os.path.join(GRN_DIR, f"{network_id}_edges.csv")
    gene_path = os.path.join(GRN_DIR, f"{network_id}_genes.csv")

    # Skip empty networks
    if not os.path.exists(edge_path) or os.path.getsize(edge_path) == 0:
        return None

    edges_df = pd.read_csv(edge_path)
    genes_df = pd.read_csv(gene_path).set_index("gene")

    if edges_df.empty:
        return None

    # Build directed networkx graph for topology features
    G = nx.DiGraph()
    G.add_nodes_from(genes_df.index.tolist())
    for _, row in edges_df.iterrows():
        G.add_edge(row["factor"], row["target"],
                   weight=abs(row["signed_strength"]),
                   signed=row["signed_strength"])

    # Compute graph-level topology features
    try:
        pagerank   = nx.pagerank(G, weight="weight")
    except Exception:
        pagerank   = {n: 0.0 for n in G.nodes()}
    try:
        betweenness = nx.betweenness_centrality(G, weight="weight")
    except Exception:
        betweenness = {n: 0.0 for n in G.nodes()}

    features = {}
    for gene in genes_df.index:
        # Structural gene parameters
        params = genes_df.loc[gene]

        # Regulatory edge statistics for this gene as TARGET
        in_edges   = edges_df[edges_df["target"] == gene]
        out_edges  = edges_df[edges_df["factor"] == gene]

        n_activators    = int((in_edges["type"] == "activator").sum())
        n_repressors    = int((in_edges["type"] == "repressor").sum())
        total_activ     = float(in_edges[in_edges["type"] == "activator"]["strength"].sum())
        total_repres    = float(in_edges[in_edges["type"] == "repressor"]["strength"].sum())
        net_regulation  = total_activ - total_repres

        features[gene] = {
            # Structural gene params
            "basal_expression": float(params["basal_expression"]),
            "mrna_decay":       float(params["mrna_decay"]),
            "protein_decay":    float(params["protein_decay"]),
            # GRN topology features
            "grn_in_degree":    int(G.in_degree(gene)),
            "grn_out_degree":   int(G.out_degree(gene)),
            "grn_n_activators": n_activators,
            "grn_n_repressors": n_repressors,
            "grn_total_activation": total_activ,
            "grn_total_repression": total_repres,
            "grn_net_regulation":   net_regulation,
            "grn_pagerank":     float(pagerank.get(gene, 0.0)),
            "grn_betweenness":  float(betweenness.get(gene, 0.0)),
        }
    return features


# ── 4. BUILD DATASET ─────────────────────────────────────────────────────────
def build_dataset(network_ids):
    """
    Returns X_params, X_grn, y for all genes across all conditions
    in the given network IDs.
    X_params — structural params only  (3 features)
    X_grn    — structural params + GRN (12 features)
    y        — log1p(mRNA)
    """
    rows_params = []
    rows_grn    = []
    targets     = []

    skipped = 0
    for net_id in network_ids:
        grn_feats = compute_grn_features(net_id)
        if grn_feats is None:
            skipped += 1
            continue

        net_rows = expr_df[expr_df["network_id"] == net_id]
        genes_sorted = sorted(grn_feats.keys(), key=lambda g: int(g[1:]))

        for _, row in net_rows.iterrows():
            for g in genes_sorted:
                m_col = f"{g}_mRNA"
                if m_col not in row.index or pd.isna(row[m_col]):
                    continue

                f = grn_feats[g]

                # Params-only feature vector
                rows_params.append([
                    f["basal_expression"],
                    f["mrna_decay"],
                    f["protein_decay"],
                ])

                # Params + GRN feature vector
                rows_grn.append([
                    f["basal_expression"],
                    f["mrna_decay"],
                    f["protein_decay"],
                    f["grn_in_degree"],
                    f["grn_out_degree"],
                    f["grn_n_activators"],
                    f["grn_n_repressors"],
                    f["grn_total_activation"],
                    f["grn_total_repression"],
                    f["grn_net_regulation"],
                    f["grn_pagerank"],
                    f["grn_betweenness"],
                ])

                targets.append(np.log1p(float(row[m_col])))

    if skipped > 0:
        print(f"  Skipped {skipped} networks with empty edge files")

    return (np.array(rows_params, dtype=np.float32),
            np.array(rows_grn,    dtype=np.float32),
            np.array(targets,     dtype=np.float32))


print("Building tabular datasets...")
X_train_p, X_train_g, y_train = build_dataset(list(train_nets))
X_test_p,  X_test_g,  y_test  = build_dataset(list(test_nets))
print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
print(f"Params features: {X_train_p.shape[1]}, GRN features: {X_train_g.shape[1]}")

# ── 5. SCALE FEATURES ────────────────────────────────────────────────────────
scaler_p = StandardScaler()
scaler_g = StandardScaler()

X_train_p_sc = scaler_p.fit_transform(X_train_p)
X_test_p_sc  = scaler_p.transform(X_test_p)
X_train_g_sc = scaler_g.fit_transform(X_train_g)
X_test_g_sc  = scaler_g.transform(X_test_g)

# ── 6. DEFINE AND RUN MODELS ─────────────────────────────────────────────────
models = {
    "Ridge":  Ridge(alpha=1.0),
    "RF":     RandomForestRegressor(n_estimators=300, max_depth=None,
                                    n_jobs=-1, random_state=RANDOM_SEED),
}

results = {}

print("\n── Training models ─────────────────────────────────")
for model_name, model in models.items():
    for feat_set, X_tr, X_te in [
        ("ParamsOnly", X_train_p_sc, X_test_p_sc),
        ("Params+GRN", X_train_g_sc, X_test_g_sc),
    ]:
        run_name = f"{model_name}_{feat_set}"
        clf = model.__class__(**model.get_params())
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        metrics = {
            "R2":  round(float(r2_score(y_test, y_pred)), 4),
            "MSE": round(float(mean_squared_error(y_test, y_pred)), 4),
            "MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
        }
        results[run_name] = metrics
        print(f"  {run_name:30s} → R²={metrics['R2']:.4f}  "
                f"MSE={metrics['MSE']:.4f}  MAE={metrics['MAE']:.4f}")

# ── 7. ADD GNN RESULTS FOR COMPLETE COMPARISON TABLE ────────────────────────
# Manually add locked GNN results so everything is in one place
results["GNN_ExprOnly"] = {"R2": 0.0436, "MSE": 1.2181, "MAE": 1.0656}
results["GNN_GRNAware"] = {"R2": 0.5911, "MSE": 0.5208, "MAE": 0.5056}

# ── 8. SAVE RESULTS ──────────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "mrna_baselines_results.json"), "w") as f:
    json.dump(results, f, indent=2)

results_df = pd.DataFrame(results).T.reset_index()
results_df.columns = ["Model", "R2", "MSE", "MAE"]
results_df.to_csv(os.path.join(RESULTS_DIR, "mrna_baselines_results.csv"), index=False)
print(f"\nResults saved → {RESULTS_DIR}/mrna_baselines_results.csv")

# ── 9. COMPARISON PLOT ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

model_order = [
    "Ridge_ParamsOnly", "Ridge_Params+GRN",
    "RF_ParamsOnly",    "RF_Params+GRN",
    "GNN_ExprOnly",     "GNN_GRNAware",
]
colors = [
    "#aec6cf", "#2e86ab",   # Ridge: light/dark blue
    "#f4a460", "#d2691e",   # RF: light/dark orange
    "#b5ead7", "#2d6a4f",   # GNN: light/dark green
]
labels = [
    "Ridge\n(Params)", "Ridge\n(+GRN)",
    "RF\n(Params)",    "RF\n(+GRN)",
    "GNN\n(No edges)", "GNN\n(GRN edges)",
]

r2_vals  = [results[m]["R2"]  for m in model_order]
mse_vals = [results[m]["MSE"] for m in model_order]

# R² bar chart
ax = axes[0]
bars = ax.bar(labels, r2_vals, color=colors, edgecolor="black", width=0.6)
ax.set_ylabel("R² Score (higher = better)", fontsize=11)
ax.set_title("mRNA Prediction: R² Comparison\n(Tabular vs GNN, with and without GRN)", fontsize=11)
ax.set_ylim(0, 1.0)
ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.015,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# MSE bar chart
ax = axes[1]
bars = ax.bar(labels, mse_vals, color=colors, edgecolor="black", width=0.6)
ax.set_ylabel("MSE (lower = better)", fontsize=11)
ax.set_title("mRNA Prediction: MSE Comparison\n(Tabular vs GNN, with and without GRN)", fontsize=11)
for bar, val in zip(bars, mse_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mrna_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved → {RESULTS_DIR}/mrna_comparison.png")

# ── 10. PRINT FINAL COMPARISON TABLE ─────────────────────────────────────────
print("\n══ FINAL COMPARISON TABLE ══════════════════════════════")
print(f"{'Model':<30} {'R²':>8} {'MSE':>8} {'MAE':>8}")
print("─" * 58)
for m in model_order:
    r = results[m]
    print(f"{m:<30} {r['R2']:>8.4f} {r['MSE']:>8.4f} {r['MAE']:>8.4f}")
print("═" * 58)
