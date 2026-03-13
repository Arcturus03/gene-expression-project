"""
build_pergene_dataset.py — Transform expression profiles into a per-gene ML dataset.

WHAT THIS DOES:
1. Loads expression_profiles.csv (500 profiles × ~20-60 genes each)
2. Loads GRN edge lists to extract structural features for each gene
3. Creates a "per-gene" dataset where each row = one gene from one profile

OUTPUT FORMAT (pergene_dataset.csv):
    sample_id, network_id, gene_name, [mRNA features], [GRN features], target_protein

USAGE:
    python experiments/build_pergene_dataset.py

WHY PER-GENE?
    Instead of predicting all genes at once (multi-output), we predict ONE gene's
    protein level at a time. This lets us add gene-specific GRN features and
    directly measure which genes benefit most from structural information.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_transsys")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml_ready")

# Feature engineering settings
USE_ALL_MRNA_AS_FEATURES = True   # Use all other genes' mRNA as input features
USE_GRN_FEATURES = True           # Extract and include GRN structural features


# =============================================================================
# GRN FEATURE EXTRACTION
# =============================================================================

def load_network_edges(network_id: str, edges_dir: str) -> pd.DataFrame:
    """Load edge list for a specific network."""
    edge_file = os.path.join(edges_dir, f"{network_id}_edges.csv")
    if os.path.exists(edge_file):
        return pd.read_csv(edge_file)
    return pd.DataFrame(columns=['factor', 'target', 'type', 'strength', 'signed_strength'])


def build_networkx_graph(edges_df: pd.DataFrame, gene_names: List[str]) -> nx.DiGraph:
    """Build a NetworkX directed graph from edge dataframe."""
    G = nx.DiGraph()
    G.add_nodes_from(gene_names)
    
    for _, row in edges_df.iterrows():
        G.add_edge(
            row['factor'], 
            row['target'],
            effect_type=row['type'],
            strength=row['strength'],
            signed_strength=row['signed_strength']
        )
    
    return G


def extract_grn_features_for_gene(
    gene_name: str,
    edges_df: pd.DataFrame,
    G: nx.DiGraph,
    pagerank: Dict[str, float],
    betweenness: Dict[str, float]
) -> Dict[str, float]:
    """
    Extract GRN structural features for a single gene.
    
    Features extracted:
    - in_degree: number of regulators targeting this gene
    - out_degree: number of genes this gene regulates
    - n_activators: how many activating edges point to this gene
    - n_repressors: how many repressing edges point to this gene
    - total_activation: sum of positive edge weights targeting this gene
    - total_repression: sum of negative edge weights targeting this gene
    - net_regulation: total_activation - |total_repression|
    - pagerank: PageRank centrality (importance in the network)
    - betweenness: betweenness centrality (how often on shortest paths)
    """
    # Filter edges targeting this gene
    incoming = edges_df[edges_df['target'] == gene_name]
    outgoing = edges_df[edges_df['factor'] == gene_name]
    
    # Basic degree features
    in_degree = len(incoming)
    out_degree = len(outgoing)
    
    # Activation vs repression counts
    n_activators = len(incoming[incoming['type'] == 'activator'])
    n_repressors = len(incoming[incoming['type'] == 'repressor'])
    
    # Weighted sums
    activator_edges = incoming[incoming['type'] == 'activator']
    repressor_edges = incoming[incoming['type'] == 'repressor']
    
    total_activation = activator_edges['strength'].sum() if len(activator_edges) > 0 else 0.0
    total_repression = repressor_edges['strength'].sum() if len(repressor_edges) > 0 else 0.0
    net_regulation = total_activation - total_repression
    
    # Centrality metrics (pre-computed for efficiency)
    pr = pagerank.get(gene_name, 0.0)
    bc = betweenness.get(gene_name, 0.0)
    
    return {
        'in_degree': in_degree,
        'out_degree': out_degree,
        'n_activators': n_activators,
        'n_repressors': n_repressors,
        'total_activation': total_activation,
        'total_repression': total_repression,
        'net_regulation': net_regulation,
        'pagerank': pr,
        'betweenness': bc
    }


# =============================================================================
# MAIN DATASET BUILDER
# =============================================================================

def build_pergene_dataset():
    """Build the complete per-gene ML dataset."""
    
    print("=" * 60)
    print("BUILDING PER-GENE ML DATASET")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # STEP 1: Load expression profiles and metadata
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading expression profiles and metadata...")
    
    profiles_path = os.path.join(DATA_DIR, "expression_profiles.csv")
    metadata_path = os.path.join(DATA_DIR, "network_metadata.csv")
    edges_dir = os.path.join(DATA_DIR, "grn_edges")
    
    profiles_df = pd.read_csv(profiles_path)
    metadata_df = pd.read_csv(metadata_path)
    
    print(f"  Loaded {len(profiles_df)} expression profiles")
    print(f"  Columns: {profiles_df.columns.tolist()[:5]}... ({len(profiles_df.columns)} total)")
    
    # -------------------------------------------------------------------------
    # STEP 2: Identify gene columns and structure
    # -------------------------------------------------------------------------
    print("\n[2/4] Parsing gene columns...")
    
    # Find all mRNA and protein columns
    mrna_cols = [c for c in profiles_df.columns if c.endswith('_mRNA')]
    protein_cols = [c for c in profiles_df.columns if c.endswith('_protein')]
    
    # Extract gene names (strip _mRNA suffix)
    gene_names_from_mrna = [c.replace('_mRNA', '') for c in mrna_cols]
    gene_names_from_protein = [c.replace('_protein', '') for c in protein_cols]
    
    # The actual genes present depends on the max gene count across networks
    # We'll handle variable-size networks by using NaN for missing genes
    all_genes = sorted(set(gene_names_from_mrna) | set(gene_names_from_protein))
    
    print(f"  Found {len(mrna_cols)} mRNA columns, {len(protein_cols)} protein columns")
    print(f"  Gene names: {all_genes[:5]}... ({len(all_genes)} total)")
    
    # -------------------------------------------------------------------------
    # STEP 3: Build per-gene samples
    # -------------------------------------------------------------------------
    print("\n[3/4] Building per-gene samples...")
    
    samples = []
    n_profiles = len(profiles_df)
    
    # Cache for network-level computations (edges, graph, centralities)
    network_cache = {}
    
    for profile_idx in range(n_profiles):
        if (profile_idx + 1) % 100 == 0:
            print(f"  Processing profile {profile_idx + 1}/{n_profiles}...")
        
        row = profiles_df.iloc[profile_idx]
        meta = metadata_df.iloc[profile_idx]
        
        network_id = meta['network_id']
        n_genes_in_network = int(meta['n_genes'])
        
        # Load/cache network data
        if network_id not in network_cache:
            edges_df = load_network_edges(network_id, edges_dir)
            
            # Get gene names for this specific network
            net_genes = [f"G{i}" for i in range(n_genes_in_network)]
            
            G = build_networkx_graph(edges_df, net_genes)
            
            # Pre-compute centralities (expensive, so we cache them)
            try:
                pagerank = nx.pagerank(G, max_iter=100)
            except:
                pagerank = {g: 0.0 for g in net_genes}
            
            try:
                betweenness = nx.betweenness_centrality(G)
            except:
                betweenness = {g: 0.0 for g in net_genes}
            
            network_cache[network_id] = {
                'edges_df': edges_df,
                'G': G,
                'pagerank': pagerank,
                'betweenness': betweenness,
                'genes': net_genes
            }
        
        cache = network_cache[network_id]
        net_genes = cache['genes']
        
        # Get all mRNA values for this profile (for input features)
        mrna_values = {}
        for gene in net_genes:
            col = f"{gene}_mRNA"
            if col in row.index:
                mrna_values[gene] = row[col]
            else:
                mrna_values[gene] = 0.0
        
        # Create one sample per gene
        for gene in net_genes:
            protein_col = f"{gene}_protein"
            mrna_col = f"{gene}_mRNA"
            
            # Skip if target doesn't exist
            if protein_col not in row.index:
                continue
            
            target_protein = row[protein_col]
            own_mrna = row[mrna_col] if mrna_col in row.index else 0.0
            
            # Skip samples with NaN target
            if pd.isna(target_protein):
                continue
            
            sample = {
                'sample_id': f"{network_id}_seed{meta['sim_seed']}_{gene}",
                'network_id': network_id,
                'seed': meta['sim_seed'],
                'gene_name': gene,
                'n_genes_in_network': n_genes_in_network,
                'own_mRNA': own_mrna,
                'target_protein': target_protein
            }
            
            # Add other genes' mRNA as features (excluding own)
            if USE_ALL_MRNA_AS_FEATURES:
                for other_gene in net_genes:
                    if other_gene != gene:
                        sample[f"other_{other_gene}_mRNA"] = mrna_values.get(other_gene, 0.0)
            
            # Add GRN structural features
            if USE_GRN_FEATURES:
                grn_features = extract_grn_features_for_gene(
                    gene,
                    cache['edges_df'],
                    cache['G'],
                    cache['pagerank'],
                    cache['betweenness']
                )
                for feat_name, feat_val in grn_features.items():
                    sample[f"grn_{feat_name}"] = feat_val
            
            samples.append(sample)
    
    # -------------------------------------------------------------------------
    # STEP 4: Save dataset
    # -------------------------------------------------------------------------
    print(f"\n[4/4] Saving {len(samples)} per-gene samples...")
    
    dataset_df = pd.DataFrame(samples)
    
    # Reorder columns for clarity
    id_cols = ['sample_id', 'network_id', 'seed', 'gene_name', 'n_genes_in_network']
    target_col = ['target_protein']
    own_mrna_col = ['own_mRNA']
    
    other_mrna_cols = sorted([c for c in dataset_df.columns if c.startswith('other_')])
    grn_cols = sorted([c for c in dataset_df.columns if c.startswith('grn_')])
    
    ordered_cols = id_cols + own_mrna_col + other_mrna_cols + grn_cols + target_col
    
    # Only keep columns that exist
    ordered_cols = [c for c in ordered_cols if c in dataset_df.columns]
    dataset_df = dataset_df[ordered_cols]
    
    # Fill NaN in other_* columns with 0 (genes that don't exist in smaller networks)
    for col in other_mrna_cols:
        if col in dataset_df.columns:
            dataset_df[col] = dataset_df[col].fillna(0.0)
    
    output_path = os.path.join(OUTPUT_DIR, "pergene_dataset.csv")
    dataset_df.to_csv(output_path, index=False)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("DATASET BUILD COMPLETE!")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"  Samples: {len(dataset_df)}")
    print(f"  Features: {len(dataset_df.columns) - len(id_cols) - 1}")  # -1 for target
    print(f"  Time: {elapsed:.1f}s")
    
    # Print feature summary
    print(f"\nFeature breakdown:")
    print(f"  - own_mRNA: 1")
    print(f"  - other genes' mRNA: {len(other_mrna_cols)}")
    print(f"  - GRN structural: {len(grn_cols)}")
    
    # Print sample statistics
    print(f"\nTarget (protein) statistics:")
    print(f"  Mean: {dataset_df['target_protein'].mean():.4f}")
    print(f"  Std:  {dataset_df['target_protein'].std():.4f}")
    print(f"  Min:  {dataset_df['target_protein'].min():.4f}")
    print(f"  Max:  {dataset_df['target_protein'].max():.4f}")
    
    return dataset_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    build_pergene_dataset()
