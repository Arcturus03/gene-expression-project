"""
generate_synthetic_dataset.py — Generate synthetic steady-state expression data
from random GRNs using the Transsys-style simulator.

WHAT THIS DOES:
1. Generates N random GRNs with varying size, edge density, activation ratio
2. For each GRN, simulates K different initial conditions to steady state
3. Saves everything to data/synthetic_transsys/:
    - expression_profiles.csv  (all steady-state expression vectors)
    - network_metadata.csv     (which network each profile came from + network stats)
    - grn_edges/               (edge lists for each network, for later feature extraction)

USAGE:
    python experiments/generate_synthetic_dataset.py
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd

# Add project root so we can import the simulator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulators.transsys_simulator import generate_random_grn, TranssysSimulator


# =============================================================================
# CONFIGURATION — Tweak these to change dataset size and diversity
# =============================================================================

N_NETWORKS = 100              # Number of random GRNs to generate
SEEDS_PER_NETWORK = 50       # Initial conditions per GRN
T_END = 200.0                # Simulation duration (long enough to reach steady state)
N_EVAL_POINTS = 500          # Time resolution for ODE solver

# Network parameter ranges (sampled uniformly for each network)
GENE_COUNT_RANGE = (5, 30)          # Number of genes per network
EDGE_PROB_RANGE = (0.15, 0.6)         # Edge density (sparse to dense)
ACTIVATION_RATIO_RANGE = (0.2, 0.8)  # Fraction of edges that are activators

# Here the activation ratio is the fraction of edges that are activators vs repressors.
# For example, if edge_prob=0.3 and activation_ratio=0.5,
# then for each possible edge, we have:
# - 30% chance of an edge existing
# - If it exists, 50% chance it's an activator, 50% chance it's a repressor
# An edge is basically the regulatory effect of one gene's protein on another gene's transcription, which can be either activating or repressing.
# So 0.3 is the chance that one gene's protein has some regulatory effect on another gene's transcription, 
# and then the activation ratio determines how many of those effects are activating vs repressing.
# And 0.8 is the chance that an edge is an activator, so in that case most edges would be activating and fewer would be repressing.
# So how do we decide whether edges will be repressors or activators? We can use the activation ratio to control this. 
# For example, if we set the activation ratio to 0.8, 
# then 80% of the edges will be activators and 20% will be repressors. 
# If we set it to 0.5, then half of the edges will be activators and half will be repressors. 
# This allows us to create networks with different regulatory architectures, 
# which can lead to different expression patterns and dynamics.


# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_transsys")


def generate_dataset():
    #Generate the full synthetic dataset.
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    edges_dir = os.path.join(OUTPUT_DIR, "grn_edges")
    os.makedirs(edges_dir, exist_ok=True)
    
    # Master RNG for reproducibility of the whole dataset
    master_rng = np.random.default_rng(seed=42)
    
    all_profiles = []       # Each row: one steady-state expression vector
    all_metadata = []       # Each row: network_id, seed, n_genes, n_edges, etc.
    
    print(f"Generating {N_NETWORKS} networks x {SEEDS_PER_NETWORK} seeds "
          f"= {N_NETWORKS * SEEDS_PER_NETWORK} expression profiles\n")
    
    start_time = time.time()
    
    for net_idx in range(N_NETWORKS):
        # ----- Sample random network parameters -----
        n_genes = master_rng.integers(*GENE_COUNT_RANGE, endpoint=True)
        edge_prob = master_rng.uniform(*EDGE_PROB_RANGE)
        activation_ratio = master_rng.uniform(*ACTIVATION_RATIO_RANGE)
        
        # Create a per-network RNG (derived from master, so fully reproducible)
        net_seed = master_rng.integers(0, 2**31)
        net_rng = np.random.default_rng(net_seed)
        
        # ----- Generate the random GRN -----
        network = generate_random_grn(
            n_genes=int(n_genes),
            edge_prob=edge_prob,
            activation_ratio=activation_ratio,
            rng=net_rng,
            network_name=f"net_{net_idx:03d}"
        )
        
        # Count edges
        n_edges = sum(len(g.regulators) for g in network.genes.values())
        gene_names = network.gene_names
        
        # ----- Save edge list for this network -----
        edges = []
        for gene_name, gene in network.genes.items():
            for inter in gene.regulators:
                edges.append({
                    "factor": inter.factor_name,
                    "target": inter.target_gene,
                    "type": inter.effect_type,
                    "strength": round(inter.strength, 4),
                    "signed_strength": round(inter.get_signed_strength(), 4)
                })
        
        edges_df = pd.DataFrame(edges)
        edges_df.to_csv(os.path.join(edges_dir, f"net_{net_idx:03d}_edges.csv"), index=False)
        
        # Also save gene parameters
        gene_params = []
        for gname in gene_names:
            g = network.genes[gname]
            gene_params.append({
                "gene": gname,
                "basal_expression": round(g.basal_expression, 4),
                "mrna_decay": round(g.mrna_decay, 4),
                "protein_decay": round(g.protein_decay, 4),
                "translation_rate": round(g.translation_rate, 4)
            })
        params_df = pd.DataFrame(gene_params)
        params_df.to_csv(os.path.join(edges_dir, f"net_{net_idx:03d}_genes.csv"), index=False)
        
        # ----- Simulate with multiple initial conditions -----
        sim = TranssysSimulator(network)
        
        for seed_idx in range(SEEDS_PER_NETWORK):
            sim_seed = int(net_seed + seed_idx + 1)
            
            try:
                final_state = sim.simulate_to_steady_state(
                    t_end=T_END,
                    n_eval_points=N_EVAL_POINTS,
                    random_seed=sim_seed
                )
                
                # final_state = [mRNA_G0, ..., mRNA_Gn, protein_G0, ..., protein_Gn]
                n = len(gene_names)
                mrna_values = final_state[:n]
                protein_values = final_state[n:]
                
                # Build one row of expression data
                profile = {}
                for i, gname in enumerate(gene_names):
                    profile[f"{gname}_mRNA"] = round(mrna_values[i], 6)
                    profile[f"{gname}_protein"] = round(protein_values[i], 6)
                
                profile["network_id"] = f"net_{net_idx:03d}"
                profile["sim_seed"] = sim_seed
                all_profiles.append(profile)
                
                # Metadata row
                all_metadata.append({
                    "network_id": f"net_{net_idx:03d}",
                    "sim_seed": sim_seed,
                    "n_genes": int(n_genes),
                    "n_edges": n_edges,
                    "edge_prob": round(edge_prob, 3),
                    "activation_ratio": round(activation_ratio, 3),
                    "net_seed": net_seed
                })
                
            except RuntimeError as e:
                print(f"  WARNING: Simulation failed for net_{net_idx:03d} seed {sim_seed}: {e}")
                continue
        
        # Progress update every 10 networks
        if (net_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  [{net_idx + 1}/{N_NETWORKS}] networks done ({elapsed:.1f}s elapsed)")
    
    # ----- Save everything -----
    print(f"\nSaving {len(all_profiles)} expression profiles...")
    
    # Expression profiles — wide format (each column is a gene's mRNA or protein)
    profiles_df = pd.DataFrame(all_profiles)
    profiles_df.to_csv(os.path.join(OUTPUT_DIR, "expression_profiles.csv"), index=False)
    
    # Network metadata
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(os.path.join(OUTPUT_DIR, "network_metadata.csv"), index=False)
    
    # Save generation config for reproducibility
    config = {
        "n_networks": N_NETWORKS,
        "seeds_per_network": SEEDS_PER_NETWORK,
        "t_end": T_END,
        "n_eval_points": N_EVAL_POINTS,
        "gene_count_range": list(GENE_COUNT_RANGE),
        "edge_prob_range": list(EDGE_PROB_RANGE),
        "activation_ratio_range": list(ACTIVATION_RATIO_RANGE),
        "master_seed": 42,
        "total_profiles": len(all_profiles),
        "total_networks": N_NETWORKS
    }
    with open(os.path.join(OUTPUT_DIR, "generation_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\nDone! Generated {len(all_profiles)} profiles from {N_NETWORKS} networks in {elapsed:.1f}s")
    print(f"Output saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"\nFiles created:")
    print(f"  expression_profiles.csv  — {len(all_profiles)} rows x {len(profiles_df.columns)} columns")
    print(f"  network_metadata.csv     — network stats for each profile")
    print(f"  grn_edges/               — {N_NETWORKS} edge list CSVs + {N_NETWORKS} gene param CSVs")
    print(f"  generation_config.json   — reproducibility config")


if __name__ == "__main__":
    generate_dataset()
