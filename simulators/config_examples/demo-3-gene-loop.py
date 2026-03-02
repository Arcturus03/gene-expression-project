import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transsys_simulator import build_grn_from_dict, TranssysSimulator
import numpy as np
import matplotlib.pyplot as plt

# Basal value is: how hard the gene tries to turn on by default, not how much mRNA it starts with.
# We can set an initial mRNA level for each gene, but the basal value is more like a built-in tendency to produce mRNA, which can be influenced by regulators.
# For now we'll set the initial mRNA levels randomly, and the basal values given by us in this demo -
# will determine how much they produce over time in response to the feedback loop interactions.



config = {
    "genes": {
        "G1": {"basal": 0.3, "mrna_decay": 0.18, "protein_decay": 0.05},
        "G2": {"basal": 0.1, "mrna_decay": 0.12, "protein_decay": 0.06},
        "G3": {"basal": 0.05, "mrna_decay": 0.15, "protein_decay": 0.04},
    },
    "interactions": [
        {"factor": "G1", "target": "G2", "type": "activator", "strength": 0.9},
        {"factor": "G2", "target": "G3", "type": "activator", "strength": 0.8},
        {"factor": "G3", "target": "G1", "type": "repressor", "strength": 0.7},
    ],
}

network = build_grn_from_dict(config, network_name="3gene_feedback")
sim = TranssysSimulator(network)

sim.set_initial_conditions(random_seed=0)
t, y = sim.simulate(t_span=(0, 500), t_eval=np.linspace(0, 500, 300)) # simulate from time 0 to 200, with 300 time points for smooth curves

n = network.n_genes
print("Final mRNA:", np.round(y[-1, :n], 4))
print("Final protein:", np.round(y[-1, n:], 4))


# Plotting 
n = network.n_genes
names = network.gene_names

# Plot mRNA over time
plt.figure(figsize=(10, 4))
for i, name in enumerate(names):
    plt.plot(t, y[:, i], label=f"{name} mRNA")
plt.xlabel("Time")
plt.ylabel("mRNA level")
plt.title("mRNA dynamics in 3-gene feedback loop")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot protein over time
plt.figure(figsize=(10, 4))
for i, name in enumerate(names):
    plt.plot(t, y[:, n + i], label=f"{name} protein", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Protein level")
plt.title("Protein dynamics in 3-gene feedback loop")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# When you look at these plots, you’re seeing:
# each curve = how one gene’s mRNA or protein changes over time as it responds to regulators,
# the right‑hand end of each curve = the corresponding entry in final_state.
# the initial mrna levels are random, 
# but the system eventually settles into a stable pattern of expression (steady state) due to the feedback loop.
# The feedback loop creates a dynamic interplay where G1 activates G2, G2 activates G3, and G3 represses G1, leading to complex temporal patterns before reaching equilibrium.
# the interactions in the code after genes initialization is basically the feedback loop,
# which is saying - "Hey G1, produce more G2!" (activation) or "Hey G3, slow down G1!" (repression).
# and the values of strengths, (decided by us) determine how strongly these interactions influence the production rates, which in turn shapes the dynamics of the system.