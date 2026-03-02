import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transsys_simulator import generate_random_grn, TranssysSimulator
import numpy as np
import matplotlib.pyplot as plt

net = generate_random_grn(10, edge_prob=0.3, network_name="rand10") 
# the above calls the function generate_random_grn to create a random gene regulatory network:
# with 10 genes and a 30% chance of interaction between any two genes.
# this is given by edge_prob=0.3, which means that for each possible pair of genes, there is a 30% chance that one regulates the other.

# And then from the randomly generated network, we create a TranssysSimulator instance, 
# which will allow us to simulate the dynamics of gene expression based on the interactions defined in the network.

sim = TranssysSimulator(net)

#  Here the final_state variable will contain the steady-state mRNA and protein levels for all 10 genes in the network after simulating until the system reaches equilibrium.
# For now the simulation ends at a fixed time (t_end=200.0), but in future we can implement a more dynamic stopping criterion based on changes in expression levels to ensure we truly reach steady state.
# The random_seed=42 argument ensures that the initial conditions for the simulation are reproducible, 
# meaning that every time you run this code with the same seed, 
# you'll get the same final state, which is useful for debugging and comparing results across runs.

final_state = sim.simulate_to_steady_state(t_end=200.0, random_seed=42)

n = net.n_genes     # this is just a convenient way to get the number of genes in the network, which is 10 in this case, and helps us separate mRNA and protein levels from the final_state array.
names = net.gene_names      # this retrieves the list of gene names from the network, which we can use for labeling our plots and interpreting results.

#sim.set_initial_conditions(random_seed=0)   # this sets the initial mRNA and protein levels for all genes in the network to random values, but using a fixed seed (0) to ensure reproducibility.
t, y = sim.simulate(t_span=(0, 200), t_eval=np.linspace(0, 200, 300))   
# here in t_eval = np.linspace(0, 200, 300) we are specifying that:
# we want to evaluate the gene expression levels at 300 evenly spaced time points between 0 and 200, 
# which allows us to capture the dynamics of the system over time in a smooth manner for plotting and analysis.


# The mRNA levels for the 10 genes will be in the first 10 entries of final_state (final_state[:n]),
# and the protein levels will be in the next 10 entries (final_state[n:]). 
print("Final mRNA (first 10):", np.round(final_state[:n], 4)) 
print("Final protein (first 10):", np.round(final_state[n:], 4))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting the mRNA and protein levels for the 10 genes in the network.
# The code below creates two separate plots: one for mRNA levels and one for protein levels, with each gene represented by a different colored line.
# The x-axis represents time, and the y-axis represents the expression levels.

# Plot mRNA over time for first 4 genes
plt.figure(figsize=(10, 4))
for i, name in enumerate(names[:4]):    # plotting only the first 4 genes for better visibility, but you can change this to names to plot all 10 genes
    plt.plot(t, y[:, i], label=f"{name:4} mRNA")
plt.xlabel("Time")
plt.ylabel("mRNA level")
plt.title("mRNA dynamics for first 4 genes (random 10-gene network)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot protein over time first 4 genes
plt.figure(figsize=(10, 4))
for i, name in enumerate(names[:4]):    # plotting only the first 4 genes for better visibility, but you can change this to names to plot all 10 genes
    plt.plot(t, y[:, n + i], label=f"{name:4} protein", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Protein level")
plt.title("Protein dynamics for first 4 genes (random 10-gene network)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# The code below visualizes the structure of the randomly generated gene regulatory network using NetworkX and Matplotlib.
# It creates a directed graph where nodes represent genes and edges represent regulatory interactions (activations or repressions) between genes.
# The layout of the graph is determined by the spring_layout function, which positions nodes in a way that minimizes edge crossings and evenly distributes nodes in space for better visualization.
# The resulting plot will show the 10 genes as light blue nodes, with arrows indicating the direction of regulation (from regulator to target), and labels for each gene.

import networkx as nx

G = net.to_networkx()
plt.figure(figsize=(7, 7))
pos = nx.spring_layout(G, seed=0)
nx.draw(G, pos, with_labels=True, node_size=800, node_color="lightblue", arrows=True)
plt.title("Random 10-gene GRN structure")
plt.tight_layout()
plt.show()

