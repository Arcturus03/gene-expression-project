"""
transsys_simulator.py — GRN simulator inspired by the Transsys framework.

ANALOGY: A city of factories (genes) that produce products (proteins).
Factories send signals to each other: "produce more!" (activation) or
"slow down!" (repression). We use differential equations to simulate how
concentrations evolve until the system reaches steady state.

WHY: By simulating networks where we know the ground truth, we can generate
training data for ML models and test if they learn network structure.

CLASSES: RegulatoryInteraction, Gene, GRNNetwork, TranssysSimulator
UTILITIES: build_grn_from_dict(), generate_random_grn()

Author: Hrithik Chandra | Based on: Transsys framework (Kim, 2001)
"""

import numpy as np
from scipy.integrate import solve_ivp
import networkx as nx   # Used for graph analysis and ML feature extraction
from typing import Dict, List, Tuple, Optional


# =============================================================================
# CLASS 1: RegulatoryInteraction
# =============================================================================

class RegulatoryInteraction:
    """
    A single regulatory connection between two genes (a directed edge).
    
    ANALOGY: A phone line between two factories.
    - factor_name: the caller (regulator gene)
    - target_gene: the receiver (regulated gene)
    - effect_type: 'activator' (green light) or 'repressor' (red light)
    - strength: how loud the signal is (0.0–1.0)
    """
    
    def __init__(
        self,
        factor_name: str,
        target_gene: str,
        effect_type: str,
        strength: float
    ):
        """
        Args:
            factor_name: gene whose protein regulates (the "boss")
            target_gene: gene being regulated (the "employee")
            effect_type: 'activator' or 'repressor'
            strength: effect magnitude (>= 0, typically 0.0–1.0)
        """
        # Validate inputs
        if effect_type not in ('activator', 'repressor'):
            raise ValueError(
                f"effect_type must be 'activator' or 'repressor', got '{effect_type}'"
            )
        
        if strength < 0:
            raise ValueError(
                f"strength must be non-negative, got {strength}"
            )
        
        self.factor_name = factor_name
        self.target_gene = target_gene
        self.effect_type = effect_type
        self.strength = strength
    
    def get_signed_strength(self) -> float:
        """
        Returns +strength for activators, -strength for repressors.
        This lets us sum all signals easily: total = sum of signed strengths.
        """
        if self.effect_type == 'activator':
            return self.strength
        else:  # repressor
            return -self.strength
    
    def __repr__(self):
        """String representation for debugging."""
        arrow = "→+" if self.effect_type == 'activator' else "→-"
        return f"{self.factor_name} {arrow} {self.target_gene} (s={self.strength:.2f})"


# =============================================================================
# CLASS 2: Gene
# =============================================================================

class Gene:
    """
    A single gene with its production parameters and current state.
    
    ANALOGY: A factory with two production lines:
    1. mRNA line — prints instruction sheets (rate set by basal_expression + regulators)
    2. Protein line — workers read mRNA sheets and build products (translation_rate)
    
    Both mRNA and protein decay over time. Proteins are the important output:
    they act as transcription factors that regulate OTHER genes.
    
    State variables: mrna, protein (updated during simulation)
    """
    
    def __init__(
        self,
        name: str,
        basal_expression: float = 0.1,
        mrna_decay: float = 0.1,
        protein_decay: float = 0.05,
        translation_rate: float = 1.0
    ):
        """
        Args:
            name: unique gene identifier (e.g., "G1", "TP53")
            basal_expression: baseline production rate (0=off, 0.5=half, 1.0=full)
            mrna_decay: mRNA degradation rate (higher = less stable)
            protein_decay: protein degradation rate (usually < mrna_decay)
            translation_rate: mRNA → protein conversion efficiency
        """
        self.name = name
        self.basal_expression = basal_expression
        self.mrna_decay = mrna_decay
        self.protein_decay = protein_decay
        self.translation_rate = translation_rate
        
        # Current concentrations (updated during simulation)
        self.mrna = 0.0
        self.protein = 0.0
        
        # Regulators: interactions from other genes targeting this one
        self.regulators: List[RegulatoryInteraction] = []
    
    def reset_state(self, mrna_init: float = 0.0, protein_init: float = 0.0):
        """Reset mRNA and protein to given initial values (called before each sim run)."""
        self.mrna = mrna_init
        self.protein = protein_init
    
    def regulatory_input(self, factor_levels: Dict[str, float]) -> float:
        """
        Calculate net regulatory signal arriving at this gene.
        
        For each regulator: signal += regulator_protein * signed_strength
        Positive total = net activation, negative = net repression.
        
        Args:
            factor_levels: {gene_name: protein_level} for all genes
        Returns:
            float: net regulatory signal
        """
        total_input = 0.0
        
        for interaction in self.regulators:
            # Get the protein level of the regulating gene
            # (If the regulator doesn't exist in factor_levels, assume 0)
            regulator_protein = factor_levels.get(interaction.factor_name, 0.0)
            
            contribution = regulator_protein * interaction.get_signed_strength()
            
            total_input += contribution
        
        return total_input
    
    def production_rate(self, factor_levels: Dict[str, float]) -> float:
        """
        Calculate mRNA production rate using a sigmoid activation function.
        
        Sigmoid maps the combined signal to [0, 1] — like a dial that maxes out.
        Formula: production = 1 / (1 + e^(-(basal + regulatory_input)))
        
        Args:
            factor_levels: {gene_name: protein_level} for all genes
        Returns:
            float: Production rate in [0, 1]
        """
        raw_signal = self.basal_expression + self.regulatory_input(factor_levels)
        production = 1.0 / (1.0 + np.exp(-raw_signal))
        return production
    
    def mrna_derivative(self, factor_levels: Dict[str, float]) -> float:
        """
        Rate of change of mRNA: d[mRNA]/dt = production - decay * [mRNA].
        
        At steady state (d[mRNA]/dt=0): steady_mRNA = production / decay_rate.
        
        Args:
            factor_levels: current protein concentrations
        Returns:
            float: d[mRNA]/dt
        """
        production = self.production_rate(factor_levels)
        decay = self.mrna_decay * self.mrna
        return production - decay
    
    def protein_derivative(self) -> float:
        """
        Rate of change of protein: d[Protein]/dt = translation * [mRNA] - decay * [Protein].
        
        Depends only on this gene's own mRNA; other genes affect protein
        indirectly through mRNA production.
        
        Returns:
            float: d[Protein]/dt
        """
        production = self.translation_rate * self.mrna
        decay = self.protein_decay * self.protein
        return production - decay
    
    def __repr__(self):
        """String representation for debugging."""
        return (f"Gene({self.name}, basal={self.basal_expression:.2f}, "
                f"mRNA={self.mrna:.3f}, protein={self.protein:.3f})")


# =============================================================================
# CLASS 3: GRNNetwork
# =============================================================================

class GRNNetwork:
    """
    The complete Gene Regulatory Network — holds all genes and their connections.
    
    ANALOGY: This is like a MAP of an industrial district showing:
    - All the factories (genes)
    - All the communication lines between them (regulatory interactions)
    
    The network doesn't "do" anything by itself — it's just the structure.
    The TranssysSimulator uses this map to run the simulation.
    
    EXAMPLE NETWORK:
    ----------------
       G1 ──activates──> G2 ──activates──> G3
        ^                                   │
        │                                   │
        └────────represses─────────────────┘
    
    This creates a feedback loop:
    - G1 turns on G2
    - G2 turns on G3  
    - G3 turns off G1 (completing the cycle)
    """
    
    def __init__(self, name: str = "GRN"):
        """Create an empty network with the given name."""
        self.name = name
        self.genes: Dict[str, Gene] = {}  # gene_name → Gene object
    
    @property
    def n_genes(self) -> int:
        """Number of genes in the network."""
        return len(self.genes)
    
    @property
    def gene_names(self) -> List[str]:
        """Ordered list of gene names (for consistent indexing)."""
        return list(self.genes.keys())
    
    def add_gene(self, gene: Gene):
        """Add a gene to the network. Raises ValueError if name already exists."""
        if gene.name in self.genes:
            raise ValueError(f"Gene '{gene.name}' already exists in the network")
        self.genes[gene.name] = gene
    
    def add_interaction(self, interaction: RegulatoryInteraction):
        """
        Add a regulatory interaction (edge) to the network.
        Links the interaction to the target gene's regulators list.
        Raises ValueError if factor or target gene doesn't exist.
        """
        if interaction.factor_name not in self.genes:
            raise ValueError(
                f"Factor gene '{interaction.factor_name}' not found in network"
            )
        if interaction.target_gene not in self.genes:
            raise ValueError(
                f"Target gene '{interaction.target_gene}' not found in network"
            )
        
        # Add to target gene's regulators
        self.genes[interaction.target_gene].regulators.append(interaction)
    
    def get_factor_levels(self) -> Dict[str, float]:
        """Return {gene_name: protein_concentration} snapshot for all genes."""
        return {name: gene.protein for name, gene in self.genes.items()}
    
    def state_vector(self) -> np.ndarray:
        """
        Pack all mRNA and protein values into a flat 1D array for the ODE solver.
        
        Format: [mRNA_G1, ..., mRNA_Gn, protein_G1, ..., protein_Gn]
        Returns: np.ndarray of shape (2 * n_genes,)
        """
        n = self.n_genes
        state = np.zeros(2 * n)
        
        for i, name in enumerate(self.gene_names):
            state[i] = self.genes[name].mrna
            state[n + i] = self.genes[name].protein
        
        return state
    
    def set_state_from_vector(self, y: np.ndarray):
        """Inverse of state_vector(): unpack flat array back into gene mRNA/protein values."""
        n = self.n_genes
        
        for i, name in enumerate(self.gene_names):
            self.genes[name].mrna = y[i]
            self.genes[name].protein = y[n + i]
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Export as a NetworkX directed graph for analysis (degree, centrality, etc.).
        
        Edge attributes: 'effect_type', 'strength', 'signed_strength'.
        These graph metrics become features for ML models.
        """
        G = nx.DiGraph()
        
        # Add genes as nodes
        G.add_nodes_from(self.gene_names)
        
        # Add interactions as directed edges
        for gene_name, gene in self.genes.items():
            for interaction in gene.regulators:
                G.add_edge(
                    interaction.factor_name,
                    interaction.target_gene,
                    effect_type=interaction.effect_type,
                    strength=interaction.strength,
                    signed_strength=interaction.get_signed_strength()
                )
        
        return G
    
    def __repr__(self):
        """String representation."""
        n_interactions = sum(len(g.regulators) for g in self.genes.values())
        return f"GRNNetwork({self.name}, {self.n_genes} genes, {n_interactions} interactions)"


# =============================================================================
# CLASS 4: TranssysSimulator
# =============================================================================

class TranssysSimulator:
    """
    Runs ODE simulation for a GRNNetwork — the "time machine" for the factory district.
    
    Give it a network + initial state, and it calculates how mRNA/protein
    concentrations evolve over time using scipy's solve_ivp (RK45).
    
    Usage:
        sim = TranssysSimulator(my_network)
        sim.set_initial_conditions(random_seed=42)
        t, y = sim.simulate(t_span=(0, 100))
        final_state = y[-1]
    """
    
    def __init__(self, network: GRNNetwork):
        """Create a simulator for the given network."""
        self.network = network
    
    def set_initial_conditions(
        self,
        mrna_init: Optional[Dict[str, float]] = None,
        protein_init: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Set starting mRNA and protein levels before simulation.
        
        Args:
            mrna_init: {gene: value} for specific genes (others get random)
            protein_init: {gene: value} for specific genes (others get random)
            random_seed: for reproducibility
        
        Defaults: mRNA ~ U[0, 0.3], protein ~ U[0, 0.1]
        """
        # Set up random number generator
        rng = np.random.default_rng(random_seed)
        if mrna_init is None:
            mrna_init = {}
        if protein_init is None:
            protein_init = {}
        
        # Set initial state for each gene
        for name, gene in self.network.genes.items():
            m_init = mrna_init.get(name, rng.uniform(0, 0.3))
            p_init = protein_init.get(name, rng.uniform(0, 0.1))
            gene.reset_state(m_init, p_init)
    
    def _derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calculate all derivatives (dy/dt) at current state. Called by solve_ivp.
        
        Steps: sync gene objects with solver state → get protein snapshot →
        compute d(mRNA)/dt and d(protein)/dt for each gene.
        
        Args:
            t: current time (unused; system is autonomous)
            y: state vector [mRNA_G1..Gn, protein_G1..Gn]
        Returns:
            np.ndarray: derivative vector of same shape
        """
        n = self.network.n_genes
        
        # Update Gene objects with solver's current values
        self.network.set_state_from_vector(y)
        factor_levels = self.network.get_factor_levels()
        
        derivatives = np.zeros(2 * n)
        
        for i, name in enumerate(self.network.gene_names):
            gene = self.network.genes[name]
            
            derivatives[i] = gene.mrna_derivative(factor_levels)
            derivatives[n + i] = gene.protein_derivative()
        
        return derivatives
    
    def simulate(
        self,
        t_span: Tuple[float, float] = (0, 100),
        t_eval: Optional[np.ndarray] = None,
        method: str = 'RK45'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the ODE simulation and return time-series data.
        
        Args:
            t_span: (start, end) time interval
            t_eval: specific time points to record (None = solver picks)
            method: ODE solver algorithm (default RK45)
        Returns:
            t: np.ndarray shape (T,) — time points
            y: np.ndarray shape (T, 2*n_genes) — state at each time.
               y[:, :n_genes] = mRNA, y[:, n_genes:] = protein
        Raises:
            RuntimeError: if solver fails
        """
        y0 = self.network.state_vector()
        
        solution = solve_ivp(
            fun=self._derivatives,
            t_span=t_span,
            y0=y0,
            method=method,
            t_eval=t_eval,
            dense_output=False
        )
        
        if not solution.success:
            raise RuntimeError(f"ODE solver failed: {solution.message}")
        
        # solution.y is (2*n_genes, T) — transpose to (T, 2*n_genes)
        return solution.t, solution.y.T
    
    def simulate_to_steady_state(
        self,
        t_end: float = 200.0,
        n_eval_points: int = 500,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate from random initial conditions to steady state; return final state.
        
        Most common use case for dataset generation: run many seeds, collect
        the final expression profiles as training samples.
        
        Args:
            t_end: simulation duration (should be long enough to stabilize)
            n_eval_points: time resolution (doesn't affect final state)
            random_seed: for reproducible initial conditions
        Returns:
            np.ndarray shape (2*n_genes,): [mRNA_G1..Gn, protein_G1..Gn]
        """
        self.set_initial_conditions(random_seed=random_seed)
        
        t_eval = np.linspace(0, t_end, n_eval_points)
        t, y = self.simulate(t_span=(0, t_end), t_eval=t_eval)
        
        # Return final state and sync network objects
        final_state = y[-1]
        self.network.set_state_from_vector(final_state)
        
        return final_state


# =============================================================================
# UTILITY FUNCTION 1: build_grn_from_dict
# =============================================================================

def build_grn_from_dict(config: dict, network_name: str = "GRN") -> GRNNetwork:
    """
    Build a GRNNetwork from a config dictionary — the primary way to create networks.
    
    Expected format:
        {
            "genes": {
                "G1": {"basal": 0.2, "mrna_decay": 0.1, "protein_decay": 0.05, "translation_rate": 1.0},
                "G2": {}  # all defaults
            },
            "interactions": [
                {"factor": "G1", "target": "G2", "type": "activator", "strength": 0.8}
            ]
        }
    All gene parameters are optional (missing keys use Gene defaults).
    """
    network = GRNNetwork(name=network_name)
    
    # Create genes
    for gene_name, params in config.get("genes", {}).items():
        gene = Gene(
            name=gene_name,
            basal_expression=params.get("basal", 0.1),
            mrna_decay=params.get("mrna_decay", 0.1),
            protein_decay=params.get("protein_decay", 0.05),
            translation_rate=params.get("translation_rate", 1.0)
        )
        network.add_gene(gene)
    
    # Create interactions
    for inter in config.get("interactions", []):
        interaction = RegulatoryInteraction(
            factor_name=inter["factor"],
            target_gene=inter["target"],
            effect_type=inter["type"],
            strength=inter["strength"]
        )
        network.add_interaction(interaction)
    
    return network


# =============================================================================
# UTILITY FUNCTION 2: generate_random_grn
# =============================================================================

def generate_random_grn(
    n_genes: int,
    edge_prob: float = 0.3,
    activation_ratio: float = 0.6,
    basal_range: Tuple[float, float] = (0.05, 0.3),
    strength_range: Tuple[float, float] = (0.3, 1.0),
    mrna_decay_range: Tuple[float, float] = (0.05, 0.2),
    protein_decay_range: Tuple[float, float] = (0.02, 0.1),
    network_name: str = "random_GRN",
    rng: Optional[np.random.Generator] = None
) -> GRNNetwork:
    """
    Generate a random Gene Regulatory Network — useful for creating diverse training data.
    
    Args:
        n_genes: number of genes (named G0, G1, G2, ...)
        edge_prob: probability of each directed edge (0.1=sparse, 0.3=moderate, 0.5=dense)
        activation_ratio: fraction of edges that are activators (rest are repressors)
        basal_range: range for random basal expression
        strength_range: range for random interaction strengths
        mrna_decay_range: range for mRNA decay rates
        protein_decay_range: range for protein decay rates
        network_name: label for the network
        rng: np.random.Generator for reproducibility
    Returns:
        GRNNetwork: a randomly generated network
    """
    # Use provided RNG or create a new one
    if rng is None:
        rng = np.random.default_rng()
    
    network = GRNNetwork(name=network_name)
    
    # Create genes with random parameters
    gene_names = [f"G{i}" for i in range(n_genes)]
    
    for name in gene_names:
        gene = Gene(
            name=name,
            basal_expression=rng.uniform(*basal_range),
            mrna_decay=rng.uniform(*mrna_decay_range),
            protein_decay=rng.uniform(*protein_decay_range),
            translation_rate=1.0  # Keep translation rate fixed for simplicity
        )
        network.add_gene(gene)
    
    # Randomly create edges between gene pairs (no self-loops)
    for i, factor in enumerate(gene_names):
        for j, target in enumerate(gene_names):
            if i == j:
                continue
            
            if rng.random() < edge_prob:
                # Decide activator vs repressor
                if rng.random() < activation_ratio:
                    effect_type = 'activator'
                else:
                    effect_type = 'repressor'
                
                strength = rng.uniform(*strength_range)
                
                interaction = RegulatoryInteraction(
                    factor_name=factor,
                    target_gene=target,
                    effect_type=effect_type,
                    strength=strength
                )
                network.add_interaction(interaction)
    
    return network


# =============================================================================
# DEMONSTRATION (runs if you execute this file directly)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TRANSSYS SIMULATOR DEMO")
    print("=" * 60)
    
    # Example 1: Build a simple 3-gene feedback network from config
    print("\n--- Example 1: 3-gene feedback loop ---")
    
    config = {
        "genes": {
            "G1": {"basal": 0.3, "mrna_decay": 0.10, "protein_decay": 0.05},
            "G2": {"basal": 0.1, "mrna_decay": 0.12, "protein_decay": 0.06},
            "G3": {"basal": 0.05, "mrna_decay": 0.08, "protein_decay": 0.04},
        },
        "interactions": [
            {"factor": "G1", "target": "G2", "type": "activator", "strength": 0.9},
            {"factor": "G2", "target": "G3", "type": "activator", "strength": 0.8},
            {"factor": "G3", "target": "G1", "type": "repressor", "strength": 0.7},
        ],
    }
    
    network = build_grn_from_dict(config, "3gene_feedback")
    print(f"Created: {network}")
    
    # Simulate
    sim = TranssysSimulator(network)
    sim.set_initial_conditions(mrna_init={"G1": 0.5}, protein_init={"G1": 0.2}, random_seed=42)
    t, y = sim.simulate(t_span=(0, 100), t_eval=np.linspace(0, 100, 200))
    
    print(f"Simulation ran from t=0 to t=100")
    print(f"Final mRNA levels: G1={y[-1,0]:.3f}, G2={y[-1,1]:.3f}, G3={y[-1,2]:.3f}")
    print(f"Final protein levels: G1={y[-1,3]:.3f}, G2={y[-1,4]:.3f}, G3={y[-1,5]:.3f}")
    
    # Example 2: Generate a random network
    print("\n--- Example 2: Random 10-gene network ---")
    
    rng = np.random.default_rng(123)
    random_net = generate_random_grn(n_genes=10, edge_prob=0.25, rng=rng)
    print(f"Created: {random_net}")
    
    # Get NetworkX graph for feature extraction
    G = random_net.to_networkx()
    print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Simulate to steady state
    sim2 = TranssysSimulator(random_net)
    final_state = sim2.simulate_to_steady_state(t_end=200, random_seed=42)
    
    print(f"Steady-state mRNA: {final_state[:10].round(3)}")
    print(f"Steady-state protein: {final_state[10:].round(3)}")
    
    print("\n" + "=" * 60)
    print("Demo complete! The simulator is ready for use.")
    print("=" * 60)
