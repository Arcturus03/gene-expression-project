"""
transsys_simulator.py
=====================
A Gene Regulatory Network (GRN) simulator inspired by the Transsys framework.

WHAT IS THIS?
-------------
Imagine a city of factories. Each factory (gene) produces a specific product (protein).
Some factories can influence other factories — they might send signals that say 
"Hey, produce more!" (activation) or "Slow down!" (repression).

This simulator models how the production levels in all factories change over time,
given the network of influence between them. We use math (differential equations)
to calculate how concentrations rise and fall until the system settles into a 
stable state (steady state).

WHY DO WE NEED THIS?
--------------------
In real biology, we often don't know the "true" network of gene interactions.
By SIMULATING networks where we DO know the truth, we can:
1. Generate training data for machine learning models
2. Test whether ML models can learn to use network information
3. Have a controlled environment where we know the right answers

CLASSES IN THIS FILE:
---------------------
1. RegulatoryInteraction - A single arrow in the network (Gene A affects Gene B)
2. Gene - A single gene/factory with its production rules
3. GRNNetwork - The whole city of factories and their connections
4. TranssysSimulator - The "time machine" that runs the simulation forward

UTILITY FUNCTIONS:
------------------
- build_grn_from_dict(): Create a network from a simple config dictionary
- generate_random_grn(): Generate a random network for experiments

Author: Hrithik Chandra
Based on: Transsys framework (Kim, 2001)
"""

import numpy as np
from scipy.integrate import solve_ivp
import networkx as nx
from typing import Dict, List, Tuple, Optional


# =============================================================================
# CLASS 1: RegulatoryInteraction
# =============================================================================

class RegulatoryInteraction:
    """
    Represents a single regulatory connection between two genes.
    
    ANALOGY: Think of this as a phone line between two factories.
    - The CALLER (factor_name) is the factory sending the message
    - The RECEIVER (target_gene) is the factory getting the message
    - The MESSAGE TYPE (effect_type) is either:
        - "activator": "Please produce MORE!" (green light)
        - "repressor": "Please produce LESS!" (red light)
    - The VOLUME (strength) is how loud the message is (0.0 = whisper, 1.0 = shout)
    
    EXAMPLE:
    If Gene A activates Gene B with strength 0.8, it means:
    "When Gene A makes its protein, that protein tells Gene B to ramp up production,
    and the signal is pretty strong (0.8 out of 1.0)."
    """
    
    def __init__(
        self,
        factor_name: str,
        target_gene: str,
        effect_type: str,
        strength: float
    ):
        """
        Create a new regulatory interaction (a directed edge in the network).
        
        Parameters:
        -----------
        factor_name : str
            The gene whose protein does the regulating (the "boss" gene)
        target_gene : str
            The gene being regulated (the "employee" gene)
        effect_type : str
            Must be 'activator' (speeds up) or 'repressor' (slows down)
        strength : float
            How strong the effect is (0.0 to 1.0 typically, must be >= 0)
        
        Raises:
        -------
        ValueError: If effect_type is invalid or strength is negative
        """
        # Validate effect_type — only two options allowed!
        if effect_type not in ('activator', 'repressor'):
            raise ValueError(
                f"effect_type must be 'activator' or 'repressor', got '{effect_type}'"
            )
        
        # Validate strength — can't have negative influence strength
        if strength < 0:
            raise ValueError(
                f"strength must be non-negative, got {strength}"
            )
        
        self.factor_name = factor_name   # Who sends the signal
        self.target_gene = target_gene   # Who receives the signal
        self.effect_type = effect_type   # Activate or repress?
        self.strength = strength         # How strong?
    
    def get_signed_strength(self) -> float:
        """
        Returns the strength with a sign: positive for activators, negative for repressors.
        
        WHY SIGNED?
        -----------
        When we calculate the total signal arriving at a gene, we want to add up
        all the "speed up" signals and subtract all the "slow down" signals.
        
        This method makes that math easy:
        - Activator with strength 0.8 → returns +0.8
        - Repressor with strength 0.6 → returns -0.6
        
        Then we can just sum: total_signal = +0.8 + (-0.6) = +0.2 (net activation)
        
        Returns:
        --------
        float: Positive for activators, negative for repressors
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
    Represents a single gene with its production parameters and current state.
    
    ANALOGY: Each gene is like a FACTORY with two production lines:
    
    PRODUCTION LINE 1: Makes mRNA (the blueprint)
    - Think of mRNA as instruction sheets that get printed
    - The factory has a BASE PRINTING RATE (basal_expression) — it always prints
      some sheets even when no one tells it to
    - Other factories can call and say "print more!" or "print less!"
    - The sheets naturally decay/get destroyed over time (mrna_decay)
    
    PRODUCTION LINE 2: Makes Protein (the actual product)
    - Workers read the mRNA instruction sheets and build proteins
    - More mRNA sheets = more proteins being built (translation_rate)
    - Proteins also decay over time, but usually slower than mRNA (protein_decay)
    
    THE PROTEIN IS THE IMPORTANT PART:
    Proteins are what actually DO things in the cell. They're the transcription
    factors that go to OTHER genes and tell them to speed up or slow down.
    
    STATE VARIABLES (what changes during simulation):
    - mrna: Current amount of mRNA instruction sheets
    - protein: Current amount of protein product
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
        Create a new gene with its production parameters.
        
        Parameters:
        -----------
        name : str
            Unique identifier for this gene (e.g., "Gene1", "TP53", "MYC")
        
        basal_expression : float, default=0.1
            The baseline production rate when no regulators are active.
            Think of it as: "How much does this factory produce when no one 
            is telling it what to do?"
            - 0.0 = factory is OFF by default
            - 0.5 = factory runs at half speed by default
            - 1.0 = factory runs at full speed by default
        
        mrna_decay : float, default=0.1
            How fast mRNA degrades. Higher = mRNA disappears faster.
            - 0.05 = stable mRNA (lasts a while)
            - 0.2 = unstable mRNA (disappears quickly)
        
        protein_decay : float, default=0.05
            How fast protein degrades. Usually LOWER than mrna_decay because
            proteins are more stable than mRNA in real cells.
        
        translation_rate : float, default=1.0
            How efficiently mRNA is converted to protein.
            - 1.0 = one unit of mRNA produces one unit of protein per time step
            - 2.0 = very efficient translation
        """
        self.name = name
        self.basal_expression = basal_expression
        self.mrna_decay = mrna_decay
        self.protein_decay = protein_decay
        self.translation_rate = translation_rate
        
        # Current molecular concentrations (updated during simulation)
        self.mrna = 0.0      # How much mRNA is currently in the cell
        self.protein = 0.0   # How much protein is currently in the cell
        
        # List of regulatory interactions targeting THIS gene
        # (Other genes that control this one)
        self.regulators: List[RegulatoryInteraction] = []
    
    def reset_state(self, mrna_init: float = 0.0, protein_init: float = 0.0):
        """
        Reset mRNA and protein to specified initial values.
        
        Called before each simulation run to start fresh.
        
        Parameters:
        -----------
        mrna_init : float
            Starting mRNA concentration
        protein_init : float
            Starting protein concentration
        
        EXAMPLE:
        --------
        # Start with no mRNA but some protein already present
        gene.reset_state(mrna_init=0.0, protein_init=0.5)
        """
        self.mrna = mrna_init
        self.protein = protein_init
    
    def regulatory_input(self, factor_levels: Dict[str, float]) -> float:
        """
        Calculate the NET regulatory signal arriving at this gene.
        
        ANALOGY: Imagine your phone is ringing with calls from multiple factories:
        - Factory A says: "Speed up! (+0.4)"
        - Factory B says: "Speed up! (+0.3)"
        - Factory C says: "Slow down! (-0.5)"
        
        Total signal = +0.4 + 0.3 - 0.5 = +0.2 (net "speed up" signal)
        
        Parameters:
        -----------
        factor_levels : Dict[str, float]
            Current protein concentration of each gene in the network.
            Example: {"Gene1": 0.5, "Gene2": 0.8, "Gene3": 0.2}
        
        Returns:
        --------
        float: Net regulatory signal (positive = activation, negative = repression)
        
        HOW IT WORKS:
        -------------
        For each regulator:
        1. Look up how much protein the regulator has (how loud they're shouting)
        2. Multiply by signed strength (positive for activators, negative for repressors)
        3. Sum everything up
        """
        total_input = 0.0
        
        for interaction in self.regulators:
            # Get the protein level of the regulating gene
            # (If the regulator doesn't exist in factor_levels, assume 0)
            regulator_protein = factor_levels.get(interaction.factor_name, 0.0)
            
            # Multiply: [how much protein regulator has] × [signed effect strength]
            # If Gene1 is an activator with strength 0.8 and has protein level 0.5:
            # contribution = 0.5 × (+0.8) = +0.4
            contribution = regulator_protein * interaction.get_signed_strength()
            
            total_input += contribution
        
        return total_input
    
    def production_rate(self, factor_levels: Dict[str, float]) -> float:
        """
        Calculate how fast this gene is producing mRNA right now.
        
        Uses a SIGMOID function to convert signal to production rate.
        
        WHY SIGMOID?
        ------------
        In real biology, production doesn't increase forever with more signal.
        There's a maximum rate (cell has limited resources).
        
        The sigmoid function is shaped like an "S":
        - Very negative input → production ≈ 0 (factory nearly shut down)
        - Zero input → production ≈ 0.5 (factory at half speed)
        - Very positive input → production ≈ 1.0 (factory at max speed)
        
        FORMULA:
        --------
        raw_signal = basal_expression + regulatory_input
        production = 1 / (1 + e^(-raw_signal))
        
        Parameters:
        -----------
        factor_levels : Dict[str, float]
            Current protein concentrations in the network
        
        Returns:
        --------
        float: Production rate between 0 and 1
        """
        # Step 1: Calculate the raw signal
        # = baseline activity + sum of all regulatory inputs
        raw_signal = self.basal_expression + self.regulatory_input(factor_levels)
        
        # Step 2: Pass through sigmoid to get bounded production rate
        # sigmoid(x) = 1 / (1 + e^(-x))
        production = 1.0 / (1.0 + np.exp(-raw_signal))
        
        return production
    
    def mrna_derivative(self, factor_levels: Dict[str, float]) -> float:
        """
        Calculate how fast mRNA concentration is changing (d[mRNA]/dt).
        
        THE ODE FOR mRNA:
        -----------------
        d[mRNA]/dt = production_rate - decay_rate × [mRNA]
        
        In plain English:
        "mRNA increases because we're making new mRNA (production),
        and decreases because old mRNA is being destroyed (decay)."
        
        AT STEADY STATE:
        ----------------
        When d[mRNA]/dt = 0, the system is stable.
        This happens when: production_rate = decay_rate × [mRNA]
        So: steady_mRNA = production_rate / decay_rate
        
        Parameters:
        -----------
        factor_levels : Dict[str, float]
            Current protein concentrations
        
        Returns:
        --------
        float: Rate of change of mRNA (positive = increasing, negative = decreasing)
        """
        # How much new mRNA we're making
        production = self.production_rate(factor_levels)
        
        # How much mRNA is decaying (proportional to current amount)
        decay = self.mrna_decay * self.mrna
        
        # Net change = new stuff - destroyed stuff
        return production - decay
    
    def protein_derivative(self) -> float:
        """
        Calculate how fast protein concentration is changing (d[Protein]/dt).
        
        THE ODE FOR PROTEIN:
        --------------------
        d[Protein]/dt = translation_rate × [mRNA] - decay_rate × [Protein]
        
        In plain English:
        "Protein increases because we're translating mRNA into protein,
        and decreases because protein is being destroyed."
        
        Note: This depends ONLY on this gene's own mRNA — not on other genes directly.
        (Other genes affect protein levels INDIRECTLY by affecting mRNA production.)
        
        Returns:
        --------
        float: Rate of change of protein
        """
        # How much new protein we're making from mRNA
        production = self.translation_rate * self.mrna
        
        # How much protein is decaying
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
        """
        Create an empty network.
        
        Parameters:
        -----------
        name : str
            A human-readable label for this network (e.g., "3gene_feedback")
        """
        self.name = name
        self.genes: Dict[str, Gene] = {}  # Maps gene_name → Gene object
    
    @property
    def n_genes(self) -> int:
        """Number of genes in the network."""
        return len(self.genes)
    
    @property
    def gene_names(self) -> List[str]:
        """Ordered list of gene names (for consistent indexing)."""
        return list(self.genes.keys())
    
    def add_gene(self, gene: Gene):
        """
        Add a gene to the network.
        
        Parameters:
        -----------
        gene : Gene
            The Gene object to add
        
        Raises:
        -------
        ValueError: If a gene with this name already exists
        
        EXAMPLE:
        --------
        network = GRNNetwork("my_network")
        network.add_gene(Gene("G1", basal_expression=0.2))
        network.add_gene(Gene("G2", basal_expression=0.1))
        """
        if gene.name in self.genes:
            raise ValueError(f"Gene '{gene.name}' already exists in the network")
        self.genes[gene.name] = gene
    
    def add_interaction(self, interaction: RegulatoryInteraction):
        """
        Add a regulatory interaction (edge) to the network.
        
        This links the interaction to the TARGET gene's list of regulators.
        
        Parameters:
        -----------
        interaction : RegulatoryInteraction
            The interaction to add
        
        Raises:
        -------
        ValueError: If either the factor or target gene doesn't exist
        
        EXAMPLE:
        --------
        # G1's protein activates G2
        network.add_interaction(
            RegulatoryInteraction("G1", "G2", "activator", strength=0.8)
        )
        """
        # Check that both genes exist
        if interaction.factor_name not in self.genes:
            raise ValueError(
                f"Factor gene '{interaction.factor_name}' not found in network"
            )
        if interaction.target_gene not in self.genes:
            raise ValueError(
                f"Target gene '{interaction.target_gene}' not found in network"
            )
        
        # Add to the target gene's list of regulators
        self.genes[interaction.target_gene].regulators.append(interaction)
    
    def get_factor_levels(self) -> Dict[str, float]:
        """
        Get current protein concentrations for all genes.
        
        This snapshot is used when calculating regulatory inputs.
        We need to know "how much protein does each gene have RIGHT NOW?"
        to determine how strongly each gene is signaling to others.
        
        Returns:
        --------
        Dict[str, float]: {gene_name: protein_concentration}
        
        EXAMPLE OUTPUT:
        ---------------
        {"G1": 0.42, "G2": 0.18, "G3": 0.75}
        """
        return {name: gene.protein for name, gene in self.genes.items()}
    
    def state_vector(self) -> np.ndarray:
        """
        Pack all mRNA and protein values into a flat array.
        
        WHY FLAT ARRAY?
        ---------------
        The ODE solver (scipy's solve_ivp) needs the system state as a 1D array.
        This method converts our nice structured Gene objects into that format.
        
        FORMAT:
        -------
        [mRNA_G1, mRNA_G2, ..., mRNA_Gn, protein_G1, protein_G2, ..., protein_Gn]
        
        For a 3-gene network (G1, G2, G3):
        [mRNA_G1, mRNA_G2, mRNA_G3, protein_G1, protein_G2, protein_G3]
        
        So index 0-2 are mRNA, index 3-5 are protein.
        
        Returns:
        --------
        np.ndarray: Shape (2 * n_genes,)
        """
        n = self.n_genes
        state = np.zeros(2 * n)
        
        for i, name in enumerate(self.gene_names):
            state[i] = self.genes[name].mrna           # First half: mRNA
            state[n + i] = self.genes[name].protein    # Second half: protein
        
        return state
    
    def set_state_from_vector(self, y: np.ndarray):
        """
        Unpack a flat array back into gene mRNA and protein values.
        
        This is the INVERSE of state_vector().
        Called by the ODE solver to update our Gene objects with new values.
        
        Parameters:
        -----------
        y : np.ndarray
            State vector with shape (2 * n_genes,)
        """
        n = self.n_genes
        
        for i, name in enumerate(self.gene_names):
            self.genes[name].mrna = y[i]           # First half: mRNA
            self.genes[name].protein = y[n + i]    # Second half: protein
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Export the network as a NetworkX directed graph.
        
        WHY NetworkX?
        -------------
        NetworkX is a popular Python library for graph analysis.
        Once we have the network as a NetworkX graph, we can easily compute:
        - In-degree (how many genes regulate this one?)
        - Out-degree (how many genes does this one regulate?)
        - Centrality scores (how "important" is this gene in the network?)
        
        These become FEATURES for our machine learning models!
        
        Returns:
        --------
        nx.DiGraph: Directed graph with genes as nodes and interactions as edges
        
        EDGE ATTRIBUTES:
        ----------------
        - 'effect_type': 'activator' or 'repressor'
        - 'strength': The interaction strength
        - 'signed_strength': Positive for activators, negative for repressors
        """
        G = nx.DiGraph()
        
        # Add all genes as nodes
        G.add_nodes_from(self.gene_names)
        
        # Add all interactions as directed edges
        for gene_name, gene in self.genes.items():
            for interaction in gene.regulators:
                G.add_edge(
                    interaction.factor_name,  # FROM this gene
                    interaction.target_gene,   # TO this gene
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
    Runs the ODE simulation for a GRNNetwork.
    
    ANALOGY: This is the TIME MACHINE that runs our factory district forward in time.
    
    Give it a network (the map of factories and connections) and an initial state
    (how much mRNA and protein each factory starts with), and it will calculate
    how everything evolves over time until the system reaches a steady state.
    
    HOW IT WORKS:
    -------------
    1. Start with initial concentrations
    2. At each tiny time step, calculate:
       - How fast is each mRNA changing? (production - decay)
       - How fast is each protein changing? (translation - decay)
    3. Update all concentrations based on those rates
    4. Repeat until we've simulated enough time
    
    The math is handled by scipy's solve_ivp (a very accurate ODE solver).
    
    TYPICAL USAGE:
    --------------
    sim = TranssysSimulator(my_network)
    sim.set_initial_conditions(random_seed=42)  # Random starting point
    t, y = sim.simulate(t_span=(0, 100))        # Simulate 100 time units
    final_state = y[-1]                         # Get the final (steady) state
    """
    
    def __init__(self, network: GRNNetwork):
        """
        Create a simulator for a given network.
        
        Parameters:
        -----------
        network : GRNNetwork
            The network to simulate
        """
        self.network = network
    
    def set_initial_conditions(
        self,
        mrna_init: Optional[Dict[str, float]] = None,
        protein_init: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Set starting mRNA and protein levels before simulation.
        
        Parameters:
        -----------
        mrna_init : Dict[str, float], optional
            Initial mRNA for each gene. Example: {"G1": 0.5, "G2": 0.3}
            Genes not in the dict get random values.
        
        protein_init : Dict[str, float], optional
            Initial protein for each gene.
            Genes not in the dict get random values.
        
        random_seed : int, optional
            For reproducibility. Same seed = same random initial conditions.
        
        WHEN TO USE WHAT:
        -----------------
        - For a specific experiment: provide exact mrna_init and protein_init
        - For dataset generation: set random_seed for reproducible random starts
        - For completely random exploration: leave everything as None
        
        DEFAULT BEHAVIOR (all None):
        ----------------------------
        - mRNA: random uniform in [0, 0.3] — low initial expression
        - Protein: random uniform in [0, 0.1] — even lower (protein takes time to build)
        """
        # Set up random number generator
        rng = np.random.default_rng(random_seed)
        
        # Default to empty dicts if not provided
        if mrna_init is None:
            mrna_init = {}
        if protein_init is None:
            protein_init = {}
        
        # Set initial state for each gene
        for name, gene in self.network.genes.items():
            # Use provided value or generate random
            m_init = mrna_init.get(name, rng.uniform(0, 0.3))
            p_init = protein_init.get(name, rng.uniform(0, 0.1))
            gene.reset_state(m_init, p_init)
    
    def _derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calculate all derivatives (dy/dt) at the current state.
        
        THIS IS THE HEART OF THE SIMULATION.
        
        Called automatically by scipy's solve_ivp at each time step.
        It calculates: "Given where we are now, how fast is everything changing?"
        
        Parameters:
        -----------
        t : float
            Current time (not used directly since our system is autonomous,
            meaning the equations don't explicitly depend on time)
        
        y : np.ndarray
            Current state vector [mRNA_G1, ..., mRNA_Gn, protein_G1, ..., protein_Gn]
        
        Returns:
        --------
        np.ndarray: Derivative vector [d(mRNA_G1)/dt, ..., d(protein_Gn)/dt]
        
        THE ALGORITHM:
        --------------
        1. Sync our Gene objects with the solver's current state
        2. Get current protein levels (these affect gene regulation)
        3. For each gene, compute d(mRNA)/dt and d(protein)/dt
        4. Pack derivatives into an array and return
        """
        n = self.network.n_genes
        
        # Step 1: Update Gene objects with current values from solver
        self.network.set_state_from_vector(y)
        
        # Step 2: Get snapshot of all protein concentrations
        factor_levels = self.network.get_factor_levels()
        
        # Step 3: Calculate all derivatives
        derivatives = np.zeros(2 * n)
        
        for i, name in enumerate(self.network.gene_names):
            gene = self.network.genes[name]
            
            # d(mRNA)/dt = production - decay
            derivatives[i] = gene.mrna_derivative(factor_levels)
            
            # d(protein)/dt = translation - decay
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
        
        Parameters:
        -----------
        t_span : tuple, default=(0, 100)
            (start_time, end_time) — how long to simulate
        
        t_eval : np.ndarray, optional
            Specific time points to record. If None, solver picks automatically.
            Example: np.linspace(0, 100, 500) for 500 evenly spaced points.
        
        method : str, default='RK45'
            ODE solver algorithm. RK45 (Runge-Kutta 4/5) is accurate and robust.
        
        Returns:
        --------
        t : np.ndarray, shape (T,)
            Time points where solution was recorded
        
        y : np.ndarray, shape (T, 2*n_genes)
            State at each time point.
            y[i, :] is the state at time t[i]
            y[i, 0:n_genes] = mRNA values
            y[i, n_genes:] = protein values
        
        Raises:
        -------
        RuntimeError: If the solver fails to converge
        
        EXAMPLE:
        --------
        sim = TranssysSimulator(network)
        sim.set_initial_conditions(random_seed=42)
        
        # Simulate and get 200 time points
        t, y = sim.simulate(t_span=(0, 100), t_eval=np.linspace(0, 100, 200))
        
        # Plot mRNA of first gene over time
        import matplotlib.pyplot as plt
        plt.plot(t, y[:, 0])  # y[:, 0] is mRNA of first gene
        """
        # Get initial state vector
        y0 = self.network.state_vector()
        
        # Run the ODE solver
        solution = solve_ivp(
            fun=self._derivatives,    # Our derivative function
            t_span=t_span,            # Time interval
            y0=y0,                    # Initial state
            method=method,            # Solver algorithm
            t_eval=t_eval,            # When to record
            dense_output=False
        )
        
        # Check if solver succeeded
        if not solution.success:
            raise RuntimeError(f"ODE solver failed: {solution.message}")
        
        # solution.t has shape (T,) — time points
        # solution.y has shape (2*n_genes, T) — we transpose to (T, 2*n_genes)
        return solution.t, solution.y.T
    
    def simulate_to_steady_state(
        self,
        t_end: float = 200.0,
        n_eval_points: int = 500,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Convenience method: simulate from random initial conditions to steady state.
        
        This is the most common use case for dataset generation:
        1. Start from a random initial state
        2. Simulate long enough for the system to stabilize
        3. Return only the FINAL state (the steady-state expression profile)
        
        Parameters:
        -----------
        t_end : float, default=200.0
            How long to simulate. Should be long enough to reach steady state.
            If unsure, use a longer time — it just takes more computation.
        
        n_eval_points : int, default=500
            Number of points to record (doesn't affect final answer, just resolution)
        
        random_seed : int, optional
            For reproducible random initial conditions
        
        Returns:
        --------
        np.ndarray: Shape (2 * n_genes,)
            Final state: [mRNA_G1, ..., mRNA_Gn, protein_G1, ..., protein_Gn]
        
        EXAMPLE — Generate 100 training samples:
        ----------------------------------------
        samples = []
        for seed in range(100):
            final_state = sim.simulate_to_steady_state(t_end=200, random_seed=seed)
            samples.append(final_state)
        """
        # Set up random initial conditions
        self.set_initial_conditions(random_seed=random_seed)
        
        # Run simulation
        t_eval = np.linspace(0, t_end, n_eval_points)
        t, y = self.simulate(t_span=(0, t_end), t_eval=t_eval)
        
        # Return only the final state
        # Also sync the network's Gene objects with this final state
        final_state = y[-1]
        self.network.set_state_from_vector(final_state)
        
        return final_state


# =============================================================================
# UTILITY FUNCTION 1: build_grn_from_dict
# =============================================================================

def build_grn_from_dict(config: dict, network_name: str = "GRN") -> GRNNetwork:
    """
    Build a GRNNetwork from a configuration dictionary.
    
    This is the PRIMARY WAY to create networks — just define your network
    as a simple Python dictionary (or load from JSON) and this function
    does the rest.
    
    Parameters:
    -----------
    config : dict
        Configuration with "genes" and "interactions" keys.
        See format below.
    
    network_name : str
        A label for the network
    
    Returns:
    --------
    GRNNetwork: The constructed network, ready for simulation
    
    EXPECTED CONFIG FORMAT:
    -----------------------
    {
        "genes": {
            "G1": {"basal": 0.2, "mrna_decay": 0.1, "protein_decay": 0.05, "translation_rate": 1.0},
            "G2": {"basal": 0.1, "mrna_decay": 0.15, "protein_decay": 0.07},
            "G3": {}  # Uses all defaults
        },
        "interactions": [
            {"factor": "G1", "target": "G2", "type": "activator", "strength": 0.8},
            {"factor": "G2", "target": "G3", "type": "activator", "strength": 0.7},
            {"factor": "G3", "target": "G1", "type": "repressor", "strength": 0.6}
        ]
    }
    
    Note: All gene parameters are OPTIONAL — missing keys use Gene defaults.
    
    SIMPLE EXAMPLE:
    ---------------
    config = {
        "genes": {"A": {}, "B": {}},
        "interactions": [{"factor": "A", "target": "B", "type": "activator", "strength": 0.5}]
    }
    network = build_grn_from_dict(config, "simple_AB")
    """
    network = GRNNetwork(name=network_name)
    
    # Step 1: Create all genes
    for gene_name, params in config.get("genes", {}).items():
        gene = Gene(
            name=gene_name,
            basal_expression=params.get("basal", 0.1),
            mrna_decay=params.get("mrna_decay", 0.1),
            protein_decay=params.get("protein_decay", 0.05),
            translation_rate=params.get("translation_rate", 1.0)
        )
        network.add_gene(gene)
    
    # Step 2: Create all interactions
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
    Generate a random Gene Regulatory Network.
    
    This is super useful for creating diverse training data:
    generate many random networks with different structures,
    simulate each one, and you have a varied dataset!
    
    Parameters:
    -----------
    n_genes : int
        Number of genes in the network. Genes are named G0, G1, G2, ...
    
    edge_prob : float, default=0.3
        Probability that any directed edge exists.
        - 0.1 = very sparse network (few connections)
        - 0.3 = moderate connectivity  
        - 0.5 = dense network (many connections)
    
    activation_ratio : float, default=0.6
        Fraction of edges that are activators (rest are repressors).
        - 0.5 = equal activators and repressors
        - 0.8 = mostly activators (cooperative network)
        - 0.2 = mostly repressors (competitive network)
    
    basal_range : tuple, default=(0.05, 0.3)
        Range for random basal expression values
    
    strength_range : tuple, default=(0.3, 1.0)
        Range for random interaction strengths
    
    mrna_decay_range : tuple, default=(0.05, 0.2)
        Range for mRNA decay rates
    
    protein_decay_range : tuple, default=(0.02, 0.1)
        Range for protein decay rates
    
    network_name : str, default="random_GRN"
        Label for the network
    
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
        Example: rng = np.random.default_rng(42)
    
    Returns:
    --------
    GRNNetwork: A randomly generated network
    
    EXAMPLE — Generate 10 different random networks:
    -------------------------------------------------
    networks = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        net = generate_random_grn(n_genes=20, edge_prob=0.25, rng=rng)
        networks.append(net)
    
    EXAMPLE — Sparse 10-gene network:
    ----------------------------------
    rng = np.random.default_rng(42)
    network = generate_random_grn(10, edge_prob=0.2, rng=rng)
    print(network)  # Shows number of genes and interactions
    """
    # Use provided RNG or create a new one
    if rng is None:
        rng = np.random.default_rng()
    
    network = GRNNetwork(name=network_name)
    
    # Step 1: Create genes with random parameters
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
    
    # Step 2: Randomly create edges
    # For each pair of genes (i, j) where i != j, flip a coin
    for i, factor in enumerate(gene_names):
        for j, target in enumerate(gene_names):
            if i == j:
                continue  # No self-loops
            
            # Does this edge exist?
            if rng.random() < edge_prob:
                # Decide if activator or repressor
                if rng.random() < activation_ratio:
                    effect_type = 'activator'
                else:
                    effect_type = 'repressor'
                
                # Random strength
                strength = rng.uniform(*strength_range)
                
                # Add the interaction
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
