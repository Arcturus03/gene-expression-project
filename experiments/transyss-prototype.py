import numpy as np
import matplotlib.pyplot as plt

# This is a simple simulation of a gene expression system using the "transsys" logic.
# But this is quite hardcoded and not very flexible. 
# In a real implementation, we want to make this more modular and data-driven.


class CellSimulator:
    def __init__(self):
        # Our "Kitchen Storage" - stores the current amount of each protein
        self.factors = {
            "Protein_A": 0.2,  # Starting with a little bit of Protein A (the Chef's trigger)
            "Protein_B": 0.0
        }
        
        # Settings for our proteins (The Physics)
        # Decay: How fast the food rots/disappears if you don't make more (0.1 = 10% lost per step)
        self.decay_rates = {
            "Protein_A": 0.15, 
            "Protein_B": 0.15   
        }

    def michaelis_menten(self, amount, vmax, km):
        """
        This is the 'Biological Logic' formula. 
        It says: as you have more 'Ingredient X', the speed of cooking increases, 
        but only up to a maximum speed (Vmax).
        """
        if amount == 0: return 0
        return (vmax * amount) / (km + amount)

    def run_simulation(self, steps):
        history = {"Protein_A": [], "Protein_B": []}
        
        # Starting conditions: Someone left a little bit of Protein A in the fridge
        self.factors["Protein_A"] = 0.5
        
        for t in range(steps):
            # 1. Record the current state
            for name in self.factors:
                history[name].append(self.factors[name])
            
            # 2. Calculate "Production" (The Gene Logic)
            # LOGIC: 
            # Gene A is 'Constitutive' (always on a little bit, like a pilot light).
            # Gene B is 'Activated' by Protein A (Protein A is the Chef telling B to start).
            
            prod_A = 0.05  # Constant slow drip of A
            
            # B depends on how much A is currently in the cell
            prod_B = self.michaelis_menten(self.factors["Protein_A"], vmax=0.2, km=0.1)
            
            # 3. Calculate "Decay" (The Rotting)
            # amount_lost = current_amount * decay_rate
            decay_A = self.factors["Protein_A"] * self.decay_rates["Protein_A"]
            decay_B = self.factors["Protein_B"] * self.decay_rates["Protein_B"]
            
            # 4. Update the storage (New = Old + Created - Rotted)
            self.factors["Protein_A"] += (prod_A - decay_A)
            self.factors["Protein_B"] += (prod_B - decay_B)
            
        return history

# --- Let's run it and see the "Gene Expression" ---

sim = CellSimulator()
data = sim.run_simulation(steps=200)

# Visualizing the results
plt.figure(figsize=(10, 5))
plt.plot(data["Protein_A"], label="Protein A (The Trigger)", color="blue")
plt.plot(data["Protein_B"], label="Protein B (The Result)", color="green")
plt.title("Simulated Gene Expression (transsys logic)")
plt.xlabel("Time Steps")
plt.ylabel("Concentration in Cell")
plt.legend()
plt.grid(True)
plt.show()


# The above code is hard-coded to show a simple gene expression system where 
# Protein A is produced at a constant rate and activates the production of Protein B. 
# Both proteins decay over time. The Michaelis-Menten function models the activation of Protein B by Protein A, 
# showing how the production of B increases with more A but eventually saturates.
