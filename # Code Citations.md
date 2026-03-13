# Code Citations

## License: GPL-3.0
https://github.com/cKNUSPeR/i2nca/blob/a13e08a4db2ae0463402a6896862fdc296267e24/i2nca/qctools/visualization.py

```


Great questions. Let me answer each one, then build the solution.

---

## Answers to Your Questions

### Is sklearn MLP the same as D-GEX?
**Yes, conceptually.** D-GEX is just an MLP that predicts gene expression from other genes' expression. The sklearn `MLPRegressor` with `hidden_layer_sizes=(128, 64)` and ReLU activation IS a D-GEX-style model. The only difference is that the original D-GEX paper used a custom implementation with more control over training. So:

- **sklearn MLP** = D-GEX baseline (quick version)
- **PyTorch MLP** = D-GEX baseline (proper version with full control)

Both are D-GEX. Both should give similar results when properly tuned.

### Should we use both sklearn and PyTorch?
**Yes.** Keep both. In your report you can write: *"We implemented the D-GEX architecture in both scikit-learn and PyTorch and confirmed comparable performance, validating our implementation."* This shows rigour.

### File Organisation
Your instinct is correct. Here's my recommended structure:

```
experiments/
├── build_pergene_dataset.py          # Data preparation (already done)
├── train_pergene_baselines.py        # All per-gene models (Ridge, Lasso, ElasticNet, RF, MLP)
├── train_dgex_pytorch.py             # PyTorch D-GEX (Friday's task)
├── generate_synthetic_dataset.py     # Data generation (already done)
├── generate_fixed_size_dataset.py    # Fixed-size data (Friday's task)
└── plotting/
    └── plot_pergene_results.py       # ALL plots in one reusable file
```

The plotting file will be **standalone** — any training script can call it by passing results DataFrames. This avoids duplicating plotting code across model files.

---

Now let me build both files. First the updated training script, then the plotting module.

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\train_pergene_baselines.py
"""
train_pergene_baselines.py — Train and evaluate all per-gene baseline models.

PURPOSE:
    Answer the core research question: "Does adding GRN structural features
    improve gene expression prediction compared to expression-only models?"

MODELS (in order of complexity):
    1. Ridge Regression       — Linear baseline with L2 penalty (shrinks weights)
    2. Lasso Regression       — Linear baseline with L1 penalty (kills useless features)
    3. Elastic Net             — Combines L1 + L2 (best linear baseline for correlated features)
    4. Random Forest           — Non-linear, tree-based, gives feature importance
    5. MLP (sklearn)           — D-GEX-style neural network (quick version)

EACH MODEL IS TESTED TWICE:
    A. Expression-only:  input = [own_mRNA + other genes' mRNA]
    B. Expression + GRN: input = [own_mRNA + other genes' mRNA + 9 GRN features]

    If B beats A consistently → GRN features carry predictive power. QED.

OUTPUT:
    results/tables/pergene_baseline_results.csv   — All scores
    results/figures/*.png                          — All plots (via plot_pergene_results.py)

USAGE:
    python experiments/train_pergene_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plotting"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings for cleaner output

# Import the standalone plotting module
from plot_pergene_results import PergeneResultsPlotter


# =============================================================================
# CONFIGURATION
# =============================================================================
# All paths and settings in one place for easy modification.

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_transsys")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create output directories
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Columns that are NOT features (metadata + target)
META_COLS = ["sample_id", "network_id", "gene_name"]
TARGET_COL = "target_protein"

# GRN feature columns — these are the 9 structural features extracted from network topology.
# They are the "treatment" in our controlled experiment.
GRN_FEATURE_PREFIXES = ["grn_"]

# Test split — 20% held out, never seen during training
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pergene_data():
    """
    Load the per-gene dataset and split into expression-only and full feature sets.
    
    Returns two feature sets:
      - X_expr: only mRNA columns (the "control group")
      - X_full: mRNA + GRN columns (the "treatment group")
      - y: target protein levels
      - meta: sample_id, network_id, gene_name for later analysis
      - feature_names: dict with 'expr' and 'full' feature lists
    """
    path = os.path.join(DATA_DIR, "pergene_dataset.csv")
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"  Dataset shape: {df.shape[0]} samples × {df.shape[1]} columns")
    
    # Separate metadata, features, and target
    meta = df[META_COLS]
    y = df[TARGET_COL]
    
    # All feature columns (everything except metadata and target)
    all_features = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    
    # Split into expression-only and GRN features
    grn_cols = [c for c in all_features if any(c.startswith(p) for p in GRN_FEATURE_PREFIXES)]
    expr_cols = [c for c in all_features if c not in grn_cols]
    
    print(f"  Expression features: {len(expr_cols)}")
    print(f"  GRN features: {len(grn_cols)} → {grn_cols}")
    
    X_expr = df[expr_cols]
    X_full = df[expr_cols + grn_cols]
    
    feature_names = {"expr": expr_cols, "full": expr_cols + grn_cols, "grn": grn_cols}
    
    return X_expr, X_full, y, meta, feature_names


# =============================================================================
# MODEL DEFINITIONS
# Each function creates, trains, and evaluates one model.
# All follow the same interface: (X_train, X_test, y_train, y_test, name) → dict
# =============================================================================

def train_ridge(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Ridge Regression — Linear model with L2 penalty.
    
    WHY RIDGE:
        The simplest possible baseline. If a neural network can't beat this,
        something is wrong with the neural network. Ridge is the "floor".
    
    WHAT alpha DOES:
        alpha=1.0 controls how much we penalise large weights.
        Higher alpha = more conservative model (smaller weights, less overfitting).
        Lower alpha = closer to standard linear regression.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (try 0.1, 1.0, 10.0)
    """
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_lasso(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Lasso Regression — Linear model with L1 penalty.
    
    WHY LASSO:
        Unlike Ridge (which shrinks all weights), Lasso can drive weights to
        EXACTLY ZERO. This means it automatically selects which features matter
        and which are useless. Perfect for our data where 30+ columns might be
        padding zeros from variable-size networks.
    
    WHAT LASSO REVEALS:
        After training, check model.coef_ — any coefficient that is exactly 0.0
        means Lasso decided that feature is worthless for prediction. If ALL GRN
        features survive (non-zero), that's strong evidence they carry real signal.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (higher = more features killed)
          0.001 is gentle, 0.1 is aggressive. We use 0.001 to start soft.
    """
    model = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Count how many features Lasso kept vs killed
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_elasticnet(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Elastic Net — Combines Ridge (L2) and Lasso (L1) penalties.
    
    WHY ELASTIC NET:
        Our GRN features are correlated (e.g., grn_in_degree and grn_n_activators
        overlap — a gene with high in-degree likely has many activators).
        
        - Lasso alone would randomly pick ONE from a group of correlated features
          and kill the rest. This loses information.
        - Ridge alone keeps everything but can't eliminate truly useless features.
        - Elastic Net keeps groups of correlated features together (Ridge behaviour)
          while still killing genuinely useless ones (Lasso behaviour).
    
    WHAT l1_ratio CONTROLS:
        - l1_ratio=0.0 → pure Ridge
        - l1_ratio=1.0 → pure Lasso
        - l1_ratio=0.5 → equal mix (our default)
    
    CUSTOMISABLE PARAMETERS:
        - alpha: overall regularisation strength
        - l1_ratio: balance between L1 and L2 (0.0 to 1.0)
    """
    model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_random_forest(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Random Forest — Ensemble of 100 decision trees.
    
    WHY RANDOM FOREST:
        First non-linear baseline. Can learn rules like "IF mRNA > 0.5 AND
        in_degree > 3 THEN protein is high". Linear models can't learn this.
        Also provides feature_importances_ for free.
    
    CUSTOMISABLE PARAMETERS:
        - n_estimators: number of trees (100 is standard, 200+ for more stability)
        - max_depth: how deep each tree can go (None = unlimited)
        - min_samples_leaf: minimum samples in each leaf (higher = simpler tree)
    """
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_STATE, 
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_mlp(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    MLP (Multi-Layer Perceptron) — D-GEX-style neural network.
    
    WHY MLP:
        This IS the D-GEX baseline. The original D-GEX paper (Chen et al., 2016)
        used an MLP to predict gene expression. We replicate their approach here
        with sklearn for speed, and later with PyTorch for full control.
    
    WHAT WAS CHANGED FROM DEFAULTS (and why):
        - hidden_layer_sizes: (100,) → (128, 64)
          DEFAULT had only 1 hidden layer with 100 neurons.
          CHANGED to 2 layers (128 then 64) to match D-GEX architecture.
          The funnel shape (128→64→1) forces the network to compress information.
        
        - max_iter: 200 → 1000
          DEFAULT cut training short after 200 epochs.
          CHANGED to give the model enough time to converge. Without this,
          the model barely started learning, causing negative R².
        
        - early_stopping: False → True
          DEFAULT trained for all max_iter epochs even if overfitting.
          CHANGED to monitor validation error and stop when it plateaus.
          This is like pulling the cake out of the oven before it burns.
        
        - validation_fraction: 0.1 → 0.15
          15% of training data is held out to monitor overfitting.
        
        - n_iter_no_change: 10 → 20
          Wait 20 epochs without improvement before stopping.
          Being patient helps avoid stopping too early on noisy data.
        
        - learning_rate: 'constant' → 'adaptive'
          DEFAULT kept the same learning rate forever.
          CHANGED to automatically reduce it when loss plateaus.
          Like slowing down as you approach a parking spot — big steps first,
          fine adjustments at the end.
        
        - batch_size: 200 → 64
          DEFAULT processed 200 samples per gradient update.
          CHANGED to 64 for noisier but more frequent updates.
          Helps the model explore more of the loss landscape.
    
    CUSTOMISABLE PARAMETERS TO TUNE:
        - hidden_layer_sizes: try (256, 128), (128, 64, 32), (64, 32)
        - learning_rate_init: try 0.0001, 0.001, 0.01
        - batch_size: try 32, 64, 128
        - alpha: L2 penalty strength (default 0.0001)
    """
    # MLP requires scaled features — without this, large-valued columns dominate
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=64,
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (stopped at epoch {model.n_iter_})")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("PER-GENE BASELINE EXPERIMENTS")
    print("=" * 70)
    
    # 1. Load data
    X_expr, X_full, y, meta, feature_names = load_pergene_data()
    
    # 2. Train-test split (same split for all models — fair comparison)
    X_expr_train, X_expr_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X_expr, y, meta, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_full_train, X_full_test = train_test_split(
        X_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )[:2]
    
    # 3. Define all model trainers
    # Each entry: (train_function, display_name)
    model_trainers = [
        (train_ridge,         "Ridge"),
        (train_lasso,         "Lasso"),
        (train_elasticnet,    "ElasticNet"),
        (train_random_forest, "RandomForest"),
        (train_mlp,           "MLP_DGEX"),
    ]
    
    # 4. Run all experiments: expression-only (A) vs expression+GRN (B)
    all_results = []
    all_predictions = {}  # Store for plotting
    rf_full_result = None  # Need this for feature importance
    lasso_full_result = None  # Need this for feature selection analysis
    
    for train_fn, base_name in model_trainers:
        # Experiment A: Expression-only
        print(f"\n--- {base_name} (Expression Only) ---")
        result_a = train_fn(X_expr_train.values, X_expr_test.values, 
                           y_train.values, y_test.values,
                           f"{base_name}_ExprOnly")
        all_results.append(result_a)
        all_predictions[f"{base_name}_ExprOnly"] = result_a["predictions"]
        
        # Experiment B: Expression + GRN
        print(f"--- {base_name} (Expression + GRN) ---")
        result_b = train_fn(X_full_train.values, X_full_test.values,
                           y_train.values, y_test.values,
                           f"{base_name}_ExprGRN")
        all_results.append(result_b)
        all_predictions[f"{base_name}_ExprGRN"] = result_b["predictions"]
        
        # Track special results for analysis
        if base_name == "RandomForest":
            rf_full_result = result_b
        if base_name == "Lasso":
            lasso_full_result = result_b
    
    # 5. Build results DataFrame
    results_df = pd.DataFrame([{
        "model": r["model"], "mse": r["mse"], "r2": r["r2"]
    } for r in all_results])
    
    # Save results table
    results_path = os.path.join(TABLES_DIR, "pergene_baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # 6. Print summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY: Expression-Only vs Expression+GRN")
    print("=" * 70)
    print(f"{'Model':<22} {'MSE (Expr)':<14} {'MSE (GRN)':<14} {'Δ MSE':<12} {'R² (Expr)':<12} {'R² (GRN)':<12}")
    print("-" * 86)
    
    for _, base_name in model_trainers:
        expr_row = results_df[results_df["model"] == f"{base_name}_ExprOnly"].iloc[0]
        grn_row = results_df[results_df["model"] == f"{base_name}_ExprGRN"].iloc[0]
        delta_mse = ((grn_row["mse"] - expr_row["mse"]) / expr_row["mse"]) * 100
        print(f"{base_name:<22} {expr_row['mse']:<14.6f} {grn_row['mse']:<14.6f} {delta_mse:<12.1f}% {expr_row['r2']:<12.4f} {grn_row['r2']:<12.4f}")
    
    # 7. Lasso feature selection analysis
    if lasso_full_result:
        print("\n" + "=" * 70)
        print("LASSO FEATURE SELECTION ANALYSIS")
        print("Which features did Lasso keep vs kill?")
        print("=" * 70)
        lasso_model = lasso_full_result["instance"]
        full_feature_names = feature_names["full"]
        
        for fname, coef in sorted(zip(full_feature_names, lasso_model.coef_), 
                                    key=lambda x: abs(x[1]), reverse=True):
            status = "✓ KEPT" if abs(coef) > 1e-10 else "✗ KILLED"
            if fname.startswith("grn_"):
                print(f"  [GRN] {fname:<30} coef={coef:>10.6f}  {status}")
        
        # Count GRN features that survived
        grn_feature_names = feature_names["grn"]
        grn_coefs = [lasso_model.coef_[full_feature_names.index(g)] for g in grn_feature_names]
        grn_alive = sum(1 for c in grn_coefs if abs(c) > 1e-10)
        print(f"\n  GRN features surviving Lasso: {grn_alive}/{len(grn_feature_names)}")
    
    # 8. Generate all plots using the standalone plotting module
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plotter = PergeneResultsPlotter(FIGURES_DIR)
    
    # Plot 1: Model comparison bar chart (MSE and R² side by side)
    plotter.plot_model_comparison(results_df)
    
    # Plot 2: Predicted vs Actual scatter for each model
    plotter.plot_predicted_vs_actual(all_predictions, y_test.values)
    
    # Plot 3: Residual distributions
    plotter.plot_residual_distributions(all_predictions, y_test.values)
    
    # Plot 4: Feature importance from Random Forest
    if rf_full_result:
        plotter.plot_feature_importance(
            rf_full_result["instance"], 
            feature_names["full"],
            top_n=20
        )
    
    # Plot 5: GRN feature correlation matrix
    plotter.plot_grn_correlation_matrix(
        pd.concat([X_full_train, X_full_test]),
        feature_names["grn"],
        y
    )
    
    # Plot 6: Per-network performance heatmap
    plotter.plot_per_network_performance(
        all_predictions, y_test.values, meta_test
    )
    
    print(f"\n✓ All plots saved to {FIGURES_DIR}")
    print("=" * 70)
    print("DONE. Next step: train_dgex_pytorch.py for the full D-GEX comparison.")
    print("=" * 70)


if __name__ == "__main__":
    main()
````

Now the standalone plotting module:

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\plotting\plot_pergene_results.py
"""
plot_pergene_results.py — Standalone plotting module for per-gene baseline results.

PURPOSE:
    Generates all analysis plots for the per-gene baseline experiments.
    This file is SEPARATE from the training code so that:
    1. Any training script can import and use it
    2. Plots can be regenerated without retraining models
    3. New plot types can be added without touching model code

PLOTS GENERATED:
    1. Model Comparison Bar Chart  — MSE and R² for all models, side by side
    2. Predicted vs Actual Scatter — How close predictions are to reality
    3. Residual Distributions      — Are errors random or systematically biased?
    4. Feature Importance          — Which features does Random Forest rely on?
    5. GRN Correlation Matrix      — How correlated are GRN features with each other?
    6. Per-Network Heatmap         — Which networks are easy/hard to predict?

USAGE:
    from plot_pergene_results import PergeneResultsPlotter
    plotter = PergeneResultsPlotter("path/to/figures")
    plotter.plot_model_comparison(results_df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves files without opening windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class PergeneResultsPlotter:
    """
    All plotting logic for per-gene baseline experiments.
    
    Each method takes raw data (predictions, scores, etc.) and saves a .png file.
    Methods are independent — you can call any one without the others.
    """
    
    def __init__(self, figures_dir: str):
        """
        Args:
            figures_dir: directory where all .png files will be saved
        """
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        
        # Consistent styling across all plots
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })
    
    def _save(self, fig, filename: str):
        """Helper: save figure and close to free memory."""
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {filename}")
    
    # =========================================================================
    # PLOT 1: MODEL COMPARISON BAR CHART
    # =========================================================================
    
    def plot_model_comparison(self, results_df: pd.DataFrame):
        """
        Side-by-side bar chart comparing MSE and R² for all models.
        
        WHY THIS PLOT:
            The single most important plot in the experiment. At a glance, you can
            see whether adding GRN features (darker bars) consistently improves
            over expression-only (lighter bars) across all model types.
        
        WHAT TO LOOK FOR:
            - Dark bars lower than light bars (MSE chart) = GRN helps
            - Dark bars higher than light bars (R² chart) = GRN helps
            - Consistent pattern across ALL models = strong evidence
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract model base names (without _ExprOnly/_ExprGRN suffix)
        base_names = []
        for name in results_df["model"]:
            base = name.replace("_ExprOnly", "").replace("_ExprGRN", "")
            if base not in base_names:
                base_names.append(base)
        
        x = np.arange(len(base_names))
        width = 0.35
        
        for ax, metric, title in zip(axes, ["mse", "r2"], ["MSE (lower is better)", "R² (higher is better)"]):
            expr_vals = [results_df[results_df["model"] == f"{b}_ExprOnly"][metric].values[0] for b in base_names]
            grn_vals = [results_df[results_df["model"] == f"{b}_ExprGRN"][metric].values[0] for b in base_names]
            
            bars1 = ax.bar(x - width/2, expr_vals, width, label="Expression Only", color="#78909C", alpha=0.85)
            bars2 = ax.bar(x + width/2, grn_vals, width, label="Expression + GRN", color="#26A69A", alpha=0.85)
            
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(base_names, rotation=15)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
```


## License: GPL-3.0
https://github.com/cKNUSPeR/i2nca/blob/a13e08a4db2ae0463402a6896862fdc296267e24/i2nca/qctools/visualization.py

```


Great questions. Let me answer each one, then build the solution.

---

## Answers to Your Questions

### Is sklearn MLP the same as D-GEX?
**Yes, conceptually.** D-GEX is just an MLP that predicts gene expression from other genes' expression. The sklearn `MLPRegressor` with `hidden_layer_sizes=(128, 64)` and ReLU activation IS a D-GEX-style model. The only difference is that the original D-GEX paper used a custom implementation with more control over training. So:

- **sklearn MLP** = D-GEX baseline (quick version)
- **PyTorch MLP** = D-GEX baseline (proper version with full control)

Both are D-GEX. Both should give similar results when properly tuned.

### Should we use both sklearn and PyTorch?
**Yes.** Keep both. In your report you can write: *"We implemented the D-GEX architecture in both scikit-learn and PyTorch and confirmed comparable performance, validating our implementation."* This shows rigour.

### File Organisation
Your instinct is correct. Here's my recommended structure:

```
experiments/
├── build_pergene_dataset.py          # Data preparation (already done)
├── train_pergene_baselines.py        # All per-gene models (Ridge, Lasso, ElasticNet, RF, MLP)
├── train_dgex_pytorch.py             # PyTorch D-GEX (Friday's task)
├── generate_synthetic_dataset.py     # Data generation (already done)
├── generate_fixed_size_dataset.py    # Fixed-size data (Friday's task)
└── plotting/
    └── plot_pergene_results.py       # ALL plots in one reusable file
```

The plotting file will be **standalone** — any training script can call it by passing results DataFrames. This avoids duplicating plotting code across model files.

---

Now let me build both files. First the updated training script, then the plotting module.

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\train_pergene_baselines.py
"""
train_pergene_baselines.py — Train and evaluate all per-gene baseline models.

PURPOSE:
    Answer the core research question: "Does adding GRN structural features
    improve gene expression prediction compared to expression-only models?"

MODELS (in order of complexity):
    1. Ridge Regression       — Linear baseline with L2 penalty (shrinks weights)
    2. Lasso Regression       — Linear baseline with L1 penalty (kills useless features)
    3. Elastic Net             — Combines L1 + L2 (best linear baseline for correlated features)
    4. Random Forest           — Non-linear, tree-based, gives feature importance
    5. MLP (sklearn)           — D-GEX-style neural network (quick version)

EACH MODEL IS TESTED TWICE:
    A. Expression-only:  input = [own_mRNA + other genes' mRNA]
    B. Expression + GRN: input = [own_mRNA + other genes' mRNA + 9 GRN features]

    If B beats A consistently → GRN features carry predictive power. QED.

OUTPUT:
    results/tables/pergene_baseline_results.csv   — All scores
    results/figures/*.png                          — All plots (via plot_pergene_results.py)

USAGE:
    python experiments/train_pergene_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plotting"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings for cleaner output

# Import the standalone plotting module
from plot_pergene_results import PergeneResultsPlotter


# =============================================================================
# CONFIGURATION
# =============================================================================
# All paths and settings in one place for easy modification.

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_transsys")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create output directories
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Columns that are NOT features (metadata + target)
META_COLS = ["sample_id", "network_id", "gene_name"]
TARGET_COL = "target_protein"

# GRN feature columns — these are the 9 structural features extracted from network topology.
# They are the "treatment" in our controlled experiment.
GRN_FEATURE_PREFIXES = ["grn_"]

# Test split — 20% held out, never seen during training
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pergene_data():
    """
    Load the per-gene dataset and split into expression-only and full feature sets.
    
    Returns two feature sets:
      - X_expr: only mRNA columns (the "control group")
      - X_full: mRNA + GRN columns (the "treatment group")
      - y: target protein levels
      - meta: sample_id, network_id, gene_name for later analysis
      - feature_names: dict with 'expr' and 'full' feature lists
    """
    path = os.path.join(DATA_DIR, "pergene_dataset.csv")
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"  Dataset shape: {df.shape[0]} samples × {df.shape[1]} columns")
    
    # Separate metadata, features, and target
    meta = df[META_COLS]
    y = df[TARGET_COL]
    
    # All feature columns (everything except metadata and target)
    all_features = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    
    # Split into expression-only and GRN features
    grn_cols = [c for c in all_features if any(c.startswith(p) for p in GRN_FEATURE_PREFIXES)]
    expr_cols = [c for c in all_features if c not in grn_cols]
    
    print(f"  Expression features: {len(expr_cols)}")
    print(f"  GRN features: {len(grn_cols)} → {grn_cols}")
    
    X_expr = df[expr_cols]
    X_full = df[expr_cols + grn_cols]
    
    feature_names = {"expr": expr_cols, "full": expr_cols + grn_cols, "grn": grn_cols}
    
    return X_expr, X_full, y, meta, feature_names


# =============================================================================
# MODEL DEFINITIONS
# Each function creates, trains, and evaluates one model.
# All follow the same interface: (X_train, X_test, y_train, y_test, name) → dict
# =============================================================================

def train_ridge(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Ridge Regression — Linear model with L2 penalty.
    
    WHY RIDGE:
        The simplest possible baseline. If a neural network can't beat this,
        something is wrong with the neural network. Ridge is the "floor".
    
    WHAT alpha DOES:
        alpha=1.0 controls how much we penalise large weights.
        Higher alpha = more conservative model (smaller weights, less overfitting).
        Lower alpha = closer to standard linear regression.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (try 0.1, 1.0, 10.0)
    """
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_lasso(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Lasso Regression — Linear model with L1 penalty.
    
    WHY LASSO:
        Unlike Ridge (which shrinks all weights), Lasso can drive weights to
        EXACTLY ZERO. This means it automatically selects which features matter
        and which are useless. Perfect for our data where 30+ columns might be
        padding zeros from variable-size networks.
    
    WHAT LASSO REVEALS:
        After training, check model.coef_ — any coefficient that is exactly 0.0
        means Lasso decided that feature is worthless for prediction. If ALL GRN
        features survive (non-zero), that's strong evidence they carry real signal.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (higher = more features killed)
          0.001 is gentle, 0.1 is aggressive. We use 0.001 to start soft.
    """
    model = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Count how many features Lasso kept vs killed
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_elasticnet(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Elastic Net — Combines Ridge (L2) and Lasso (L1) penalties.
    
    WHY ELASTIC NET:
        Our GRN features are correlated (e.g., grn_in_degree and grn_n_activators
        overlap — a gene with high in-degree likely has many activators).
        
        - Lasso alone would randomly pick ONE from a group of correlated features
          and kill the rest. This loses information.
        - Ridge alone keeps everything but can't eliminate truly useless features.
        - Elastic Net keeps groups of correlated features together (Ridge behaviour)
          while still killing genuinely useless ones (Lasso behaviour).
    
    WHAT l1_ratio CONTROLS:
        - l1_ratio=0.0 → pure Ridge
        - l1_ratio=1.0 → pure Lasso
        - l1_ratio=0.5 → equal mix (our default)
    
    CUSTOMISABLE PARAMETERS:
        - alpha: overall regularisation strength
        - l1_ratio: balance between L1 and L2 (0.0 to 1.0)
    """
    model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_random_forest(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Random Forest — Ensemble of 100 decision trees.
    
    WHY RANDOM FOREST:
        First non-linear baseline. Can learn rules like "IF mRNA > 0.5 AND
        in_degree > 3 THEN protein is high". Linear models can't learn this.
        Also provides feature_importances_ for free.
    
    CUSTOMISABLE PARAMETERS:
        - n_estimators: number of trees (100 is standard, 200+ for more stability)
        - max_depth: how deep each tree can go (None = unlimited)
        - min_samples_leaf: minimum samples in each leaf (higher = simpler tree)
    """
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_STATE, 
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_mlp(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    MLP (Multi-Layer Perceptron) — D-GEX-style neural network.
    
    WHY MLP:
        This IS the D-GEX baseline. The original D-GEX paper (Chen et al., 2016)
        used an MLP to predict gene expression. We replicate their approach here
        with sklearn for speed, and later with PyTorch for full control.
    
    WHAT WAS CHANGED FROM DEFAULTS (and why):
        - hidden_layer_sizes: (100,) → (128, 64)
          DEFAULT had only 1 hidden layer with 100 neurons.
          CHANGED to 2 layers (128 then 64) to match D-GEX architecture.
          The funnel shape (128→64→1) forces the network to compress information.
        
        - max_iter: 200 → 1000
          DEFAULT cut training short after 200 epochs.
          CHANGED to give the model enough time to converge. Without this,
          the model barely started learning, causing negative R².
        
        - early_stopping: False → True
          DEFAULT trained for all max_iter epochs even if overfitting.
          CHANGED to monitor validation error and stop when it plateaus.
          This is like pulling the cake out of the oven before it burns.
        
        - validation_fraction: 0.1 → 0.15
          15% of training data is held out to monitor overfitting.
        
        - n_iter_no_change: 10 → 20
          Wait 20 epochs without improvement before stopping.
          Being patient helps avoid stopping too early on noisy data.
        
        - learning_rate: 'constant' → 'adaptive'
          DEFAULT kept the same learning rate forever.
          CHANGED to automatically reduce it when loss plateaus.
          Like slowing down as you approach a parking spot — big steps first,
          fine adjustments at the end.
        
        - batch_size: 200 → 64
          DEFAULT processed 200 samples per gradient update.
          CHANGED to 64 for noisier but more frequent updates.
          Helps the model explore more of the loss landscape.
    
    CUSTOMISABLE PARAMETERS TO TUNE:
        - hidden_layer_sizes: try (256, 128), (128, 64, 32), (64, 32)
        - learning_rate_init: try 0.0001, 0.001, 0.01
        - batch_size: try 32, 64, 128
        - alpha: L2 penalty strength (default 0.0001)
    """
    # MLP requires scaled features — without this, large-valued columns dominate
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=64,
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (stopped at epoch {model.n_iter_})")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("PER-GENE BASELINE EXPERIMENTS")
    print("=" * 70)
    
    # 1. Load data
    X_expr, X_full, y, meta, feature_names = load_pergene_data()
    
    # 2. Train-test split (same split for all models — fair comparison)
    X_expr_train, X_expr_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X_expr, y, meta, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_full_train, X_full_test = train_test_split(
        X_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )[:2]
    
    # 3. Define all model trainers
    # Each entry: (train_function, display_name)
    model_trainers = [
        (train_ridge,         "Ridge"),
        (train_lasso,         "Lasso"),
        (train_elasticnet,    "ElasticNet"),
        (train_random_forest, "RandomForest"),
        (train_mlp,           "MLP_DGEX"),
    ]
    
    # 4. Run all experiments: expression-only (A) vs expression+GRN (B)
    all_results = []
    all_predictions = {}  # Store for plotting
    rf_full_result = None  # Need this for feature importance
    lasso_full_result = None  # Need this for feature selection analysis
    
    for train_fn, base_name in model_trainers:
        # Experiment A: Expression-only
        print(f"\n--- {base_name} (Expression Only) ---")
        result_a = train_fn(X_expr_train.values, X_expr_test.values, 
                           y_train.values, y_test.values,
                           f"{base_name}_ExprOnly")
        all_results.append(result_a)
        all_predictions[f"{base_name}_ExprOnly"] = result_a["predictions"]
        
        # Experiment B: Expression + GRN
        print(f"--- {base_name} (Expression + GRN) ---")
        result_b = train_fn(X_full_train.values, X_full_test.values,
                           y_train.values, y_test.values,
                           f"{base_name}_ExprGRN")
        all_results.append(result_b)
        all_predictions[f"{base_name}_ExprGRN"] = result_b["predictions"]
        
        # Track special results for analysis
        if base_name == "RandomForest":
            rf_full_result = result_b
        if base_name == "Lasso":
            lasso_full_result = result_b
    
    # 5. Build results DataFrame
    results_df = pd.DataFrame([{
        "model": r["model"], "mse": r["mse"], "r2": r["r2"]
    } for r in all_results])
    
    # Save results table
    results_path = os.path.join(TABLES_DIR, "pergene_baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # 6. Print summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY: Expression-Only vs Expression+GRN")
    print("=" * 70)
    print(f"{'Model':<22} {'MSE (Expr)':<14} {'MSE (GRN)':<14} {'Δ MSE':<12} {'R² (Expr)':<12} {'R² (GRN)':<12}")
    print("-" * 86)
    
    for _, base_name in model_trainers:
        expr_row = results_df[results_df["model"] == f"{base_name}_ExprOnly"].iloc[0]
        grn_row = results_df[results_df["model"] == f"{base_name}_ExprGRN"].iloc[0]
        delta_mse = ((grn_row["mse"] - expr_row["mse"]) / expr_row["mse"]) * 100
        print(f"{base_name:<22} {expr_row['mse']:<14.6f} {grn_row['mse']:<14.6f} {delta_mse:<12.1f}% {expr_row['r2']:<12.4f} {grn_row['r2']:<12.4f}")
    
    # 7. Lasso feature selection analysis
    if lasso_full_result:
        print("\n" + "=" * 70)
        print("LASSO FEATURE SELECTION ANALYSIS")
        print("Which features did Lasso keep vs kill?")
        print("=" * 70)
        lasso_model = lasso_full_result["instance"]
        full_feature_names = feature_names["full"]
        
        for fname, coef in sorted(zip(full_feature_names, lasso_model.coef_), 
                                    key=lambda x: abs(x[1]), reverse=True):
            status = "✓ KEPT" if abs(coef) > 1e-10 else "✗ KILLED"
            if fname.startswith("grn_"):
                print(f"  [GRN] {fname:<30} coef={coef:>10.6f}  {status}")
        
        # Count GRN features that survived
        grn_feature_names = feature_names["grn"]
        grn_coefs = [lasso_model.coef_[full_feature_names.index(g)] for g in grn_feature_names]
        grn_alive = sum(1 for c in grn_coefs if abs(c) > 1e-10)
        print(f"\n  GRN features surviving Lasso: {grn_alive}/{len(grn_feature_names)}")
    
    # 8. Generate all plots using the standalone plotting module
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plotter = PergeneResultsPlotter(FIGURES_DIR)
    
    # Plot 1: Model comparison bar chart (MSE and R² side by side)
    plotter.plot_model_comparison(results_df)
    
    # Plot 2: Predicted vs Actual scatter for each model
    plotter.plot_predicted_vs_actual(all_predictions, y_test.values)
    
    # Plot 3: Residual distributions
    plotter.plot_residual_distributions(all_predictions, y_test.values)
    
    # Plot 4: Feature importance from Random Forest
    if rf_full_result:
        plotter.plot_feature_importance(
            rf_full_result["instance"], 
            feature_names["full"],
            top_n=20
        )
    
    # Plot 5: GRN feature correlation matrix
    plotter.plot_grn_correlation_matrix(
        pd.concat([X_full_train, X_full_test]),
        feature_names["grn"],
        y
    )
    
    # Plot 6: Per-network performance heatmap
    plotter.plot_per_network_performance(
        all_predictions, y_test.values, meta_test
    )
    
    print(f"\n✓ All plots saved to {FIGURES_DIR}")
    print("=" * 70)
    print("DONE. Next step: train_dgex_pytorch.py for the full D-GEX comparison.")
    print("=" * 70)


if __name__ == "__main__":
    main()
````

Now the standalone plotting module:

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\plotting\plot_pergene_results.py
"""
plot_pergene_results.py — Standalone plotting module for per-gene baseline results.

PURPOSE:
    Generates all analysis plots for the per-gene baseline experiments.
    This file is SEPARATE from the training code so that:
    1. Any training script can import and use it
    2. Plots can be regenerated without retraining models
    3. New plot types can be added without touching model code

PLOTS GENERATED:
    1. Model Comparison Bar Chart  — MSE and R² for all models, side by side
    2. Predicted vs Actual Scatter — How close predictions are to reality
    3. Residual Distributions      — Are errors random or systematically biased?
    4. Feature Importance          — Which features does Random Forest rely on?
    5. GRN Correlation Matrix      — How correlated are GRN features with each other?
    6. Per-Network Heatmap         — Which networks are easy/hard to predict?

USAGE:
    from plot_pergene_results import PergeneResultsPlotter
    plotter = PergeneResultsPlotter("path/to/figures")
    plotter.plot_model_comparison(results_df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves files without opening windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class PergeneResultsPlotter:
    """
    All plotting logic for per-gene baseline experiments.
    
    Each method takes raw data (predictions, scores, etc.) and saves a .png file.
    Methods are independent — you can call any one without the others.
    """
    
    def __init__(self, figures_dir: str):
        """
        Args:
            figures_dir: directory where all .png files will be saved
        """
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        
        # Consistent styling across all plots
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })
    
    def _save(self, fig, filename: str):
        """Helper: save figure and close to free memory."""
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {filename}")
    
    # =========================================================================
    # PLOT 1: MODEL COMPARISON BAR CHART
    # =========================================================================
    
    def plot_model_comparison(self, results_df: pd.DataFrame):
        """
        Side-by-side bar chart comparing MSE and R² for all models.
        
        WHY THIS PLOT:
            The single most important plot in the experiment. At a glance, you can
            see whether adding GRN features (darker bars) consistently improves
            over expression-only (lighter bars) across all model types.
        
        WHAT TO LOOK FOR:
            - Dark bars lower than light bars (MSE chart) = GRN helps
            - Dark bars higher than light bars (R² chart) = GRN helps
            - Consistent pattern across ALL models = strong evidence
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract model base names (without _ExprOnly/_ExprGRN suffix)
        base_names = []
        for name in results_df["model"]:
            base = name.replace("_ExprOnly", "").replace("_ExprGRN", "")
            if base not in base_names:
                base_names.append(base)
        
        x = np.arange(len(base_names))
        width = 0.35
        
        for ax, metric, title in zip(axes, ["mse", "r2"], ["MSE (lower is better)", "R² (higher is better)"]):
            expr_vals = [results_df[results_df["model"] == f"{b}_ExprOnly"][metric].values[0] for b in base_names]
            grn_vals = [results_df[results_df["model"] == f"{b}_ExprGRN"][metric].values[0] for b in base_names]
            
            bars1 = ax.bar(x - width/2, expr_vals, width, label="Expression Only", color="#78909C", alpha=0.85)
            bars2 = ax.bar(x + width/2, grn_vals, width, label="Expression + GRN", color="#26A69A", alpha=0.85)
            
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(base_names, rotation=15)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
```


## License: GPL-3.0
https://github.com/cKNUSPeR/i2nca/blob/a13e08a4db2ae0463402a6896862fdc296267e24/i2nca/qctools/visualization.py

```


Great questions. Let me answer each one, then build the solution.

---

## Answers to Your Questions

### Is sklearn MLP the same as D-GEX?
**Yes, conceptually.** D-GEX is just an MLP that predicts gene expression from other genes' expression. The sklearn `MLPRegressor` with `hidden_layer_sizes=(128, 64)` and ReLU activation IS a D-GEX-style model. The only difference is that the original D-GEX paper used a custom implementation with more control over training. So:

- **sklearn MLP** = D-GEX baseline (quick version)
- **PyTorch MLP** = D-GEX baseline (proper version with full control)

Both are D-GEX. Both should give similar results when properly tuned.

### Should we use both sklearn and PyTorch?
**Yes.** Keep both. In your report you can write: *"We implemented the D-GEX architecture in both scikit-learn and PyTorch and confirmed comparable performance, validating our implementation."* This shows rigour.

### File Organisation
Your instinct is correct. Here's my recommended structure:

```
experiments/
├── build_pergene_dataset.py          # Data preparation (already done)
├── train_pergene_baselines.py        # All per-gene models (Ridge, Lasso, ElasticNet, RF, MLP)
├── train_dgex_pytorch.py             # PyTorch D-GEX (Friday's task)
├── generate_synthetic_dataset.py     # Data generation (already done)
├── generate_fixed_size_dataset.py    # Fixed-size data (Friday's task)
└── plotting/
    └── plot_pergene_results.py       # ALL plots in one reusable file
```

The plotting file will be **standalone** — any training script can call it by passing results DataFrames. This avoids duplicating plotting code across model files.

---

Now let me build both files. First the updated training script, then the plotting module.

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\train_pergene_baselines.py
"""
train_pergene_baselines.py — Train and evaluate all per-gene baseline models.

PURPOSE:
    Answer the core research question: "Does adding GRN structural features
    improve gene expression prediction compared to expression-only models?"

MODELS (in order of complexity):
    1. Ridge Regression       — Linear baseline with L2 penalty (shrinks weights)
    2. Lasso Regression       — Linear baseline with L1 penalty (kills useless features)
    3. Elastic Net             — Combines L1 + L2 (best linear baseline for correlated features)
    4. Random Forest           — Non-linear, tree-based, gives feature importance
    5. MLP (sklearn)           — D-GEX-style neural network (quick version)

EACH MODEL IS TESTED TWICE:
    A. Expression-only:  input = [own_mRNA + other genes' mRNA]
    B. Expression + GRN: input = [own_mRNA + other genes' mRNA + 9 GRN features]

    If B beats A consistently → GRN features carry predictive power. QED.

OUTPUT:
    results/tables/pergene_baseline_results.csv   — All scores
    results/figures/*.png                          — All plots (via plot_pergene_results.py)

USAGE:
    python experiments/train_pergene_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plotting"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings for cleaner output

# Import the standalone plotting module
from plot_pergene_results import PergeneResultsPlotter


# =============================================================================
# CONFIGURATION
# =============================================================================
# All paths and settings in one place for easy modification.

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_transsys")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create output directories
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Columns that are NOT features (metadata + target)
META_COLS = ["sample_id", "network_id", "gene_name"]
TARGET_COL = "target_protein"

# GRN feature columns — these are the 9 structural features extracted from network topology.
# They are the "treatment" in our controlled experiment.
GRN_FEATURE_PREFIXES = ["grn_"]

# Test split — 20% held out, never seen during training
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pergene_data():
    """
    Load the per-gene dataset and split into expression-only and full feature sets.
    
    Returns two feature sets:
      - X_expr: only mRNA columns (the "control group")
      - X_full: mRNA + GRN columns (the "treatment group")
      - y: target protein levels
      - meta: sample_id, network_id, gene_name for later analysis
      - feature_names: dict with 'expr' and 'full' feature lists
    """
    path = os.path.join(DATA_DIR, "pergene_dataset.csv")
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"  Dataset shape: {df.shape[0]} samples × {df.shape[1]} columns")
    
    # Separate metadata, features, and target
    meta = df[META_COLS]
    y = df[TARGET_COL]
    
    # All feature columns (everything except metadata and target)
    all_features = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    
    # Split into expression-only and GRN features
    grn_cols = [c for c in all_features if any(c.startswith(p) for p in GRN_FEATURE_PREFIXES)]
    expr_cols = [c for c in all_features if c not in grn_cols]
    
    print(f"  Expression features: {len(expr_cols)}")
    print(f"  GRN features: {len(grn_cols)} → {grn_cols}")
    
    X_expr = df[expr_cols]
    X_full = df[expr_cols + grn_cols]
    
    feature_names = {"expr": expr_cols, "full": expr_cols + grn_cols, "grn": grn_cols}
    
    return X_expr, X_full, y, meta, feature_names


# =============================================================================
# MODEL DEFINITIONS
# Each function creates, trains, and evaluates one model.
# All follow the same interface: (X_train, X_test, y_train, y_test, name) → dict
# =============================================================================

def train_ridge(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Ridge Regression — Linear model with L2 penalty.
    
    WHY RIDGE:
        The simplest possible baseline. If a neural network can't beat this,
        something is wrong with the neural network. Ridge is the "floor".
    
    WHAT alpha DOES:
        alpha=1.0 controls how much we penalise large weights.
        Higher alpha = more conservative model (smaller weights, less overfitting).
        Lower alpha = closer to standard linear regression.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (try 0.1, 1.0, 10.0)
    """
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_lasso(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Lasso Regression — Linear model with L1 penalty.
    
    WHY LASSO:
        Unlike Ridge (which shrinks all weights), Lasso can drive weights to
        EXACTLY ZERO. This means it automatically selects which features matter
        and which are useless. Perfect for our data where 30+ columns might be
        padding zeros from variable-size networks.
    
    WHAT LASSO REVEALS:
        After training, check model.coef_ — any coefficient that is exactly 0.0
        means Lasso decided that feature is worthless for prediction. If ALL GRN
        features survive (non-zero), that's strong evidence they carry real signal.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (higher = more features killed)
          0.001 is gentle, 0.1 is aggressive. We use 0.001 to start soft.
    """
    model = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Count how many features Lasso kept vs killed
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_elasticnet(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Elastic Net — Combines Ridge (L2) and Lasso (L1) penalties.
    
    WHY ELASTIC NET:
        Our GRN features are correlated (e.g., grn_in_degree and grn_n_activators
        overlap — a gene with high in-degree likely has many activators).
        
        - Lasso alone would randomly pick ONE from a group of correlated features
          and kill the rest. This loses information.
        - Ridge alone keeps everything but can't eliminate truly useless features.
        - Elastic Net keeps groups of correlated features together (Ridge behaviour)
          while still killing genuinely useless ones (Lasso behaviour).
    
    WHAT l1_ratio CONTROLS:
        - l1_ratio=0.0 → pure Ridge
        - l1_ratio=1.0 → pure Lasso
        - l1_ratio=0.5 → equal mix (our default)
    
    CUSTOMISABLE PARAMETERS:
        - alpha: overall regularisation strength
        - l1_ratio: balance between L1 and L2 (0.0 to 1.0)
    """
    model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_random_forest(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Random Forest — Ensemble of 100 decision trees.
    
    WHY RANDOM FOREST:
        First non-linear baseline. Can learn rules like "IF mRNA > 0.5 AND
        in_degree > 3 THEN protein is high". Linear models can't learn this.
        Also provides feature_importances_ for free.
    
    CUSTOMISABLE PARAMETERS:
        - n_estimators: number of trees (100 is standard, 200+ for more stability)
        - max_depth: how deep each tree can go (None = unlimited)
        - min_samples_leaf: minimum samples in each leaf (higher = simpler tree)
    """
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_STATE, 
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_mlp(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    MLP (Multi-Layer Perceptron) — D-GEX-style neural network.
    
    WHY MLP:
        This IS the D-GEX baseline. The original D-GEX paper (Chen et al., 2016)
        used an MLP to predict gene expression. We replicate their approach here
        with sklearn for speed, and later with PyTorch for full control.
    
    WHAT WAS CHANGED FROM DEFAULTS (and why):
        - hidden_layer_sizes: (100,) → (128, 64)
          DEFAULT had only 1 hidden layer with 100 neurons.
          CHANGED to 2 layers (128 then 64) to match D-GEX architecture.
          The funnel shape (128→64→1) forces the network to compress information.
        
        - max_iter: 200 → 1000
          DEFAULT cut training short after 200 epochs.
          CHANGED to give the model enough time to converge. Without this,
          the model barely started learning, causing negative R².
        
        - early_stopping: False → True
          DEFAULT trained for all max_iter epochs even if overfitting.
          CHANGED to monitor validation error and stop when it plateaus.
          This is like pulling the cake out of the oven before it burns.
        
        - validation_fraction: 0.1 → 0.15
          15% of training data is held out to monitor overfitting.
        
        - n_iter_no_change: 10 → 20
          Wait 20 epochs without improvement before stopping.
          Being patient helps avoid stopping too early on noisy data.
        
        - learning_rate: 'constant' → 'adaptive'
          DEFAULT kept the same learning rate forever.
          CHANGED to automatically reduce it when loss plateaus.
          Like slowing down as you approach a parking spot — big steps first,
          fine adjustments at the end.
        
        - batch_size: 200 → 64
          DEFAULT processed 200 samples per gradient update.
          CHANGED to 64 for noisier but more frequent updates.
          Helps the model explore more of the loss landscape.
    
    CUSTOMISABLE PARAMETERS TO TUNE:
        - hidden_layer_sizes: try (256, 128), (128, 64, 32), (64, 32)
        - learning_rate_init: try 0.0001, 0.001, 0.01
        - batch_size: try 32, 64, 128
        - alpha: L2 penalty strength (default 0.0001)
    """
    # MLP requires scaled features — without this, large-valued columns dominate
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=64,
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (stopped at epoch {model.n_iter_})")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("PER-GENE BASELINE EXPERIMENTS")
    print("=" * 70)
    
    # 1. Load data
    X_expr, X_full, y, meta, feature_names = load_pergene_data()
    
    # 2. Train-test split (same split for all models — fair comparison)
    X_expr_train, X_expr_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X_expr, y, meta, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_full_train, X_full_test = train_test_split(
        X_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )[:2]
    
    # 3. Define all model trainers
    # Each entry: (train_function, display_name)
    model_trainers = [
        (train_ridge,         "Ridge"),
        (train_lasso,         "Lasso"),
        (train_elasticnet,    "ElasticNet"),
        (train_random_forest, "RandomForest"),
        (train_mlp,           "MLP_DGEX"),
    ]
    
    # 4. Run all experiments: expression-only (A) vs expression+GRN (B)
    all_results = []
    all_predictions = {}  # Store for plotting
    rf_full_result = None  # Need this for feature importance
    lasso_full_result = None  # Need this for feature selection analysis
    
    for train_fn, base_name in model_trainers:
        # Experiment A: Expression-only
        print(f"\n--- {base_name} (Expression Only) ---")
        result_a = train_fn(X_expr_train.values, X_expr_test.values, 
                           y_train.values, y_test.values,
                           f"{base_name}_ExprOnly")
        all_results.append(result_a)
        all_predictions[f"{base_name}_ExprOnly"] = result_a["predictions"]
        
        # Experiment B: Expression + GRN
        print(f"--- {base_name} (Expression + GRN) ---")
        result_b = train_fn(X_full_train.values, X_full_test.values,
                           y_train.values, y_test.values,
                           f"{base_name}_ExprGRN")
        all_results.append(result_b)
        all_predictions[f"{base_name}_ExprGRN"] = result_b["predictions"]
        
        # Track special results for analysis
        if base_name == "RandomForest":
            rf_full_result = result_b
        if base_name == "Lasso":
            lasso_full_result = result_b
    
    # 5. Build results DataFrame
    results_df = pd.DataFrame([{
        "model": r["model"], "mse": r["mse"], "r2": r["r2"]
    } for r in all_results])
    
    # Save results table
    results_path = os.path.join(TABLES_DIR, "pergene_baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # 6. Print summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY: Expression-Only vs Expression+GRN")
    print("=" * 70)
    print(f"{'Model':<22} {'MSE (Expr)':<14} {'MSE (GRN)':<14} {'Δ MSE':<12} {'R² (Expr)':<12} {'R² (GRN)':<12}")
    print("-" * 86)
    
    for _, base_name in model_trainers:
        expr_row = results_df[results_df["model"] == f"{base_name}_ExprOnly"].iloc[0]
        grn_row = results_df[results_df["model"] == f"{base_name}_ExprGRN"].iloc[0]
        delta_mse = ((grn_row["mse"] - expr_row["mse"]) / expr_row["mse"]) * 100
        print(f"{base_name:<22} {expr_row['mse']:<14.6f} {grn_row['mse']:<14.6f} {delta_mse:<12.1f}% {expr_row['r2']:<12.4f} {grn_row['r2']:<12.4f}")
    
    # 7. Lasso feature selection analysis
    if lasso_full_result:
        print("\n" + "=" * 70)
        print("LASSO FEATURE SELECTION ANALYSIS")
        print("Which features did Lasso keep vs kill?")
        print("=" * 70)
        lasso_model = lasso_full_result["instance"]
        full_feature_names = feature_names["full"]
        
        for fname, coef in sorted(zip(full_feature_names, lasso_model.coef_), 
                                    key=lambda x: abs(x[1]), reverse=True):
            status = "✓ KEPT" if abs(coef) > 1e-10 else "✗ KILLED"
            if fname.startswith("grn_"):
                print(f"  [GRN] {fname:<30} coef={coef:>10.6f}  {status}")
        
        # Count GRN features that survived
        grn_feature_names = feature_names["grn"]
        grn_coefs = [lasso_model.coef_[full_feature_names.index(g)] for g in grn_feature_names]
        grn_alive = sum(1 for c in grn_coefs if abs(c) > 1e-10)
        print(f"\n  GRN features surviving Lasso: {grn_alive}/{len(grn_feature_names)}")
    
    # 8. Generate all plots using the standalone plotting module
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plotter = PergeneResultsPlotter(FIGURES_DIR)
    
    # Plot 1: Model comparison bar chart (MSE and R² side by side)
    plotter.plot_model_comparison(results_df)
    
    # Plot 2: Predicted vs Actual scatter for each model
    plotter.plot_predicted_vs_actual(all_predictions, y_test.values)
    
    # Plot 3: Residual distributions
    plotter.plot_residual_distributions(all_predictions, y_test.values)
    
    # Plot 4: Feature importance from Random Forest
    if rf_full_result:
        plotter.plot_feature_importance(
            rf_full_result["instance"], 
            feature_names["full"],
            top_n=20
        )
    
    # Plot 5: GRN feature correlation matrix
    plotter.plot_grn_correlation_matrix(
        pd.concat([X_full_train, X_full_test]),
        feature_names["grn"],
        y
    )
    
    # Plot 6: Per-network performance heatmap
    plotter.plot_per_network_performance(
        all_predictions, y_test.values, meta_test
    )
    
    print(f"\n✓ All plots saved to {FIGURES_DIR}")
    print("=" * 70)
    print("DONE. Next step: train_dgex_pytorch.py for the full D-GEX comparison.")
    print("=" * 70)


if __name__ == "__main__":
    main()
````

Now the standalone plotting module:

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\plotting\plot_pergene_results.py
"""
plot_pergene_results.py — Standalone plotting module for per-gene baseline results.

PURPOSE:
    Generates all analysis plots for the per-gene baseline experiments.
    This file is SEPARATE from the training code so that:
    1. Any training script can import and use it
    2. Plots can be regenerated without retraining models
    3. New plot types can be added without touching model code

PLOTS GENERATED:
    1. Model Comparison Bar Chart  — MSE and R² for all models, side by side
    2. Predicted vs Actual Scatter — How close predictions are to reality
    3. Residual Distributions      — Are errors random or systematically biased?
    4. Feature Importance          — Which features does Random Forest rely on?
    5. GRN Correlation Matrix      — How correlated are GRN features with each other?
    6. Per-Network Heatmap         — Which networks are easy/hard to predict?

USAGE:
    from plot_pergene_results import PergeneResultsPlotter
    plotter = PergeneResultsPlotter("path/to/figures")
    plotter.plot_model_comparison(results_df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves files without opening windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class PergeneResultsPlotter:
    """
    All plotting logic for per-gene baseline experiments.
    
    Each method takes raw data (predictions, scores, etc.) and saves a .png file.
    Methods are independent — you can call any one without the others.
    """
    
    def __init__(self, figures_dir: str):
        """
        Args:
            figures_dir: directory where all .png files will be saved
        """
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        
        # Consistent styling across all plots
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })
    
    def _save(self, fig, filename: str):
        """Helper: save figure and close to free memory."""
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {filename}")
    
    # =========================================================================
    # PLOT 1: MODEL COMPARISON BAR CHART
    # =========================================================================
    
    def plot_model_comparison(self, results_df: pd.DataFrame):
        """
        Side-by-side bar chart comparing MSE and R² for all models.
        
        WHY THIS PLOT:
            The single most important plot in the experiment. At a glance, you can
            see whether adding GRN features (darker bars) consistently improves
            over expression-only (lighter bars) across all model types.
        
        WHAT TO LOOK FOR:
            - Dark bars lower than light bars (MSE chart) = GRN helps
            - Dark bars higher than light bars (R² chart) = GRN helps
            - Consistent pattern across ALL models = strong evidence
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract model base names (without _ExprOnly/_ExprGRN suffix)
        base_names = []
        for name in results_df["model"]:
            base = name.replace("_ExprOnly", "").replace("_ExprGRN", "")
            if base not in base_names:
                base_names.append(base)
        
        x = np.arange(len(base_names))
        width = 0.35
        
        for ax, metric, title in zip(axes, ["mse", "r2"], ["MSE (lower is better)", "R² (higher is better)"]):
            expr_vals = [results_df[results_df["model"] == f"{b}_ExprOnly"][metric].values[0] for b in base_names]
            grn_vals = [results_df[results_df["model"] == f"{b}_ExprGRN"][metric].values[0] for b in base_names]
            
            bars1 = ax.bar(x - width/2, expr_vals, width, label="Expression Only", color="#78909C", alpha=0.85)
            bars2 = ax.bar(x + width/2, grn_vals, width, label="Expression + GRN", color="#26A69A", alpha=0.85)
            
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(base_names, rotation=15)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
```


## License: GPL-3.0
https://github.com/cKNUSPeR/i2nca/blob/a13e08a4db2ae0463402a6896862fdc296267e24/i2nca/qctools/visualization.py

```


Great questions. Let me answer each one, then build the solution.

---

## Answers to Your Questions

### Is sklearn MLP the same as D-GEX?
**Yes, conceptually.** D-GEX is just an MLP that predicts gene expression from other genes' expression. The sklearn `MLPRegressor` with `hidden_layer_sizes=(128, 64)` and ReLU activation IS a D-GEX-style model. The only difference is that the original D-GEX paper used a custom implementation with more control over training. So:

- **sklearn MLP** = D-GEX baseline (quick version)
- **PyTorch MLP** = D-GEX baseline (proper version with full control)

Both are D-GEX. Both should give similar results when properly tuned.

### Should we use both sklearn and PyTorch?
**Yes.** Keep both. In your report you can write: *"We implemented the D-GEX architecture in both scikit-learn and PyTorch and confirmed comparable performance, validating our implementation."* This shows rigour.

### File Organisation
Your instinct is correct. Here's my recommended structure:

```
experiments/
├── build_pergene_dataset.py          # Data preparation (already done)
├── train_pergene_baselines.py        # All per-gene models (Ridge, Lasso, ElasticNet, RF, MLP)
├── train_dgex_pytorch.py             # PyTorch D-GEX (Friday's task)
├── generate_synthetic_dataset.py     # Data generation (already done)
├── generate_fixed_size_dataset.py    # Fixed-size data (Friday's task)
└── plotting/
    └── plot_pergene_results.py       # ALL plots in one reusable file
```

The plotting file will be **standalone** — any training script can call it by passing results DataFrames. This avoids duplicating plotting code across model files.

---

Now let me build both files. First the updated training script, then the plotting module.

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\train_pergene_baselines.py
"""
train_pergene_baselines.py — Train and evaluate all per-gene baseline models.

PURPOSE:
    Answer the core research question: "Does adding GRN structural features
    improve gene expression prediction compared to expression-only models?"

MODELS (in order of complexity):
    1. Ridge Regression       — Linear baseline with L2 penalty (shrinks weights)
    2. Lasso Regression       — Linear baseline with L1 penalty (kills useless features)
    3. Elastic Net             — Combines L1 + L2 (best linear baseline for correlated features)
    4. Random Forest           — Non-linear, tree-based, gives feature importance
    5. MLP (sklearn)           — D-GEX-style neural network (quick version)

EACH MODEL IS TESTED TWICE:
    A. Expression-only:  input = [own_mRNA + other genes' mRNA]
    B. Expression + GRN: input = [own_mRNA + other genes' mRNA + 9 GRN features]

    If B beats A consistently → GRN features carry predictive power. QED.

OUTPUT:
    results/tables/pergene_baseline_results.csv   — All scores
    results/figures/*.png                          — All plots (via plot_pergene_results.py)

USAGE:
    python experiments/train_pergene_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plotting"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings for cleaner output

# Import the standalone plotting module
from plot_pergene_results import PergeneResultsPlotter


# =============================================================================
# CONFIGURATION
# =============================================================================
# All paths and settings in one place for easy modification.

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_transsys")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create output directories
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Columns that are NOT features (metadata + target)
META_COLS = ["sample_id", "network_id", "gene_name"]
TARGET_COL = "target_protein"

# GRN feature columns — these are the 9 structural features extracted from network topology.
# They are the "treatment" in our controlled experiment.
GRN_FEATURE_PREFIXES = ["grn_"]

# Test split — 20% held out, never seen during training
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pergene_data():
    """
    Load the per-gene dataset and split into expression-only and full feature sets.
    
    Returns two feature sets:
      - X_expr: only mRNA columns (the "control group")
      - X_full: mRNA + GRN columns (the "treatment group")
      - y: target protein levels
      - meta: sample_id, network_id, gene_name for later analysis
      - feature_names: dict with 'expr' and 'full' feature lists
    """
    path = os.path.join(DATA_DIR, "pergene_dataset.csv")
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"  Dataset shape: {df.shape[0]} samples × {df.shape[1]} columns")
    
    # Separate metadata, features, and target
    meta = df[META_COLS]
    y = df[TARGET_COL]
    
    # All feature columns (everything except metadata and target)
    all_features = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    
    # Split into expression-only and GRN features
    grn_cols = [c for c in all_features if any(c.startswith(p) for p in GRN_FEATURE_PREFIXES)]
    expr_cols = [c for c in all_features if c not in grn_cols]
    
    print(f"  Expression features: {len(expr_cols)}")
    print(f"  GRN features: {len(grn_cols)} → {grn_cols}")
    
    X_expr = df[expr_cols]
    X_full = df[expr_cols + grn_cols]
    
    feature_names = {"expr": expr_cols, "full": expr_cols + grn_cols, "grn": grn_cols}
    
    return X_expr, X_full, y, meta, feature_names


# =============================================================================
# MODEL DEFINITIONS
# Each function creates, trains, and evaluates one model.
# All follow the same interface: (X_train, X_test, y_train, y_test, name) → dict
# =============================================================================

def train_ridge(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Ridge Regression — Linear model with L2 penalty.
    
    WHY RIDGE:
        The simplest possible baseline. If a neural network can't beat this,
        something is wrong with the neural network. Ridge is the "floor".
    
    WHAT alpha DOES:
        alpha=1.0 controls how much we penalise large weights.
        Higher alpha = more conservative model (smaller weights, less overfitting).
        Lower alpha = closer to standard linear regression.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (try 0.1, 1.0, 10.0)
    """
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_lasso(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Lasso Regression — Linear model with L1 penalty.
    
    WHY LASSO:
        Unlike Ridge (which shrinks all weights), Lasso can drive weights to
        EXACTLY ZERO. This means it automatically selects which features matter
        and which are useless. Perfect for our data where 30+ columns might be
        padding zeros from variable-size networks.
    
    WHAT LASSO REVEALS:
        After training, check model.coef_ — any coefficient that is exactly 0.0
        means Lasso decided that feature is worthless for prediction. If ALL GRN
        features survive (non-zero), that's strong evidence they carry real signal.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (higher = more features killed)
          0.001 is gentle, 0.1 is aggressive. We use 0.001 to start soft.
    """
    model = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Count how many features Lasso kept vs killed
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_elasticnet(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Elastic Net — Combines Ridge (L2) and Lasso (L1) penalties.
    
    WHY ELASTIC NET:
        Our GRN features are correlated (e.g., grn_in_degree and grn_n_activators
        overlap — a gene with high in-degree likely has many activators).
        
        - Lasso alone would randomly pick ONE from a group of correlated features
          and kill the rest. This loses information.
        - Ridge alone keeps everything but can't eliminate truly useless features.
        - Elastic Net keeps groups of correlated features together (Ridge behaviour)
          while still killing genuinely useless ones (Lasso behaviour).
    
    WHAT l1_ratio CONTROLS:
        - l1_ratio=0.0 → pure Ridge
        - l1_ratio=1.0 → pure Lasso
        - l1_ratio=0.5 → equal mix (our default)
    
    CUSTOMISABLE PARAMETERS:
        - alpha: overall regularisation strength
        - l1_ratio: balance between L1 and L2 (0.0 to 1.0)
    """
    model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_random_forest(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Random Forest — Ensemble of 100 decision trees.
    
    WHY RANDOM FOREST:
        First non-linear baseline. Can learn rules like "IF mRNA > 0.5 AND
        in_degree > 3 THEN protein is high". Linear models can't learn this.
        Also provides feature_importances_ for free.
    
    CUSTOMISABLE PARAMETERS:
        - n_estimators: number of trees (100 is standard, 200+ for more stability)
        - max_depth: how deep each tree can go (None = unlimited)
        - min_samples_leaf: minimum samples in each leaf (higher = simpler tree)
    """
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_STATE, 
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_mlp(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    MLP (Multi-Layer Perceptron) — D-GEX-style neural network.
    
    WHY MLP:
        This IS the D-GEX baseline. The original D-GEX paper (Chen et al., 2016)
        used an MLP to predict gene expression. We replicate their approach here
        with sklearn for speed, and later with PyTorch for full control.
    
    WHAT WAS CHANGED FROM DEFAULTS (and why):
        - hidden_layer_sizes: (100,) → (128, 64)
          DEFAULT had only 1 hidden layer with 100 neurons.
          CHANGED to 2 layers (128 then 64) to match D-GEX architecture.
          The funnel shape (128→64→1) forces the network to compress information.
        
        - max_iter: 200 → 1000
          DEFAULT cut training short after 200 epochs.
          CHANGED to give the model enough time to converge. Without this,
          the model barely started learning, causing negative R².
        
        - early_stopping: False → True
          DEFAULT trained for all max_iter epochs even if overfitting.
          CHANGED to monitor validation error and stop when it plateaus.
          This is like pulling the cake out of the oven before it burns.
        
        - validation_fraction: 0.1 → 0.15
          15% of training data is held out to monitor overfitting.
        
        - n_iter_no_change: 10 → 20
          Wait 20 epochs without improvement before stopping.
          Being patient helps avoid stopping too early on noisy data.
        
        - learning_rate: 'constant' → 'adaptive'
          DEFAULT kept the same learning rate forever.
          CHANGED to automatically reduce it when loss plateaus.
          Like slowing down as you approach a parking spot — big steps first,
          fine adjustments at the end.
        
        - batch_size: 200 → 64
          DEFAULT processed 200 samples per gradient update.
          CHANGED to 64 for noisier but more frequent updates.
          Helps the model explore more of the loss landscape.
    
    CUSTOMISABLE PARAMETERS TO TUNE:
        - hidden_layer_sizes: try (256, 128), (128, 64, 32), (64, 32)
        - learning_rate_init: try 0.0001, 0.001, 0.01
        - batch_size: try 32, 64, 128
        - alpha: L2 penalty strength (default 0.0001)
    """
    # MLP requires scaled features — without this, large-valued columns dominate
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=64,
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (stopped at epoch {model.n_iter_})")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("PER-GENE BASELINE EXPERIMENTS")
    print("=" * 70)
    
    # 1. Load data
    X_expr, X_full, y, meta, feature_names = load_pergene_data()
    
    # 2. Train-test split (same split for all models — fair comparison)
    X_expr_train, X_expr_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X_expr, y, meta, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_full_train, X_full_test = train_test_split(
        X_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )[:2]
    
    # 3. Define all model trainers
    # Each entry: (train_function, display_name)
    model_trainers = [
        (train_ridge,         "Ridge"),
        (train_lasso,         "Lasso"),
        (train_elasticnet,    "ElasticNet"),
        (train_random_forest, "RandomForest"),
        (train_mlp,           "MLP_DGEX"),
    ]
    
    # 4. Run all experiments: expression-only (A) vs expression+GRN (B)
    all_results = []
    all_predictions = {}  # Store for plotting
    rf_full_result = None  # Need this for feature importance
    lasso_full_result = None  # Need this for feature selection analysis
    
    for train_fn, base_name in model_trainers:
        # Experiment A: Expression-only
        print(f"\n--- {base_name} (Expression Only) ---")
        result_a = train_fn(X_expr_train.values, X_expr_test.values, 
                           y_train.values, y_test.values,
                           f"{base_name}_ExprOnly")
        all_results.append(result_a)
        all_predictions[f"{base_name}_ExprOnly"] = result_a["predictions"]
        
        # Experiment B: Expression + GRN
        print(f"--- {base_name} (Expression + GRN) ---")
        result_b = train_fn(X_full_train.values, X_full_test.values,
                           y_train.values, y_test.values,
                           f"{base_name}_ExprGRN")
        all_results.append(result_b)
        all_predictions[f"{base_name}_ExprGRN"] = result_b["predictions"]
        
        # Track special results for analysis
        if base_name == "RandomForest":
            rf_full_result = result_b
        if base_name == "Lasso":
            lasso_full_result = result_b
    
    # 5. Build results DataFrame
    results_df = pd.DataFrame([{
        "model": r["model"], "mse": r["mse"], "r2": r["r2"]
    } for r in all_results])
    
    # Save results table
    results_path = os.path.join(TABLES_DIR, "pergene_baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # 6. Print summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY: Expression-Only vs Expression+GRN")
    print("=" * 70)
    print(f"{'Model':<22} {'MSE (Expr)':<14} {'MSE (GRN)':<14} {'Δ MSE':<12} {'R² (Expr)':<12} {'R² (GRN)':<12}")
    print("-" * 86)
    
    for _, base_name in model_trainers:
        expr_row = results_df[results_df["model"] == f"{base_name}_ExprOnly"].iloc[0]
        grn_row = results_df[results_df["model"] == f"{base_name}_ExprGRN"].iloc[0]
        delta_mse = ((grn_row["mse"] - expr_row["mse"]) / expr_row["mse"]) * 100
        print(f"{base_name:<22} {expr_row['mse']:<14.6f} {grn_row['mse']:<14.6f} {delta_mse:<12.1f}% {expr_row['r2']:<12.4f} {grn_row['r2']:<12.4f}")
    
    # 7. Lasso feature selection analysis
    if lasso_full_result:
        print("\n" + "=" * 70)
        print("LASSO FEATURE SELECTION ANALYSIS")
        print("Which features did Lasso keep vs kill?")
        print("=" * 70)
        lasso_model = lasso_full_result["instance"]
        full_feature_names = feature_names["full"]
        
        for fname, coef in sorted(zip(full_feature_names, lasso_model.coef_), 
                                    key=lambda x: abs(x[1]), reverse=True):
            status = "✓ KEPT" if abs(coef) > 1e-10 else "✗ KILLED"
            if fname.startswith("grn_"):
                print(f"  [GRN] {fname:<30} coef={coef:>10.6f}  {status}")
        
        # Count GRN features that survived
        grn_feature_names = feature_names["grn"]
        grn_coefs = [lasso_model.coef_[full_feature_names.index(g)] for g in grn_feature_names]
        grn_alive = sum(1 for c in grn_coefs if abs(c) > 1e-10)
        print(f"\n  GRN features surviving Lasso: {grn_alive}/{len(grn_feature_names)}")
    
    # 8. Generate all plots using the standalone plotting module
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plotter = PergeneResultsPlotter(FIGURES_DIR)
    
    # Plot 1: Model comparison bar chart (MSE and R² side by side)
    plotter.plot_model_comparison(results_df)
    
    # Plot 2: Predicted vs Actual scatter for each model
    plotter.plot_predicted_vs_actual(all_predictions, y_test.values)
    
    # Plot 3: Residual distributions
    plotter.plot_residual_distributions(all_predictions, y_test.values)
    
    # Plot 4: Feature importance from Random Forest
    if rf_full_result:
        plotter.plot_feature_importance(
            rf_full_result["instance"], 
            feature_names["full"],
            top_n=20
        )
    
    # Plot 5: GRN feature correlation matrix
    plotter.plot_grn_correlation_matrix(
        pd.concat([X_full_train, X_full_test]),
        feature_names["grn"],
        y
    )
    
    # Plot 6: Per-network performance heatmap
    plotter.plot_per_network_performance(
        all_predictions, y_test.values, meta_test
    )
    
    print(f"\n✓ All plots saved to {FIGURES_DIR}")
    print("=" * 70)
    print("DONE. Next step: train_dgex_pytorch.py for the full D-GEX comparison.")
    print("=" * 70)


if __name__ == "__main__":
    main()
````

Now the standalone plotting module:

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\plotting\plot_pergene_results.py
"""
plot_pergene_results.py — Standalone plotting module for per-gene baseline results.

PURPOSE:
    Generates all analysis plots for the per-gene baseline experiments.
    This file is SEPARATE from the training code so that:
    1. Any training script can import and use it
    2. Plots can be regenerated without retraining models
    3. New plot types can be added without touching model code

PLOTS GENERATED:
    1. Model Comparison Bar Chart  — MSE and R² for all models, side by side
    2. Predicted vs Actual Scatter — How close predictions are to reality
    3. Residual Distributions      — Are errors random or systematically biased?
    4. Feature Importance          — Which features does Random Forest rely on?
    5. GRN Correlation Matrix      — How correlated are GRN features with each other?
    6. Per-Network Heatmap         — Which networks are easy/hard to predict?

USAGE:
    from plot_pergene_results import PergeneResultsPlotter
    plotter = PergeneResultsPlotter("path/to/figures")
    plotter.plot_model_comparison(results_df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves files without opening windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class PergeneResultsPlotter:
    """
    All plotting logic for per-gene baseline experiments.
    
    Each method takes raw data (predictions, scores, etc.) and saves a .png file.
    Methods are independent — you can call any one without the others.
    """
    
    def __init__(self, figures_dir: str):
        """
        Args:
            figures_dir: directory where all .png files will be saved
        """
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        
        # Consistent styling across all plots
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })
    
    def _save(self, fig, filename: str):
        """Helper: save figure and close to free memory."""
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {filename}")
    
    # =========================================================================
    # PLOT 1: MODEL COMPARISON BAR CHART
    # =========================================================================
    
    def plot_model_comparison(self, results_df: pd.DataFrame):
        """
        Side-by-side bar chart comparing MSE and R² for all models.
        
        WHY THIS PLOT:
            The single most important plot in the experiment. At a glance, you can
            see whether adding GRN features (darker bars) consistently improves
            over expression-only (lighter bars) across all model types.
        
        WHAT TO LOOK FOR:
            - Dark bars lower than light bars (MSE chart) = GRN helps
            - Dark bars higher than light bars (R² chart) = GRN helps
            - Consistent pattern across ALL models = strong evidence
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract model base names (without _ExprOnly/_ExprGRN suffix)
        base_names = []
        for name in results_df["model"]:
            base = name.replace("_ExprOnly", "").replace("_ExprGRN", "")
            if base not in base_names:
                base_names.append(base)
        
        x = np.arange(len(base_names))
        width = 0.35
        
        for ax, metric, title in zip(axes, ["mse", "r2"], ["MSE (lower is better)", "R² (higher is better)"]):
            expr_vals = [results_df[results_df["model"] == f"{b}_ExprOnly"][metric].values[0] for b in base_names]
            grn_vals = [results_df[results_df["model"] == f"{b}_ExprGRN"][metric].values[0] for b in base_names]
            
            bars1 = ax.bar(x - width/2, expr_vals, width, label="Expression Only", color="#78909C", alpha=0.85)
            bars2 = ax.bar(x + width/2, grn_vals, width, label="Expression + GRN", color="#26A69A", alpha=0.85)
            
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(base_names, rotation=15)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
```


## License: GPL-3.0
https://github.com/cKNUSPeR/i2nca/blob/a13e08a4db2ae0463402a6896862fdc296267e24/i2nca/qctools/visualization.py

```


Great questions. Let me answer each one, then build the solution.

---

## Answers to Your Questions

### Is sklearn MLP the same as D-GEX?
**Yes, conceptually.** D-GEX is just an MLP that predicts gene expression from other genes' expression. The sklearn `MLPRegressor` with `hidden_layer_sizes=(128, 64)` and ReLU activation IS a D-GEX-style model. The only difference is that the original D-GEX paper used a custom implementation with more control over training. So:

- **sklearn MLP** = D-GEX baseline (quick version)
- **PyTorch MLP** = D-GEX baseline (proper version with full control)

Both are D-GEX. Both should give similar results when properly tuned.

### Should we use both sklearn and PyTorch?
**Yes.** Keep both. In your report you can write: *"We implemented the D-GEX architecture in both scikit-learn and PyTorch and confirmed comparable performance, validating our implementation."* This shows rigour.

### File Organisation
Your instinct is correct. Here's my recommended structure:

```
experiments/
├── build_pergene_dataset.py          # Data preparation (already done)
├── train_pergene_baselines.py        # All per-gene models (Ridge, Lasso, ElasticNet, RF, MLP)
├── train_dgex_pytorch.py             # PyTorch D-GEX (Friday's task)
├── generate_synthetic_dataset.py     # Data generation (already done)
├── generate_fixed_size_dataset.py    # Fixed-size data (Friday's task)
└── plotting/
    └── plot_pergene_results.py       # ALL plots in one reusable file
```

The plotting file will be **standalone** — any training script can call it by passing results DataFrames. This avoids duplicating plotting code across model files.

---

Now let me build both files. First the updated training script, then the plotting module.

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\train_pergene_baselines.py
"""
train_pergene_baselines.py — Train and evaluate all per-gene baseline models.

PURPOSE:
    Answer the core research question: "Does adding GRN structural features
    improve gene expression prediction compared to expression-only models?"

MODELS (in order of complexity):
    1. Ridge Regression       — Linear baseline with L2 penalty (shrinks weights)
    2. Lasso Regression       — Linear baseline with L1 penalty (kills useless features)
    3. Elastic Net             — Combines L1 + L2 (best linear baseline for correlated features)
    4. Random Forest           — Non-linear, tree-based, gives feature importance
    5. MLP (sklearn)           — D-GEX-style neural network (quick version)

EACH MODEL IS TESTED TWICE:
    A. Expression-only:  input = [own_mRNA + other genes' mRNA]
    B. Expression + GRN: input = [own_mRNA + other genes' mRNA + 9 GRN features]

    If B beats A consistently → GRN features carry predictive power. QED.

OUTPUT:
    results/tables/pergene_baseline_results.csv   — All scores
    results/figures/*.png                          — All plots (via plot_pergene_results.py)

USAGE:
    python experiments/train_pergene_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plotting"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings for cleaner output

# Import the standalone plotting module
from plot_pergene_results import PergeneResultsPlotter


# =============================================================================
# CONFIGURATION
# =============================================================================
# All paths and settings in one place for easy modification.

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_transsys")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create output directories
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Columns that are NOT features (metadata + target)
META_COLS = ["sample_id", "network_id", "gene_name"]
TARGET_COL = "target_protein"

# GRN feature columns — these are the 9 structural features extracted from network topology.
# They are the "treatment" in our controlled experiment.
GRN_FEATURE_PREFIXES = ["grn_"]

# Test split — 20% held out, never seen during training
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pergene_data():
    """
    Load the per-gene dataset and split into expression-only and full feature sets.
    
    Returns two feature sets:
      - X_expr: only mRNA columns (the "control group")
      - X_full: mRNA + GRN columns (the "treatment group")
      - y: target protein levels
      - meta: sample_id, network_id, gene_name for later analysis
      - feature_names: dict with 'expr' and 'full' feature lists
    """
    path = os.path.join(DATA_DIR, "pergene_dataset.csv")
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"  Dataset shape: {df.shape[0]} samples × {df.shape[1]} columns")
    
    # Separate metadata, features, and target
    meta = df[META_COLS]
    y = df[TARGET_COL]
    
    # All feature columns (everything except metadata and target)
    all_features = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    
    # Split into expression-only and GRN features
    grn_cols = [c for c in all_features if any(c.startswith(p) for p in GRN_FEATURE_PREFIXES)]
    expr_cols = [c for c in all_features if c not in grn_cols]
    
    print(f"  Expression features: {len(expr_cols)}")
    print(f"  GRN features: {len(grn_cols)} → {grn_cols}")
    
    X_expr = df[expr_cols]
    X_full = df[expr_cols + grn_cols]
    
    feature_names = {"expr": expr_cols, "full": expr_cols + grn_cols, "grn": grn_cols}
    
    return X_expr, X_full, y, meta, feature_names


# =============================================================================
# MODEL DEFINITIONS
# Each function creates, trains, and evaluates one model.
# All follow the same interface: (X_train, X_test, y_train, y_test, name) → dict
# =============================================================================

def train_ridge(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Ridge Regression — Linear model with L2 penalty.
    
    WHY RIDGE:
        The simplest possible baseline. If a neural network can't beat this,
        something is wrong with the neural network. Ridge is the "floor".
    
    WHAT alpha DOES:
        alpha=1.0 controls how much we penalise large weights.
        Higher alpha = more conservative model (smaller weights, less overfitting).
        Lower alpha = closer to standard linear regression.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (try 0.1, 1.0, 10.0)
    """
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_lasso(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Lasso Regression — Linear model with L1 penalty.
    
    WHY LASSO:
        Unlike Ridge (which shrinks all weights), Lasso can drive weights to
        EXACTLY ZERO. This means it automatically selects which features matter
        and which are useless. Perfect for our data where 30+ columns might be
        padding zeros from variable-size networks.
    
    WHAT LASSO REVEALS:
        After training, check model.coef_ — any coefficient that is exactly 0.0
        means Lasso decided that feature is worthless for prediction. If ALL GRN
        features survive (non-zero), that's strong evidence they carry real signal.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (higher = more features killed)
          0.001 is gentle, 0.1 is aggressive. We use 0.001 to start soft.
    """
    model = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Count how many features Lasso kept vs killed
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_elasticnet(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Elastic Net — Combines Ridge (L2) and Lasso (L1) penalties.
    
    WHY ELASTIC NET:
        Our GRN features are correlated (e.g., grn_in_degree and grn_n_activators
        overlap — a gene with high in-degree likely has many activators).
        
        - Lasso alone would randomly pick ONE from a group of correlated features
          and kill the rest. This loses information.
        - Ridge alone keeps everything but can't eliminate truly useless features.
        - Elastic Net keeps groups of correlated features together (Ridge behaviour)
          while still killing genuinely useless ones (Lasso behaviour).
    
    WHAT l1_ratio CONTROLS:
        - l1_ratio=0.0 → pure Ridge
        - l1_ratio=1.0 → pure Lasso
        - l1_ratio=0.5 → equal mix (our default)
    
    CUSTOMISABLE PARAMETERS:
        - alpha: overall regularisation strength
        - l1_ratio: balance between L1 and L2 (0.0 to 1.0)
    """
    model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_random_forest(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Random Forest — Ensemble of 100 decision trees.
    
    WHY RANDOM FOREST:
        First non-linear baseline. Can learn rules like "IF mRNA > 0.5 AND
        in_degree > 3 THEN protein is high". Linear models can't learn this.
        Also provides feature_importances_ for free.
    
    CUSTOMISABLE PARAMETERS:
        - n_estimators: number of trees (100 is standard, 200+ for more stability)
        - max_depth: how deep each tree can go (None = unlimited)
        - min_samples_leaf: minimum samples in each leaf (higher = simpler tree)
    """
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_STATE, 
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_mlp(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    MLP (Multi-Layer Perceptron) — D-GEX-style neural network.
    
    WHY MLP:
        This IS the D-GEX baseline. The original D-GEX paper (Chen et al., 2016)
        used an MLP to predict gene expression. We replicate their approach here
        with sklearn for speed, and later with PyTorch for full control.
    
    WHAT WAS CHANGED FROM DEFAULTS (and why):
        - hidden_layer_sizes: (100,) → (128, 64)
          DEFAULT had only 1 hidden layer with 100 neurons.
          CHANGED to 2 layers (128 then 64) to match D-GEX architecture.
          The funnel shape (128→64→1) forces the network to compress information.
        
        - max_iter: 200 → 1000
          DEFAULT cut training short after 200 epochs.
          CHANGED to give the model enough time to converge. Without this,
          the model barely started learning, causing negative R².
        
        - early_stopping: False → True
          DEFAULT trained for all max_iter epochs even if overfitting.
          CHANGED to monitor validation error and stop when it plateaus.
          This is like pulling the cake out of the oven before it burns.
        
        - validation_fraction: 0.1 → 0.15
          15% of training data is held out to monitor overfitting.
        
        - n_iter_no_change: 10 → 20
          Wait 20 epochs without improvement before stopping.
          Being patient helps avoid stopping too early on noisy data.
        
        - learning_rate: 'constant' → 'adaptive'
          DEFAULT kept the same learning rate forever.
          CHANGED to automatically reduce it when loss plateaus.
          Like slowing down as you approach a parking spot — big steps first,
          fine adjustments at the end.
        
        - batch_size: 200 → 64
          DEFAULT processed 200 samples per gradient update.
          CHANGED to 64 for noisier but more frequent updates.
          Helps the model explore more of the loss landscape.
    
    CUSTOMISABLE PARAMETERS TO TUNE:
        - hidden_layer_sizes: try (256, 128), (128, 64, 32), (64, 32)
        - learning_rate_init: try 0.0001, 0.001, 0.01
        - batch_size: try 32, 64, 128
        - alpha: L2 penalty strength (default 0.0001)
    """
    # MLP requires scaled features — without this, large-valued columns dominate
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=64,
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (stopped at epoch {model.n_iter_})")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("PER-GENE BASELINE EXPERIMENTS")
    print("=" * 70)
    
    # 1. Load data
    X_expr, X_full, y, meta, feature_names = load_pergene_data()
    
    # 2. Train-test split (same split for all models — fair comparison)
    X_expr_train, X_expr_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X_expr, y, meta, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_full_train, X_full_test = train_test_split(
        X_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )[:2]
    
    # 3. Define all model trainers
    # Each entry: (train_function, display_name)
    model_trainers = [
        (train_ridge,         "Ridge"),
        (train_lasso,         "Lasso"),
        (train_elasticnet,    "ElasticNet"),
        (train_random_forest, "RandomForest"),
        (train_mlp,           "MLP_DGEX"),
    ]
    
    # 4. Run all experiments: expression-only (A) vs expression+GRN (B)
    all_results = []
    all_predictions = {}  # Store for plotting
    rf_full_result = None  # Need this for feature importance
    lasso_full_result = None  # Need this for feature selection analysis
    
    for train_fn, base_name in model_trainers:
        # Experiment A: Expression-only
        print(f"\n--- {base_name} (Expression Only) ---")
        result_a = train_fn(X_expr_train.values, X_expr_test.values, 
                           y_train.values, y_test.values,
                           f"{base_name}_ExprOnly")
        all_results.append(result_a)
        all_predictions[f"{base_name}_ExprOnly"] = result_a["predictions"]
        
        # Experiment B: Expression + GRN
        print(f"--- {base_name} (Expression + GRN) ---")
        result_b = train_fn(X_full_train.values, X_full_test.values,
                           y_train.values, y_test.values,
                           f"{base_name}_ExprGRN")
        all_results.append(result_b)
        all_predictions[f"{base_name}_ExprGRN"] = result_b["predictions"]
        
        # Track special results for analysis
        if base_name == "RandomForest":
            rf_full_result = result_b
        if base_name == "Lasso":
            lasso_full_result = result_b
    
    # 5. Build results DataFrame
    results_df = pd.DataFrame([{
        "model": r["model"], "mse": r["mse"], "r2": r["r2"]
    } for r in all_results])
    
    # Save results table
    results_path = os.path.join(TABLES_DIR, "pergene_baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # 6. Print summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY: Expression-Only vs Expression+GRN")
    print("=" * 70)
    print(f"{'Model':<22} {'MSE (Expr)':<14} {'MSE (GRN)':<14} {'Δ MSE':<12} {'R² (Expr)':<12} {'R² (GRN)':<12}")
    print("-" * 86)
    
    for _, base_name in model_trainers:
        expr_row = results_df[results_df["model"] == f"{base_name}_ExprOnly"].iloc[0]
        grn_row = results_df[results_df["model"] == f"{base_name}_ExprGRN"].iloc[0]
        delta_mse = ((grn_row["mse"] - expr_row["mse"]) / expr_row["mse"]) * 100
        print(f"{base_name:<22} {expr_row['mse']:<14.6f} {grn_row['mse']:<14.6f} {delta_mse:<12.1f}% {expr_row['r2']:<12.4f} {grn_row['r2']:<12.4f}")
    
    # 7. Lasso feature selection analysis
    if lasso_full_result:
        print("\n" + "=" * 70)
        print("LASSO FEATURE SELECTION ANALYSIS")
        print("Which features did Lasso keep vs kill?")
        print("=" * 70)
        lasso_model = lasso_full_result["instance"]
        full_feature_names = feature_names["full"]
        
        for fname, coef in sorted(zip(full_feature_names, lasso_model.coef_), 
                                    key=lambda x: abs(x[1]), reverse=True):
            status = "✓ KEPT" if abs(coef) > 1e-10 else "✗ KILLED"
            if fname.startswith("grn_"):
                print(f"  [GRN] {fname:<30} coef={coef:>10.6f}  {status}")
        
        # Count GRN features that survived
        grn_feature_names = feature_names["grn"]
        grn_coefs = [lasso_model.coef_[full_feature_names.index(g)] for g in grn_feature_names]
        grn_alive = sum(1 for c in grn_coefs if abs(c) > 1e-10)
        print(f"\n  GRN features surviving Lasso: {grn_alive}/{len(grn_feature_names)}")
    
    # 8. Generate all plots using the standalone plotting module
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plotter = PergeneResultsPlotter(FIGURES_DIR)
    
    # Plot 1: Model comparison bar chart (MSE and R² side by side)
    plotter.plot_model_comparison(results_df)
    
    # Plot 2: Predicted vs Actual scatter for each model
    plotter.plot_predicted_vs_actual(all_predictions, y_test.values)
    
    # Plot 3: Residual distributions
    plotter.plot_residual_distributions(all_predictions, y_test.values)
    
    # Plot 4: Feature importance from Random Forest
    if rf_full_result:
        plotter.plot_feature_importance(
            rf_full_result["instance"], 
            feature_names["full"],
            top_n=20
        )
    
    # Plot 5: GRN feature correlation matrix
    plotter.plot_grn_correlation_matrix(
        pd.concat([X_full_train, X_full_test]),
        feature_names["grn"],
        y
    )
    
    # Plot 6: Per-network performance heatmap
    plotter.plot_per_network_performance(
        all_predictions, y_test.values, meta_test
    )
    
    print(f"\n✓ All plots saved to {FIGURES_DIR}")
    print("=" * 70)
    print("DONE. Next step: train_dgex_pytorch.py for the full D-GEX comparison.")
    print("=" * 70)


if __name__ == "__main__":
    main()
````

Now the standalone plotting module:

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\plotting\plot_pergene_results.py
"""
plot_pergene_results.py — Standalone plotting module for per-gene baseline results.

PURPOSE:
    Generates all analysis plots for the per-gene baseline experiments.
    This file is SEPARATE from the training code so that:
    1. Any training script can import and use it
    2. Plots can be regenerated without retraining models
    3. New plot types can be added without touching model code

PLOTS GENERATED:
    1. Model Comparison Bar Chart  — MSE and R² for all models, side by side
    2. Predicted vs Actual Scatter — How close predictions are to reality
    3. Residual Distributions      — Are errors random or systematically biased?
    4. Feature Importance          — Which features does Random Forest rely on?
    5. GRN Correlation Matrix      — How correlated are GRN features with each other?
    6. Per-Network Heatmap         — Which networks are easy/hard to predict?

USAGE:
    from plot_pergene_results import PergeneResultsPlotter
    plotter = PergeneResultsPlotter("path/to/figures")
    plotter.plot_model_comparison(results_df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves files without opening windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class PergeneResultsPlotter:
    """
    All plotting logic for per-gene baseline experiments.
    
    Each method takes raw data (predictions, scores, etc.) and saves a .png file.
    Methods are independent — you can call any one without the others.
    """
    
    def __init__(self, figures_dir: str):
        """
        Args:
            figures_dir: directory where all .png files will be saved
        """
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        
        # Consistent styling across all plots
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })
    
    def _save(self, fig, filename: str):
        """Helper: save figure and close to free memory."""
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {filename}")
    
    # =========================================================================
    # PLOT 1: MODEL COMPARISON BAR CHART
    # =========================================================================
    
    def plot_model_comparison(self, results_df: pd.DataFrame):
        """
        Side-by-side bar chart comparing MSE and R² for all models.
        
        WHY THIS PLOT:
            The single most important plot in the experiment. At a glance, you can
            see whether adding GRN features (darker bars) consistently improves
            over expression-only (lighter bars) across all model types.
        
        WHAT TO LOOK FOR:
            - Dark bars lower than light bars (MSE chart) = GRN helps
            - Dark bars higher than light bars (R² chart) = GRN helps
            - Consistent pattern across ALL models = strong evidence
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract model base names (without _ExprOnly/_ExprGRN suffix)
        base_names = []
        for name in results_df["model"]:
            base = name.replace("_ExprOnly", "").replace("_ExprGRN", "")
            if base not in base_names:
                base_names.append(base)
        
        x = np.arange(len(base_names))
        width = 0.35
        
        for ax, metric, title in zip(axes, ["mse", "r2"], ["MSE (lower is better)", "R² (higher is better)"]):
            expr_vals = [results_df[results_df["model"] == f"{b}_ExprOnly"][metric].values[0] for b in base_names]
            grn_vals = [results_df[results_df["model"] == f"{b}_ExprGRN"][metric].values[0] for b in base_names]
            
            bars1 = ax.bar(x - width/2, expr_vals, width, label="Expression Only", color="#78909C", alpha=0.85)
            bars2 = ax.bar(x + width/2, grn_vals, width, label="Expression + GRN", color="#26A69A", alpha=0.85)
            
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(base_names, rotation=15)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
```


## License: GPL-3.0
https://github.com/cKNUSPeR/i2nca/blob/a13e08a4db2ae0463402a6896862fdc296267e24/i2nca/qctools/visualization.py

```


Great questions. Let me answer each one, then build the solution.

---

## Answers to Your Questions

### Is sklearn MLP the same as D-GEX?
**Yes, conceptually.** D-GEX is just an MLP that predicts gene expression from other genes' expression. The sklearn `MLPRegressor` with `hidden_layer_sizes=(128, 64)` and ReLU activation IS a D-GEX-style model. The only difference is that the original D-GEX paper used a custom implementation with more control over training. So:

- **sklearn MLP** = D-GEX baseline (quick version)
- **PyTorch MLP** = D-GEX baseline (proper version with full control)

Both are D-GEX. Both should give similar results when properly tuned.

### Should we use both sklearn and PyTorch?
**Yes.** Keep both. In your report you can write: *"We implemented the D-GEX architecture in both scikit-learn and PyTorch and confirmed comparable performance, validating our implementation."* This shows rigour.

### File Organisation
Your instinct is correct. Here's my recommended structure:

```
experiments/
├── build_pergene_dataset.py          # Data preparation (already done)
├── train_pergene_baselines.py        # All per-gene models (Ridge, Lasso, ElasticNet, RF, MLP)
├── train_dgex_pytorch.py             # PyTorch D-GEX (Friday's task)
├── generate_synthetic_dataset.py     # Data generation (already done)
├── generate_fixed_size_dataset.py    # Fixed-size data (Friday's task)
└── plotting/
    └── plot_pergene_results.py       # ALL plots in one reusable file
```

The plotting file will be **standalone** — any training script can call it by passing results DataFrames. This avoids duplicating plotting code across model files.

---

Now let me build both files. First the updated training script, then the plotting module.

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\train_pergene_baselines.py
"""
train_pergene_baselines.py — Train and evaluate all per-gene baseline models.

PURPOSE:
    Answer the core research question: "Does adding GRN structural features
    improve gene expression prediction compared to expression-only models?"

MODELS (in order of complexity):
    1. Ridge Regression       — Linear baseline with L2 penalty (shrinks weights)
    2. Lasso Regression       — Linear baseline with L1 penalty (kills useless features)
    3. Elastic Net             — Combines L1 + L2 (best linear baseline for correlated features)
    4. Random Forest           — Non-linear, tree-based, gives feature importance
    5. MLP (sklearn)           — D-GEX-style neural network (quick version)

EACH MODEL IS TESTED TWICE:
    A. Expression-only:  input = [own_mRNA + other genes' mRNA]
    B. Expression + GRN: input = [own_mRNA + other genes' mRNA + 9 GRN features]

    If B beats A consistently → GRN features carry predictive power. QED.

OUTPUT:
    results/tables/pergene_baseline_results.csv   — All scores
    results/figures/*.png                          — All plots (via plot_pergene_results.py)

USAGE:
    python experiments/train_pergene_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plotting"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings for cleaner output

# Import the standalone plotting module
from plot_pergene_results import PergeneResultsPlotter


# =============================================================================
# CONFIGURATION
# =============================================================================
# All paths and settings in one place for easy modification.

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_transsys")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create output directories
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Columns that are NOT features (metadata + target)
META_COLS = ["sample_id", "network_id", "gene_name"]
TARGET_COL = "target_protein"

# GRN feature columns — these are the 9 structural features extracted from network topology.
# They are the "treatment" in our controlled experiment.
GRN_FEATURE_PREFIXES = ["grn_"]

# Test split — 20% held out, never seen during training
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pergene_data():
    """
    Load the per-gene dataset and split into expression-only and full feature sets.
    
    Returns two feature sets:
      - X_expr: only mRNA columns (the "control group")
      - X_full: mRNA + GRN columns (the "treatment group")
      - y: target protein levels
      - meta: sample_id, network_id, gene_name for later analysis
      - feature_names: dict with 'expr' and 'full' feature lists
    """
    path = os.path.join(DATA_DIR, "pergene_dataset.csv")
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"  Dataset shape: {df.shape[0]} samples × {df.shape[1]} columns")
    
    # Separate metadata, features, and target
    meta = df[META_COLS]
    y = df[TARGET_COL]
    
    # All feature columns (everything except metadata and target)
    all_features = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    
    # Split into expression-only and GRN features
    grn_cols = [c for c in all_features if any(c.startswith(p) for p in GRN_FEATURE_PREFIXES)]
    expr_cols = [c for c in all_features if c not in grn_cols]
    
    print(f"  Expression features: {len(expr_cols)}")
    print(f"  GRN features: {len(grn_cols)} → {grn_cols}")
    
    X_expr = df[expr_cols]
    X_full = df[expr_cols + grn_cols]
    
    feature_names = {"expr": expr_cols, "full": expr_cols + grn_cols, "grn": grn_cols}
    
    return X_expr, X_full, y, meta, feature_names


# =============================================================================
# MODEL DEFINITIONS
# Each function creates, trains, and evaluates one model.
# All follow the same interface: (X_train, X_test, y_train, y_test, name) → dict
# =============================================================================

def train_ridge(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Ridge Regression — Linear model with L2 penalty.
    
    WHY RIDGE:
        The simplest possible baseline. If a neural network can't beat this,
        something is wrong with the neural network. Ridge is the "floor".
    
    WHAT alpha DOES:
        alpha=1.0 controls how much we penalise large weights.
        Higher alpha = more conservative model (smaller weights, less overfitting).
        Lower alpha = closer to standard linear regression.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (try 0.1, 1.0, 10.0)
    """
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_lasso(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Lasso Regression — Linear model with L1 penalty.
    
    WHY LASSO:
        Unlike Ridge (which shrinks all weights), Lasso can drive weights to
        EXACTLY ZERO. This means it automatically selects which features matter
        and which are useless. Perfect for our data where 30+ columns might be
        padding zeros from variable-size networks.
    
    WHAT LASSO REVEALS:
        After training, check model.coef_ — any coefficient that is exactly 0.0
        means Lasso decided that feature is worthless for prediction. If ALL GRN
        features survive (non-zero), that's strong evidence they carry real signal.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (higher = more features killed)
          0.001 is gentle, 0.1 is aggressive. We use 0.001 to start soft.
    """
    model = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Count how many features Lasso kept vs killed
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_elasticnet(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Elastic Net — Combines Ridge (L2) and Lasso (L1) penalties.
    
    WHY ELASTIC NET:
        Our GRN features are correlated (e.g., grn_in_degree and grn_n_activators
        overlap — a gene with high in-degree likely has many activators).
        
        - Lasso alone would randomly pick ONE from a group of correlated features
          and kill the rest. This loses information.
        - Ridge alone keeps everything but can't eliminate truly useless features.
        - Elastic Net keeps groups of correlated features together (Ridge behaviour)
          while still killing genuinely useless ones (Lasso behaviour).
    
    WHAT l1_ratio CONTROLS:
        - l1_ratio=0.0 → pure Ridge
        - l1_ratio=1.0 → pure Lasso
        - l1_ratio=0.5 → equal mix (our default)
    
    CUSTOMISABLE PARAMETERS:
        - alpha: overall regularisation strength
        - l1_ratio: balance between L1 and L2 (0.0 to 1.0)
    """
    model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_random_forest(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Random Forest — Ensemble of 100 decision trees.
    
    WHY RANDOM FOREST:
        First non-linear baseline. Can learn rules like "IF mRNA > 0.5 AND
        in_degree > 3 THEN protein is high". Linear models can't learn this.
        Also provides feature_importances_ for free.
    
    CUSTOMISABLE PARAMETERS:
        - n_estimators: number of trees (100 is standard, 200+ for more stability)
        - max_depth: how deep each tree can go (None = unlimited)
        - min_samples_leaf: minimum samples in each leaf (higher = simpler tree)
    """
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_STATE, 
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_mlp(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    MLP (Multi-Layer Perceptron) — D-GEX-style neural network.
    
    WHY MLP:
        This IS the D-GEX baseline. The original D-GEX paper (Chen et al., 2016)
        used an MLP to predict gene expression. We replicate their approach here
        with sklearn for speed, and later with PyTorch for full control.
    
    WHAT WAS CHANGED FROM DEFAULTS (and why):
        - hidden_layer_sizes: (100,) → (128, 64)
          DEFAULT had only 1 hidden layer with 100 neurons.
          CHANGED to 2 layers (128 then 64) to match D-GEX architecture.
          The funnel shape (128→64→1) forces the network to compress information.
        
        - max_iter: 200 → 1000
          DEFAULT cut training short after 200 epochs.
          CHANGED to give the model enough time to converge. Without this,
          the model barely started learning, causing negative R².
        
        - early_stopping: False → True
          DEFAULT trained for all max_iter epochs even if overfitting.
          CHANGED to monitor validation error and stop when it plateaus.
          This is like pulling the cake out of the oven before it burns.
        
        - validation_fraction: 0.1 → 0.15
          15% of training data is held out to monitor overfitting.
        
        - n_iter_no_change: 10 → 20
          Wait 20 epochs without improvement before stopping.
          Being patient helps avoid stopping too early on noisy data.
        
        - learning_rate: 'constant' → 'adaptive'
          DEFAULT kept the same learning rate forever.
          CHANGED to automatically reduce it when loss plateaus.
          Like slowing down as you approach a parking spot — big steps first,
          fine adjustments at the end.
        
        - batch_size: 200 → 64
          DEFAULT processed 200 samples per gradient update.
          CHANGED to 64 for noisier but more frequent updates.
          Helps the model explore more of the loss landscape.
    
    CUSTOMISABLE PARAMETERS TO TUNE:
        - hidden_layer_sizes: try (256, 128), (128, 64, 32), (64, 32)
        - learning_rate_init: try 0.0001, 0.001, 0.01
        - batch_size: try 32, 64, 128
        - alpha: L2 penalty strength (default 0.0001)
    """
    # MLP requires scaled features — without this, large-valued columns dominate
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=64,
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (stopped at epoch {model.n_iter_})")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("PER-GENE BASELINE EXPERIMENTS")
    print("=" * 70)
    
    # 1. Load data
    X_expr, X_full, y, meta, feature_names = load_pergene_data()
    
    # 2. Train-test split (same split for all models — fair comparison)
    X_expr_train, X_expr_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X_expr, y, meta, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_full_train, X_full_test = train_test_split(
        X_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )[:2]
    
    # 3. Define all model trainers
    # Each entry: (train_function, display_name)
    model_trainers = [
        (train_ridge,         "Ridge"),
        (train_lasso,         "Lasso"),
        (train_elasticnet,    "ElasticNet"),
        (train_random_forest, "RandomForest"),
        (train_mlp,           "MLP_DGEX"),
    ]
    
    # 4. Run all experiments: expression-only (A) vs expression+GRN (B)
    all_results = []
    all_predictions = {}  # Store for plotting
    rf_full_result = None  # Need this for feature importance
    lasso_full_result = None  # Need this for feature selection analysis
    
    for train_fn, base_name in model_trainers:
        # Experiment A: Expression-only
        print(f"\n--- {base_name} (Expression Only) ---")
        result_a = train_fn(X_expr_train.values, X_expr_test.values, 
                           y_train.values, y_test.values,
                           f"{base_name}_ExprOnly")
        all_results.append(result_a)
        all_predictions[f"{base_name}_ExprOnly"] = result_a["predictions"]
        
        # Experiment B: Expression + GRN
        print(f"--- {base_name} (Expression + GRN) ---")
        result_b = train_fn(X_full_train.values, X_full_test.values,
                           y_train.values, y_test.values,
                           f"{base_name}_ExprGRN")
        all_results.append(result_b)
        all_predictions[f"{base_name}_ExprGRN"] = result_b["predictions"]
        
        # Track special results for analysis
        if base_name == "RandomForest":
            rf_full_result = result_b
        if base_name == "Lasso":
            lasso_full_result = result_b
    
    # 5. Build results DataFrame
    results_df = pd.DataFrame([{
        "model": r["model"], "mse": r["mse"], "r2": r["r2"]
    } for r in all_results])
    
    # Save results table
    results_path = os.path.join(TABLES_DIR, "pergene_baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # 6. Print summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY: Expression-Only vs Expression+GRN")
    print("=" * 70)
    print(f"{'Model':<22} {'MSE (Expr)':<14} {'MSE (GRN)':<14} {'Δ MSE':<12} {'R² (Expr)':<12} {'R² (GRN)':<12}")
    print("-" * 86)
    
    for _, base_name in model_trainers:
        expr_row = results_df[results_df["model"] == f"{base_name}_ExprOnly"].iloc[0]
        grn_row = results_df[results_df["model"] == f"{base_name}_ExprGRN"].iloc[0]
        delta_mse = ((grn_row["mse"] - expr_row["mse"]) / expr_row["mse"]) * 100
        print(f"{base_name:<22} {expr_row['mse']:<14.6f} {grn_row['mse']:<14.6f} {delta_mse:<12.1f}% {expr_row['r2']:<12.4f} {grn_row['r2']:<12.4f}")
    
    # 7. Lasso feature selection analysis
    if lasso_full_result:
        print("\n" + "=" * 70)
        print("LASSO FEATURE SELECTION ANALYSIS")
        print("Which features did Lasso keep vs kill?")
        print("=" * 70)
        lasso_model = lasso_full_result["instance"]
        full_feature_names = feature_names["full"]
        
        for fname, coef in sorted(zip(full_feature_names, lasso_model.coef_), 
                                    key=lambda x: abs(x[1]), reverse=True):
            status = "✓ KEPT" if abs(coef) > 1e-10 else "✗ KILLED"
            if fname.startswith("grn_"):
                print(f"  [GRN] {fname:<30} coef={coef:>10.6f}  {status}")
        
        # Count GRN features that survived
        grn_feature_names = feature_names["grn"]
        grn_coefs = [lasso_model.coef_[full_feature_names.index(g)] for g in grn_feature_names]
        grn_alive = sum(1 for c in grn_coefs if abs(c) > 1e-10)
        print(f"\n  GRN features surviving Lasso: {grn_alive}/{len(grn_feature_names)}")
    
    # 8. Generate all plots using the standalone plotting module
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plotter = PergeneResultsPlotter(FIGURES_DIR)
    
    # Plot 1: Model comparison bar chart (MSE and R² side by side)
    plotter.plot_model_comparison(results_df)
    
    # Plot 2: Predicted vs Actual scatter for each model
    plotter.plot_predicted_vs_actual(all_predictions, y_test.values)
    
    # Plot 3: Residual distributions
    plotter.plot_residual_distributions(all_predictions, y_test.values)
    
    # Plot 4: Feature importance from Random Forest
    if rf_full_result:
        plotter.plot_feature_importance(
            rf_full_result["instance"], 
            feature_names["full"],
            top_n=20
        )
    
    # Plot 5: GRN feature correlation matrix
    plotter.plot_grn_correlation_matrix(
        pd.concat([X_full_train, X_full_test]),
        feature_names["grn"],
        y
    )
    
    # Plot 6: Per-network performance heatmap
    plotter.plot_per_network_performance(
        all_predictions, y_test.values, meta_test
    )
    
    print(f"\n✓ All plots saved to {FIGURES_DIR}")
    print("=" * 70)
    print("DONE. Next step: train_dgex_pytorch.py for the full D-GEX comparison.")
    print("=" * 70)


if __name__ == "__main__":
    main()
````

Now the standalone plotting module:

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\plotting\plot_pergene_results.py
"""
plot_pergene_results.py — Standalone plotting module for per-gene baseline results.

PURPOSE:
    Generates all analysis plots for the per-gene baseline experiments.
    This file is SEPARATE from the training code so that:
    1. Any training script can import and use it
    2. Plots can be regenerated without retraining models
    3. New plot types can be added without touching model code

PLOTS GENERATED:
    1. Model Comparison Bar Chart  — MSE and R² for all models, side by side
    2. Predicted vs Actual Scatter — How close predictions are to reality
    3. Residual Distributions      — Are errors random or systematically biased?
    4. Feature Importance          — Which features does Random Forest rely on?
    5. GRN Correlation Matrix      — How correlated are GRN features with each other?
    6. Per-Network Heatmap         — Which networks are easy/hard to predict?

USAGE:
    from plot_pergene_results import PergeneResultsPlotter
    plotter = PergeneResultsPlotter("path/to/figures")
    plotter.plot_model_comparison(results_df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves files without opening windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class PergeneResultsPlotter:
    """
    All plotting logic for per-gene baseline experiments.
    
    Each method takes raw data (predictions, scores, etc.) and saves a .png file.
    Methods are independent — you can call any one without the others.
    """
    
    def __init__(self, figures_dir: str):
        """
        Args:
            figures_dir: directory where all .png files will be saved
        """
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        
        # Consistent styling across all plots
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })
    
    def _save(self, fig, filename: str):
        """Helper: save figure and close to free memory."""
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {filename}")
    
    # =========================================================================
    # PLOT 1: MODEL COMPARISON BAR CHART
    # =========================================================================
    
    def plot_model_comparison(self, results_df: pd.DataFrame):
        """
        Side-by-side bar chart comparing MSE and R² for all models.
        
        WHY THIS PLOT:
            The single most important plot in the experiment. At a glance, you can
            see whether adding GRN features (darker bars) consistently improves
            over expression-only (lighter bars) across all model types.
        
        WHAT TO LOOK FOR:
            - Dark bars lower than light bars (MSE chart) = GRN helps
            - Dark bars higher than light bars (R² chart) = GRN helps
            - Consistent pattern across ALL models = strong evidence
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract model base names (without _ExprOnly/_ExprGRN suffix)
        base_names = []
        for name in results_df["model"]:
            base = name.replace("_ExprOnly", "").replace("_ExprGRN", "")
            if base not in base_names:
                base_names.append(base)
        
        x = np.arange(len(base_names))
        width = 0.35
        
        for ax, metric, title in zip(axes, ["mse", "r2"], ["MSE (lower is better)", "R² (higher is better)"]):
            expr_vals = [results_df[results_df["model"] == f"{b}_ExprOnly"][metric].values[0] for b in base_names]
            grn_vals = [results_df[results_df["model"] == f"{b}_ExprGRN"][metric].values[0] for b in base_names]
            
            bars1 = ax.bar(x - width/2, expr_vals, width, label="Expression Only", color="#78909C", alpha=0.85)
            bars2 = ax.bar(x + width/2, grn_vals, width, label="Expression + GRN", color="#26A69A", alpha=0.85)
            
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(base_names, rotation=15)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
```


## License: GPL-3.0
https://github.com/cKNUSPeR/i2nca/blob/a13e08a4db2ae0463402a6896862fdc296267e24/i2nca/qctools/visualization.py

```


Great questions. Let me answer each one, then build the solution.

---

## Answers to Your Questions

### Is sklearn MLP the same as D-GEX?
**Yes, conceptually.** D-GEX is just an MLP that predicts gene expression from other genes' expression. The sklearn `MLPRegressor` with `hidden_layer_sizes=(128, 64)` and ReLU activation IS a D-GEX-style model. The only difference is that the original D-GEX paper used a custom implementation with more control over training. So:

- **sklearn MLP** = D-GEX baseline (quick version)
- **PyTorch MLP** = D-GEX baseline (proper version with full control)

Both are D-GEX. Both should give similar results when properly tuned.

### Should we use both sklearn and PyTorch?
**Yes.** Keep both. In your report you can write: *"We implemented the D-GEX architecture in both scikit-learn and PyTorch and confirmed comparable performance, validating our implementation."* This shows rigour.

### File Organisation
Your instinct is correct. Here's my recommended structure:

```
experiments/
├── build_pergene_dataset.py          # Data preparation (already done)
├── train_pergene_baselines.py        # All per-gene models (Ridge, Lasso, ElasticNet, RF, MLP)
├── train_dgex_pytorch.py             # PyTorch D-GEX (Friday's task)
├── generate_synthetic_dataset.py     # Data generation (already done)
├── generate_fixed_size_dataset.py    # Fixed-size data (Friday's task)
└── plotting/
    └── plot_pergene_results.py       # ALL plots in one reusable file
```

The plotting file will be **standalone** — any training script can call it by passing results DataFrames. This avoids duplicating plotting code across model files.

---

Now let me build both files. First the updated training script, then the plotting module.

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\train_pergene_baselines.py
"""
train_pergene_baselines.py — Train and evaluate all per-gene baseline models.

PURPOSE:
    Answer the core research question: "Does adding GRN structural features
    improve gene expression prediction compared to expression-only models?"

MODELS (in order of complexity):
    1. Ridge Regression       — Linear baseline with L2 penalty (shrinks weights)
    2. Lasso Regression       — Linear baseline with L1 penalty (kills useless features)
    3. Elastic Net             — Combines L1 + L2 (best linear baseline for correlated features)
    4. Random Forest           — Non-linear, tree-based, gives feature importance
    5. MLP (sklearn)           — D-GEX-style neural network (quick version)

EACH MODEL IS TESTED TWICE:
    A. Expression-only:  input = [own_mRNA + other genes' mRNA]
    B. Expression + GRN: input = [own_mRNA + other genes' mRNA + 9 GRN features]

    If B beats A consistently → GRN features carry predictive power. QED.

OUTPUT:
    results/tables/pergene_baseline_results.csv   — All scores
    results/figures/*.png                          — All plots (via plot_pergene_results.py)

USAGE:
    python experiments/train_pergene_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plotting"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings for cleaner output

# Import the standalone plotting module
from plot_pergene_results import PergeneResultsPlotter


# =============================================================================
# CONFIGURATION
# =============================================================================
# All paths and settings in one place for easy modification.

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_transsys")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create output directories
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Columns that are NOT features (metadata + target)
META_COLS = ["sample_id", "network_id", "gene_name"]
TARGET_COL = "target_protein"

# GRN feature columns — these are the 9 structural features extracted from network topology.
# They are the "treatment" in our controlled experiment.
GRN_FEATURE_PREFIXES = ["grn_"]

# Test split — 20% held out, never seen during training
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pergene_data():
    """
    Load the per-gene dataset and split into expression-only and full feature sets.
    
    Returns two feature sets:
      - X_expr: only mRNA columns (the "control group")
      - X_full: mRNA + GRN columns (the "treatment group")
      - y: target protein levels
      - meta: sample_id, network_id, gene_name for later analysis
      - feature_names: dict with 'expr' and 'full' feature lists
    """
    path = os.path.join(DATA_DIR, "pergene_dataset.csv")
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"  Dataset shape: {df.shape[0]} samples × {df.shape[1]} columns")
    
    # Separate metadata, features, and target
    meta = df[META_COLS]
    y = df[TARGET_COL]
    
    # All feature columns (everything except metadata and target)
    all_features = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    
    # Split into expression-only and GRN features
    grn_cols = [c for c in all_features if any(c.startswith(p) for p in GRN_FEATURE_PREFIXES)]
    expr_cols = [c for c in all_features if c not in grn_cols]
    
    print(f"  Expression features: {len(expr_cols)}")
    print(f"  GRN features: {len(grn_cols)} → {grn_cols}")
    
    X_expr = df[expr_cols]
    X_full = df[expr_cols + grn_cols]
    
    feature_names = {"expr": expr_cols, "full": expr_cols + grn_cols, "grn": grn_cols}
    
    return X_expr, X_full, y, meta, feature_names


# =============================================================================
# MODEL DEFINITIONS
# Each function creates, trains, and evaluates one model.
# All follow the same interface: (X_train, X_test, y_train, y_test, name) → dict
# =============================================================================

def train_ridge(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Ridge Regression — Linear model with L2 penalty.
    
    WHY RIDGE:
        The simplest possible baseline. If a neural network can't beat this,
        something is wrong with the neural network. Ridge is the "floor".
    
    WHAT alpha DOES:
        alpha=1.0 controls how much we penalise large weights.
        Higher alpha = more conservative model (smaller weights, less overfitting).
        Lower alpha = closer to standard linear regression.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (try 0.1, 1.0, 10.0)
    """
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_lasso(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Lasso Regression — Linear model with L1 penalty.
    
    WHY LASSO:
        Unlike Ridge (which shrinks all weights), Lasso can drive weights to
        EXACTLY ZERO. This means it automatically selects which features matter
        and which are useless. Perfect for our data where 30+ columns might be
        padding zeros from variable-size networks.
    
    WHAT LASSO REVEALS:
        After training, check model.coef_ — any coefficient that is exactly 0.0
        means Lasso decided that feature is worthless for prediction. If ALL GRN
        features survive (non-zero), that's strong evidence they carry real signal.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (higher = more features killed)
          0.001 is gentle, 0.1 is aggressive. We use 0.001 to start soft.
    """
    model = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Count how many features Lasso kept vs killed
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_elasticnet(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Elastic Net — Combines Ridge (L2) and Lasso (L1) penalties.
    
    WHY ELASTIC NET:
        Our GRN features are correlated (e.g., grn_in_degree and grn_n_activators
        overlap — a gene with high in-degree likely has many activators).
        
        - Lasso alone would randomly pick ONE from a group of correlated features
          and kill the rest. This loses information.
        - Ridge alone keeps everything but can't eliminate truly useless features.
        - Elastic Net keeps groups of correlated features together (Ridge behaviour)
          while still killing genuinely useless ones (Lasso behaviour).
    
    WHAT l1_ratio CONTROLS:
        - l1_ratio=0.0 → pure Ridge
        - l1_ratio=1.0 → pure Lasso
        - l1_ratio=0.5 → equal mix (our default)
    
    CUSTOMISABLE PARAMETERS:
        - alpha: overall regularisation strength
        - l1_ratio: balance between L1 and L2 (0.0 to 1.0)
    """
    model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_random_forest(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Random Forest — Ensemble of 100 decision trees.
    
    WHY RANDOM FOREST:
        First non-linear baseline. Can learn rules like "IF mRNA > 0.5 AND
        in_degree > 3 THEN protein is high". Linear models can't learn this.
        Also provides feature_importances_ for free.
    
    CUSTOMISABLE PARAMETERS:
        - n_estimators: number of trees (100 is standard, 200+ for more stability)
        - max_depth: how deep each tree can go (None = unlimited)
        - min_samples_leaf: minimum samples in each leaf (higher = simpler tree)
    """
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_STATE, 
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_mlp(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    MLP (Multi-Layer Perceptron) — D-GEX-style neural network.
    
    WHY MLP:
        This IS the D-GEX baseline. The original D-GEX paper (Chen et al., 2016)
        used an MLP to predict gene expression. We replicate their approach here
        with sklearn for speed, and later with PyTorch for full control.
    
    WHAT WAS CHANGED FROM DEFAULTS (and why):
        - hidden_layer_sizes: (100,) → (128, 64)
          DEFAULT had only 1 hidden layer with 100 neurons.
          CHANGED to 2 layers (128 then 64) to match D-GEX architecture.
          The funnel shape (128→64→1) forces the network to compress information.
        
        - max_iter: 200 → 1000
          DEFAULT cut training short after 200 epochs.
          CHANGED to give the model enough time to converge. Without this,
          the model barely started learning, causing negative R².
        
        - early_stopping: False → True
          DEFAULT trained for all max_iter epochs even if overfitting.
          CHANGED to monitor validation error and stop when it plateaus.
          This is like pulling the cake out of the oven before it burns.
        
        - validation_fraction: 0.1 → 0.15
          15% of training data is held out to monitor overfitting.
        
        - n_iter_no_change: 10 → 20
          Wait 20 epochs without improvement before stopping.
          Being patient helps avoid stopping too early on noisy data.
        
        - learning_rate: 'constant' → 'adaptive'
          DEFAULT kept the same learning rate forever.
          CHANGED to automatically reduce it when loss plateaus.
          Like slowing down as you approach a parking spot — big steps first,
          fine adjustments at the end.
        
        - batch_size: 200 → 64
          DEFAULT processed 200 samples per gradient update.
          CHANGED to 64 for noisier but more frequent updates.
          Helps the model explore more of the loss landscape.
    
    CUSTOMISABLE PARAMETERS TO TUNE:
        - hidden_layer_sizes: try (256, 128), (128, 64, 32), (64, 32)
        - learning_rate_init: try 0.0001, 0.001, 0.01
        - batch_size: try 32, 64, 128
        - alpha: L2 penalty strength (default 0.0001)
    """
    # MLP requires scaled features — without this, large-valued columns dominate
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=64,
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (stopped at epoch {model.n_iter_})")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("PER-GENE BASELINE EXPERIMENTS")
    print("=" * 70)
    
    # 1. Load data
    X_expr, X_full, y, meta, feature_names = load_pergene_data()
    
    # 2. Train-test split (same split for all models — fair comparison)
    X_expr_train, X_expr_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X_expr, y, meta, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_full_train, X_full_test = train_test_split(
        X_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )[:2]
    
    # 3. Define all model trainers
    # Each entry: (train_function, display_name)
    model_trainers = [
        (train_ridge,         "Ridge"),
        (train_lasso,         "Lasso"),
        (train_elasticnet,    "ElasticNet"),
        (train_random_forest, "RandomForest"),
        (train_mlp,           "MLP_DGEX"),
    ]
    
    # 4. Run all experiments: expression-only (A) vs expression+GRN (B)
    all_results = []
    all_predictions = {}  # Store for plotting
    rf_full_result = None  # Need this for feature importance
    lasso_full_result = None  # Need this for feature selection analysis
    
    for train_fn, base_name in model_trainers:
        # Experiment A: Expression-only
        print(f"\n--- {base_name} (Expression Only) ---")
        result_a = train_fn(X_expr_train.values, X_expr_test.values, 
                           y_train.values, y_test.values,
                           f"{base_name}_ExprOnly")
        all_results.append(result_a)
        all_predictions[f"{base_name}_ExprOnly"] = result_a["predictions"]
        
        # Experiment B: Expression + GRN
        print(f"--- {base_name} (Expression + GRN) ---")
        result_b = train_fn(X_full_train.values, X_full_test.values,
                           y_train.values, y_test.values,
                           f"{base_name}_ExprGRN")
        all_results.append(result_b)
        all_predictions[f"{base_name}_ExprGRN"] = result_b["predictions"]
        
        # Track special results for analysis
        if base_name == "RandomForest":
            rf_full_result = result_b
        if base_name == "Lasso":
            lasso_full_result = result_b
    
    # 5. Build results DataFrame
    results_df = pd.DataFrame([{
        "model": r["model"], "mse": r["mse"], "r2": r["r2"]
    } for r in all_results])
    
    # Save results table
    results_path = os.path.join(TABLES_DIR, "pergene_baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # 6. Print summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY: Expression-Only vs Expression+GRN")
    print("=" * 70)
    print(f"{'Model':<22} {'MSE (Expr)':<14} {'MSE (GRN)':<14} {'Δ MSE':<12} {'R² (Expr)':<12} {'R² (GRN)':<12}")
    print("-" * 86)
    
    for _, base_name in model_trainers:
        expr_row = results_df[results_df["model"] == f"{base_name}_ExprOnly"].iloc[0]
        grn_row = results_df[results_df["model"] == f"{base_name}_ExprGRN"].iloc[0]
        delta_mse = ((grn_row["mse"] - expr_row["mse"]) / expr_row["mse"]) * 100
        print(f"{base_name:<22} {expr_row['mse']:<14.6f} {grn_row['mse']:<14.6f} {delta_mse:<12.1f}% {expr_row['r2']:<12.4f} {grn_row['r2']:<12.4f}")
    
    # 7. Lasso feature selection analysis
    if lasso_full_result:
        print("\n" + "=" * 70)
        print("LASSO FEATURE SELECTION ANALYSIS")
        print("Which features did Lasso keep vs kill?")
        print("=" * 70)
        lasso_model = lasso_full_result["instance"]
        full_feature_names = feature_names["full"]
        
        for fname, coef in sorted(zip(full_feature_names, lasso_model.coef_), 
                                    key=lambda x: abs(x[1]), reverse=True):
            status = "✓ KEPT" if abs(coef) > 1e-10 else "✗ KILLED"
            if fname.startswith("grn_"):
                print(f"  [GRN] {fname:<30} coef={coef:>10.6f}  {status}")
        
        # Count GRN features that survived
        grn_feature_names = feature_names["grn"]
        grn_coefs = [lasso_model.coef_[full_feature_names.index(g)] for g in grn_feature_names]
        grn_alive = sum(1 for c in grn_coefs if abs(c) > 1e-10)
        print(f"\n  GRN features surviving Lasso: {grn_alive}/{len(grn_feature_names)}")
    
    # 8. Generate all plots using the standalone plotting module
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plotter = PergeneResultsPlotter(FIGURES_DIR)
    
    # Plot 1: Model comparison bar chart (MSE and R² side by side)
    plotter.plot_model_comparison(results_df)
    
    # Plot 2: Predicted vs Actual scatter for each model
    plotter.plot_predicted_vs_actual(all_predictions, y_test.values)
    
    # Plot 3: Residual distributions
    plotter.plot_residual_distributions(all_predictions, y_test.values)
    
    # Plot 4: Feature importance from Random Forest
    if rf_full_result:
        plotter.plot_feature_importance(
            rf_full_result["instance"], 
            feature_names["full"],
            top_n=20
        )
    
    # Plot 5: GRN feature correlation matrix
    plotter.plot_grn_correlation_matrix(
        pd.concat([X_full_train, X_full_test]),
        feature_names["grn"],
        y
    )
    
    # Plot 6: Per-network performance heatmap
    plotter.plot_per_network_performance(
        all_predictions, y_test.values, meta_test
    )
    
    print(f"\n✓ All plots saved to {FIGURES_DIR}")
    print("=" * 70)
    print("DONE. Next step: train_dgex_pytorch.py for the full D-GEX comparison.")
    print("=" * 70)


if __name__ == "__main__":
    main()
````

Now the standalone plotting module:

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\plotting\plot_pergene_results.py
"""
plot_pergene_results.py — Standalone plotting module for per-gene baseline results.

PURPOSE:
    Generates all analysis plots for the per-gene baseline experiments.
    This file is SEPARATE from the training code so that:
    1. Any training script can import and use it
    2. Plots can be regenerated without retraining models
    3. New plot types can be added without touching model code

PLOTS GENERATED:
    1. Model Comparison Bar Chart  — MSE and R² for all models, side by side
    2. Predicted vs Actual Scatter — How close predictions are to reality
    3. Residual Distributions      — Are errors random or systematically biased?
    4. Feature Importance          — Which features does Random Forest rely on?
    5. GRN Correlation Matrix      — How correlated are GRN features with each other?
    6. Per-Network Heatmap         — Which networks are easy/hard to predict?

USAGE:
    from plot_pergene_results import PergeneResultsPlotter
    plotter = PergeneResultsPlotter("path/to/figures")
    plotter.plot_model_comparison(results_df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves files without opening windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class PergeneResultsPlotter:
    """
    All plotting logic for per-gene baseline experiments.
    
    Each method takes raw data (predictions, scores, etc.) and saves a .png file.
    Methods are independent — you can call any one without the others.
    """
    
    def __init__(self, figures_dir: str):
        """
        Args:
            figures_dir: directory where all .png files will be saved
        """
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        
        # Consistent styling across all plots
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })
    
    def _save(self, fig, filename: str):
        """Helper: save figure and close to free memory."""
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {filename}")
    
    # =========================================================================
    # PLOT 1: MODEL COMPARISON BAR CHART
    # =========================================================================
    
    def plot_model_comparison(self, results_df: pd.DataFrame):
        """
        Side-by-side bar chart comparing MSE and R² for all models.
        
        WHY THIS PLOT:
            The single most important plot in the experiment. At a glance, you can
            see whether adding GRN features (darker bars) consistently improves
            over expression-only (lighter bars) across all model types.
        
        WHAT TO LOOK FOR:
            - Dark bars lower than light bars (MSE chart) = GRN helps
            - Dark bars higher than light bars (R² chart) = GRN helps
            - Consistent pattern across ALL models = strong evidence
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract model base names (without _ExprOnly/_ExprGRN suffix)
        base_names = []
        for name in results_df["model"]:
            base = name.replace("_ExprOnly", "").replace("_ExprGRN", "")
            if base not in base_names:
                base_names.append(base)
        
        x = np.arange(len(base_names))
        width = 0.35
        
        for ax, metric, title in zip(axes, ["mse", "r2"], ["MSE (lower is better)", "R² (higher is better)"]):
            expr_vals = [results_df[results_df["model"] == f"{b}_ExprOnly"][metric].values[0] for b in base_names]
            grn_vals = [results_df[results_df["model"] == f"{b}_ExprGRN"][metric].values[0] for b in base_names]
            
            bars1 = ax.bar(x - width/2, expr_vals, width, label="Expression Only", color="#78909C", alpha=0.85)
            bars2 = ax.bar(x + width/2, grn_vals, width, label="Expression + GRN", color="#26A69A", alpha=0.85)
            
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(base_names, rotation=15)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
```


## License: GPL-3.0
https://github.com/cKNUSPeR/i2nca/blob/a13e08a4db2ae0463402a6896862fdc296267e24/i2nca/qctools/visualization.py

```


Great questions. Let me answer each one, then build the solution.

---

## Answers to Your Questions

### Is sklearn MLP the same as D-GEX?
**Yes, conceptually.** D-GEX is just an MLP that predicts gene expression from other genes' expression. The sklearn `MLPRegressor` with `hidden_layer_sizes=(128, 64)` and ReLU activation IS a D-GEX-style model. The only difference is that the original D-GEX paper used a custom implementation with more control over training. So:

- **sklearn MLP** = D-GEX baseline (quick version)
- **PyTorch MLP** = D-GEX baseline (proper version with full control)

Both are D-GEX. Both should give similar results when properly tuned.

### Should we use both sklearn and PyTorch?
**Yes.** Keep both. In your report you can write: *"We implemented the D-GEX architecture in both scikit-learn and PyTorch and confirmed comparable performance, validating our implementation."* This shows rigour.

### File Organisation
Your instinct is correct. Here's my recommended structure:

```
experiments/
├── build_pergene_dataset.py          # Data preparation (already done)
├── train_pergene_baselines.py        # All per-gene models (Ridge, Lasso, ElasticNet, RF, MLP)
├── train_dgex_pytorch.py             # PyTorch D-GEX (Friday's task)
├── generate_synthetic_dataset.py     # Data generation (already done)
├── generate_fixed_size_dataset.py    # Fixed-size data (Friday's task)
└── plotting/
    └── plot_pergene_results.py       # ALL plots in one reusable file
```

The plotting file will be **standalone** — any training script can call it by passing results DataFrames. This avoids duplicating plotting code across model files.

---

Now let me build both files. First the updated training script, then the plotting module.

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\train_pergene_baselines.py
"""
train_pergene_baselines.py — Train and evaluate all per-gene baseline models.

PURPOSE:
    Answer the core research question: "Does adding GRN structural features
    improve gene expression prediction compared to expression-only models?"

MODELS (in order of complexity):
    1. Ridge Regression       — Linear baseline with L2 penalty (shrinks weights)
    2. Lasso Regression       — Linear baseline with L1 penalty (kills useless features)
    3. Elastic Net             — Combines L1 + L2 (best linear baseline for correlated features)
    4. Random Forest           — Non-linear, tree-based, gives feature importance
    5. MLP (sklearn)           — D-GEX-style neural network (quick version)

EACH MODEL IS TESTED TWICE:
    A. Expression-only:  input = [own_mRNA + other genes' mRNA]
    B. Expression + GRN: input = [own_mRNA + other genes' mRNA + 9 GRN features]

    If B beats A consistently → GRN features carry predictive power. QED.

OUTPUT:
    results/tables/pergene_baseline_results.csv   — All scores
    results/figures/*.png                          — All plots (via plot_pergene_results.py)

USAGE:
    python experiments/train_pergene_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plotting"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings for cleaner output

# Import the standalone plotting module
from plot_pergene_results import PergeneResultsPlotter


# =============================================================================
# CONFIGURATION
# =============================================================================
# All paths and settings in one place for easy modification.

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_transsys")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create output directories
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Columns that are NOT features (metadata + target)
META_COLS = ["sample_id", "network_id", "gene_name"]
TARGET_COL = "target_protein"

# GRN feature columns — these are the 9 structural features extracted from network topology.
# They are the "treatment" in our controlled experiment.
GRN_FEATURE_PREFIXES = ["grn_"]

# Test split — 20% held out, never seen during training
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pergene_data():
    """
    Load the per-gene dataset and split into expression-only and full feature sets.
    
    Returns two feature sets:
      - X_expr: only mRNA columns (the "control group")
      - X_full: mRNA + GRN columns (the "treatment group")
      - y: target protein levels
      - meta: sample_id, network_id, gene_name for later analysis
      - feature_names: dict with 'expr' and 'full' feature lists
    """
    path = os.path.join(DATA_DIR, "pergene_dataset.csv")
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"  Dataset shape: {df.shape[0]} samples × {df.shape[1]} columns")
    
    # Separate metadata, features, and target
    meta = df[META_COLS]
    y = df[TARGET_COL]
    
    # All feature columns (everything except metadata and target)
    all_features = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    
    # Split into expression-only and GRN features
    grn_cols = [c for c in all_features if any(c.startswith(p) for p in GRN_FEATURE_PREFIXES)]
    expr_cols = [c for c in all_features if c not in grn_cols]
    
    print(f"  Expression features: {len(expr_cols)}")
    print(f"  GRN features: {len(grn_cols)} → {grn_cols}")
    
    X_expr = df[expr_cols]
    X_full = df[expr_cols + grn_cols]
    
    feature_names = {"expr": expr_cols, "full": expr_cols + grn_cols, "grn": grn_cols}
    
    return X_expr, X_full, y, meta, feature_names


# =============================================================================
# MODEL DEFINITIONS
# Each function creates, trains, and evaluates one model.
# All follow the same interface: (X_train, X_test, y_train, y_test, name) → dict
# =============================================================================

def train_ridge(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Ridge Regression — Linear model with L2 penalty.
    
    WHY RIDGE:
        The simplest possible baseline. If a neural network can't beat this,
        something is wrong with the neural network. Ridge is the "floor".
    
    WHAT alpha DOES:
        alpha=1.0 controls how much we penalise large weights.
        Higher alpha = more conservative model (smaller weights, less overfitting).
        Lower alpha = closer to standard linear regression.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (try 0.1, 1.0, 10.0)
    """
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_lasso(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Lasso Regression — Linear model with L1 penalty.
    
    WHY LASSO:
        Unlike Ridge (which shrinks all weights), Lasso can drive weights to
        EXACTLY ZERO. This means it automatically selects which features matter
        and which are useless. Perfect for our data where 30+ columns might be
        padding zeros from variable-size networks.
    
    WHAT LASSO REVEALS:
        After training, check model.coef_ — any coefficient that is exactly 0.0
        means Lasso decided that feature is worthless for prediction. If ALL GRN
        features survive (non-zero), that's strong evidence they carry real signal.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (higher = more features killed)
          0.001 is gentle, 0.1 is aggressive. We use 0.001 to start soft.
    """
    model = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Count how many features Lasso kept vs killed
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_elasticnet(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Elastic Net — Combines Ridge (L2) and Lasso (L1) penalties.
    
    WHY ELASTIC NET:
        Our GRN features are correlated (e.g., grn_in_degree and grn_n_activators
        overlap — a gene with high in-degree likely has many activators).
        
        - Lasso alone would randomly pick ONE from a group of correlated features
          and kill the rest. This loses information.
        - Ridge alone keeps everything but can't eliminate truly useless features.
        - Elastic Net keeps groups of correlated features together (Ridge behaviour)
          while still killing genuinely useless ones (Lasso behaviour).
    
    WHAT l1_ratio CONTROLS:
        - l1_ratio=0.0 → pure Ridge
        - l1_ratio=1.0 → pure Lasso
        - l1_ratio=0.5 → equal mix (our default)
    
    CUSTOMISABLE PARAMETERS:
        - alpha: overall regularisation strength
        - l1_ratio: balance between L1 and L2 (0.0 to 1.0)
    """
    model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_random_forest(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Random Forest — Ensemble of 100 decision trees.
    
    WHY RANDOM FOREST:
        First non-linear baseline. Can learn rules like "IF mRNA > 0.5 AND
        in_degree > 3 THEN protein is high". Linear models can't learn this.
        Also provides feature_importances_ for free.
    
    CUSTOMISABLE PARAMETERS:
        - n_estimators: number of trees (100 is standard, 200+ for more stability)
        - max_depth: how deep each tree can go (None = unlimited)
        - min_samples_leaf: minimum samples in each leaf (higher = simpler tree)
    """
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_STATE, 
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_mlp(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    MLP (Multi-Layer Perceptron) — D-GEX-style neural network.
    
    WHY MLP:
        This IS the D-GEX baseline. The original D-GEX paper (Chen et al., 2016)
        used an MLP to predict gene expression. We replicate their approach here
        with sklearn for speed, and later with PyTorch for full control.
    
    WHAT WAS CHANGED FROM DEFAULTS (and why):
        - hidden_layer_sizes: (100,) → (128, 64)
          DEFAULT had only 1 hidden layer with 100 neurons.
          CHANGED to 2 layers (128 then 64) to match D-GEX architecture.
          The funnel shape (128→64→1) forces the network to compress information.
        
        - max_iter: 200 → 1000
          DEFAULT cut training short after 200 epochs.
          CHANGED to give the model enough time to converge. Without this,
          the model barely started learning, causing negative R².
        
        - early_stopping: False → True
          DEFAULT trained for all max_iter epochs even if overfitting.
          CHANGED to monitor validation error and stop when it plateaus.
          This is like pulling the cake out of the oven before it burns.
        
        - validation_fraction: 0.1 → 0.15
          15% of training data is held out to monitor overfitting.
        
        - n_iter_no_change: 10 → 20
          Wait 20 epochs without improvement before stopping.
          Being patient helps avoid stopping too early on noisy data.
        
        - learning_rate: 'constant' → 'adaptive'
          DEFAULT kept the same learning rate forever.
          CHANGED to automatically reduce it when loss plateaus.
          Like slowing down as you approach a parking spot — big steps first,
          fine adjustments at the end.
        
        - batch_size: 200 → 64
          DEFAULT processed 200 samples per gradient update.
          CHANGED to 64 for noisier but more frequent updates.
          Helps the model explore more of the loss landscape.
    
    CUSTOMISABLE PARAMETERS TO TUNE:
        - hidden_layer_sizes: try (256, 128), (128, 64, 32), (64, 32)
        - learning_rate_init: try 0.0001, 0.001, 0.01
        - batch_size: try 32, 64, 128
        - alpha: L2 penalty strength (default 0.0001)
    """
    # MLP requires scaled features — without this, large-valued columns dominate
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=64,
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (stopped at epoch {model.n_iter_})")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("PER-GENE BASELINE EXPERIMENTS")
    print("=" * 70)
    
    # 1. Load data
    X_expr, X_full, y, meta, feature_names = load_pergene_data()
    
    # 2. Train-test split (same split for all models — fair comparison)
    X_expr_train, X_expr_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X_expr, y, meta, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_full_train, X_full_test = train_test_split(
        X_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )[:2]
    
    # 3. Define all model trainers
    # Each entry: (train_function, display_name)
    model_trainers = [
        (train_ridge,         "Ridge"),
        (train_lasso,         "Lasso"),
        (train_elasticnet,    "ElasticNet"),
        (train_random_forest, "RandomForest"),
        (train_mlp,           "MLP_DGEX"),
    ]
    
    # 4. Run all experiments: expression-only (A) vs expression+GRN (B)
    all_results = []
    all_predictions = {}  # Store for plotting
    rf_full_result = None  # Need this for feature importance
    lasso_full_result = None  # Need this for feature selection analysis
    
    for train_fn, base_name in model_trainers:
        # Experiment A: Expression-only
        print(f"\n--- {base_name} (Expression Only) ---")
        result_a = train_fn(X_expr_train.values, X_expr_test.values, 
                           y_train.values, y_test.values,
                           f"{base_name}_ExprOnly")
        all_results.append(result_a)
        all_predictions[f"{base_name}_ExprOnly"] = result_a["predictions"]
        
        # Experiment B: Expression + GRN
        print(f"--- {base_name} (Expression + GRN) ---")
        result_b = train_fn(X_full_train.values, X_full_test.values,
                           y_train.values, y_test.values,
                           f"{base_name}_ExprGRN")
        all_results.append(result_b)
        all_predictions[f"{base_name}_ExprGRN"] = result_b["predictions"]
        
        # Track special results for analysis
        if base_name == "RandomForest":
            rf_full_result = result_b
        if base_name == "Lasso":
            lasso_full_result = result_b
    
    # 5. Build results DataFrame
    results_df = pd.DataFrame([{
        "model": r["model"], "mse": r["mse"], "r2": r["r2"]
    } for r in all_results])
    
    # Save results table
    results_path = os.path.join(TABLES_DIR, "pergene_baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # 6. Print summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY: Expression-Only vs Expression+GRN")
    print("=" * 70)
    print(f"{'Model':<22} {'MSE (Expr)':<14} {'MSE (GRN)':<14} {'Δ MSE':<12} {'R² (Expr)':<12} {'R² (GRN)':<12}")
    print("-" * 86)
    
    for _, base_name in model_trainers:
        expr_row = results_df[results_df["model"] == f"{base_name}_ExprOnly"].iloc[0]
        grn_row = results_df[results_df["model"] == f"{base_name}_ExprGRN"].iloc[0]
        delta_mse = ((grn_row["mse"] - expr_row["mse"]) / expr_row["mse"]) * 100
        print(f"{base_name:<22} {expr_row['mse']:<14.6f} {grn_row['mse']:<14.6f} {delta_mse:<12.1f}% {expr_row['r2']:<12.4f} {grn_row['r2']:<12.4f}")
    
    # 7. Lasso feature selection analysis
    if lasso_full_result:
        print("\n" + "=" * 70)
        print("LASSO FEATURE SELECTION ANALYSIS")
        print("Which features did Lasso keep vs kill?")
        print("=" * 70)
        lasso_model = lasso_full_result["instance"]
        full_feature_names = feature_names["full"]
        
        for fname, coef in sorted(zip(full_feature_names, lasso_model.coef_), 
                                    key=lambda x: abs(x[1]), reverse=True):
            status = "✓ KEPT" if abs(coef) > 1e-10 else "✗ KILLED"
            if fname.startswith("grn_"):
                print(f"  [GRN] {fname:<30} coef={coef:>10.6f}  {status}")
        
        # Count GRN features that survived
        grn_feature_names = feature_names["grn"]
        grn_coefs = [lasso_model.coef_[full_feature_names.index(g)] for g in grn_feature_names]
        grn_alive = sum(1 for c in grn_coefs if abs(c) > 1e-10)
        print(f"\n  GRN features surviving Lasso: {grn_alive}/{len(grn_feature_names)}")
    
    # 8. Generate all plots using the standalone plotting module
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plotter = PergeneResultsPlotter(FIGURES_DIR)
    
    # Plot 1: Model comparison bar chart (MSE and R² side by side)
    plotter.plot_model_comparison(results_df)
    
    # Plot 2: Predicted vs Actual scatter for each model
    plotter.plot_predicted_vs_actual(all_predictions, y_test.values)
    
    # Plot 3: Residual distributions
    plotter.plot_residual_distributions(all_predictions, y_test.values)
    
    # Plot 4: Feature importance from Random Forest
    if rf_full_result:
        plotter.plot_feature_importance(
            rf_full_result["instance"], 
            feature_names["full"],
            top_n=20
        )
    
    # Plot 5: GRN feature correlation matrix
    plotter.plot_grn_correlation_matrix(
        pd.concat([X_full_train, X_full_test]),
        feature_names["grn"],
        y
    )
    
    # Plot 6: Per-network performance heatmap
    plotter.plot_per_network_performance(
        all_predictions, y_test.values, meta_test
    )
    
    print(f"\n✓ All plots saved to {FIGURES_DIR}")
    print("=" * 70)
    print("DONE. Next step: train_dgex_pytorch.py for the full D-GEX comparison.")
    print("=" * 70)


if __name__ == "__main__":
    main()
````

Now the standalone plotting module:

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\plotting\plot_pergene_results.py
"""
plot_pergene_results.py — Standalone plotting module for per-gene baseline results.

PURPOSE:
    Generates all analysis plots for the per-gene baseline experiments.
    This file is SEPARATE from the training code so that:
    1. Any training script can import and use it
    2. Plots can be regenerated without retraining models
    3. New plot types can be added without touching model code

PLOTS GENERATED:
    1. Model Comparison Bar Chart  — MSE and R² for all models, side by side
    2. Predicted vs Actual Scatter — How close predictions are to reality
    3. Residual Distributions      — Are errors random or systematically biased?
    4. Feature Importance          — Which features does Random Forest rely on?
    5. GRN Correlation Matrix      — How correlated are GRN features with each other?
    6. Per-Network Heatmap         — Which networks are easy/hard to predict?

USAGE:
    from plot_pergene_results import PergeneResultsPlotter
    plotter = PergeneResultsPlotter("path/to/figures")
    plotter.plot_model_comparison(results_df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves files without opening windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class PergeneResultsPlotter:
    """
    All plotting logic for per-gene baseline experiments.
    
    Each method takes raw data (predictions, scores, etc.) and saves a .png file.
    Methods are independent — you can call any one without the others.
    """
    
    def __init__(self, figures_dir: str):
        """
        Args:
            figures_dir: directory where all .png files will be saved
        """
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        
        # Consistent styling across all plots
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })
    
    def _save(self, fig, filename: str):
        """Helper: save figure and close to free memory."""
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {filename}")
    
    # =========================================================================
    # PLOT 1: MODEL COMPARISON BAR CHART
    # =========================================================================
    
    def plot_model_comparison(self, results_df: pd.DataFrame):
        """
        Side-by-side bar chart comparing MSE and R² for all models.
        
        WHY THIS PLOT:
            The single most important plot in the experiment. At a glance, you can
            see whether adding GRN features (darker bars) consistently improves
            over expression-only (lighter bars) across all model types.
        
        WHAT TO LOOK FOR:
            - Dark bars lower than light bars (MSE chart) = GRN helps
            - Dark bars higher than light bars (R² chart) = GRN helps
            - Consistent pattern across ALL models = strong evidence
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract model base names (without _ExprOnly/_ExprGRN suffix)
        base_names = []
        for name in results_df["model"]:
            base = name.replace("_ExprOnly", "").replace("_ExprGRN", "")
            if base not in base_names:
                base_names.append(base)
        
        x = np.arange(len(base_names))
        width = 0.35
        
        for ax, metric, title in zip(axes, ["mse", "r2"], ["MSE (lower is better)", "R² (higher is better)"]):
            expr_vals = [results_df[results_df["model"] == f"{b}_ExprOnly"][metric].values[0] for b in base_names]
            grn_vals = [results_df[results_df["model"] == f"{b}_ExprGRN"][metric].values[0] for b in base_names]
            
            bars1 = ax.bar(x - width/2, expr_vals, width, label="Expression Only", color="#78909C", alpha=0.85)
            bars2 = ax.bar(x + width/2, grn_vals, width, label="Expression + GRN", color="#26A69A", alpha=0.85)
            
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(base_names, rotation=15)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
```


## License: GPL-3.0
https://github.com/cKNUSPeR/i2nca/blob/a13e08a4db2ae0463402a6896862fdc296267e24/i2nca/qctools/visualization.py

```


Great questions. Let me answer each one, then build the solution.

---

## Answers to Your Questions

### Is sklearn MLP the same as D-GEX?
**Yes, conceptually.** D-GEX is just an MLP that predicts gene expression from other genes' expression. The sklearn `MLPRegressor` with `hidden_layer_sizes=(128, 64)` and ReLU activation IS a D-GEX-style model. The only difference is that the original D-GEX paper used a custom implementation with more control over training. So:

- **sklearn MLP** = D-GEX baseline (quick version)
- **PyTorch MLP** = D-GEX baseline (proper version with full control)

Both are D-GEX. Both should give similar results when properly tuned.

### Should we use both sklearn and PyTorch?
**Yes.** Keep both. In your report you can write: *"We implemented the D-GEX architecture in both scikit-learn and PyTorch and confirmed comparable performance, validating our implementation."* This shows rigour.

### File Organisation
Your instinct is correct. Here's my recommended structure:

```
experiments/
├── build_pergene_dataset.py          # Data preparation (already done)
├── train_pergene_baselines.py        # All per-gene models (Ridge, Lasso, ElasticNet, RF, MLP)
├── train_dgex_pytorch.py             # PyTorch D-GEX (Friday's task)
├── generate_synthetic_dataset.py     # Data generation (already done)
├── generate_fixed_size_dataset.py    # Fixed-size data (Friday's task)
└── plotting/
    └── plot_pergene_results.py       # ALL plots in one reusable file
```

The plotting file will be **standalone** — any training script can call it by passing results DataFrames. This avoids duplicating plotting code across model files.

---

Now let me build both files. First the updated training script, then the plotting module.

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\train_pergene_baselines.py
"""
train_pergene_baselines.py — Train and evaluate all per-gene baseline models.

PURPOSE:
    Answer the core research question: "Does adding GRN structural features
    improve gene expression prediction compared to expression-only models?"

MODELS (in order of complexity):
    1. Ridge Regression       — Linear baseline with L2 penalty (shrinks weights)
    2. Lasso Regression       — Linear baseline with L1 penalty (kills useless features)
    3. Elastic Net             — Combines L1 + L2 (best linear baseline for correlated features)
    4. Random Forest           — Non-linear, tree-based, gives feature importance
    5. MLP (sklearn)           — D-GEX-style neural network (quick version)

EACH MODEL IS TESTED TWICE:
    A. Expression-only:  input = [own_mRNA + other genes' mRNA]
    B. Expression + GRN: input = [own_mRNA + other genes' mRNA + 9 GRN features]

    If B beats A consistently → GRN features carry predictive power. QED.

OUTPUT:
    results/tables/pergene_baseline_results.csv   — All scores
    results/figures/*.png                          — All plots (via plot_pergene_results.py)

USAGE:
    python experiments/train_pergene_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plotting"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings for cleaner output

# Import the standalone plotting module
from plot_pergene_results import PergeneResultsPlotter


# =============================================================================
# CONFIGURATION
# =============================================================================
# All paths and settings in one place for easy modification.

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_transsys")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create output directories
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Columns that are NOT features (metadata + target)
META_COLS = ["sample_id", "network_id", "gene_name"]
TARGET_COL = "target_protein"

# GRN feature columns — these are the 9 structural features extracted from network topology.
# They are the "treatment" in our controlled experiment.
GRN_FEATURE_PREFIXES = ["grn_"]

# Test split — 20% held out, never seen during training
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pergene_data():
    """
    Load the per-gene dataset and split into expression-only and full feature sets.
    
    Returns two feature sets:
      - X_expr: only mRNA columns (the "control group")
      - X_full: mRNA + GRN columns (the "treatment group")
      - y: target protein levels
      - meta: sample_id, network_id, gene_name for later analysis
      - feature_names: dict with 'expr' and 'full' feature lists
    """
    path = os.path.join(DATA_DIR, "pergene_dataset.csv")
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"  Dataset shape: {df.shape[0]} samples × {df.shape[1]} columns")
    
    # Separate metadata, features, and target
    meta = df[META_COLS]
    y = df[TARGET_COL]
    
    # All feature columns (everything except metadata and target)
    all_features = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    
    # Split into expression-only and GRN features
    grn_cols = [c for c in all_features if any(c.startswith(p) for p in GRN_FEATURE_PREFIXES)]
    expr_cols = [c for c in all_features if c not in grn_cols]
    
    print(f"  Expression features: {len(expr_cols)}")
    print(f"  GRN features: {len(grn_cols)} → {grn_cols}")
    
    X_expr = df[expr_cols]
    X_full = df[expr_cols + grn_cols]
    
    feature_names = {"expr": expr_cols, "full": expr_cols + grn_cols, "grn": grn_cols}
    
    return X_expr, X_full, y, meta, feature_names


# =============================================================================
# MODEL DEFINITIONS
# Each function creates, trains, and evaluates one model.
# All follow the same interface: (X_train, X_test, y_train, y_test, name) → dict
# =============================================================================

def train_ridge(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Ridge Regression — Linear model with L2 penalty.
    
    WHY RIDGE:
        The simplest possible baseline. If a neural network can't beat this,
        something is wrong with the neural network. Ridge is the "floor".
    
    WHAT alpha DOES:
        alpha=1.0 controls how much we penalise large weights.
        Higher alpha = more conservative model (smaller weights, less overfitting).
        Lower alpha = closer to standard linear regression.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (try 0.1, 1.0, 10.0)
    """
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_lasso(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Lasso Regression — Linear model with L1 penalty.
    
    WHY LASSO:
        Unlike Ridge (which shrinks all weights), Lasso can drive weights to
        EXACTLY ZERO. This means it automatically selects which features matter
        and which are useless. Perfect for our data where 30+ columns might be
        padding zeros from variable-size networks.
    
    WHAT LASSO REVEALS:
        After training, check model.coef_ — any coefficient that is exactly 0.0
        means Lasso decided that feature is worthless for prediction. If ALL GRN
        features survive (non-zero), that's strong evidence they carry real signal.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (higher = more features killed)
          0.001 is gentle, 0.1 is aggressive. We use 0.001 to start soft.
    """
    model = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Count how many features Lasso kept vs killed
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_elasticnet(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Elastic Net — Combines Ridge (L2) and Lasso (L1) penalties.
    
    WHY ELASTIC NET:
        Our GRN features are correlated (e.g., grn_in_degree and grn_n_activators
        overlap — a gene with high in-degree likely has many activators).
        
        - Lasso alone would randomly pick ONE from a group of correlated features
          and kill the rest. This loses information.
        - Ridge alone keeps everything but can't eliminate truly useless features.
        - Elastic Net keeps groups of correlated features together (Ridge behaviour)
          while still killing genuinely useless ones (Lasso behaviour).
    
    WHAT l1_ratio CONTROLS:
        - l1_ratio=0.0 → pure Ridge
        - l1_ratio=1.0 → pure Lasso
        - l1_ratio=0.5 → equal mix (our default)
    
    CUSTOMISABLE PARAMETERS:
        - alpha: overall regularisation strength
        - l1_ratio: balance between L1 and L2 (0.0 to 1.0)
    """
    model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_random_forest(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Random Forest — Ensemble of 100 decision trees.
    
    WHY RANDOM FOREST:
        First non-linear baseline. Can learn rules like "IF mRNA > 0.5 AND
        in_degree > 3 THEN protein is high". Linear models can't learn this.
        Also provides feature_importances_ for free.
    
    CUSTOMISABLE PARAMETERS:
        - n_estimators: number of trees (100 is standard, 200+ for more stability)
        - max_depth: how deep each tree can go (None = unlimited)
        - min_samples_leaf: minimum samples in each leaf (higher = simpler tree)
    """
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_STATE, 
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_mlp(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    MLP (Multi-Layer Perceptron) — D-GEX-style neural network.
    
    WHY MLP:
        This IS the D-GEX baseline. The original D-GEX paper (Chen et al., 2016)
        used an MLP to predict gene expression. We replicate their approach here
        with sklearn for speed, and later with PyTorch for full control.
    
    WHAT WAS CHANGED FROM DEFAULTS (and why):
        - hidden_layer_sizes: (100,) → (128, 64)
          DEFAULT had only 1 hidden layer with 100 neurons.
          CHANGED to 2 layers (128 then 64) to match D-GEX architecture.
          The funnel shape (128→64→1) forces the network to compress information.
        
        - max_iter: 200 → 1000
          DEFAULT cut training short after 200 epochs.
          CHANGED to give the model enough time to converge. Without this,
          the model barely started learning, causing negative R².
        
        - early_stopping: False → True
          DEFAULT trained for all max_iter epochs even if overfitting.
          CHANGED to monitor validation error and stop when it plateaus.
          This is like pulling the cake out of the oven before it burns.
        
        - validation_fraction: 0.1 → 0.15
          15% of training data is held out to monitor overfitting.
        
        - n_iter_no_change: 10 → 20
          Wait 20 epochs without improvement before stopping.
          Being patient helps avoid stopping too early on noisy data.
        
        - learning_rate: 'constant' → 'adaptive'
          DEFAULT kept the same learning rate forever.
          CHANGED to automatically reduce it when loss plateaus.
          Like slowing down as you approach a parking spot — big steps first,
          fine adjustments at the end.
        
        - batch_size: 200 → 64
          DEFAULT processed 200 samples per gradient update.
          CHANGED to 64 for noisier but more frequent updates.
          Helps the model explore more of the loss landscape.
    
    CUSTOMISABLE PARAMETERS TO TUNE:
        - hidden_layer_sizes: try (256, 128), (128, 64, 32), (64, 32)
        - learning_rate_init: try 0.0001, 0.001, 0.01
        - batch_size: try 32, 64, 128
        - alpha: L2 penalty strength (default 0.0001)
    """
    # MLP requires scaled features — without this, large-valued columns dominate
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=64,
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (stopped at epoch {model.n_iter_})")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("PER-GENE BASELINE EXPERIMENTS")
    print("=" * 70)
    
    # 1. Load data
    X_expr, X_full, y, meta, feature_names = load_pergene_data()
    
    # 2. Train-test split (same split for all models — fair comparison)
    X_expr_train, X_expr_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X_expr, y, meta, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_full_train, X_full_test = train_test_split(
        X_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )[:2]
    
    # 3. Define all model trainers
    # Each entry: (train_function, display_name)
    model_trainers = [
        (train_ridge,         "Ridge"),
        (train_lasso,         "Lasso"),
        (train_elasticnet,    "ElasticNet"),
        (train_random_forest, "RandomForest"),
        (train_mlp,           "MLP_DGEX"),
    ]
    
    # 4. Run all experiments: expression-only (A) vs expression+GRN (B)
    all_results = []
    all_predictions = {}  # Store for plotting
    rf_full_result = None  # Need this for feature importance
    lasso_full_result = None  # Need this for feature selection analysis
    
    for train_fn, base_name in model_trainers:
        # Experiment A: Expression-only
        print(f"\n--- {base_name} (Expression Only) ---")
        result_a = train_fn(X_expr_train.values, X_expr_test.values, 
                           y_train.values, y_test.values,
                           f"{base_name}_ExprOnly")
        all_results.append(result_a)
        all_predictions[f"{base_name}_ExprOnly"] = result_a["predictions"]
        
        # Experiment B: Expression + GRN
        print(f"--- {base_name} (Expression + GRN) ---")
        result_b = train_fn(X_full_train.values, X_full_test.values,
                           y_train.values, y_test.values,
                           f"{base_name}_ExprGRN")
        all_results.append(result_b)
        all_predictions[f"{base_name}_ExprGRN"] = result_b["predictions"]
        
        # Track special results for analysis
        if base_name == "RandomForest":
            rf_full_result = result_b
        if base_name == "Lasso":
            lasso_full_result = result_b
    
    # 5. Build results DataFrame
    results_df = pd.DataFrame([{
        "model": r["model"], "mse": r["mse"], "r2": r["r2"]
    } for r in all_results])
    
    # Save results table
    results_path = os.path.join(TABLES_DIR, "pergene_baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # 6. Print summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY: Expression-Only vs Expression+GRN")
    print("=" * 70)
    print(f"{'Model':<22} {'MSE (Expr)':<14} {'MSE (GRN)':<14} {'Δ MSE':<12} {'R² (Expr)':<12} {'R² (GRN)':<12}")
    print("-" * 86)
    
    for _, base_name in model_trainers:
        expr_row = results_df[results_df["model"] == f"{base_name}_ExprOnly"].iloc[0]
        grn_row = results_df[results_df["model"] == f"{base_name}_ExprGRN"].iloc[0]
        delta_mse = ((grn_row["mse"] - expr_row["mse"]) / expr_row["mse"]) * 100
        print(f"{base_name:<22} {expr_row['mse']:<14.6f} {grn_row['mse']:<14.6f} {delta_mse:<12.1f}% {expr_row['r2']:<12.4f} {grn_row['r2']:<12.4f}")
    
    # 7. Lasso feature selection analysis
    if lasso_full_result:
        print("\n" + "=" * 70)
        print("LASSO FEATURE SELECTION ANALYSIS")
        print("Which features did Lasso keep vs kill?")
        print("=" * 70)
        lasso_model = lasso_full_result["instance"]
        full_feature_names = feature_names["full"]
        
        for fname, coef in sorted(zip(full_feature_names, lasso_model.coef_), 
                                    key=lambda x: abs(x[1]), reverse=True):
            status = "✓ KEPT" if abs(coef) > 1e-10 else "✗ KILLED"
            if fname.startswith("grn_"):
                print(f"  [GRN] {fname:<30} coef={coef:>10.6f}  {status}")
        
        # Count GRN features that survived
        grn_feature_names = feature_names["grn"]
        grn_coefs = [lasso_model.coef_[full_feature_names.index(g)] for g in grn_feature_names]
        grn_alive = sum(1 for c in grn_coefs if abs(c) > 1e-10)
        print(f"\n  GRN features surviving Lasso: {grn_alive}/{len(grn_feature_names)}")
    
    # 8. Generate all plots using the standalone plotting module
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plotter = PergeneResultsPlotter(FIGURES_DIR)
    
    # Plot 1: Model comparison bar chart (MSE and R² side by side)
    plotter.plot_model_comparison(results_df)
    
    # Plot 2: Predicted vs Actual scatter for each model
    plotter.plot_predicted_vs_actual(all_predictions, y_test.values)
    
    # Plot 3: Residual distributions
    plotter.plot_residual_distributions(all_predictions, y_test.values)
    
    # Plot 4: Feature importance from Random Forest
    if rf_full_result:
        plotter.plot_feature_importance(
            rf_full_result["instance"], 
            feature_names["full"],
            top_n=20
        )
    
    # Plot 5: GRN feature correlation matrix
    plotter.plot_grn_correlation_matrix(
        pd.concat([X_full_train, X_full_test]),
        feature_names["grn"],
        y
    )
    
    # Plot 6: Per-network performance heatmap
    plotter.plot_per_network_performance(
        all_predictions, y_test.values, meta_test
    )
    
    print(f"\n✓ All plots saved to {FIGURES_DIR}")
    print("=" * 70)
    print("DONE. Next step: train_dgex_pytorch.py for the full D-GEX comparison.")
    print("=" * 70)


if __name__ == "__main__":
    main()
````

Now the standalone plotting module:

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\plotting\plot_pergene_results.py
"""
plot_pergene_results.py — Standalone plotting module for per-gene baseline results.

PURPOSE:
    Generates all analysis plots for the per-gene baseline experiments.
    This file is SEPARATE from the training code so that:
    1. Any training script can import and use it
    2. Plots can be regenerated without retraining models
    3. New plot types can be added without touching model code

PLOTS GENERATED:
    1. Model Comparison Bar Chart  — MSE and R² for all models, side by side
    2. Predicted vs Actual Scatter — How close predictions are to reality
    3. Residual Distributions      — Are errors random or systematically biased?
    4. Feature Importance          — Which features does Random Forest rely on?
    5. GRN Correlation Matrix      — How correlated are GRN features with each other?
    6. Per-Network Heatmap         — Which networks are easy/hard to predict?

USAGE:
    from plot_pergene_results import PergeneResultsPlotter
    plotter = PergeneResultsPlotter("path/to/figures")
    plotter.plot_model_comparison(results_df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves files without opening windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class PergeneResultsPlotter:
    """
    All plotting logic for per-gene baseline experiments.
    
    Each method takes raw data (predictions, scores, etc.) and saves a .png file.
    Methods are independent — you can call any one without the others.
    """
    
    def __init__(self, figures_dir: str):
        """
        Args:
            figures_dir: directory where all .png files will be saved
        """
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        
        # Consistent styling across all plots
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })
    
    def _save(self, fig, filename: str):
        """Helper: save figure and close to free memory."""
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {filename}")
    
    # =========================================================================
    # PLOT 1: MODEL COMPARISON BAR CHART
    # =========================================================================
    
    def plot_model_comparison(self, results_df: pd.DataFrame):
        """
        Side-by-side bar chart comparing MSE and R² for all models.
        
        WHY THIS PLOT:
            The single most important plot in the experiment. At a glance, you can
            see whether adding GRN features (darker bars) consistently improves
            over expression-only (lighter bars) across all model types.
        
        WHAT TO LOOK FOR:
            - Dark bars lower than light bars (MSE chart) = GRN helps
            - Dark bars higher than light bars (R² chart) = GRN helps
            - Consistent pattern across ALL models = strong evidence
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract model base names (without _ExprOnly/_ExprGRN suffix)
        base_names = []
        for name in results_df["model"]:
            base = name.replace("_ExprOnly", "").replace("_ExprGRN", "")
            if base not in base_names:
                base_names.append(base)
        
        x = np.arange(len(base_names))
        width = 0.35
        
        for ax, metric, title in zip(axes, ["mse", "r2"], ["MSE (lower is better)", "R² (higher is better)"]):
            expr_vals = [results_df[results_df["model"] == f"{b}_ExprOnly"][metric].values[0] for b in base_names]
            grn_vals = [results_df[results_df["model"] == f"{b}_ExprGRN"][metric].values[0] for b in base_names]
            
            bars1 = ax.bar(x - width/2, expr_vals, width, label="Expression Only", color="#78909C", alpha=0.85)
            bars2 = ax.bar(x + width/2, grn_vals, width, label="Expression + GRN", color="#26A69A", alpha=0.85)
            
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(base_names, rotation=15)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
```


## License: GPL-3.0
https://github.com/cKNUSPeR/i2nca/blob/a13e08a4db2ae0463402a6896862fdc296267e24/i2nca/qctools/visualization.py

```


Great questions. Let me answer each one, then build the solution.

---

## Answers to Your Questions

### Is sklearn MLP the same as D-GEX?
**Yes, conceptually.** D-GEX is just an MLP that predicts gene expression from other genes' expression. The sklearn `MLPRegressor` with `hidden_layer_sizes=(128, 64)` and ReLU activation IS a D-GEX-style model. The only difference is that the original D-GEX paper used a custom implementation with more control over training. So:

- **sklearn MLP** = D-GEX baseline (quick version)
- **PyTorch MLP** = D-GEX baseline (proper version with full control)

Both are D-GEX. Both should give similar results when properly tuned.

### Should we use both sklearn and PyTorch?
**Yes.** Keep both. In your report you can write: *"We implemented the D-GEX architecture in both scikit-learn and PyTorch and confirmed comparable performance, validating our implementation."* This shows rigour.

### File Organisation
Your instinct is correct. Here's my recommended structure:

```
experiments/
├── build_pergene_dataset.py          # Data preparation (already done)
├── train_pergene_baselines.py        # All per-gene models (Ridge, Lasso, ElasticNet, RF, MLP)
├── train_dgex_pytorch.py             # PyTorch D-GEX (Friday's task)
├── generate_synthetic_dataset.py     # Data generation (already done)
├── generate_fixed_size_dataset.py    # Fixed-size data (Friday's task)
└── plotting/
    └── plot_pergene_results.py       # ALL plots in one reusable file
```

The plotting file will be **standalone** — any training script can call it by passing results DataFrames. This avoids duplicating plotting code across model files.

---

Now let me build both files. First the updated training script, then the plotting module.

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\train_pergene_baselines.py
"""
train_pergene_baselines.py — Train and evaluate all per-gene baseline models.

PURPOSE:
    Answer the core research question: "Does adding GRN structural features
    improve gene expression prediction compared to expression-only models?"

MODELS (in order of complexity):
    1. Ridge Regression       — Linear baseline with L2 penalty (shrinks weights)
    2. Lasso Regression       — Linear baseline with L1 penalty (kills useless features)
    3. Elastic Net             — Combines L1 + L2 (best linear baseline for correlated features)
    4. Random Forest           — Non-linear, tree-based, gives feature importance
    5. MLP (sklearn)           — D-GEX-style neural network (quick version)

EACH MODEL IS TESTED TWICE:
    A. Expression-only:  input = [own_mRNA + other genes' mRNA]
    B. Expression + GRN: input = [own_mRNA + other genes' mRNA + 9 GRN features]

    If B beats A consistently → GRN features carry predictive power. QED.

OUTPUT:
    results/tables/pergene_baseline_results.csv   — All scores
    results/figures/*.png                          — All plots (via plot_pergene_results.py)

USAGE:
    python experiments/train_pergene_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plotting"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings for cleaner output

# Import the standalone plotting module
from plot_pergene_results import PergeneResultsPlotter


# =============================================================================
# CONFIGURATION
# =============================================================================
# All paths and settings in one place for easy modification.

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_transsys")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create output directories
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Columns that are NOT features (metadata + target)
META_COLS = ["sample_id", "network_id", "gene_name"]
TARGET_COL = "target_protein"

# GRN feature columns — these are the 9 structural features extracted from network topology.
# They are the "treatment" in our controlled experiment.
GRN_FEATURE_PREFIXES = ["grn_"]

# Test split — 20% held out, never seen during training
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pergene_data():
    """
    Load the per-gene dataset and split into expression-only and full feature sets.
    
    Returns two feature sets:
      - X_expr: only mRNA columns (the "control group")
      - X_full: mRNA + GRN columns (the "treatment group")
      - y: target protein levels
      - meta: sample_id, network_id, gene_name for later analysis
      - feature_names: dict with 'expr' and 'full' feature lists
    """
    path = os.path.join(DATA_DIR, "pergene_dataset.csv")
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"  Dataset shape: {df.shape[0]} samples × {df.shape[1]} columns")
    
    # Separate metadata, features, and target
    meta = df[META_COLS]
    y = df[TARGET_COL]
    
    # All feature columns (everything except metadata and target)
    all_features = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    
    # Split into expression-only and GRN features
    grn_cols = [c for c in all_features if any(c.startswith(p) for p in GRN_FEATURE_PREFIXES)]
    expr_cols = [c for c in all_features if c not in grn_cols]
    
    print(f"  Expression features: {len(expr_cols)}")
    print(f"  GRN features: {len(grn_cols)} → {grn_cols}")
    
    X_expr = df[expr_cols]
    X_full = df[expr_cols + grn_cols]
    
    feature_names = {"expr": expr_cols, "full": expr_cols + grn_cols, "grn": grn_cols}
    
    return X_expr, X_full, y, meta, feature_names


# =============================================================================
# MODEL DEFINITIONS
# Each function creates, trains, and evaluates one model.
# All follow the same interface: (X_train, X_test, y_train, y_test, name) → dict
# =============================================================================

def train_ridge(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Ridge Regression — Linear model with L2 penalty.
    
    WHY RIDGE:
        The simplest possible baseline. If a neural network can't beat this,
        something is wrong with the neural network. Ridge is the "floor".
    
    WHAT alpha DOES:
        alpha=1.0 controls how much we penalise large weights.
        Higher alpha = more conservative model (smaller weights, less overfitting).
        Lower alpha = closer to standard linear regression.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (try 0.1, 1.0, 10.0)
    """
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_lasso(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Lasso Regression — Linear model with L1 penalty.
    
    WHY LASSO:
        Unlike Ridge (which shrinks all weights), Lasso can drive weights to
        EXACTLY ZERO. This means it automatically selects which features matter
        and which are useless. Perfect for our data where 30+ columns might be
        padding zeros from variable-size networks.
    
    WHAT LASSO REVEALS:
        After training, check model.coef_ — any coefficient that is exactly 0.0
        means Lasso decided that feature is worthless for prediction. If ALL GRN
        features survive (non-zero), that's strong evidence they carry real signal.
    
    CUSTOMISABLE PARAMETERS:
        - alpha: regularisation strength (higher = more features killed)
          0.001 is gentle, 0.1 is aggressive. We use 0.001 to start soft.
    """
    model = Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Count how many features Lasso kept vs killed
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_elasticnet(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Elastic Net — Combines Ridge (L2) and Lasso (L1) penalties.
    
    WHY ELASTIC NET:
        Our GRN features are correlated (e.g., grn_in_degree and grn_n_activators
        overlap — a gene with high in-degree likely has many activators).
        
        - Lasso alone would randomly pick ONE from a group of correlated features
          and kill the rest. This loses information.
        - Ridge alone keeps everything but can't eliminate truly useless features.
        - Elastic Net keeps groups of correlated features together (Ridge behaviour)
          while still killing genuinely useless ones (Lasso behaviour).
    
    WHAT l1_ratio CONTROLS:
        - l1_ratio=0.0 → pure Ridge
        - l1_ratio=1.0 → pure Lasso
        - l1_ratio=0.5 → equal mix (our default)
    
    CUSTOMISABLE PARAMETERS:
        - alpha: overall regularisation strength
        - l1_ratio: balance between L1 and L2 (0.0 to 1.0)
    """
    model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    n_alive = np.sum(np.abs(model.coef_) > 1e-10)
    n_dead = np.sum(np.abs(model.coef_) <= 1e-10)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (features kept: {n_alive}, killed: {n_dead})")
    
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model,
            "n_features_alive": n_alive, "n_features_dead": n_dead}


def train_random_forest(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    Random Forest — Ensemble of 100 decision trees.
    
    WHY RANDOM FOREST:
        First non-linear baseline. Can learn rules like "IF mRNA > 0.5 AND
        in_degree > 3 THEN protein is high". Linear models can't learn this.
        Also provides feature_importances_ for free.
    
    CUSTOMISABLE PARAMETERS:
        - n_estimators: number of trees (100 is standard, 200+ for more stability)
        - max_depth: how deep each tree can go (None = unlimited)
        - min_samples_leaf: minimum samples in each leaf (higher = simpler tree)
    """
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_STATE, 
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f}")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


def train_mlp(X_train, X_test, y_train, y_test, name: str) -> dict:
    """
    MLP (Multi-Layer Perceptron) — D-GEX-style neural network.
    
    WHY MLP:
        This IS the D-GEX baseline. The original D-GEX paper (Chen et al., 2016)
        used an MLP to predict gene expression. We replicate their approach here
        with sklearn for speed, and later with PyTorch for full control.
    
    WHAT WAS CHANGED FROM DEFAULTS (and why):
        - hidden_layer_sizes: (100,) → (128, 64)
          DEFAULT had only 1 hidden layer with 100 neurons.
          CHANGED to 2 layers (128 then 64) to match D-GEX architecture.
          The funnel shape (128→64→1) forces the network to compress information.
        
        - max_iter: 200 → 1000
          DEFAULT cut training short after 200 epochs.
          CHANGED to give the model enough time to converge. Without this,
          the model barely started learning, causing negative R².
        
        - early_stopping: False → True
          DEFAULT trained for all max_iter epochs even if overfitting.
          CHANGED to monitor validation error and stop when it plateaus.
          This is like pulling the cake out of the oven before it burns.
        
        - validation_fraction: 0.1 → 0.15
          15% of training data is held out to monitor overfitting.
        
        - n_iter_no_change: 10 → 20
          Wait 20 epochs without improvement before stopping.
          Being patient helps avoid stopping too early on noisy data.
        
        - learning_rate: 'constant' → 'adaptive'
          DEFAULT kept the same learning rate forever.
          CHANGED to automatically reduce it when loss plateaus.
          Like slowing down as you approach a parking spot — big steps first,
          fine adjustments at the end.
        
        - batch_size: 200 → 64
          DEFAULT processed 200 samples per gradient update.
          CHANGED to 64 for noisier but more frequent updates.
          Helps the model explore more of the loss landscape.
    
    CUSTOMISABLE PARAMETERS TO TUNE:
        - hidden_layer_sizes: try (256, 128), (128, 64, 32), (64, 32)
        - learning_rate_init: try 0.0001, 0.001, 0.01
        - batch_size: try 32, 64, 128
        - alpha: L2 penalty strength (default 0.0001)
    """
    # MLP requires scaled features — without this, large-valued columns dominate
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=64,
        random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  {name}: MSE={mse:.6f}, R²={r2:.4f} (stopped at epoch {model.n_iter_})")
    return {"model": name, "mse": mse, "r2": r2, "predictions": preds, "instance": model}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("PER-GENE BASELINE EXPERIMENTS")
    print("=" * 70)
    
    # 1. Load data
    X_expr, X_full, y, meta, feature_names = load_pergene_data()
    
    # 2. Train-test split (same split for all models — fair comparison)
    X_expr_train, X_expr_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X_expr, y, meta, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_full_train, X_full_test = train_test_split(
        X_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )[:2]
    
    # 3. Define all model trainers
    # Each entry: (train_function, display_name)
    model_trainers = [
        (train_ridge,         "Ridge"),
        (train_lasso,         "Lasso"),
        (train_elasticnet,    "ElasticNet"),
        (train_random_forest, "RandomForest"),
        (train_mlp,           "MLP_DGEX"),
    ]
    
    # 4. Run all experiments: expression-only (A) vs expression+GRN (B)
    all_results = []
    all_predictions = {}  # Store for plotting
    rf_full_result = None  # Need this for feature importance
    lasso_full_result = None  # Need this for feature selection analysis
    
    for train_fn, base_name in model_trainers:
        # Experiment A: Expression-only
        print(f"\n--- {base_name} (Expression Only) ---")
        result_a = train_fn(X_expr_train.values, X_expr_test.values, 
                           y_train.values, y_test.values,
                           f"{base_name}_ExprOnly")
        all_results.append(result_a)
        all_predictions[f"{base_name}_ExprOnly"] = result_a["predictions"]
        
        # Experiment B: Expression + GRN
        print(f"--- {base_name} (Expression + GRN) ---")
        result_b = train_fn(X_full_train.values, X_full_test.values,
                           y_train.values, y_test.values,
                           f"{base_name}_ExprGRN")
        all_results.append(result_b)
        all_predictions[f"{base_name}_ExprGRN"] = result_b["predictions"]
        
        # Track special results for analysis
        if base_name == "RandomForest":
            rf_full_result = result_b
        if base_name == "Lasso":
            lasso_full_result = result_b
    
    # 5. Build results DataFrame
    results_df = pd.DataFrame([{
        "model": r["model"], "mse": r["mse"], "r2": r["r2"]
    } for r in all_results])
    
    # Save results table
    results_path = os.path.join(TABLES_DIR, "pergene_baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # 6. Print summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY: Expression-Only vs Expression+GRN")
    print("=" * 70)
    print(f"{'Model':<22} {'MSE (Expr)':<14} {'MSE (GRN)':<14} {'Δ MSE':<12} {'R² (Expr)':<12} {'R² (GRN)':<12}")
    print("-" * 86)
    
    for _, base_name in model_trainers:
        expr_row = results_df[results_df["model"] == f"{base_name}_ExprOnly"].iloc[0]
        grn_row = results_df[results_df["model"] == f"{base_name}_ExprGRN"].iloc[0]
        delta_mse = ((grn_row["mse"] - expr_row["mse"]) / expr_row["mse"]) * 100
        print(f"{base_name:<22} {expr_row['mse']:<14.6f} {grn_row['mse']:<14.6f} {delta_mse:<12.1f}% {expr_row['r2']:<12.4f} {grn_row['r2']:<12.4f}")
    
    # 7. Lasso feature selection analysis
    if lasso_full_result:
        print("\n" + "=" * 70)
        print("LASSO FEATURE SELECTION ANALYSIS")
        print("Which features did Lasso keep vs kill?")
        print("=" * 70)
        lasso_model = lasso_full_result["instance"]
        full_feature_names = feature_names["full"]
        
        for fname, coef in sorted(zip(full_feature_names, lasso_model.coef_), 
                                    key=lambda x: abs(x[1]), reverse=True):
            status = "✓ KEPT" if abs(coef) > 1e-10 else "✗ KILLED"
            if fname.startswith("grn_"):
                print(f"  [GRN] {fname:<30} coef={coef:>10.6f}  {status}")
        
        # Count GRN features that survived
        grn_feature_names = feature_names["grn"]
        grn_coefs = [lasso_model.coef_[full_feature_names.index(g)] for g in grn_feature_names]
        grn_alive = sum(1 for c in grn_coefs if abs(c) > 1e-10)
        print(f"\n  GRN features surviving Lasso: {grn_alive}/{len(grn_feature_names)}")
    
    # 8. Generate all plots using the standalone plotting module
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plotter = PergeneResultsPlotter(FIGURES_DIR)
    
    # Plot 1: Model comparison bar chart (MSE and R² side by side)
    plotter.plot_model_comparison(results_df)
    
    # Plot 2: Predicted vs Actual scatter for each model
    plotter.plot_predicted_vs_actual(all_predictions, y_test.values)
    
    # Plot 3: Residual distributions
    plotter.plot_residual_distributions(all_predictions, y_test.values)
    
    # Plot 4: Feature importance from Random Forest
    if rf_full_result:
        plotter.plot_feature_importance(
            rf_full_result["instance"], 
            feature_names["full"],
            top_n=20
        )
    
    # Plot 5: GRN feature correlation matrix
    plotter.plot_grn_correlation_matrix(
        pd.concat([X_full_train, X_full_test]),
        feature_names["grn"],
        y
    )
    
    # Plot 6: Per-network performance heatmap
    plotter.plot_per_network_performance(
        all_predictions, y_test.values, meta_test
    )
    
    print(f"\n✓ All plots saved to {FIGURES_DIR}")
    print("=" * 70)
    print("DONE. Next step: train_dgex_pytorch.py for the full D-GEX comparison.")
    print("=" * 70)


if __name__ == "__main__":
    main()
````

Now the standalone plotting module:

````python
// filepath: c:\Users\hrith\OneDrive\05_L6_UH25\00-Level6\00_AI_project_module_backup\x-main-code\gene-expression-project\experiments\plotting\plot_pergene_results.py
"""
plot_pergene_results.py — Standalone plotting module for per-gene baseline results.

PURPOSE:
    Generates all analysis plots for the per-gene baseline experiments.
    This file is SEPARATE from the training code so that:
    1. Any training script can import and use it
    2. Plots can be regenerated without retraining models
    3. New plot types can be added without touching model code

PLOTS GENERATED:
    1. Model Comparison Bar Chart  — MSE and R² for all models, side by side
    2. Predicted vs Actual Scatter — How close predictions are to reality
    3. Residual Distributions      — Are errors random or systematically biased?
    4. Feature Importance          — Which features does Random Forest rely on?
    5. GRN Correlation Matrix      — How correlated are GRN features with each other?
    6. Per-Network Heatmap         — Which networks are easy/hard to predict?

USAGE:
    from plot_pergene_results import PergeneResultsPlotter
    plotter = PergeneResultsPlotter("path/to/figures")
    plotter.plot_model_comparison(results_df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves files without opening windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class PergeneResultsPlotter:
    """
    All plotting logic for per-gene baseline experiments.
    
    Each method takes raw data (predictions, scores, etc.) and saves a .png file.
    Methods are independent — you can call any one without the others.
    """
    
    def __init__(self, figures_dir: str):
        """
        Args:
            figures_dir: directory where all .png files will be saved
        """
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        
        # Consistent styling across all plots
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })
    
    def _save(self, fig, filename: str):
        """Helper: save figure and close to free memory."""
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {filename}")
    
    # =========================================================================
    # PLOT 1: MODEL COMPARISON BAR CHART
    # =========================================================================
    
    def plot_model_comparison(self, results_df: pd.DataFrame):
        """
        Side-by-side bar chart comparing MSE and R² for all models.
        
        WHY THIS PLOT:
            The single most important plot in the experiment. At a glance, you can
            see whether adding GRN features (darker bars) consistently improves
            over expression-only (lighter bars) across all model types.
        
        WHAT TO LOOK FOR:
            - Dark bars lower than light bars (MSE chart) = GRN helps
            - Dark bars higher than light bars (R² chart) = GRN helps
            - Consistent pattern across ALL models = strong evidence
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract model base names (without _ExprOnly/_ExprGRN suffix)
        base_names = []
        for name in results_df["model"]:
            base = name.replace("_ExprOnly", "").replace("_ExprGRN", "")
            if base not in base_names:
                base_names.append(base)
        
        x = np.arange(len(base_names))
        width = 0.35
        
        for ax, metric, title in zip(axes, ["mse", "r2"], ["MSE (lower is better)", "R² (higher is better)"]):
            expr_vals = [results_df[results_df["model"] == f"{b}_ExprOnly"][metric].values[0] for b in base_names]
            grn_vals = [results_df[results_df["model"] == f"{b}_ExprGRN"][metric].values[0] for b in base_names]
            
            bars1 = ax.bar(x - width/2, expr_vals, width, label="Expression Only", color="#78909C", alpha=0.85)
            bars2 = ax.bar(x + width/2, grn_vals, width, label="Expression + GRN", color="#26A69A", alpha=0.85)
            
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(base_names, rotation=15)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
```

