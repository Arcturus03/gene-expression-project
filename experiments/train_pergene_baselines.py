"""
train_pergene_baselines.py — Train and compare per-gene baseline models.

PURPOSE:
    This script answers the project's central research question:
    "Does encoding GRN structure as features improve ML prediction of gene expression?"

    It does this by running a CONTROLLED EXPERIMENT:
    - Group A (Expression-Only): Models see only mRNA levels of other genes
    - Group B (Expression + GRN): Same models, but also see 9 GRN structural features
    If Group B consistently outperforms Group A -> GRN features carry predictive value.

WHY THESE 4 MODELS?
    1. Ridge Regression — Linear baseline with L2 penalty. Shrinks weights but keeps
       all features. Good when all features might contribute a little.
    2. Elastic Net — Linear baseline with L1 + L2 penalty. Can zero out useless features
       (like Lasso) while handling correlated features gracefully (like Ridge).
       This is superior to pure Lasso when features are correlated (which GRN features are).
    3. Random Forest — Non-linear ensemble that captures interactions between features.
       Provides built-in feature importance rankings.
    4. MLP (sklearn) — Lightweight neural network. Bridge between simple baselines
       and the full PyTorch D-GEX model we build next.

TRAIN/TEST SPLIT STRATEGY:
    We split by NETWORK_ID, not by row. This ensures the model is tested on
    entirely unseen network topologies — a much harder and more realistic test
    than random row splitting (which would leak network structure into the test set).

OUTPUTS:
    - Console: Full comparison table with MSE, MAE, R² for all 8 experiments
    - CSV: results/tables/pergene_baseline_results.csv
    - CSV: results/tables/pergene_feature_importance.csv
    - CSV: results/tables/pergene_predictions.csv (for plotting)
    - Plots are generated separately via experiments/plotting.py

Author: Hrithik Chandra
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')  # Suppress sklearn convergence warnings


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml_ready", "pergene_dataset.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Train/test split ratio (by network, not by row)
TEST_NETWORK_FRACTION = 0.2  # 20% of networks held out for testing

# Random seed for reproducibility across all experiments
RANDOM_SEED = 42

# Log-transform flag — set True to stabilise MLP training on high-variance targets.
# All metrics (MSE, MAE, R²) will be computed in log-space for model comparison.
# Predictions CSV will store BOTH log-space and original-scale values for plotting.
LOG_TRANSFORM_TARGET = True


# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================

ID_COLS = ['sample_id', 'network_id', 'seed', 'gene_name', 'n_genes_in_network']
TARGET_COL = 'target_protein'


# =============================================================================
# DATA LOADING & SPLITTING
# =============================================================================

def load_and_split_data(data_path: str, test_fraction: float, seed: int):
    """
    Load the per-gene dataset and split into train/test BY NETWORK.

    WHY SPLIT BY NETWORK?
    If we split randomly by row, test rows from network X would share the same
    GRN structure as training rows from network X. The model could memorize
    network-specific patterns instead of learning generalizable relationships.
    Splitting by network forces the model to generalize to unseen topologies.

    Returns:
        train_df, test_df: DataFrames with all columns preserved
        expr_cols: list of expression feature column names
        grn_cols: list of GRN feature column names
    """
    print("[1/6] Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"      Loaded {len(df)} samples, {len(df.columns)} columns")
    
    # LOG-TRANSFORM the protein target to stabilise MLP training
    # log1p(x) = log(x + 1), handles zeros safely
    df['target_protein'] = np.log1p(df['target_protein'])

    print(f"  Target after log-transform: min={df['target_protein'].min():.2f}, "
    f"max={df['target_protein'].max():.2f}, mean={df['target_protein'].mean():.2f}")
    

    # Identify feature columns dynamically from the CSV header
    expr_cols = ['own_mRNA'] + sorted([c for c in df.columns if c.startswith('other_')])
    grn_cols = sorted([c for c in df.columns if c.startswith('grn_')])

    print(f"      Expression features: {len(expr_cols)} (own_mRNA + {len(expr_cols)-1} other genes)")
    print(f"      GRN features: {len(grn_cols)} ({', '.join(grn_cols)})")

    # Split by network ID (not by row)
    unique_networks = df['network_id'].unique()
    n_test = max(1, int(len(unique_networks) * test_fraction))

    rng = np.random.default_rng(seed)
    test_networks = set(rng.choice(unique_networks, size=n_test, replace=False))
    train_networks = set(unique_networks) - test_networks

    train_df = df[df['network_id'].isin(train_networks)].copy().reset_index(drop=True)
    test_df  = df[df['network_id'].isin(test_networks)].copy().reset_index(drop=True)


    print(f"      Train: {len(train_df)} samples from {len(train_networks)} networks")
    print(f"      Test:  {len(test_df)} samples from {len(test_networks)} networks")

    return train_df, test_df, expr_cols, grn_cols


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def train_and_evaluate(
    model,
    model_name: str,
    feature_cols: list,
    feature_set_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: StandardScaler = None
) -> dict:
    """
    Train a single model on given features and evaluate on test set.

    WHY STANDARDSCALER?
    Ridge, Elastic Net, and MLP are sensitive to feature scale. A gene's mRNA 
    might range 0–700 while PageRank ranges 0–0.2. Without scaling, the model 
    would over-weight mRNA simply because its numbers are bigger, not because
    it's more informative. StandardScaler normalises all features to mean=0, std=1.

    Returns:
        dict with model name, feature set, MSE, MAE, R², training time, and fitted model
    """
    # Extract features and target
    X_train = train_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values
    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # Scale features (fit on train only, transform both — prevents data leakage)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Predict and score
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'model': model_name,
        'features': feature_set_name,
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'train_time_s': train_time,
        'n_features': len(feature_cols),
        'fitted_model': model,  # Keep for feature importance analysis
        'y_test': y_test,       # Keep for plotting
        'y_pred': y_pred        # Keep for plotting
    }


def run_all_experiments(train_df, test_df, expr_cols, grn_cols) -> tuple:
    """
    Run the complete A/B experiment matrix: 4 models × 2 feature sets = 8 experiments.

    The experiment matrix:
    ┌──────────────────┬─────────────────────┬──────────────────────────────┐
    │ Model            │ Expression Only (A) │ Expression + GRN (B)         │
    ├──────────────────┼─────────────────────┼──────────────────────────────┤
    │ Ridge Regression │ Experiment 1        │ Experiment 2                 │
    │ Elastic Net      │ Experiment 3        │ Experiment 4                 │
    │ Random Forest    │ Experiment 5        │ Experiment 6                 │
    │ MLP (sklearn)    │ Experiment 7        │ Experiment 8                 │
    └──────────────────┴─────────────────────┴──────────────────────────────┘

    If Experiments 2,4,6,8 consistently beat 1,3,5,7 → GRN features help.
    """
    print("\n[2/6] Running 8 experiments (4 models × 2 feature sets)...\n")

    all_features = expr_cols + grn_cols  # Combined feature set for Group B
    results = []

    # =========================================================================
    # MODEL 1: Ridge Regression (L2 penalty)
    # =========================================================================
    # WHY RIDGE?
    # - Adds a penalty term: Loss = MSE + α × Σ(weights²)
    # - This shrinks all weights toward zero but never exactly to zero
    # - Good baseline when you believe all features contribute some signal
    # - Handles multicollinearity (correlated features) by distributing weight
    #
    # HYPERPARAMETERS:
    # - alpha (default=1.0): Regularisation strength. Higher = more shrinkage.
    #   Try: 0.01, 0.1, 1.0, 10.0, 100.0

    print("      [1/8] Ridge — Expression Only")
    results.append(train_and_evaluate(
        model=Ridge(alpha=1.0),
        model_name="Ridge",
        feature_cols=expr_cols,
        feature_set_name="Expression Only",
        train_df=train_df,
        test_df=test_df,
        scaler=StandardScaler()
    ))

    print("      [2/8] Ridge — Expression + GRN")
    results.append(train_and_evaluate(
        model=Ridge(alpha=1.0),
        model_name="Ridge",
        feature_cols=all_features,
        feature_set_name="Expression + GRN",
        train_df=train_df,
        test_df=test_df,
        scaler=StandardScaler()
    ))

    # =========================================================================
    # MODEL 2: Elastic Net (L1 + L2 penalty combined)
    # =========================================================================
    # WHY ELASTIC NET (instead of pure Lasso)?
    # - Combines L1 (sparsity) and L2 (stability): Loss = MSE + α × [ρ×|w| + (1-ρ)×w²]
    # - l1_ratio=0.5 means 50% Lasso, 50% Ridge
    # - Superior to pure Lasso when features are CORRELATED (which ours are!)
    #   Example: grn_in_degree and grn_n_activators are highly correlated.
    #   Pure Lasso would arbitrarily pick one and zero the other.
    #   Elastic Net keeps both with reduced weights.
    # - Can perform FEATURE SELECTION (set some weights exactly to 0)
    #   This tells us which features are truly useless.
    #
    # HYPERPARAMETERS:
    # - alpha (default=0.1): Overall regularisation strength (higher = more penalty)
    #   Try: 0.001, 0.01, 0.1, 1.0
    # - l1_ratio (default=0.5): Balance between L1 and L2
    #   0.0 = pure Ridge, 1.0 = pure Lasso, 0.5 = balanced
    #   Try: 0.1, 0.5, 0.7, 0.9
    # - max_iter: Increase if you get convergence warnings

    print("      [3/8] Elastic Net — Expression Only")
    results.append(train_and_evaluate(
        model=ElasticNet(
            alpha=0.1,           # Regularisation strength
            l1_ratio=0.5,        # 50% L1 (sparsity) + 50% L2 (stability)
            max_iter=10000,      # Enough iterations to converge
            random_state=RANDOM_SEED
        ),
        model_name="Elastic Net",
        feature_cols=expr_cols,
        feature_set_name="Expression Only",
        train_df=train_df,
        test_df=test_df,
        scaler=StandardScaler()
    ))

    print("      [4/8] Elastic Net — Expression + GRN")
    results.append(train_and_evaluate(
        model=ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=10000,
            random_state=RANDOM_SEED
        ),
        model_name="Elastic Net",
        feature_cols=all_features,
        feature_set_name="Expression + GRN",
        train_df=train_df,
        test_df=test_df,
        scaler=StandardScaler()
    ))

    # =========================================================================
    # MODEL 3: Random Forest (ensemble of decision trees)
    # =========================================================================
    # WHY RANDOM FOREST?
    # - Non-linear: Can capture "if mRNA > 0.5 AND in_degree > 3, then protein high"
    # - Feature importance: Built-in ranking of which features reduce error most
    # - Robust: Doesn't need feature scaling, handles outliers well
    # - Ensemble: 200 trees vote, reducing variance from any single tree's quirks
    #
    # HYPERPARAMETERS:
    # - n_estimators (default=200): Number of trees. More = better but slower.
    #   Try: 100, 200, 500
    # - max_depth (default=15): Max tree depth. Limits overfitting.
    #   Try: 10, 15, 20, None (unlimited)
    # - min_samples_leaf (default=1): Min samples per leaf. Higher = smoother.
    #   Try: 1, 5, 10
    # - n_jobs=-1: Use all CPU cores

    print("      [5/8] Random Forest — Expression Only")
    results.append(train_and_evaluate(
        model=RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        model_name="Random Forest",
        feature_cols=expr_cols,
        feature_set_name="Expression Only",
        train_df=train_df,
        test_df=test_df,
        scaler=None  # RF doesn't need scaling — splits on thresholds, not magnitudes
    ))

    print("      [6/8] Random Forest — Expression + GRN")
    results.append(train_and_evaluate(
        model=RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        model_name="Random Forest",
        feature_cols=all_features,
        feature_set_name="Expression + GRN",
        train_df=train_df,
        test_df=test_df,
        scaler=None
    ))

    # =========================================================================
    # MODEL 4: MLP (Multi-Layer Perceptron) — sklearn version
    # =========================================================================
    # WHY MLP?
    # - Universal approximator: Can learn any continuous function given enough neurons
    # - Bridge to D-GEX: The D-GEX paper uses a deep MLP for gene expression prediction.
    #   This sklearn version validates the approach before we build the PyTorch version.
    # - Non-linear: ReLU activations allow learning complex relationships
    #
    # RELATIONSHIP TO D-GEX:
    # - D-GEX (Chen et al., 2016) uses 3+ hidden layers with 3000-9000 neurons
    # - Our sklearn MLP is a SIMPLIFIED version (2 layers, 128+64 neurons)
    # - We'll build a proper PyTorch D-GEX later for fair comparison
    #
    # ARCHITECTURE: Input → 128 neurons → 64 neurons → 1 output
    #
    # =====================================================================
    # FIXES APPLIED TO ACHIEVE POSITIVE R² (previously was negative):
    # =====================================================================
    # 1. StandardScaler: CRITICAL — MLPs use gradient descent, which is very
    #    sensitive to feature scale. Without scaling, large features dominate.
    # 2. early_stopping=True: Stops training when validation loss plateaus,
    #    preventing overfitting to training data.
    # 3. learning_rate='adaptive': Automatically reduces learning rate when
    #    loss stops improving, allowing finer convergence.
    # 4. max_iter=500: Default (200) often insufficient for convergence.
    # 5. alpha=0.001: L2 regularisation prevents weight explosion.
    # 6. validation_fraction=0.1: Uses 10% of training data to monitor overfitting.
    # =====================================================================
    #
    # HYPERPARAMETERS TO TUNE:
    # - hidden_layer_sizes: Network architecture
    #   Try: (64, 32), (128, 64), (256, 128), (256, 128, 64)
    # - alpha: L2 regularisation (higher = more regularisation)
    #   Try: 0.0001, 0.001, 0.01
    # - learning_rate_init: Starting step size
    #   Try: 0.0001, 0.001, 0.01
    # - batch_size: Samples per gradient update (affects training stability)
    #   Try: 32, 64, 128, 'auto' (default=min(200, n_samples))

    print("      [7/8] MLP (sklearn) — Expression Only")
    results.append(train_and_evaluate(
        model=MLPRegressor(
            hidden_layer_sizes=( 128, 64),    # 3 hidden layers
            activation='relu',               # ReLU: max(0, x)
            solver='adam',                   # Adam optimiser (adaptive learning rate)
            alpha=0.01,                     # L2 regularisation strength
            batch_size=32,               # Mini-batch size
            learning_rate='adaptive',        # Reduce LR when loss plateaus
            learning_rate_init=0.0005,        # Starting learning rate
            max_iter=2000,                    # Max training epochs
            early_stopping=False,             # Stop if validation loss stops improving
            validation_fraction=0.15,         # 10% of training data for validation
            n_iter_no_change=50,             # Patience: epochs without improvement
            random_state=RANDOM_SEED,
            verbose=False
        ),
        model_name="MLP (sklearn)",
        feature_cols=expr_cols,
        feature_set_name="Expression Only",
        train_df=train_df,
        test_df=test_df,
        scaler=StandardScaler()  # CRITICAL for MLP convergence
    ))

    print("      [8/8] MLP (sklearn) — Expression + GRN")
    results.append(train_and_evaluate(
        model=MLPRegressor(
            hidden_layer_sizes=( 128, 64),    # 3 hidden layers
            activation='relu',               # ReLU: max(0, x)
            solver='adam',                   # Adam optimiser (adaptive learning rate)
            alpha=0.01,                     # L2 regularisation strength
            batch_size=32,               # Mini-batch size
            learning_rate='adaptive',        # Reduce LR when loss plateaus
            learning_rate_init=0.0005,        # Starting learning rate
            max_iter=2000,                    # Max training epochs
            early_stopping=False,             # Stop if validation loss stops improving
            validation_fraction=0.15,         # 10% of training data for validation
            n_iter_no_change=50,             # Patience: epochs without improvement
            random_state=RANDOM_SEED,
            verbose=False
        ),
        model_name="MLP (sklearn)",
        feature_cols=all_features,
        feature_set_name="Expression + GRN",
        train_df=train_df,
        test_df=test_df,
        scaler=StandardScaler()
    ))

    # Build results dataframe (exclude non-serializable columns for CSV)
    csv_cols = ['model', 'features', 'MSE', 'MAE', 'R2', 'train_time_s', 'n_features']
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k in csv_cols} for r in results])

    return results_df, results


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_feature_importance(results: list, expr_cols: list, grn_cols: list) -> pd.DataFrame:
    """
    Extract feature importance from Random Forest (Expression + GRN) model.

    WHY ONLY RANDOM FOREST?
    RF computes importance as the average reduction in impurity (MSE reduction)
    caused by each feature across all trees. This is:
    - More reliable than Ridge/Elastic Net coefficients (which are scale-dependent)
    - More interpretable than MLP weights (which are opaque due to non-linearity)

    KEY QUESTION ANSWERED:
    "Which specific GRN features matter most for predicting protein levels?"
    """
    print("\n[3/6] Analyzing feature importance...")

    # Find the RF model trained on Expression + GRN
    rf_grn_result = None
    for r in results:
        if r['model'] == 'Random Forest' and r['features'] == 'Expression + GRN':
            rf_grn_result = r
            break

    if rf_grn_result is None:
        print("      WARNING: Random Forest (Expression + GRN) not found. Skipping.")
        return None

    rf_model = rf_grn_result['fitted_model']
    all_features = expr_cols + grn_cols
    importances = rf_model.feature_importances_

    # Build importance dataframe
    importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': importances,
        'category': ['Expression'] * len(expr_cols) + ['GRN'] * len(grn_cols)
    }).sort_values('importance', ascending=False)

    # Print top 15 features
    print("\n      Top 15 Most Important Features:")
    print("      " + "-" * 55)
    for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
        bar = "█" * int(row['importance'] * 200)
        tag = " ← GRN" if row['category'] == 'GRN' else ""
        print(f"      {i+1:2d}. {row['feature']:25s} {row['importance']:.4f} {bar}{tag}")

    # Calculate total importance by category
    expr_total = importance_df[importance_df['category'] == 'Expression']['importance'].sum()
    grn_total = importance_df[importance_df['category'] == 'GRN']['importance'].sum()

    print(f"\n      Total importance — Expression: {expr_total:.3f} ({expr_total*100:.1f}%)")
    print(f"      Total importance — GRN:        {grn_total:.3f} ({grn_total*100:.1f}%)")

    return importance_df


# =============================================================================
# ELASTIC NET COEFFICIENT ANALYSIS
# =============================================================================

def analyze_elastic_net_coefficients(results: list, expr_cols: list, grn_cols: list) -> pd.DataFrame:
    """
    Extract and analyze coefficients from Elastic Net (Expression + GRN) model.

    WHY ANALYZE ELASTIC NET COEFFICIENTS?
    - Elastic Net with L1 penalty can set coefficients EXACTLY to zero
    - Non-zero coefficients = features the model found useful
    - Zero coefficients = features the model deemed uninformative
    - Sign of coefficient indicates direction of relationship

    INTERPRETATION:
    - Positive coefficient: Higher feature value → higher protein level
    - Negative coefficient: Higher feature value → lower protein level
    - Zero coefficient: Feature has no predictive value (was "zeroed out")

    This provides a different perspective than Random Forest importance:
    - RF importance = how much each feature reduces prediction error
    - ElasticNet coef = linear relationship strength and direction
    """
    print("\n[4/6] Analyzing Elastic Net coefficients...")

    # Find the Elastic Net model trained on Expression + GRN
    en_grn_result = None
    for r in results:
        if r['model'] == 'Elastic Net' and r['features'] == 'Expression + GRN':
            en_grn_result = r
            break

    if en_grn_result is None:
        print("      WARNING: Elastic Net (Expression + GRN) not found. Skipping.")
        return None

    en_model = en_grn_result['fitted_model']
    all_features = expr_cols + grn_cols
    coefficients = en_model.coef_

    # Build coefficient dataframe
    coef_df = pd.DataFrame({
        'feature': all_features,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients),
        'category': ['Expression'] * len(expr_cols) + ['GRN'] * len(grn_cols)
    }).sort_values('abs_coefficient', ascending=False)

    # Count non-zero coefficients
    n_nonzero = np.sum(coefficients != 0)
    n_zero = len(coefficients) - n_nonzero
    n_grn_nonzero = np.sum((coef_df['category'] == 'GRN') & (coef_df['coefficient'] != 0))

    print(f"\n      Elastic Net Feature Selection Summary:")
    print(f"      - Non-zero coefficients: {n_nonzero}/{len(coefficients)}")
    print(f"      - Zeroed-out features: {n_zero}")
    print(f"      - GRN features retained: {n_grn_nonzero}/{len(grn_cols)}")

    # Print top 10 features by absolute coefficient
    print("\n      Top 10 Features by |Coefficient|:")
    print("      " + "-" * 60)
    for i, (_, row) in enumerate(coef_df.head(10).iterrows()):
        sign = "+" if row['coefficient'] > 0 else "-" if row['coefficient'] < 0 else "0"
        tag = " ← GRN" if row['category'] == 'GRN' else ""
        print(f"      {i+1:2d}. {row['feature']:25s} {sign}{row['abs_coefficient']:.4f}{tag}")

    return coef_df


# =============================================================================
# SAVE PREDICTIONS FOR PLOTTING
# =============================================================================

def save_predictions(results: list, test_df: pd.DataFrame,
                    log_transformed: bool = False) -> pd.DataFrame:
    """
    Save predictions from all models for later plotting.

    WHY SAVE PREDICTIONS?
    - Enables detailed diagnostic plots (scatter, residuals, per-network)
    - Allows regenerating plots without retraining
    - Facilitates comparison across multiple runs

    NOTE ON LOG TRANSFORM:
    If log_transformed=True, y_test and y_pred are in log1p-space.
    We reverse-transform to original protein units for plots so that
    axis labels (0–800) remain interpretable. Metrics (MSE/R²) are
    kept in log-space inside results_df — that is intentional and
    standard practice (log-space metrics are used for model selection,
    original-space values are used for human-readable reporting).
    """
    print("\n[5/6] Saving predictions for plotting...")

    predictions_data = []

    for r in results:
        model_name  = r['model']
        feature_set = r['features']
        y_test      = r['y_test']   # log-space if log_transformed=True
        y_pred      = r['y_pred']   # log-space if log_transformed=True

        for i, (actual_log, predicted_log) in enumerate(zip(y_test, y_pred)):

            # Reverse log1p transform → original protein scale
            if log_transformed:
                actual_orig    = np.expm1(actual_log)
                predicted_orig = np.expm1(predicted_log)
            else:
                actual_orig    = actual_log
                predicted_orig = predicted_log

            predictions_data.append({
                'model':            model_name,
                'features':         feature_set,
                # Original-scale values for plotting
                'actual':           actual_orig,
                'predicted':        predicted_orig,
                'residual':         actual_orig - predicted_orig,
                # Log-space values saved separately — useful for log-space diagnostic plots
                'actual_log':       actual_log,
                'predicted_log':    predicted_log,
                'residual_log':     actual_log - predicted_log,
                'network_id':       test_df.iloc[i]['network_id'] if i < len(test_df) else None
            })

    predictions_df = pd.DataFrame(predictions_data)
    print(f"   Saved {len(predictions_df)} prediction rows "
        f"({'log-space reversed to original units' if log_transformed else 'original units'})")
    return predictions_df


# =============================================================================
# RESULTS SUMMARY
# =============================================================================

def print_results_summary(results_df: pd.DataFrame, log_transformed: bool = False):
    """
    Print a clean comparison table and calculate the improvement from adding GRN features.

    The "Δ MSE" and "Δ R²" columns directly quantify the benefit of GRN features.
    - Negative Δ MSE = error decreased (GOOD)
    - Positive Δ R² = explained variance increased (GOOD)
    """
    print("\n[6/6] Results Summary")
    print("=" * 85)

    # Clearly label whether metrics are in log-space
    if log_transformed:
        print("  NOTE: MSE and MAE are in log1p-space (target was log-transformed).")
        print("  R² is unitless and directly comparable across runs.")
        print("  To convert MSE to original protein units, use: np.expm1(sqrt(MSE))²")
    print()

    display_cols = ['model', 'features', 'n_features', 'MSE', 'MAE', 'R2', 'train_time_s']
    print(results_df[display_cols].to_string(index=False, float_format='%.4f'))

    print("\n" + "-" * 85)
    print("GRN Feature Impact (Expression+GRN vs Expression Only):")
    print("-" * 85)

    models = results_df['model'].unique()
    for model in models:
        expr = results_df[(results_df['model'] == model) & (results_df['features'] == 'Expression Only')]
        grn  = results_df[(results_df['model'] == model) & (results_df['features'] == 'Expression + GRN')]

        if len(expr) == 0 or len(grn) == 0:
            continue

        mse_baseline = expr['MSE'].values[0]
        mse_grn      = grn['MSE'].values[0]
        r2_baseline  = expr['R2'].values[0]
        r2_grn       = grn['R2'].values[0]

        mse_change = mse_grn - mse_baseline
        r2_change  = r2_grn  - r2_baseline
        mse_pct    = (mse_change / mse_baseline) * 100 if mse_baseline != 0 else 0

        direction = "↓ IMPROVED" if mse_change < 0 else "↑ WORSENED"

        print(f"  {model:20s}  MSE: {mse_change:+.4f} ({mse_pct:+.1f}%) {direction}  |  R²: {r2_change:+.4f}")

    print("=" * 85)


def save_results(results_df: pd.DataFrame, importance_df: pd.DataFrame, 
                coef_df: pd.DataFrame, predictions_df: pd.DataFrame):
    """
    Save all results to CSV files for later use by plotting script.
    """
    print("\nSaving results to CSV...")

    os.makedirs(TABLES_DIR, exist_ok=True)

    # Save main results
    results_path = os.path.join(TABLES_DIR, "pergene_baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"      Saved: {results_path}")

    # Save feature importance
    if importance_df is not None:
        importance_path = os.path.join(TABLES_DIR, "pergene_feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        print(f"      Saved: {importance_path}")

    # Save Elastic Net coefficients
    if coef_df is not None:
        coef_path = os.path.join(TABLES_DIR, "pergene_elasticnet_coefficients.csv")
        coef_df.to_csv(coef_path, index=False)
        print(f"      Saved: {coef_path}")

    # Save predictions
    if predictions_df is not None:
        pred_path = os.path.join(TABLES_DIR, "pergene_predictions.csv")
        predictions_df.to_csv(pred_path, index=False)
        print(f"      Saved: {pred_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 85)
    print("PER-GENE BASELINE EXPERIMENTS")
    print("Research Question: Does GRN structure improve gene expression prediction?")
    if LOG_TRANSFORM_TARGET:
        print("  Target transformation: log1p(target_protein) — metrics reported in log-space")
    print("=" * 85)

    total_start = time.time()

    train_df, test_df, expr_cols, grn_cols = load_and_split_data(
        DATA_PATH, TEST_NETWORK_FRACTION, RANDOM_SEED
    )

    results_df, results_list = run_all_experiments(train_df, test_df, expr_cols, grn_cols)

    importance_df = analyze_feature_importance(results_list, expr_cols, grn_cols)

    coef_df = analyze_elastic_net_coefficients(results_list, expr_cols, grn_cols)

    # Pass log_transformed flag so predictions are reversed to original protein units
    predictions_df = save_predictions(
        results_list,
        test_df,
        log_transformed=LOG_TRANSFORM_TARGET   # ← NEW
    )

    # Pass flag so summary labels metrics correctly
    print_results_summary(results_df, log_transformed=LOG_TRANSFORM_TARGET)   # ← NEW

    save_results(results_df, importance_df, coef_df, predictions_df)

    total_time = time.time() - total_start
    print(f"\nTotal runtime: {total_time:.1f}s")
    print("\nTo generate plots, run: python experiments/plotting.py")


if __name__ == "__main__":
    main()
