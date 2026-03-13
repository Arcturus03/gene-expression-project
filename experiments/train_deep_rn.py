"""
train_deep_rn.py — Train and evaluate DeepRN-Expr and DeepRN-Struct.

Compares:
  DeepRN-Expr   (expression only)  — deep expression baseline
  DeepRN-Struct (expression + GRN) — GRN-aware deep model

Same train/test network split as baselines for fair comparison.
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.deep_rn_model import DeepRNExpr, DeepRNStruct

# =============================================================================
# CONFIG — must match train_pergene_baselines.py
# =============================================================================
DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "ml_ready", "pergene_dataset.csv")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR   = os.path.join(RESULTS_DIR, "tables")
RANDOM_SEED  = 42
TEST_NETWORK_FRACTION = 0.2
LOG_TRANSFORM_TARGET  = True

# Training hyperparameters
EPOCHS      = 300
BATCH_SIZE  = 128
LR          = 0.001
DROPOUT     = 0.3
WEIGHT_DECAY = 1e-4
PATIENCE    = 60        # early stopping patience

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# DATA LOADING (same split logic as baselines)
# =============================================================================

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples, {len(df.columns)} columns")

    if LOG_TRANSFORM_TARGET:
        df['target_protein'] = np.log1p(df['target_protein'])

    expr_cols = ['own_mRNA'] + sorted([c for c in df.columns if c.startswith('other_')])
    grn_cols  = sorted([c for c in df.columns if c.startswith('grn_')])
    print(f"Expression features: {len(expr_cols)}, GRN features: {len(grn_cols)}")

    # Same network-based split
    unique_networks = df['network_id'].unique()
    n_test = max(1, int(len(unique_networks) * TEST_NETWORK_FRACTION))

    rng = np.random.default_rng(RANDOM_SEED)
    test_networks  = set(rng.choice(unique_networks, size=n_test, replace=False))
    train_networks = set(unique_networks) - test_networks

    train_df = df[df['network_id'].isin(train_networks)].copy().reset_index(drop=True)
    test_df  = df[df['network_id'].isin(test_networks)].copy().reset_index(drop=True)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    return train_df, test_df, expr_cols, grn_cols


# =============================================================================
# DATASET BUILDER
# =============================================================================

def build_tensors(train_df, test_df, expr_cols, grn_cols):
    """Scale features and return PyTorch tensors."""
    
    # Fit scalers on train only (prevent data leakage)
    expr_scaler = StandardScaler()
    grn_scaler  = StandardScaler()

    X_expr_train = expr_scaler.fit_transform(train_df[expr_cols].fillna(0).values)
    X_grn_train  = grn_scaler.fit_transform(train_df[grn_cols].fillna(0).values)
    y_train      = train_df['target_protein'].values

    X_expr_test  = expr_scaler.transform(test_df[expr_cols].fillna(0).values)
    X_grn_test   = grn_scaler.transform(test_df[grn_cols].fillna(0).values)
    y_test       = test_df['target_protein'].values

    # Convert to tensors
    to_t = lambda x: torch.tensor(x, dtype=torch.float32)

    train_dataset = TensorDataset(to_t(X_expr_train), to_t(X_grn_train), to_t(y_train))
    test_dataset  = TensorDataset(to_t(X_expr_test),  to_t(X_grn_test),  to_t(y_test))

    return (train_dataset, test_dataset,
            len(expr_cols), len(grn_cols),
            y_test)


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_model(model, train_dataset, test_dataset,
                model_name: str, y_test_raw: np.ndarray) -> dict:
    """Generic training loop for any DeepRN model."""
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    model = model.to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', patience=20, factor=0.4      # what this does is: if the validation loss doesn't improve for 20 epochs, reduce the learning rate by 0.4. This helps the model converge better when it hits a plateau.
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state    = None
    patience_ctr  = 0
    
    print(f"\n  Training {model_name}...")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        
        # --- Train ---
        model.train()
        for batch in train_loader:
            optimiser.zero_grad()
            if len(batch) == 3:
                x_e, x_g, y = [b.to(DEVICE) for b in batch]
                pred = model(x_e, x_g) if isinstance(model, DeepRNStruct) else model(x_e)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        # --- Validate ---
        model.eval()
        val_losses, all_preds = [], []
        with torch.no_grad():
            for batch in test_loader:
                x_e, x_g, y = [b.to(DEVICE) for b in batch]
                pred = model(x_e, x_g) if isinstance(model, DeepRNStruct) else model(x_e)
                val_losses.append(criterion(pred, y).item())
                all_preds.append(pred.cpu().numpy())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:     # Print every 10 epochs
            preds_log = np.concatenate(all_preds)
            r2_log    = r2_score(y_test_raw, preds_log)
            print(f"    Epoch {epoch:3d} | Val Loss: {val_loss:.4f} | R² (log): {r2_log:.4f}")

    train_time = time.time() - start
    
    # --- Final evaluation with best weights ---
    model.load_state_dict(best_state)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            x_e, x_g, y = [b.to(DEVICE) for b in batch]
            pred = model(x_e, x_g) if isinstance(model, DeepRNStruct) else model(x_e)
            all_preds.append(pred.cpu().numpy())

    preds_log = np.concatenate(all_preds)

    # Metrics in log-space (matches baselines)
    mse_log = mean_squared_error(y_test_raw, preds_log)
    mae_log = mean_absolute_error(y_test_raw, preds_log)
    r2_log  = r2_score(y_test_raw, preds_log)

    # Metrics in original space (for reporting/plotting)
    preds_orig = np.expm1(preds_log)
    actuals_orig = np.expm1(y_test_raw)
    r2_orig = r2_score(actuals_orig, preds_orig)

    print(f"\n  {model_name} RESULTS:")
    print(f"    MSE (log-space): {mse_log:.4f}")
    print(f"    MAE (log-space): {mae_log:.4f}")
    print(f"    R²  (log-space): {r2_log:.4f}")
    print(f"    R²  (orig-space): {r2_orig:.4f}")
    print(f"    Train time: {train_time:.1f}s")

    return {
        'model':        model_name,
        'MSE':          mse_log,
        'MAE':          mae_log,
        'R2':           r2_log,
        'R2_orig':      r2_orig,
        'train_time_s': train_time,
        'y_test':       y_test_raw,
        'y_pred':       preds_log,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("DeepRN-Expr vs DeepRN-Struct Comparison")
    print("=" * 70)

    train_df, test_df, expr_cols, grn_cols = load_data()

    (train_dataset, test_dataset,
    n_expr, n_grn, y_test_raw) = build_tensors(
        train_df, test_df, expr_cols, grn_cols
    )

    results = []

    # --- DeepRN-Expr (expression only) ---
    expr_model = DeepRNExpr(
        n_expr_features=n_expr,
        expr_hidden=128,
        fusion_hidden=[128, 64],
        dropout=DROPOUT
    )
    results.append(train_model(
        expr_model, train_dataset, test_dataset,
        "DeepRN-Expr", y_test_raw
    ))

    # --- DeepRN-Struct (expression + GRN) ---
    struct_model = DeepRNStruct(
        n_expr_features=n_expr,
        n_grn_features=n_grn,
        expr_hidden=128,
        grn_hidden=64,
        fusion_hidden=[128, 64],
        dropout=DROPOUT
    )
    results.append(train_model(
        struct_model, train_dataset, test_dataset,
        "DeepRN-Struct", y_test_raw
    ))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("DEEPRN SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r['model']:20s}  R²(log): {r['R2']:.4f}  "
            f"R²(orig): {r['R2_orig']:.4f}  MSE: {r['MSE']:.4f}")

    expr_r2   = results[0]['R2']
    struct_r2 = results[1]['R2']
    delta     = struct_r2 - expr_r2
    direction = "IMPROVED ✅" if delta > 0 else "WORSENED ❌"
    print(f"\n  GRN stream impact: ΔR² = {delta:+.4f}  ({direction})")

    # Save to CSV
    os.makedirs(TABLES_DIR, exist_ok=True)
    out_df = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ('y_test', 'y_pred')}
                            for r in results])
    out_path = os.path.join(TABLES_DIR, "deep_rn_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
