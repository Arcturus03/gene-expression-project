"""
plotting.py — Comprehensive visualisation module for baseline model results.

PURPOSE:
    Generates all diagnostic and comparison plots from saved CSV results.
    Separated from training script so that:
    1. Training can run quickly without rendering plots
    2. Plots can be regenerated or customised without retraining
    3. Different plot styles can be tested easily

PLOTS GENERATED (6 total):
    1. Model Comparison Bar Chart — Side-by-side MSE/R² comparison
    2. Feature Importance Plot — Which features matter most (Random Forest)
    3. GRN Improvement Summary — Bar chart of MSE reduction per model
    4. Predicted vs Actual Scatter — Diagnostic for best model
    5. Residual Distribution — Error analysis histogram
    6. GRN Feature Correlation Matrix — Correlation between GRN features and target

USAGE:
    python experiments/plotting.py

    Or import specific functions:
    from plotting import plot_model_comparison, plot_residual_distribution

Author: Hrithik Chandra
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml_ready")

# Plot style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'expr_only': '#90CAF9',      # Light blue
    'expr_grn': '#FF8A65',       # Light orange
    'expression': '#2196F3',     # Blue
    'grn': '#FF5722',            # Orange
    'best': '#4CAF50',           # Green
    'worst': '#F44336',          # Red
    'neutral': '#9E9E9E'         # Grey
}


# =============================================================================
# PLOT 1: Model Comparison Bar Chart
# =============================================================================

def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    """
    Create grouped bar chart comparing all models on MSE, MAE, and R².

    WHY THIS PLOT?
    ==============
    - The single most important figure for the dissertation results section
    - Visually answers: "Did GRN features help?" for each model type
    - Easy to see patterns: if orange bars are consistently better, GRN helps

    WHAT TO LOOK FOR:
    =================
    - MSE/MAE: Lower is better. Orange bars should be SHORTER than blue.
    - R²: Higher is better. Orange bars should be TALLER than blue.
    - Consistency: If 3/4 models improve with GRN, the signal is robust.
    - R² > 0.5 = good model, R² > 0.8 = excellent model

    INTERPRETATION GUIDE:
    =====================
    - All orange bars better → Strong evidence GRN features help
    - Mixed results → GRN features help some models but not others
    - All blue bars better → GRN features hurt (unlikely if real signal exists)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['MSE', 'MAE', 'R2']
    titles = [
        'Mean Squared Error\n(lower = better)',
        'Mean Absolute Error\n(lower = better)',
        'R² Score\n(higher = better)'
    ]

    models = results_df['model'].unique()
    x = np.arange(len(models))
    width = 0.35

    for ax, metric, title in zip(axes, metrics, titles):
        # Get values for each feature set
        expr_vals = []
        grn_vals = []
        for model in models:
            expr_row = results_df[(results_df['model'] == model) & 
                                   (results_df['features'] == 'Expression Only')]
            grn_row = results_df[(results_df['model'] == model) & 
                                  (results_df['features'] == 'Expression + GRN')]
            expr_vals.append(expr_row[metric].values[0] if len(expr_row) > 0 else 0)
            grn_vals.append(grn_row[metric].values[0] if len(grn_row) > 0 else 0)

        # Create bars
        bars1 = ax.bar(x - width/2, expr_vals, width, 
                       label='Expression Only', color=COLORS['expr_only'], edgecolor='black')
        bars2 = ax.bar(x + width/2, grn_vals, width, 
                       label='Expression + GRN', color=COLORS['expr_grn'], edgecolor='black')

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_title(title, fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

        # Add reference lines
        if metric == 'R2':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.3, linewidth=1)
            ax.text(0.02, 0.02, 'R²=0 (random)', transform=ax.transAxes, 
                    fontsize=8, color='red', alpha=0.7)

    plt.suptitle('Per-Gene Baseline Comparison: Expression Only vs Expression + GRN', 
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"      Saved: {save_path}")
    
    plt.close()
    return fig


# =============================================================================
# PLOT 2: Feature Importance (Random Forest)
# =============================================================================

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, 
                            save_path: str = None):
    """
    Horizontal bar chart of feature importances from Random Forest.

    WHY THIS PLOT?
    ==============
    - Shows which features the model actually relied on for predictions
    - GRN features highlighted in orange — easy to see their contribution
    - Directly answers: "Which specific GRN features matter most?"

    WHAT TO LOOK FOR:
    =================
    - own_mRNA should dominate (biological ground truth: protein comes from mRNA)
    - GRN features in top 10-15 = strong evidence they carry signal
    - Total GRN contribution (shown in legend) quantifies overall impact

    INTERPRETATION:
    ===============
    - Importance = mean decrease in impurity (MSE reduction) when feature is used
    - Higher bar = feature was more useful for making accurate predictions
    - If GRN features have ~0 importance, they're not helping the model
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Take top N features
    top_features = importance_df.head(top_n).copy()

    # Color by category
    colors = [COLORS['expression'] if c == 'Expression' else COLORS['grn'] 
              for c in top_features['category']]

    # Horizontal bar chart
    bars = ax.barh(range(len(top_features)), top_features['importance'].values, 
                   color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Importances — Random Forest (Expression + GRN)', 
                 fontsize=12)

    # Calculate totals for legend
    expr_total = importance_df[importance_df['category'] == 'Expression']['importance'].sum()
    grn_total = importance_df[importance_df['category'] == 'GRN']['importance'].sum()

    # Legend with category totals
    legend_elements = [
        Patch(facecolor=COLORS['expression'], edgecolor='black', 
              label=f'Expression ({expr_total:.1%})'),
        Patch(facecolor=COLORS['grn'], edgecolor='black', 
              label=f'GRN Structure ({grn_total:.1%})')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"      Saved: {save_path}")

    plt.close()
    return fig


# =============================================================================
# PLOT 3: GRN Feature Improvement Summary
# =============================================================================

def plot_grn_improvement(results_df: pd.DataFrame, save_path: str = None):
    """
    Bar chart showing MSE reduction (%) from adding GRN features for each model.

    WHY THIS PLOT?
    ==============
    - Single clearest answer to "Do GRN features help?"
    - Positive bars = GRN features reduced error (good)
    - If all bars are positive, the conclusion is strong

    INTERPRETATION:
    ===============
    - Bar height = percentage reduction in MSE
    - 5-10% reduction = modest improvement
    - 10-20% reduction = meaningful improvement
    - >20% reduction = strong evidence GRN features matter
    - Negative bar = GRN features made predictions WORSE (overfitting or noise)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = results_df['model'].unique()
    improvements = []
    model_names = []

    for model in models:
        expr = results_df[(results_df['model'] == model) & 
                           (results_df['features'] == 'Expression Only')]
        grn = results_df[(results_df['model'] == model) & 
                          (results_df['features'] == 'Expression + GRN')]

        if len(expr) > 0 and len(grn) > 0:
            mse_expr = expr['MSE'].values[0]
            mse_grn = grn['MSE'].values[0]
            # Positive = improvement (MSE decreased)
            improvement = ((mse_expr - mse_grn) / mse_expr) * 100
            improvements.append(improvement)
            model_names.append(model)

    # Color bars: green if improved, red if worsened
    colors = [COLORS['best'] if imp > 0 else COLORS['worst'] for imp in improvements]

    bars = ax.bar(model_names, improvements, color=colors, edgecolor='black')

    # Add value labels
    for bar, imp in zip(bars, improvements):
        label = f'{imp:+.1f}%'
        y_pos = bar.get_height() + 0.5 if bar.get_height() >= 0 else bar.get_height() - 1.5
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, label, 
                ha='center', va='bottom' if bar.get_height() >= 0 else 'top', fontsize=11)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('MSE Reduction (%)', fontsize=11)
    ax.set_title('Impact of Adding GRN Features\n(Positive = Lower Error = Better)', fontsize=12)
    ax.set_xlabel('Model', fontsize=11)

    # Add annotation
    avg_improvement = np.mean(improvements)
    ax.text(0.98, 0.98, f'Average improvement: {avg_improvement:+.1f}%', 
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"      Saved: {save_path}")

    plt.close()
    return fig


# =============================================================================
# PLOT 4: Predicted vs Actual Scatter
# =============================================================================

def plot_predicted_vs_actual(predictions_df: pd.DataFrame, model_name: str = None,
                              feature_set: str = None, save_path: str = None):
    """
    Scatter plot of predicted vs actual protein levels.

    WHY THIS PLOT?
    ==============
    - The most intuitive diagnostic for regression model quality
    - Perfect model: all points lie exactly on the diagonal y=x
    - Shows whether errors are random or systematic (e.g., consistent underestimation)

    WHAT TO LOOK FOR:
    =================
    - Points clustered tightly around diagonal = good model
    - Horizontal/vertical spread = poor predictions
    - Curved pattern = model is missing a non-linear relationship
    - Outliers = samples the model struggles with (investigate these!)

    INTERPRETATION:
    ===============
    - Diagonal line = perfect prediction (predicted = actual)
    - Points above diagonal = overestimation
    - Points below diagonal = underestimation
    - R² shown in corner quantifies how close points are to diagonal
    """
    # Filter to specific model/feature set if provided
    plot_df = predictions_df.copy()
    if model_name:
        plot_df = plot_df[plot_df['model'] == model_name]
    if feature_set:
        plot_df = plot_df[plot_df['features'] == feature_set]

    # If no filter, use best performing model
    if model_name is None and feature_set is None:
        # Default to Random Forest + GRN (usually best)
        plot_df = predictions_df[
            (predictions_df['model'] == 'Random Forest') & 
            (predictions_df['features'] == 'Expression + GRN')
        ]
        model_name = 'Random Forest'
        feature_set = 'Expression + GRN'

    fig, ax = plt.subplots(figsize=(8, 8))

    actual = plot_df['actual'].values
    predicted = plot_df['predicted'].values

    # Scatter plot with transparency for overlapping points
    ax.scatter(actual, predicted, alpha=0.3, s=10, c=COLORS['expression'], edgecolors='none')

    # Add perfect prediction line (diagonal)
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction (y=x)')

    # Calculate R² for annotation
    from sklearn.metrics import r2_score
    r2 = r2_score(actual, predicted)

    ax.set_xlabel('Actual Protein Level', fontsize=11)
    ax.set_ylabel('Predicted Protein Level', fontsize=11)
    ax.set_title(f'Predicted vs Actual — {model_name} ({feature_set})', fontsize=12)
    ax.legend(loc='upper left')

    # Add R² annotation
    ax.text(0.95, 0.05, f'R² = {r2:.4f}', transform=ax.transAxes, ha='right', va='bottom',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"      Saved: {save_path}")

    plt.close()
    return fig


# =============================================================================
# PLOT 5: Residual Distribution
# =============================================================================

def plot_residual_distribution(predictions_df: pd.DataFrame, model_name: str = None,
                                feature_set: str = None, save_path: str = None):
    """
    Histogram of prediction residuals (actual - predicted).

    WHY THIS PLOT?
    ==============
    - Shows the distribution of errors
    - Should be centred at 0 (unbiased) and roughly normal (well-behaved)
    - Skewed distribution = systematic bias in one direction

    WHAT TO LOOK FOR:
    =================
    - Centre at 0: predictions are unbiased (no systematic over/under-estimation)
    - Narrow spread: predictions are precise (low variance)
    - Bell curve shape: errors are random, not systematic
    - Long tails: some samples are very hard to predict (outliers)
    - Skewness: model consistently over- or under-predicts

    INTERPRETATION:
    ===============
    - Mean ≈ 0: Unbiased model
    - Small std: Precise predictions
    - Normal distribution: Errors are random (good)
    - Bimodal: Two different "regimes" the model handles differently
    """
    # Filter to specific model/feature set
    plot_df = predictions_df.copy()
    if model_name:
        plot_df = plot_df[plot_df['model'] == model_name]
    if feature_set:
        plot_df = plot_df[plot_df['features'] == feature_set]

    if model_name is None and feature_set is None:
        plot_df = predictions_df[
            (predictions_df['model'] == 'Random Forest') & 
            (predictions_df['features'] == 'Expression + GRN')
        ]
        model_name = 'Random Forest'
        feature_set = 'Expression + GRN'

    fig, ax = plt.subplots(figsize=(10, 6))

    residuals = plot_df['residual'].values

    # Histogram with KDE overlay
    ax.hist(residuals, bins=50, density=True, alpha=0.7, 
            color=COLORS['expression'], edgecolor='black', label='Residuals')

    # Add vertical lines for statistics
    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals)

    ax.axvline(x=0, color='green', linestyle='-', linewidth=2, label='Zero (ideal)')
    ax.axvline(x=mean_resid, color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_resid:.4f}')
    ax.axvline(x=std_resid, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.axvline(x=-std_resid, color='orange', linestyle=':', alpha=0.7, linewidth=1.5,
               label=f'±1 std = {std_resid:.4f}')

    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Residual Distribution — {model_name} ({feature_set})', fontsize=12)
    ax.legend(loc='upper right')

    # Add skewness annotation
    from scipy.stats import skew
    skewness = skew(residuals)
    ax.text(0.02, 0.98, f'Skewness: {skewness:.3f}\n(0 = symmetric)', 
            transform=ax.transAxes, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"      Saved: {save_path}")

    plt.close()
    return fig


# =============================================================================
# PLOT 6: GRN Feature Correlation Matrix
# =============================================================================

def plot_grn_correlation_matrix(data_path: str = None, save_path: str = None):
    """
    Heatmap showing correlations between GRN features and the target.

    WHY THIS PLOT?
    ==============
    - Shows how correlated GRN features are with each other
    - Highly correlated features may be redundant (Elastic Net handles this)
    - Shows which GRN features correlate most strongly with target

    WHAT TO LOOK FOR:
    =================
    - Diagonal = 1.0 (each feature perfectly correlates with itself)
    - Off-diagonal high values = correlated features (potential redundancy)
    - Target column: which features correlate with protein level?
    - Clustering: groups of related features

    INTERPRETATION:
    ===============
    - |r| > 0.7: Strong correlation (features may be redundant)
    - |r| > 0.3: Moderate correlation (worth investigating)
    - r ≈ 0: No linear relationship (but could still have non-linear relationship)
    - Negative r: Inverse relationship (high X → low Y)
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR, "pergene_dataset.csv")

    if not os.path.exists(data_path):
        print(f"      WARNING: Data file not found: {data_path}")
        return None

    df = pd.read_csv(data_path)

    # Select only GRN features and target
    grn_cols = sorted([c for c in df.columns if c.startswith('grn_')])
    cols_to_plot = grn_cols + ['target_protein']

    # Calculate correlation matrix
    corr_matrix = df[cols_to_plot].corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Upper triangle mask

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation'},
        ax=ax
    )

    ax.set_title('GRN Feature Correlation Matrix\n(Lower triangle only)', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"      Saved: {save_path}")

    plt.close()
    return fig


# =============================================================================
# PLOT 7: Elastic Net Coefficient Plot
# =============================================================================

def plot_elasticnet_coefficients(coef_df: pd.DataFrame, top_n: int = 20, 
                                  save_path: str = None):
    """
    Bar chart of Elastic Net coefficients (signed, showing direction).

    WHY THIS PLOT?
    ==============
    - Unlike RF importance, coefficients show DIRECTION of relationship
    - Positive = feature increases protein level
    - Negative = feature decreases protein level
    - Zero = feature was eliminated by L1 penalty (deemed useless)

    WHAT TO LOOK FOR:
    =================
    - own_mRNA should have large positive coefficient (more mRNA → more protein)
    - GRN features with non-zero coefficients = features Elastic Net found useful
    - Sign of GRN coefficients reveals biological interpretation

    INTERPRETATION:
    ===============
    - Positive coef for grn_n_activators: More activators → more protein (makes sense!)
    - Negative coef for grn_n_repressors: More repressors → less protein (makes sense!)
    - Zero coef: Feature was "zeroed out" (L1 penalty deemed it useless)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Take top N by absolute value, but keep sign
    top_features = coef_df.head(top_n).copy()

    # Color by sign AND category
    colors = []
    for _, row in top_features.iterrows():
        if row['coefficient'] > 0:
            colors.append(COLORS['best'])  # Green for positive
        elif row['coefficient'] < 0:
            colors.append(COLORS['worst'])  # Red for negative
        else:
            colors.append(COLORS['neutral'])  # Grey for zero

    # Horizontal bar chart (use actual signed coefficient)
    bars = ax.barh(range(len(top_features)), top_features['coefficient'].values, 
                   color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(top_features)))

    # Add GRN indicator to labels
    labels = []
    for _, row in top_features.iterrows():
        label = row['feature']
        if row['category'] == 'GRN':
            label += ' ★'  # Star indicates GRN feature
        labels.append(label)

    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Coefficient Value (Scaled Features)', fontsize=11)
    ax.set_title(f'Top {top_n} Elastic Net Coefficients\n(★ = GRN feature)', fontsize=12)

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS['best'], edgecolor='black', label='Positive (increases protein)'),
        Patch(facecolor=COLORS['worst'], edgecolor='black', label='Negative (decreases protein)'),
        Patch(facecolor=COLORS['neutral'], edgecolor='black', label='Zero (eliminated)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"      Saved: {save_path}")

    plt.close()
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Load saved results and generate all plots.
    """
    print("=" * 60)
    print("GENERATING PLOTS FROM SAVED RESULTS")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Check for required files
    results_path = os.path.join(TABLES_DIR, "pergene_baseline_results.csv")
    importance_path = os.path.join(TABLES_DIR, "pergene_feature_importance.csv")
    coef_path = os.path.join(TABLES_DIR, "pergene_elasticnet_coefficients.csv")
    pred_path = os.path.join(TABLES_DIR, "pergene_predictions.csv")

    if not os.path.exists(results_path):
        print(f"\nERROR: Results file not found: {results_path}")
        print("Please run train_pergene_baselines.py first.")
        return

    # Load all data
    print("\n[1/7] Loading saved results...")
    results_df = pd.read_csv(results_path)
    print(f"      Loaded {len(results_df)} experiment results")

    # Plot 1: Model comparison
    print("\n[2/7] Plot 1: Model Comparison...")
    plot_model_comparison(
        results_df, 
        save_path=os.path.join(FIGURES_DIR, "01_model_comparison.png")
    )

    # Plot 2: GRN improvement summary
    print("\n[3/7] Plot 2: GRN Improvement Summary...")
    plot_grn_improvement(
        results_df,
        save_path=os.path.join(FIGURES_DIR, "02_grn_improvement.png")
    )

    # Plot 3: Feature importance
    print("\n[4/7] Plot 3: Feature Importance...")
    if os.path.exists(importance_path):
        importance_df = pd.read_csv(importance_path)
        plot_feature_importance(
            importance_df,
            save_path=os.path.join(FIGURES_DIR, "03_feature_importance.png")
        )
    else:
        print(f"      SKIPPED: {importance_path} not found")

    # Plot 4: Elastic Net coefficients
    print("\n[5/7] Plot 4: Elastic Net Coefficients...")
    if os.path.exists(coef_path):
        coef_df = pd.read_csv(coef_path)
        plot_elasticnet_coefficients(
            coef_df,
            save_path=os.path.join(FIGURES_DIR, "04_elasticnet_coefficients.png")
        )
    else:
        print(f"      SKIPPED: {coef_path} not found")

    # Plot 5: Predicted vs Actual
    print("\n[6/7] Plot 5: Predicted vs Actual...")
    if os.path.exists(pred_path):
        predictions_df = pd.read_csv(pred_path)
        plot_predicted_vs_actual(
            predictions_df,
            save_path=os.path.join(FIGURES_DIR, "05_predicted_vs_actual.png")
        )
    else:
        print(f"      SKIPPED: {pred_path} not found")

    # Plot 6: Residual Distribution
    print("\n[7/7] Plot 6: Residual Distribution...")
    if os.path.exists(pred_path):
        plot_residual_distribution(
            predictions_df,
            save_path=os.path.join(FIGURES_DIR, "06_residual_distribution.png")
        )
    else:
        print(f"      SKIPPED: {pred_path} not found")

    # Bonus: Correlation matrix (uses raw data)
    print("\n[BONUS] Plot 7: GRN Correlation Matrix...")
    plot_grn_correlation_matrix(
        save_path=os.path.join(FIGURES_DIR, "07_grn_correlation_matrix.png")
    )

    print("\n" + "=" * 60)
    print(f"All plots saved to: {os.path.abspath(FIGURES_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()