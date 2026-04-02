"""
plot_mrna_baselines.py — Visualisations for mRNA tabular baselines.

Uses outputs from train_mrna_baseline.py:
- mrna_baselines_results_long.csv
- mrna_feature_importance.csv
- mrna_rf_grn_predictions.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "mrna_baselines")

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "expr_only": "#90CAF9",
    "expr_grn": "#FF8A65",
    "expression": "#2196F3",
    "grn": "#FF5722",
    "best": "#4CAF50",
    "worst": "#F44336",
    "neutral": "#9E9E9E",
}


def plot_model_comparison(results_long, save_path=None):
    """Grouped bar chart: Ridge/RF × (Params vs Params+GRN)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["MSE", "MAE", "R2"]
    titles = [
        "Mean Squared Error\n(lower = better)",
        "Mean Absolute Error\n(lower = better)",
        "R² Score\n(higher = better)",
    ]

    models = ["Ridge", "RF"]
    x = np.arange(len(models))
    width = 0.35

    for ax, metric, title in zip(axes, metrics, titles):
        expr_vals = []
        grn_vals = []
        for m in models:
            expr_row = results_long[
                (results_long["model"] == m)
                & (results_long["features"] == "Params Only")
            ]
            grn_row = results_long[
                (results_long["model"] == m)
                & (results_long["features"] == "Params + GRN")
            ]
            expr_vals.append(expr_row[metric].values[0])
            grn_vals.append(grn_row[metric].values[0])

        bars1 = ax.bar(
            x - width / 2,
            expr_vals,
            width,
            label="Params Only",
            color=COLORS["expr_only"],
            edgecolor="black",
        )
        bars2 = ax.bar(
            x + width / 2,
            grn_vals,
            width,
            label="Params + GRN",
            color=COLORS["expr_grn"],
            edgecolor="black",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        # Make room for the legend and floating text without overlapping bars
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        if ymin < 0:
            ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.25 * yrange)
        else:
            ax.set_ylim(0, ymax + 0.2 * ymax)
            
        ax.legend(loc="upper right")

        for bar in list(bars1) + list(bars2):
            yval = bar.get_height()
            
            # Place negative text below the bar, positive text above
            if yval < 0:
                ypos = yval - (0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
                va = "top"
            else:
                ypos = yval + (0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
                va = "bottom"
                
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"{yval:.3f}",
                ha="center",
                va=va,
                fontsize=8,
            )

        if metric == "R2":
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.text(
                0.02,
                0.02,
                "R²=0 (random)",
                transform=ax.transAxes,
                fontsize=8,
                color="red",
                alpha=0.7,
            )

    plt.suptitle(
        "mRNA Baseline Comparison: Params Only vs Params + GRN", fontsize=14, y=1.02
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_grn_improvement(results_long, save_path=None):
    """Bar chart: percentage MSE reduction from adding GRN features."""
    models = ["Ridge", "RF"]
    improvements = []

    for m in models:
        expr_mse = results_long[
            (results_long["model"] == m) & (results_long["features"] == "Params Only")
        ]["MSE"].values[0]
        grn_mse = results_long[
            (results_long["model"] == m) & (results_long["features"] == "Params + GRN")
        ]["MSE"].values[0]
        reduction = (expr_mse - grn_mse) / expr_mse * 100.0
        improvements.append(reduction)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, improvements, color=COLORS["best"], edgecolor="black")

    ax.set_ylabel("MSE Reduction (%)", fontsize=11)
    ax.set_title(
        "Impact of Adding GRN Features\n(Positive = Lower Error = Better)", fontsize=13
    )
    ax.axhline(y=0, color="black", linewidth=1)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    avg_impr = np.mean(improvements)
    ax.text(
        0.99,
        0.95,
        f"Average improvement: {avg_impr:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_feature_importance(fi_df, save_path=None):
    """RF feature importance: Expression vs GRN (matches Appendix style)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    top_features = fi_df.head(20).copy()
    colors = [
        COLORS["expression"] if c == "Expression" else COLORS["grn"]
        for c in top_features["category"]
    ]

    ax.barh(
        range(len(top_features)),
        top_features["importance"].values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title(
        "Top 20 Feature Importances — Random Forest (mRNA, Params + GRN)", fontsize=12
    )

    expr_total = fi_df[fi_df["category"] == "Expression"]["importance"].sum()
    grn_total = fi_df[fi_df["category"] == "GRN"]["importance"].sum()

    legend_elements = [
        Patch(
            facecolor=COLORS["expression"],
            edgecolor="black",
            label=f"Expression / params ({expr_total:.1%})",
        ),
        Patch(
            facecolor=COLORS["grn"],
            edgecolor="black",
            label=f"GRN structure ({grn_total:.1%})",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_pred_vs_actual_and_residuals(pred_df, save_prefix=None):
    """Predicted vs actual and residual histogram for RF + GRN."""
    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values
    residuals = y_true - y_pred

    # Predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#2196F3")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction (y=x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual log1p(mRNA)")
    ax.set_ylabel("Predicted log1p(mRNA)")
    r2 = pred_df["y_true"].corr(pred_df["y_pred"]) ** 2
    ax.text(
        0.97,
        0.03,
        f"R² ≈ {r2:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_title("Predicted vs Actual — Random Forest (Params + GRN)")
    ax.legend()
    plt.tight_layout()
    if save_prefix:
        path = os.path.join(RESULTS_DIR, f"{save_prefix}_pred_vs_actual.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    # Residuals
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(residuals, bins=40, kde=False, ax=ax, color="#64B5F6")
    ax.axvline(0, color="green", linewidth=2, label="Zero (ideal)")
    mean = residuals.mean()
    std = residuals.std()
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean = {mean:.2f}")
    ax.axvline(mean + std, color="orange", linestyle=":", alpha=0.7)
    ax.axvline(mean - std, color="orange", linestyle=":", alpha=0.7)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution — Random Forest (Params + GRN)")
    skewness = pd.Series(residuals).skew()
    ax.text(
        0.02,
        0.98,
        f"Skewness: {skewness:.3f}\n(0 = symmetric)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend()
    plt.tight_layout()
    if save_prefix:
        path = os.path.join(RESULTS_DIR, f"{save_prefix}_residuals.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_long_path = os.path.join(
        RESULTS_DIR, "mrna_baselines_results_long.csv"
    )
    fi_path = os.path.join(RESULTS_DIR, "mrna_feature_importance.csv")
    pred_path = os.path.join(RESULTS_DIR, "mrna_rf_grn_predictions.csv")

    if not os.path.exists(results_long_path):
        print(f"ERROR: {results_long_path} not found. Run train_mrna_baseline.py first.")
        return

    results_long = pd.read_csv(results_long_path)

    print("[1/4] Plotting model comparison...")
    plot_model_comparison(
        results_long,
        save_path=os.path.join(RESULTS_DIR, "mrna_model_comparison.png"),
    )

    print("[2/4] Plotting GRN improvement summary...")
    plot_grn_improvement(
        results_long,
        save_path=os.path.join(RESULTS_DIR, "mrna_grn_improvement.png"),
    )

    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path)
        print("[3/4] Plotting feature importances...")
        plot_feature_importance(
            fi_df,
            save_path=os.path.join(RESULTS_DIR, "mrna_feature_importance.png"),
        )
    else:
        print(f"WARNING: {fi_path} not found; skipping feature-importance plot.")

    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        print("[4/4] Plotting predicted vs actual and residuals...")
        plot_pred_vs_actual_and_residuals(pred_df, save_prefix="mrna_rf_grn")
    else:
        print(f"WARNING: {pred_path} not found; skipping diagnostics plots.")


if __name__ == "__main__":
    main()