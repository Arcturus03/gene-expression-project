"""
deep_rn_model.py — DeepRN-Struct: Dual-stream MLP for GRN-aware protein prediction.

ARCHITECTURE DESIGN RATIONALE:
Expression features (mRNA levels) and GRN features (graph topology metrics) 
have fundamentally different distributions, scales, and biological meanings.
Naive concatenation forces the same early layers to process both simultaneously,
which leads to interference — as observed with the sklearn MLP baseline.

The dual-stream design gives each feature group its own encoder:
- Expression encoder: learns mRNA co-expression patterns
- GRN encoder: learns which regulatory topologies matter
- Fusion layer: learns how the two interact to predict protein output

This is structurally analogous to BioM2 (Zhang et al. 2024), which also uses
biologically-informed multi-stage feature processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepRNStruct(nn.Module):
    """
    Dual-stream MLP: separate encoders for expression and GRN features,
    fused before the prediction head.
    
    Args:
        n_expr_features:   number of expression input features (e.g. 31)
        n_grn_features:    number of GRN structural features (e.g. 9)
        expr_hidden:       hidden size of expression encoder
        grn_hidden:        hidden size of GRN encoder  
        fusion_hidden:     hidden sizes of fusion MLP (list)
        dropout:           dropout probability applied after each layer
    """

    def __init__(
        self,
        n_expr_features: int,
        n_grn_features: int,
        expr_hidden: int = 128,
        grn_hidden: int = 64,
        fusion_hidden: list = [128, 64],
        dropout: float = 0.3,
    ):
        super(DeepRNStruct, self).__init__()

        # --- Expression stream ---
        self.expr_encoder = nn.Sequential(
            nn.Linear(n_expr_features, expr_hidden),    # grn_hidden=64
            nn.BatchNorm1d(expr_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expr_hidden, expr_hidden // 2),
            nn.BatchNorm1d(expr_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        expr_out_dim = expr_hidden // 2  # = 64

        # --- GRN stream ---
        self.grn_encoder = nn.Sequential(
            nn.Linear(n_grn_features, grn_hidden),     # 9 → 64
            nn.BatchNorm1d(grn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(grn_hidden, grn_hidden // 4),    # 64 → 16  ← much smaller output
            nn.BatchNorm1d(grn_hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        grn_out_dim = grn_hidden // 4  # = 16  ← proportional to 2% signal

        # --- Fusion stream ---
        fusion_in_dim = expr_out_dim + grn_out_dim  # = 128

        fusion_layers = []
        prev_dim = fusion_in_dim
        for hidden_dim in fusion_hidden:
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Final prediction: single output (log1p protein level)
        fusion_layers.append(nn.Linear(prev_dim, 1))

        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, x_expr: torch.Tensor, x_grn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_expr: [batch, n_expr_features] — scaled mRNA expression values
            x_grn:  [batch, n_grn_features]  — scaled GRN structural features
        Returns:
            pred:   [batch] — predicted log1p(protein)
        """
        expr_repr = self.expr_encoder(x_expr)   # [batch, 64]
        grn_repr  = self.grn_encoder(x_grn)     # [batch, 64]
        fused     = torch.cat([expr_repr, grn_repr], dim=1)  # [batch, 128]
        out       = self.fusion(fused)           # [batch, 1]
        return out.squeeze(1)                    # [batch]


class DeepRNExpr(nn.Module):
    """
    Expression-only version of DeepRN — the deep expression baseline.
    Identical architecture to DeepRNStruct minus the GRN stream, so any
    performance difference is attributable solely to GRN features.
    
    This is your DeepRN-Expr (deep expression-only baseline).
    """

    def __init__(
        self,
        n_expr_features: int,
        expr_hidden: int = 128,
        fusion_hidden: list = [128, 64],
        dropout: float = 0.3,
    ):
        super(DeepRNExpr, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_expr_features, expr_hidden),
            nn.BatchNorm1d(expr_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expr_hidden, expr_hidden // 2),
            nn.BatchNorm1d(expr_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        encoder_out = expr_hidden // 2  # 64

        fusion_layers = []
        prev_dim = encoder_out
        for h in fusion_hidden:
            fusion_layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        fusion_layers.append(nn.Linear(prev_dim, 1))
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, x_expr: torch.Tensor) -> torch.Tensor:
        return self.fusion(self.encoder(x_expr)).squeeze(1)
