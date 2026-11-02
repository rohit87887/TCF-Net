"""
TCF-Net: Tri-Modal Credibility Fusion Network (with Code-Mixed MuRIL & Multi-Task Hate Speech Support)
Architecture:
  - Modality A: Semantic stream (DistilBERT or google/muril-base-cased + optional TF-IDF/SVD)
  - Modality B: Stylometric & Deception linguistics (22-D)
  - Modality C: Propagation SIR cascade simulator (8-D)
  - Cross-Modal Gated Fusion with dynamic per-sample softmax attention and anti-collapse regularizer
  - SelectiveNet Abstention Mechanism:
      * f(x): Veracity Head (Real vs. Fake)
      * g(x): Selection Head (Confidence score in [0, 1])
      * h(x): Auxiliary Veracity Head
  - Multi-Task Hate Speech Head:
      * f_hate(x): Hate Speech Head (Not-Hateful vs. Hateful)
      * h_hate(x): Auxiliary Hate Speech Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class GatedFusion(nn.Module):
    """
    Learns dynamic per-sample attention weights across Semantic, Stylometric, and Propagation streams.
    """
    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Linear(d, 3)
        )

    def forward(self, zt: torch.Tensor, zs: torch.Tensor, zp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate projected representations: (B, 3*d)
        cat = torch.cat([zt, zs, zp], dim=-1)
        alpha = torch.softmax(self.gate(cat), dim=-1) # (B, 3)
        
        # Weighted combination of modalities
        fused = (alpha[:, 0:1] * zt + alpha[:, 1:2] * zs + alpha[:, 2:3] * zp)
        return fused, alpha


class TCFNet(nn.Module):
    """
    Tri-Modal Credibility Fusion Network with Selective Abstention and Multi-Task Hate Speech Head.
    """
    def __init__(
        self,
        d_text: int = 768,
        d_style: int = 22,
        d_prop: int = 8,
        d_model: int = 256,
        dropout: float = 0.3,
        use_style: bool = True,
        use_prop: bool = True,
        enable_hate_head: bool = True
    ):
        super().__init__()
        self.d_text = d_text
        self.d_style = d_style
        self.d_prop = d_prop
        self.d_model = d_model
        self.use_style = use_style
        self.use_prop = use_prop
        self.enable_hate_head = enable_hate_head

        # Projections to common d_model dimension
        def make_proj(in_dim: int):
            return nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        self.pt = make_proj(d_text)
        self.ps = make_proj(d_style)
        self.pp = make_proj(d_prop)

        # Dynamic cross-modal gated fusion
        self.fuse = GatedFusion(d_model)

        # Shared representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # --- Task 1: Veracity Selective Prediction (SelectiveNet) ---
        self.head_f = nn.Linear(d_model, 2)                                      # Main Veracity Head (0: Real, 1: Fake)
        self.head_g = nn.Sequential(                                             # Selection / Abstention Head
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        self.head_h = nn.Linear(d_model, 2)                                      # Auxiliary Veracity Head

        # --- Task 2: Multi-Task Hate Speech Detection ---
        if self.enable_hate_head:
            self.head_hate_f = nn.Linear(d_model, 2)                             # Hate Head (0: Not Hateful, 1: Hateful)
            self.head_hate_h = nn.Linear(d_model, 2)                             # Auxiliary Hate Head
        else:
            self.head_hate_f = None
            self.head_hate_h = None

    def forward(
        self,
        xt: torch.Tensor,
        xs: Optional[torch.Tensor] = None,
        xp: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            xt: Semantic tensor (B, d_text)
            xs: Stylometric tensor (B, d_style)
            xp: Propagation tensor (B, d_prop)
        Returns:
            Dict containing:
                'logits_veracity': (B, 2)
                'selection': (B,) confidence scores in [0, 1]
                'aux_veracity': (B, 2)
                'alpha': (B, 3) modality fusion weights
                'logits_hate': (B, 2) [if hate head enabled]
                'aux_hate': (B, 2) [if hate head enabled]
        """
        zt = self.pt(xt)
        zs = self.ps(xs) if (self.use_style and xs is not None) else torch.zeros_like(zt)
        zp = self.pp(xp) if (self.use_prop and xp is not None) else torch.zeros_like(zt)

        fused, alpha = self.fuse(zt, zs, zp)
        trunk_repr = self.trunk(fused)

        # Veracity branch
        logits_f = self.head_f(trunk_repr)
        selection = self.head_g(trunk_repr).squeeze(-1)
        logits_h = self.head_h(trunk_repr)

        out = {
            'logits_veracity': logits_f,
            'selection': selection,
            'aux_veracity': logits_h,
            'alpha': alpha,
            'trunk': trunk_repr
        }

        # Multi-task Hate branch
        if self.enable_hate_head:
            out['logits_hate'] = self.head_hate_f(trunk_repr)
            out['aux_hate'] = self.head_hate_h(trunk_repr)

        return out


def selective_loss(
    logits_veracity: torch.Tensor,
    selection: torch.Tensor,
    aux_veracity: torch.Tensor,
    alpha: torch.Tensor,
    y_veracity: torch.Tensor,
    target_coverage: float = 0.85,
    lambda_cov: float = 32.0,
    alpha_sel: float = 0.5,
    gate_entropy_weight: float = 0.02,
    logits_hate: Optional[torch.Tensor] = None,
    aux_hate: Optional[torch.Tensor] = None,
    y_hate: Optional[torch.Tensor] = None,
    lambda_hate: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes SelectiveNet multi-objective loss with optional multi-task hate speech classification.
    """
    # 1. Selective risk on answered veracity samples
    ce_veracity = F.cross_entropy(logits_veracity, y_veracity, reduction='none')
    phi = selection.mean()
    selective_risk = (selection * ce_veracity).mean() / (phi + 1e-8)
    
    # Coverage constraint penalty
    cov_penalty = lambda_cov * (torch.clamp(target_coverage - phi, min=0.0) ** 2)
    
    # Auxiliary cross entropy for veracity
    aux_ce_veracity = F.cross_entropy(aux_veracity, y_veracity)
    veracity_loss = alpha_sel * (selective_risk + cov_penalty) + (1.0 - alpha_sel) * aux_ce_veracity

    # 2. Gate entropy penalty to prevent collapse onto single modality
    # H(alpha) = - sum(alpha * log(alpha))
    gate_entropy = - (alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean()
    entropy_penalty = - gate_entropy_weight * gate_entropy

    total_loss = veracity_loss + entropy_penalty

    metrics = {
        'loss_veracity': veracity_loss.item(),
        'selective_risk': selective_risk.item(),
        'coverage': phi.item(),
        'cov_penalty': cov_penalty.item(),
        'gate_entropy': gate_entropy.item()
    }

    # 3. Optional multi-task hate speech classification
    if logits_hate is not None and y_hate is not None:
        # Ignore mask for samples where hate label is unobserved (e.g. -1)
        valid_mask = (y_hate >= 0)
        if valid_mask.any():
            hate_ce = F.cross_entropy(logits_hate[valid_mask], y_hate[valid_mask])
            aux_hate_ce = F.cross_entropy(aux_hate[valid_mask], y_hate[valid_mask]) if aux_hate is not None else 0.0
            hate_total = 0.7 * hate_ce + 0.3 * aux_hate_ce
            total_loss = total_loss + lambda_hate * hate_total
            metrics['loss_hate'] = hate_total.item()

    metrics['loss_total'] = total_loss.item()
    return total_loss, metrics

