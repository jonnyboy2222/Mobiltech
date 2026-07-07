# v3_bayesian_3dcd_proto/likelihood_heads.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float = 0.1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),

        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),

        nn.Linear(hidden_dim, output_dim),
    )


class DirectLRHead(nn.Module):
    """
    v3-A: likelihood ratio 직접 근사.

    Output:
        log_ratio ≈ log p(e | H1) - log p(e | H0)

    실제 학습은 BCEWithLogitsLoss로 하지만,
    해석은 learned log-likelihood ratio로 둔다.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = make_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout,
        )

    def forward(self, evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            evidence: [B, N, C]

        Returns:
            {
                "log_ratio": [B, N]
            }
        """
        log_ratio = self.net(evidence).squeeze(-1)

        return {
            "log_ratio": log_ratio,
        }


class SeparateNPHead(nn.Module):
    """
    v3-B: H0/H1 energy를 따로 근사한 뒤 NP lemma 적용.

    Energy 해석:
        log p(e | H0) ≈ -E0(e)
        log p(e | H1) ≈ -E1(e)

    따라서:
        log Lambda
        = log p(e | H1) - log p(e | H0) + log prior_ratio
        = -E1 - (-E0) + prior
        = E0 - E1 + prior
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        prior_logit: float = 0.0,
    ) -> None:
        super().__init__()

        self.h0_energy_net = make_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout,
        )

        self.h1_energy_net = make_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout,
        )

        self.register_buffer(
            "prior_logit",
            torch.tensor(float(prior_logit), dtype=torch.float32),
        )

    def forward(self, evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            evidence: [B, N, C]

        Returns:
            {
                "h0_energy": [B, N],
                "h1_energy": [B, N],
                "log_p_h0": [B, N],
                "log_p_h1": [B, N],
                "log_ratio": [B, N]
            }
        """
        # softplus로 energy를 non-negative하게 둠
        h0_energy = F.softplus(self.h0_energy_net(evidence).squeeze(-1))
        h1_energy = F.softplus(self.h1_energy_net(evidence).squeeze(-1))

        log_p_h0 = -h0_energy
        log_p_h1 = -h1_energy

        log_ratio = log_p_h1 - log_p_h0 + self.prior_logit
        # same as: h0_energy - h1_energy + prior

        return {
            "h0_energy": h0_energy,
            "h1_energy": h1_energy,
            "log_p_h0": log_p_h0,
            "log_p_h1": log_p_h1,
            "log_ratio": log_ratio,
        }