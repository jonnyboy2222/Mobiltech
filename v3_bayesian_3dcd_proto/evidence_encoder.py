# v3_bayesian_3dcd_proto/evidence_encoder.py

from __future__ import annotations

import torch
import torch.nn as nn


class EvidenceEncoder(nn.Module):
    """
    Raw geometry/RGB evidence를 latent evidence로 변환하는 point-wise MLP.

    Input:
        evidence: [B, N, C_in]

    Output:
        encoded_evidence: [B, N, C_out]

    역할:
        GeometryEvidenceBuilder가 만든 raw evidence를 바로 decision head에 넣지 않고,
        한 번 non-linear representation으로 변환한다.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1,
        use_residual: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_residual = use_residual and (input_dim == output_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            evidence: [B, N, C]

        Returns:
            encoded evidence: [B, N, D]
        """
        if evidence.dim() != 3:
            raise ValueError(
                f"evidence must be [B, N, C], got {evidence.shape}"
            )

        out = self.net(evidence)

        if self.use_residual:
            out = out + evidence

        return out