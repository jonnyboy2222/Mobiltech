# v3_bayesian_3dcd_proto/point_encoder.py

from __future__ import annotations

import torch
import torch.nn as nn


class PointMLPEncoder(nn.Module):
    """
    Shared point-wise encoder.

    Input:
        point_feat: [B, N, C]
            C = 3 if xyz only
            C = 6 if xyz + rgb

    Output:
        encoded_feat: [B, N, D]
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        feature_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, point_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_feat: [B, N, C]

        Returns:
            encoded_feat: [B, N, D]
        """
        if point_feat.dim() != 3:
            raise ValueError(f"point_feat must be [B, N, C], got {point_feat.shape}")

        b, n, c = point_feat.shape

        x = point_feat.reshape(b * n, c)
        x = self.net(x)
        x = x.reshape(b, n, -1)

        return x