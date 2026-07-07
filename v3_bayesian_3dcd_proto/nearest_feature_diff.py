# v3_bayesian_3dcd_proto/nearest_feature_diff.py

from __future__ import annotations

import torch
import torch.nn as nn


class NearestFeatureDifference(nn.Module):
    """
    Memory-safe nearest feature difference module.

    기존 문제:
        dist = torch.cdist(target_xyz, source_xyz)
        -> [B, Nt, Ns] 전체 distance matrix 생성
        -> Nt, Ns가 크면 OOM

    수정:
        target point를 chunk 단위로 나눠 cdist 계산.
    """

    def __init__(
        self,
        k_neighbors: int = 8,
        chunk_size: int = 1024,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.k_neighbors = k_neighbors
        self.chunk_size = chunk_size
        self.eps = eps

    def _forward_chunk(
        self,
        target_xyz: torch.Tensor,
        target_feat: torch.Tensor,
        source_xyz: torch.Tensor,
        source_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            target_xyz:  [B, Nc, 3]
            target_feat: [B, Nc, D]
            source_xyz:  [B, Ns, 3]
            source_feat: [B, Ns, D]
        """
        b, nc, _ = target_xyz.shape
        _, ns, _ = source_xyz.shape
        _, _, d = source_feat.shape

        k = min(self.k_neighbors, ns)

        # KNN index는 미분할 필요 없음
        with torch.no_grad():
            # [B, Nc, Ns]
            dist = torch.cdist(target_xyz, source_xyz)

            # [B, Nc, K]
            knn_dist, knn_idx = torch.topk(
                dist,
                k=k,
                dim=-1,
                largest=False,
            )

        # feature gather
        source_feat_expanded = source_feat.unsqueeze(1).expand(-1, nc, -1, -1)
        idx_feat = knn_idx.unsqueeze(-1).expand(-1, -1, -1, d)

        # [B, Nc, K, D]
        knn_feat = torch.gather(
            source_feat_expanded,
            dim=2,
            index=idx_feat,
        )

        # xyz gather
        source_xyz_expanded = source_xyz.unsqueeze(1).expand(-1, nc, -1, -1)
        idx_xyz = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)

        # [B, Nc, K, 3]
        knn_xyz = torch.gather(
            source_xyz_expanded,
            dim=2,
            index=idx_xyz,
        )

        # inverse-distance weighting
        weight = 1.0 / (knn_dist + self.eps)
        weight = weight / (weight.sum(dim=-1, keepdim=True) + self.eps)

        # [B, Nc, D]
        agg_source_feat = (knn_feat * weight.unsqueeze(-1)).sum(dim=2)

        # [B, Nc, 3]
        agg_source_xyz = (knn_xyz * weight.unsqueeze(-1)).sum(dim=2)

        rel_xyz = target_xyz - agg_source_xyz
        nearest_dist = knn_dist[..., 0]
        mean_dist = knn_dist.mean(dim=-1)

        feat_diff = target_feat - agg_source_feat
        abs_feat_diff = torch.abs(feat_diff)

        evidence = torch.cat(
            [
                target_feat,
                agg_source_feat,
                feat_diff,
                abs_feat_diff,
                rel_xyz,
                nearest_dist.unsqueeze(-1),
                mean_dist.unsqueeze(-1),
            ],
            dim=-1,
        )

        return {
            "evidence": evidence,
            "nearest_dist": nearest_dist,
            "mean_dist": mean_dist,
            "rel_xyz": rel_xyz,
        }

    def forward(
        self,
        target_xyz: torch.Tensor,
        target_feat: torch.Tensor,
        source_xyz: torch.Tensor,
        source_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            target_xyz:  [B, Nt, 3]
            target_feat: [B, Nt, D]
            source_xyz:  [B, Ns, 3]
            source_feat: [B, Ns, D]

        Returns:
            evidence: [B, Nt, 4D + 5]
        """
        if target_xyz.dim() != 3 or source_xyz.dim() != 3:
            raise ValueError("target_xyz and source_xyz must be [B, N, 3]")

        _, nt, _ = target_xyz.shape

        evidence_chunks = []
        nearest_dist_chunks = []
        mean_dist_chunks = []
        rel_xyz_chunks = []

        for start in range(0, nt, self.chunk_size):
            end = min(start + self.chunk_size, nt)

            out = self._forward_chunk(
                target_xyz=target_xyz[:, start:end, :],
                target_feat=target_feat[:, start:end, :],
                source_xyz=source_xyz,
                source_feat=source_feat,
            )

            evidence_chunks.append(out["evidence"])
            nearest_dist_chunks.append(out["nearest_dist"])
            mean_dist_chunks.append(out["mean_dist"])
            rel_xyz_chunks.append(out["rel_xyz"])

        evidence = torch.cat(evidence_chunks, dim=1)
        nearest_dist = torch.cat(nearest_dist_chunks, dim=1)
        mean_dist = torch.cat(mean_dist_chunks, dim=1)
        rel_xyz = torch.cat(rel_xyz_chunks, dim=1)

        return {
            "evidence": evidence,
            "nearest_dist": nearest_dist,
            "mean_dist": mean_dist,
            "rel_xyz": rel_xyz,
        }