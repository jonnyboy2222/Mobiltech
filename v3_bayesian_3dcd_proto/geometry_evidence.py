# v3_bayesian_3dcd_proto/geometry_evidence.py

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GeometryEvidenceBuilder(nn.Module):
    """
    Geometry-only evidence builder.

    목적:
        nearest_feature_diff.py의 GPU cdist 병목을 피하고,
        lightweight geometry evidence만 구성한다.

    Evidence:
        rel_xyz              3
        nearest_dist         1
        mean_dist            1
        std_dist             1
        inv_mean_dist        1
        -----------------------
        geometry total       7

    Optional RGB evidence:
        target_rgb           3
        agg_source_rgb       3
        abs_rgb_diff         3
        -----------------------
        rgb total            9

    Total:
        use_rgb=False -> 7
        use_rgb=True  -> 16
    """

    def __init__(
        self,
        k_neighbors: int = 8,
        use_rgb: bool = True,
        backend: str = "cpu_kdtree",
        source_chunk_size: int = 8192,
        eps: float = 1e-8,
        use_axis_evidence: bool = True,
        use_support_evidence: bool = False,
        use_structure_evidence: bool = False
    ) -> None:
        super().__init__()

        if backend not in ["cpu_kdtree", "torch_cpu_chunk"]:
            raise ValueError(
                f"Unknown geometry backend: {backend}. "
                f"Expected ['cpu_kdtree', 'torch_cpu_chunk']."
            )

        self.k_neighbors = k_neighbors
        self.use_rgb = use_rgb
        self.backend = backend
        self.source_chunk_size = source_chunk_size
        self.eps = eps
        self.use_axis_evidence = use_axis_evidence
        self.use_support_evidence = use_support_evidence
        self.use_structure_evidence = use_structure_evidence

        # base geometry:
        # rel_xyz, nearest_dist, mean_dist, std_dist, inv_mean_dist = 7
        base_dim = 7
        # rgb:
        # target_rgb, agg_source_rgb, abs_rgb_diff = 9
        rgb_dim = 9 if use_rgb else 0
        axis_dim = 8 if use_axis_evidence else 0
        support_dim = 6 if use_support_evidence else 0
        structure_dim = 10 if use_structure_evidence else 0

        self.evidence_dim = (
            base_dim
            + rgb_dim
            + axis_dim
            + support_dim
            + structure_dim
        )

    @staticmethod
    def _to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def _knn_cpu_kdtree(
        self,
        target_xyz: torch.Tensor,
        source_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CPU cKDTree 기반 KNN.

        Args:
            target_xyz: [N, 3]
            source_xyz: [M, 3]

        Returns:
            knn_dist: [N, K]
            knn_idx:  [N, K]
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError as e:
            raise ImportError(
                "geo_backend='cpu_kdtree'를 쓰려면 scipy가 필요합니다. "
                "설치: pip install scipy"
            ) from e

        device = target_xyz.device

        target_np = self._to_numpy(target_xyz).astype(np.float32)
        source_np = self._to_numpy(source_xyz).astype(np.float32)

        n_source = source_np.shape[0]
        k = min(self.k_neighbors, n_source)

        tree = cKDTree(source_np)

        # scipy 버전에 따라 workers 인자가 없을 수 있음
        try:
            dist_np, idx_np = tree.query(target_np, k=k, workers=-1)
        except TypeError:
            dist_np, idx_np = tree.query(target_np, k=k)

        if k == 1:
            dist_np = dist_np[:, None]
            idx_np = idx_np[:, None]

        knn_dist = torch.as_tensor(
            dist_np,
            device=device,
            dtype=torch.float32,
        )

        knn_idx = torch.as_tensor(
            idx_np,
            device=device,
            dtype=torch.long,
        )

        return knn_dist, knn_idx

    def _knn_torch_cpu_chunk(
        self,
        target_xyz: torch.Tensor,
        source_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        scipy가 없을 때 쓰는 CPU torch chunk KNN.

        정확한 KNN이지만 느릴 수 있음.
        GPU VRAM을 쓰지 않도록 CPU에서 cdist를 수행한다.
        """
        device = target_xyz.device

        target_cpu = target_xyz.detach().cpu().float()
        source_cpu = source_xyz.detach().cpu().float()

        n_target = target_cpu.shape[0]
        n_source = source_cpu.shape[0]
        k = min(self.k_neighbors, n_source)

        best_dist = None
        best_idx = None

        for s0 in range(0, n_source, self.source_chunk_size):
            s1 = min(s0 + self.source_chunk_size, n_source)

            source_chunk = source_cpu[s0:s1]

            # [N, Sc] on CPU
            dist = torch.cdist(target_cpu, source_chunk)

            local_k = min(k, s1 - s0)

            local_dist, local_idx = torch.topk(
                dist,
                k=local_k,
                dim=-1,
                largest=False,
            )
            local_idx = local_idx + s0

            if best_dist is None:
                best_dist = local_dist
                best_idx = local_idx
            else:
                merged_dist = torch.cat([best_dist, local_dist], dim=-1)
                merged_idx = torch.cat([best_idx, local_idx], dim=-1)

                best_dist, select_idx = torch.topk(
                    merged_dist,
                    k=k,
                    dim=-1,
                    largest=False,
                )
                best_idx = torch.gather(
                    merged_idx,
                    dim=-1,
                    index=select_idx,
                )

            del dist, local_dist, local_idx

        return best_dist.to(device), best_idx.to(device)

    def _knn(
        self,
        target_xyz: torch.Tensor,
        source_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "cpu_kdtree":
            return self._knn_cpu_kdtree(target_xyz, source_xyz)

        return self._knn_torch_cpu_chunk(target_xyz, source_xyz)

    def _build_one(
        self,
        target_xyz: torch.Tensor,
        source_xyz: torch.Tensor,
        target_rgb: torch.Tensor | None = None,
        source_rgb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            target_xyz: [Nt, 3]
            source_xyz: [Ns, 3]
            target_rgb: [Nt, 3] or None
            source_rgb: [Ns, 3] or None

        Returns:
            evidence: [Nt, C]
        """
        knn_dist, knn_idx = self._knn(
            target_xyz=target_xyz,
            source_xyz=source_xyz,
        )

        nt, k = knn_idx.shape

        # gather source xyz
        knn_xyz = source_xyz[knn_idx]  # [Nt, K, 3]

        weight = 1.0 / (knn_dist + self.eps)
        weight = weight / (weight.sum(dim=-1, keepdim=True) + self.eps)

        agg_source_xyz = (knn_xyz * weight.unsqueeze(-1)).sum(dim=1)

        rel_xyz = target_xyz - agg_source_xyz
        nearest_dist = knn_dist[:, 0:1]
        mean_dist = knn_dist.mean(dim=1, keepdim=True)
        std_dist = knn_dist.std(dim=1, keepdim=True, unbiased=False)
        inv_mean_dist = 1.0 / (mean_dist + self.eps)

        evidence_list = [
            rel_xyz,
            nearest_dist,
            mean_dist,
            std_dist,
            inv_mean_dist,
        ]

        if self.use_support_evidence:
            # KNN distance distribution support evidence.
            # knn_dist: [N, K]
            sorted_dist, _ = torch.sort(knn_dist, dim=1)
            k_eff = sorted_dist.shape[1]

            max_dist = sorted_dist[:, -1:]

            median_idx = k_eff // 2
            median_dist = sorted_dist[:, median_idx:median_idx + 1]

            q75_idx = min(int(0.75 * (k_eff - 1)), k_eff - 1)
            q75_dist = sorted_dist[:, q75_idx:q75_idx + 1]

            nearest_mean_ratio = nearest_dist / (mean_dist + self.eps)

            # 0~1 근처 스케일. 가까운 이웃이 많을수록 커짐.
            support_density = torch.exp(
                -((knn_dist / (mean_dist + self.eps)) ** 2)
            ).mean(dim=1, keepdim=True)

            # 평균 거리보다 가까운 이웃의 비율.
            effective_support = (
                knn_dist <= mean_dist
            ).float().mean(dim=1, keepdim=True)

            evidence_list.extend(
                [
                    max_dist,
                    median_dist,
                    q75_dist,
                    nearest_mean_ratio,
                    support_density,
                    effective_support,
                ]
            )

        if self.use_axis_evidence:
            abs_rel_xyz = torch.abs(rel_xyz)  # [N, 3]

            rel_norm = torch.norm(
                rel_xyz,
                dim=-1,
                keepdim=True,
            )  # [N, 1]

            rel_xy_norm = torch.norm(
                rel_xyz[:, 0:2],
                dim=-1,
                keepdim=True,
            )  # [N, 1]

            rel_z_abs = torch.abs(
                rel_xyz[:, 2:3]
            )  # [N, 1]

            vertical_ratio = rel_z_abs / (rel_norm + self.eps)

            dist_cv = std_dist / (mean_dist + self.eps)

            evidence_list.extend(
                [
                    abs_rel_xyz,
                    rel_norm,
                    rel_xy_norm,
                    rel_z_abs,
                    vertical_ratio,
                    dist_cv,
                ]
            )

        if self.use_structure_evidence:
        # --------------------------------------------------------
        # GPU-safe structural evidence without eigen decomposition
        # --------------------------------------------------------
        # Exact PCA/eigh is avoided because CUDA batched eigh can be
        # unstable or unsupported under AMP.
        #
        # This block uses only covariance entries and residual
        # projection energy, so it stays GPU-friendly.
        # --------------------------------------------------------

        # AMP를 쓰더라도 structural statistics는 float32로 계산.
        # 단, CPU로 보내지는 않는다.
            if knn_xyz.is_cuda:
                autocast_ctx = torch.amp.autocast(
                    device_type="cuda",
                    enabled=False,
                )
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()

            with autocast_ctx:
                knn_xyz_f = torch.nan_to_num(
                    knn_xyz.float(),
                    nan=0.0,
                    posinf=1e4,
                    neginf=-1e4,
                )

                rel_xyz_f = torch.nan_to_num(
                    rel_xyz.float(),
                    nan=0.0,
                    posinf=1e4,
                    neginf=-1e4,
                )

                # ----------------------------------------------------
                # 1. Local covariance of source KNN
                # ----------------------------------------------------
                knn_center = knn_xyz_f.mean(dim=1, keepdim=True)
                knn_centered = knn_xyz_f - knn_center

                denom = float(max(k - 1, 1))

                cov = torch.matmul(
                    knn_centered.transpose(1, 2),
                    knn_centered,
                ) / denom  # [N, 3, 3]

                cov = 0.5 * (cov + cov.transpose(-1, -2))

                cov = torch.nan_to_num(
                    cov,
                    nan=0.0,
                    posinf=1e4,
                    neginf=-1e4,
                )

                # ----------------------------------------------------
                # 2. Diagonal variance statistics
                # ----------------------------------------------------
                var_x = cov[:, 0:1, 0]
                var_y = cov[:, 1:2, 1]
                var_z = cov[:, 2:3, 2]

                trace = var_x + var_y + var_z + self.eps

                var_x_ratio = var_x / trace
                var_y_ratio = var_y / trace
                var_z_ratio = var_z / trace

                local_spread = torch.sqrt(trace + self.eps)

                diag_stack = torch.cat(
                    [var_x, var_y, var_z],
                    dim=-1,
                )  # [N, 3]

                diag_max = diag_stack.max(dim=-1, keepdim=True).values
                diag_min = diag_stack.min(dim=-1, keepdim=True).values

                diag_anisotropy = (diag_max - diag_min) / (trace + self.eps)

                # ----------------------------------------------------
                # 3. Off-diagonal covariance correlation
                # ----------------------------------------------------
                cov_xy = cov[:, 0:1, 1]
                cov_xz = cov[:, 0:1, 2]
                cov_yz = cov[:, 1:2, 2]

                corr_xy_abs = torch.abs(cov_xy) / torch.sqrt(
                    var_x * var_y + self.eps
                )

                corr_xz_abs = torch.abs(cov_xz) / torch.sqrt(
                    var_x * var_z + self.eps
                )

                corr_yz_abs = torch.abs(cov_yz) / torch.sqrt(
                    var_y * var_z + self.eps
                )

                corr_xy_abs = torch.clamp(corr_xy_abs, 0.0, 1.0)
                corr_xz_abs = torch.clamp(corr_xz_abs, 0.0, 1.0)
                corr_yz_abs = torch.clamp(corr_yz_abs, 0.0, 1.0)

                # ----------------------------------------------------
                # 4. Residual direction structural support
                # ----------------------------------------------------
                rel_norm = torch.norm(
                    rel_xyz_f,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(self.eps)  # [N, 1]

                rel_dir = rel_xyz_f / rel_norm  # [N, 3]

                # u^T C u
                cov_rel = torch.matmul(
                    cov,
                    rel_dir.unsqueeze(-1),
                ).squeeze(-1)  # [N, 3]

                residual_dir_var = (
                    rel_dir * cov_rel
                ).sum(dim=-1, keepdim=True)  # [N, 1]

                residual_dir_var = torch.clamp(
                    residual_dir_var,
                    min=0.0,
                )

                # normalized variance along residual direction
                residual_supported_var = residual_dir_var / (trace + self.eps)

                # residual magnitude relative to local structural std
                residual_to_structure = torch.log1p(
                    rel_norm / (torch.sqrt(residual_dir_var + self.eps) + self.eps)
                )

                structure_features = [
                    var_x_ratio,
                    var_y_ratio,
                    var_z_ratio,
                    local_spread,
                    corr_xy_abs,
                    corr_xz_abs,
                    corr_yz_abs,
                    diag_anisotropy,
                    residual_supported_var,
                    residual_to_structure,
                ]

            structure_features = [
                feat.to(dtype=target_xyz.dtype)
                for feat in structure_features
            ]

            evidence_list.extend(structure_features)


        if self.use_rgb:
            if target_rgb is None:
                target_rgb = torch.zeros_like(target_xyz)

            if source_rgb is None:
                source_rgb = torch.zeros_like(source_xyz)

            target_rgb = target_rgb.float()
            source_rgb = source_rgb.float()

            if target_rgb.numel() > 0 and target_rgb.max() > 1.5:
                target_rgb = target_rgb / 255.0

            if source_rgb.numel() > 0 and source_rgb.max() > 1.5:
                source_rgb = source_rgb / 255.0

            knn_rgb = source_rgb[knn_idx]  # [Nt, K, 3]
            agg_source_rgb = (knn_rgb * weight.unsqueeze(-1)).sum(dim=1)
            abs_rgb_diff = torch.abs(target_rgb - agg_source_rgb)

            evidence_list.extend(
                [
                    target_rgb,
                    agg_source_rgb,
                    abs_rgb_diff,
                ]
            )

        evidence = torch.cat(evidence_list, dim=-1)

        return {
            "evidence": evidence,
            "nearest_dist": nearest_dist.squeeze(-1),
            "mean_dist": mean_dist.squeeze(-1),
            "rel_xyz": rel_xyz,
        }

    def forward(
        self,
        target_xyz: torch.Tensor,
        source_xyz: torch.Tensor,
        target_rgb: torch.Tensor | None = None,
        source_rgb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            target_xyz: [B, Nt, 3] or [Nt, 3]
            source_xyz: [B, Ns, 3] or [Ns, 3]
            target_rgb: [B, Nt, 3] or [Nt, 3]
            source_rgb: [B, Ns, 3] or [Ns, 3]
        """
        if target_xyz.dim() == 2:
            target_xyz = target_xyz.unsqueeze(0)
        if source_xyz.dim() == 2:
            source_xyz = source_xyz.unsqueeze(0)

        if target_rgb is not None and target_rgb.dim() == 2:
            target_rgb = target_rgb.unsqueeze(0)
        if source_rgb is not None and source_rgb.dim() == 2:
            source_rgb = source_rgb.unsqueeze(0)

        b = target_xyz.shape[0]

        evidence_all = []
        nearest_all = []
        mean_all = []
        rel_all = []

        for bi in range(b):
            out = self._build_one(
                target_xyz=target_xyz[bi],
                source_xyz=source_xyz[bi],
                target_rgb=None if target_rgb is None else target_rgb[bi],
                source_rgb=None if source_rgb is None else source_rgb[bi],
            )

            evidence_all.append(out["evidence"])
            nearest_all.append(out["nearest_dist"])
            mean_all.append(out["mean_dist"])
            rel_all.append(out["rel_xyz"])

        return {
            "evidence": torch.stack(evidence_all, dim=0),
            "nearest_dist": torch.stack(nearest_all, dim=0),
            "mean_dist": torch.stack(mean_all, dim=0),
            "rel_xyz": torch.stack(rel_all, dim=0),
        }