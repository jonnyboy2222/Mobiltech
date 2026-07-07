# v3_bayesian_3dcd_proto/voxel_evidence.py

from __future__ import annotations

import torch
import torch.nn as nn


class VoxelEvidenceAggregator(nn.Module):
    """
    Point-wise evidence를 voxel-wise latent evidence로 aggregate한다.

    역할:
        point evidence e_i를 직접 head에 넣지 않고,
        같은 voxel에 속한 evidence들을 통계적으로 요약하여
        voxel latent evidence z_j를 만든다.

    Input:
        target_xyz_raw: [B, N, 3]
            - voxelization 기준 좌표
            - metric/raw xyz 사용 권장

        point_evidence: [B, N, C]
            - GeometryEvidenceBuilder + direction token 이후의 point evidence

    Output:
        voxel_evidence: [B, V, C_voxel]
            - selected aggregation statistics

        inverse_indices: [B, N]
            - 각 point가 속한 voxel index
            - voxel output을 point-wise output으로 scatter할 때 사용

    Supported aggregation:
        mean:
            voxel 내 평균 evidence

        std:
            voxel 내 evidence 분산성

        max:
            voxel 내 강한 evidence 보존
            added/removed가 일부 point에만 강하게 나타나는 경우 유용

        min:
            future option. 기본 false 권장

        count:
            log_count = log(1 + number of points in voxel)
    """

    def __init__(
        self,
        voxel_size: float = 0.2,
        origin_mode: str = "zero",
        eps: float = 1e-6,

        # aggregation options
        use_mean: bool = True,
        use_std: bool = True,
        use_max: bool = False,
        use_min: bool = False,
        use_count: bool = True,
    ) -> None:
        super().__init__()

        if origin_mode not in ["zero", "min"]:
            raise ValueError(
                f"origin_mode must be 'zero' or 'min', got {origin_mode}"
            )

        if not any([use_mean, use_std, use_max, use_min, use_count]):
            raise ValueError(
                "At least one voxel aggregation option must be enabled."
            )

        self.voxel_size = float(voxel_size)
        self.origin_mode = origin_mode
        self.eps = eps

        self.use_mean = use_mean
        self.use_std = use_std
        self.use_max = use_max
        self.use_min = use_min
        self.use_count = use_count

    def output_dim(self, input_dim: int) -> int:
        """
        point evidence dim C가 들어왔을 때
        voxel evidence dim을 계산한다.

        Example:
            mean + std + count:
                2C + 1

            mean + std + max + count:
                3C + 1
        """
        dim = 0

        if self.use_mean:
            dim += input_dim

        if self.use_std:
            dim += input_dim

        if self.use_max:
            dim += input_dim

        if self.use_min:
            dim += input_dim

        if self.use_count:
            dim += 1

        return dim

    @property
    def agg_name(self) -> str:
        names = []

        if self.use_mean:
            names.append("mean")

        if self.use_std:
            names.append("std")

        if self.use_max:
            names.append("max")

        if self.use_min:
            names.append("min")

        if self.use_count:
            names.append("count")

        return "_".join(names)

    def _scatter_reduce(
        self,
        evidence: torch.Tensor,   # [N, C]
        inverse: torch.Tensor,    # [N]
        num_voxels: int,
        reduce: str,
    ) -> torch.Tensor:
        """
        Voxel-wise max/min aggregation.

        reduce:
            'amax' or 'amin'
        """
        if reduce not in ["amax", "amin"]:
            raise ValueError(f"Unsupported reduce type: {reduce}")

        device = evidence.device
        dtype = evidence.dtype
        n, c = evidence.shape

        if reduce == "amax":
            init_value = -float("inf")
        else:
            init_value = float("inf")

        out = torch.full(
            (num_voxels, c),
            init_value,
            device=device,
            dtype=dtype,
        )

        # PyTorch 1.12+ / 2.x path
        if hasattr(out, "scatter_reduce_"):
            index = inverse.view(-1, 1).expand(-1, c)

            out.scatter_reduce_(
                dim=0,
                index=index,
                src=evidence,
                reduce=reduce,
                include_self=True,
            )

        # Fallback path
        elif hasattr(out, "index_reduce_"):
            out.index_reduce_(
                dim=0,
                index=inverse,
                source=evidence,
                reduce=reduce,
                include_self=True,
            )

        else:
            # 매우 구버전 PyTorch fallback.
            # 느리지만 correctness를 위해 남긴다.
            for vi in range(num_voxels):
                mask = inverse == vi

                if not mask.any():
                    continue

                if reduce == "amax":
                    out[vi] = evidence[mask].max(dim=0).values
                else:
                    out[vi] = evidence[mask].min(dim=0).values

        # 빈 voxel은 원칙적으로 unique inverse 때문에 없어야 하지만,
        # 혹시 모를 inf를 0으로 정리한다.
        out = torch.nan_to_num(
            out,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return out

    def _forward_one(
        self,
        xyz: torch.Tensor,       # [N, 3]
        evidence: torch.Tensor,  # [N, C]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = xyz.device
        dtype = evidence.dtype

        if xyz.dim() != 2 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be [N, 3], got {xyz.shape}")

        if evidence.dim() != 2:
            raise ValueError(f"evidence must be [N, C], got {evidence.shape}")

        if xyz.shape[0] != evidence.shape[0]:
            raise ValueError(
                f"xyz and evidence must have same N, "
                f"got xyz={xyz.shape}, evidence={evidence.shape}"
            )

        # ----------------------------------------------------
        # 1. Voxel indexing
        # ----------------------------------------------------
        if self.origin_mode == "zero":
            origin = torch.zeros(
                3,
                device=device,
                dtype=xyz.dtype,
            )
        else:
            # per-sample local origin
            origin = xyz.min(dim=0).values

        voxel_key = torch.floor(
            (xyz - origin) / self.voxel_size
        ).long()  # [N, 3]

        # unique voxel id
        _, inverse = torch.unique(
            voxel_key,
            dim=0,
            return_inverse=True,
        )  # inverse: [N]

        num_voxels = int(inverse.max().item()) + 1
        num_channels = evidence.shape[1]

        # ----------------------------------------------------
        # 2. Count
        # ----------------------------------------------------
        count = torch.bincount(
            inverse,
            minlength=num_voxels,
        ).to(device=device, dtype=dtype).clamp_min(1.0)  # [V]

        # ----------------------------------------------------
        # 3. Mean
        # ----------------------------------------------------
        sum_e = torch.zeros(
            num_voxels,
            num_channels,
            device=device,
            dtype=dtype,
        )
        sum_e.index_add_(0, inverse, evidence)

        mean_e = sum_e / count.unsqueeze(-1)

        # ----------------------------------------------------
        # 4. Std
        # ----------------------------------------------------
        # std는 mean을 기준으로 계산.
        # count=1인 voxel은 거의 0에 가까운 std를 갖는다.
        diff = evidence - mean_e[inverse]

        sum_sq = torch.zeros(
            num_voxels,
            num_channels,
            device=device,
            dtype=dtype,
        )
        sum_sq.index_add_(0, inverse, diff * diff)

        var_e = sum_sq / count.unsqueeze(-1)
        std_e = torch.sqrt(torch.clamp(var_e, min=0.0) + self.eps)

        # ----------------------------------------------------
        # 5. Max / Min
        # ----------------------------------------------------
        if self.use_max:
            max_e = self._scatter_reduce(
                evidence=evidence,
                inverse=inverse,
                num_voxels=num_voxels,
                reduce="amax",
            )
        else:
            max_e = None

        if self.use_min:
            min_e = self._scatter_reduce(
                evidence=evidence,
                inverse=inverse,
                num_voxels=num_voxels,
                reduce="amin",
            )
        else:
            min_e = None

        # ----------------------------------------------------
        # 6. Compose voxel evidence
        # ----------------------------------------------------
        voxel_features = []

        if self.use_mean:
            voxel_features.append(mean_e)

        if self.use_std:
            voxel_features.append(std_e)

        if self.use_max:
            voxel_features.append(max_e)

        if self.use_min:
            voxel_features.append(min_e)

        if self.use_count:
            log_count = torch.log1p(count).unsqueeze(-1)
            voxel_features.append(log_count)

        voxel_evidence = torch.cat(
            voxel_features,
            dim=-1,
        )

        return voxel_evidence, inverse

    def forward(
        self,
        target_xyz_raw: torch.Tensor,
        point_evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target_xyz_raw:
                [B, N, 3] or [N, 3]

            point_evidence:
                [B, N, C] or [N, C]

        Returns:
            voxel_evidence:
                [B, V, C_voxel]

            inverse:
                [B, N]
        """
        if target_xyz_raw.dim() == 2:
            target_xyz_raw = target_xyz_raw.unsqueeze(0)

        if point_evidence.dim() == 2:
            point_evidence = point_evidence.unsqueeze(0)

        if target_xyz_raw.dim() != 3:
            raise ValueError(
                f"target_xyz_raw must be [B, N, 3], got {target_xyz_raw.shape}"
            )

        if point_evidence.dim() != 3:
            raise ValueError(
                f"point_evidence must be [B, N, C], got {point_evidence.shape}"
            )

        if target_xyz_raw.shape[0] != point_evidence.shape[0]:
            raise ValueError(
                f"Batch size mismatch: xyz={target_xyz_raw.shape}, "
                f"evidence={point_evidence.shape}"
            )

        if target_xyz_raw.shape[1] != point_evidence.shape[1]:
            raise ValueError(
                f"Point count mismatch: xyz={target_xyz_raw.shape}, "
                f"evidence={point_evidence.shape}"
            )

        batch_size = target_xyz_raw.shape[0]

        # 현재 train script가 variable-size point cloud 때문에
        # batch_size=1을 전제로 하고 있으므로 우선 B=1만 지원.
        if batch_size != 1:
            raise ValueError(
                "VoxelEvidenceAggregator currently supports batch_size=1 only."
            )

        voxel_evidence, inverse = self._forward_one(
            xyz=target_xyz_raw[0],
            evidence=point_evidence[0],
        )

        return voxel_evidence.unsqueeze(0), inverse.unsqueeze(0)