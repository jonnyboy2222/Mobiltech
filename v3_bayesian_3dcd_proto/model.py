# v3_bayesian_3dcd_proto/model.py

from __future__ import annotations

import torch
import torch.nn as nn

from .point_encoder import PointMLPEncoder
from .nearest_feature_diff import NearestFeatureDifference
from .geometry_evidence import GeometryEvidenceBuilder
from .likelihood_heads import DirectLRHead, SeparateNPHead
from .evidence_encoder import EvidenceEncoder
from .voxel_evidence import VoxelEvidenceAggregator


class V3BayesianLRModel(nn.Module):
    """
    v3 Bayesian 3DCD prototype.

    mode:
        direct_lr:
            evidence -> log likelihood ratio directly

        separate_np:
            evidence -> H0 energy, H1 energy
            log ratio = log p(e|H1) - log p(e|H0) + prior

    evidence_type:
        geometry_only:
            no point encoder, no GPU cdist feature matching.
            geometry/RGB evidence only.

        feature_diff:
            point encoder + nearest feature difference.
            stronger but heavier.
    """

    def __init__(
        self,
        mode: str = "separate_np",
        evidence_type: str = "geometry_only",
        use_rgb: bool = True,
        normalize_xyz: bool = True,
        hidden_dim: int = 64,
        feature_dim: int = 128,
        head_hidden_dim: int = 128,
        use_evidence_encoder: bool = False,
        evidence_hidden_dim: int = 128,
        evidence_latent_dim: int = 128,
        dropout: float = 0.1,
        k_neighbors: int = 16,
        prior_logit: float = 0.0,
        nfd_chunk_size: int = 1024,
        geo_k_neighbors: int = 8,
        geo_backend: str = "cpu_kdtree",
        geo_source_chunk_size: int = 8192,

        # new
        share_change_head: bool = False,
        use_direction_token: bool = False,
        direction_token_type: str = "scalar",
        
        # voxel
        use_voxel_evidence: bool = False,
        voxel_size: float = 0.2,
        voxel_origin_mode: str = "zero",
        voxel_use_mean: bool = True,
        voxel_use_std: bool = True,
        voxel_use_max: bool = False,
        voxel_use_min: bool = False,
        voxel_use_count: bool = True,

        # evidence
        geo_use_axis_evidence: bool = True,
        geo_use_support_evidence: bool = False,
        geo_use_structure_evidence: bool = False,
    ) -> None:
        super().__init__()

        if mode not in ["direct_lr", "separate_np"]:
            raise ValueError(
                f"Unknown mode: {mode}. "
                f"Expected ['direct_lr', 'separate_np']."
            )

        if evidence_type not in ["geometry_only", "feature_diff"]:
            raise ValueError(
                f"Unknown evidence_type: {evidence_type}. "
                f"Expected ['geometry_only', 'feature_diff']."
            )

        self.mode = mode
        self.evidence_type = evidence_type
        self.use_rgb = use_rgb
        self.normalize_xyz = normalize_xyz
        self.share_change_head = share_change_head
        self.use_evidence_encoder = use_evidence_encoder
        self.use_direction_token = use_direction_token
        self.direction_token_type = direction_token_type
        self.use_voxel_evidence = use_voxel_evidence


        if direction_token_type not in ["scalar", "onehot"]:
            raise ValueError(
                f"direction_token_type must be 'scalar' or 'onehot', got {direction_token_type}"
            )

        if evidence_type == "geometry_only":
            self.geometry_builder = GeometryEvidenceBuilder(
                k_neighbors=geo_k_neighbors,
                use_rgb=use_rgb,
                backend=geo_backend,
                source_chunk_size=geo_source_chunk_size,
                use_axis_evidence=geo_use_axis_evidence,
                use_support_evidence=geo_use_support_evidence,
                use_structure_evidence=geo_use_structure_evidence,
            )

            evidence_dim = self.geometry_builder.evidence_dim

            self.encoder = None
            self.nfd = None

        else:
            input_dim = 6 if use_rgb else 3

            self.encoder = PointMLPEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                feature_dim=feature_dim,
                dropout=dropout,
            )

            self.nfd = NearestFeatureDifference(
                k_neighbors=k_neighbors,
                chunk_size=nfd_chunk_size,
            )

            # target_feat, agg_source_feat, diff, abs_diff, rel_xyz, nearest, mean
            evidence_dim = 4 * feature_dim + 5

            self.geometry_builder = None

        raw_evidence_dim = evidence_dim
        if self.use_direction_token:
            if self.direction_token_type == "scalar":
                raw_evidence_dim += 1
            else:
                raw_evidence_dim += 2

        if self.use_voxel_evidence:
            self.voxel_aggregator = VoxelEvidenceAggregator(
                voxel_size=voxel_size,
                origin_mode=voxel_origin_mode,
                use_mean=voxel_use_mean,
                use_std=voxel_use_std,
                use_max=voxel_use_max,
                use_min=voxel_use_min,
                use_count=voxel_use_count,
            )

            encoder_input_dim = self.voxel_aggregator.output_dim(
                raw_evidence_dim
            )
        else:
            self.voxel_aggregator = None
            encoder_input_dim = raw_evidence_dim

        # encode evidence
        if use_evidence_encoder:
            self.evidence_encoder = EvidenceEncoder(
                input_dim=encoder_input_dim,
                hidden_dim=evidence_hidden_dim,
                output_dim=evidence_latent_dim,
                dropout=dropout,
                use_residual=False,
            )
            head_input_dim = evidence_latent_dim
        else:
            self.evidence_encoder = nn.Identity()
            head_input_dim = encoder_input_dim


        def make_head() -> nn.Module:
            if mode == "direct_lr":
                return DirectLRHead(
                    input_dim=head_input_dim,
                    hidden_dim=head_hidden_dim,
                    dropout=dropout,
                )

            return SeparateNPHead(
                input_dim=head_input_dim,
                hidden_dim=head_hidden_dim,
                dropout=dropout,
                prior_logit=prior_logit,
            )


        if share_change_head:
            shared_head = make_head()
            self.removed_head = shared_head
            self.added_head = shared_head
        else:
            self.removed_head = make_head()
            self.added_head = make_head()

    @staticmethod
    def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(0)
        if x.dim() == 3:
            return x
        raise ValueError(f"Expected [N, C] or [B, N, C], got {x.shape}")

    def _pair_normalize_xyz(
        self,
        ref_xyz: torch.Tensor,
        query_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_xyz = torch.cat([ref_xyz, query_xyz], dim=1)

        center = all_xyz.mean(dim=1, keepdim=True)
        centered = all_xyz - center

        scale = torch.norm(centered, dim=-1).amax(dim=1, keepdim=True)
        scale = scale.clamp(min=1e-6).unsqueeze(-1)

        ref_xyz_norm = (ref_xyz - center) / scale
        query_xyz_norm = (query_xyz - center) / scale

        return ref_xyz_norm, query_xyz_norm

    def _prepare_rgb(
        self,
        rgb: torch.Tensor | None,
        xyz_like: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.use_rgb:
            return None

        if rgb is None:
            return torch.zeros_like(xyz_like)

        rgb = self._ensure_batch(rgb).float()

        if rgb.numel() > 0 and rgb.max() > 1.5:
            rgb = rgb / 255.0

        return rgb

    def _build_input(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_rgb:
            if rgb is None:
                rgb = torch.zeros_like(xyz)
            return torch.cat([xyz, rgb], dim=-1)

        return xyz
    
    def _append_direction_token(
        self,
        evidence: torch.Tensor,
        direction: str,
    ) -> torch.Tensor:
        """
        Args:
            evidence: [B, N, C]
            direction:
                "removed" or "added"

        Returns:
            evidence with direction token
        """
        if not self.use_direction_token:
            return evidence

        if direction not in ["removed", "added"]:
            raise ValueError(f"Unknown direction: {direction}")

        b, n, _ = evidence.shape

        if self.direction_token_type == "scalar":
            # removed = -1, added = +1
            value = -1.0 if direction == "removed" else 1.0

            token = torch.full(
                (b, n, 1),
                value,
                device=evidence.device,
                dtype=evidence.dtype,
            )

        elif self.direction_token_type == "onehot":
            # removed = [1, 0], added = [0, 1]
            token = torch.zeros(
                b,
                n,
                2,
                device=evidence.device,
                dtype=evidence.dtype,
            )

            if direction == "removed":
                token[..., 0] = 1.0
            else:
                token[..., 1] = 1.0

        else:
            raise ValueError(
                f"Unknown direction_token_type: {self.direction_token_type}"
            )

        return torch.cat([evidence, token], dim=-1)

    def _head_forward(
        self,
        head: nn.Module,
        evidence: torch.Tensor,
        nearest_dist: torch.Tensor,
        mean_dist: torch.Tensor,
        rel_xyz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        encoded_evidence = self.evidence_encoder(evidence)
        head_out = head(encoded_evidence)

        return {
            **head_out,
            "score": torch.sigmoid(head_out["log_ratio"]),
            "nearest_dist": nearest_dist,
            "mean_dist": mean_dist,
            "rel_xyz": rel_xyz,
        }
    
    def _head_only(
        self,
        head: nn.Module,
        evidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded_evidence = self.evidence_encoder(evidence)
        return head(encoded_evidence)


    def _scatter_head_output(
        self,
        voxel_out: dict[str, torch.Tensor],
        inverse: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        voxel_out: 각 key [B, V]
        inverse: [B, N]
        return: 각 key [B, N]
        """
        point_out = {}

        for key, value in voxel_out.items():
            if value.dim() == 2:
                point_out[key] = torch.gather(
                    value,
                    dim=1,
                    index=inverse,
                )
            else:
                point_out[key] = value

        return point_out
    

    def _run_direction_geometry(
        self,
        target_xyz_raw: torch.Tensor,
        target_xyz: torch.Tensor,
        source_xyz: torch.Tensor,
        target_rgb: torch.Tensor | None,
        source_rgb: torch.Tensor | None,
        head: nn.Module,
        direction: str,
    ) -> dict[str, torch.Tensor]:
        geo_out = self.geometry_builder(
            target_xyz=target_xyz,
            source_xyz=source_xyz,
            target_rgb=target_rgb,
            source_rgb=source_rgb,
        )

        evidence = self._append_direction_token(
            geo_out["evidence"],
            direction=direction,
        )

        if self.use_voxel_evidence:
            voxel_evidence, inverse = self.voxel_aggregator(
                target_xyz_raw=target_xyz_raw,
                point_evidence=evidence,
            )

            voxel_out = self._head_only(
                head=head,
                evidence=voxel_evidence,
            )

            head_out = self._scatter_head_output(
                voxel_out=voxel_out,
                inverse=inverse,
            )
        else:
            head_out = self._head_only(
                head=head,
                evidence=evidence,
            )

        return {
            **head_out,
            "score": torch.sigmoid(head_out["log_ratio"]),
            "nearest_dist": geo_out["nearest_dist"],
            "mean_dist": geo_out["mean_dist"],
            "rel_xyz": geo_out["rel_xyz"],
        }
    

    def _run_direction_feature(
        self,
        target_xyz: torch.Tensor,
        target_feat: torch.Tensor,
        source_xyz: torch.Tensor,
        source_feat: torch.Tensor,
        head: nn.Module,
    ) -> dict[str, torch.Tensor]:
        nfd_out = self.nfd(
            target_xyz=target_xyz,
            target_feat=target_feat,
            source_xyz=source_xyz,
            source_feat=source_feat,
        )

        return self._head_forward(
            head=head,
            evidence=nfd_out["evidence"],
            nearest_dist=nfd_out["nearest_dist"],
            mean_dist=nfd_out["mean_dist"],
            rel_xyz=nfd_out["rel_xyz"],
        )

    def forward(
        self,
        ref_xyz: torch.Tensor,
        query_xyz: torch.Tensor,
        ref_rgb: torch.Tensor | None = None,
        query_rgb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ref_xyz = self._ensure_batch(ref_xyz).float()
        query_xyz = self._ensure_batch(query_xyz).float()

        if self.normalize_xyz:
            ref_xyz_in, query_xyz_in = self._pair_normalize_xyz(ref_xyz, query_xyz)
        else:
            ref_xyz_in, query_xyz_in = ref_xyz, query_xyz

        ref_rgb_in = self._prepare_rgb(ref_rgb, ref_xyz_in)
        query_rgb_in = self._prepare_rgb(query_rgb, query_xyz_in)

        if self.evidence_type == "geometry_only":
            removed = self._run_direction_geometry(
                target_xyz=ref_xyz_in,
                source_xyz=query_xyz_in,
                target_xyz_raw=ref_xyz,
                target_rgb=ref_rgb_in,
                source_rgb=query_rgb_in,
                head=self.removed_head,
                direction="removed",
            )

            added = self._run_direction_geometry(
                target_xyz=query_xyz_in,
                source_xyz=ref_xyz_in,
                target_xyz_raw=query_xyz,
                target_rgb=query_rgb_in,
                source_rgb=ref_rgb_in,
                head=self.added_head,
                direction="added",
            )

        else:
            ref_input = self._build_input(ref_xyz_in, ref_rgb_in)
            query_input = self._build_input(query_xyz_in, query_rgb_in)

            ref_feat = self.encoder(ref_input)
            query_feat = self.encoder(query_input)

            removed = self._run_direction_feature(
                target_xyz=ref_xyz_in,
                target_feat=ref_feat,
                source_xyz=query_xyz_in,
                source_feat=query_feat,
                head=self.removed_head,
            )

            added = self._run_direction_feature(
                target_xyz=query_xyz_in,
                target_feat=query_feat,
                source_xyz=ref_xyz_in,
                source_feat=ref_feat,
                head=self.added_head,
            )

        output = {
            "removed_logit": removed["log_ratio"],
            "removed_score": removed["score"],
            "removed_nearest_dist": removed["nearest_dist"],
            "removed_mean_dist": removed["mean_dist"],

            "added_logit": added["log_ratio"],
            "added_score": added["score"],
            "added_nearest_dist": added["nearest_dist"],
            "added_mean_dist": added["mean_dist"],
        }

        if self.mode == "separate_np":
            output.update(
                {
                    "removed_h0_energy": removed["h0_energy"],
                    "removed_h1_energy": removed["h1_energy"],
                    "removed_log_p_h0": removed["log_p_h0"],
                    "removed_log_p_h1": removed["log_p_h1"],

                    "added_h0_energy": added["h0_energy"],
                    "added_h1_energy": added["h1_energy"],
                    "added_log_p_h0": added["log_p_h0"],
                    "added_log_p_h1": added["log_p_h1"],
                }
            )

        return output


def build_v3_model_from_config(cfg: dict) -> V3BayesianLRModel:
    model_cfg = cfg.get("model", {})

    return V3BayesianLRModel(
        mode=model_cfg.get("mode", "separate_np"),
        evidence_type=model_cfg.get("evidence_type", "geometry_only"),
        use_rgb=model_cfg.get("use_rgb", True),
        normalize_xyz=model_cfg.get("normalize_xyz", True),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        feature_dim=model_cfg.get("feature_dim", 64),
        head_hidden_dim=model_cfg.get("head_hidden_dim", 64),

        use_evidence_encoder=model_cfg.get("use_evidence_encoder", False),
        evidence_hidden_dim=model_cfg.get("evidence_hidden_dim", 128),
        evidence_latent_dim=model_cfg.get("evidence_latent_dim", 128),

        dropout=model_cfg.get("dropout", 0.1),
        k_neighbors=model_cfg.get("k_neighbors", 8),
        prior_logit=model_cfg.get("prior_logit", 0.0),
        nfd_chunk_size=model_cfg.get("nfd_chunk_size", 512),
        geo_k_neighbors=model_cfg.get("geo_k_neighbors", 8),
        geo_backend=model_cfg.get("geo_backend", "cpu_kdtree"),
        geo_source_chunk_size=model_cfg.get("geo_source_chunk_size", 8192),

        # new
        share_change_head=model_cfg.get("share_change_head", False),
        use_direction_token=model_cfg.get("use_direction_token", False),
        direction_token_type=model_cfg.get("direction_token_type", "scalar"),

        # voxel
        use_voxel_evidence=model_cfg.get("use_voxel_evidence", True),
        voxel_size=model_cfg.get("voxel_size", 0.2),
        voxel_origin_mode=model_cfg.get("voxel_origin_mode", "zero"),
        voxel_use_mean=model_cfg.get("voxel_use_mean", True),
        voxel_use_std=model_cfg.get("voxel_use_std", True),
        voxel_use_max=model_cfg.get("voxel_use_max", False),
        voxel_use_min=model_cfg.get("voxel_use_min", False),
        voxel_use_count=model_cfg.get("voxel_use_count", True),

        # evidence
        geo_use_axis_evidence=model_cfg.get("geo_use_axis_evidence", True),
        geo_use_support_evidence=model_cfg.get("geo_use_support_evidence", True),
        geo_use_structure_evidence=model_cfg.get("geo_use_structure_evidence", False),
    )