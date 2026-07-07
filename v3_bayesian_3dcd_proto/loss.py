from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class V3BayesianLoss(nn.Module):
    """
    V3 Bayesian 3DCD loss.

    기본 구조:
        L = BCE_removed
          + lambda_added_bce * BCE_added
          + lambda_tversky_removed * Tversky_removed
          + lambda_tversky_added * Tversky_added
          + lambda_margin * Margin

    Label:
        0 = background / unchanged / H0
        1 = changed / H1

    mode:
        direct_lr:
            log_ratio에 대해 BCE + Tversky만 사용

        separate_np:
            log_ratio에 대해 BCE + Tversky 사용
            추가로 H0/H1 energy 사이의 margin loss 사용 가능

    주의:
        기존 energy_loss는 제거함.
        absolute energy를 낮추는 항은 log-ratio scale을 흔들 수 있어서
        여기서는 margin만 optional하게 유지한다.
    """

    def __init__(
        self,
        mode: str = "separate_np",

        # BCE
        pos_weight: float | None = None,
        pos_weight_removed: float | None = None,
        pos_weight_added: float | None = None,
        lambda_removed_bce: float = 1.0,
        lambda_added_bce: float = 1.5,

        # Tversky
        lambda_tversky_removed: float = 0.3,
        lambda_tversky_added: float = 0.7,
        tversky_alpha_removed: float = 0.5,
        tversky_beta_removed: float = 0.5,
        tversky_alpha_added: float = 0.3,
        tversky_beta_added: float = 0.7,
        tversky_eps: float = 1e-6,

        # NP margin
        lambda_margin: float = 0.05,
        margin: float = 1.0,

        # backward compatibility
        lambda_energy: float = 0.0,
    ) -> None:
        super().__init__()

        if mode not in ["direct_lr", "separate_np"]:
            raise ValueError(
                f"Unknown mode: {mode}. "
                f"Expected one of ['direct_lr', 'separate_np']."
            )

        self.mode = mode

        self.lambda_removed_bce = float(lambda_removed_bce)
        self.lambda_added_bce = float(lambda_added_bce)

        self.lambda_tversky_removed = float(lambda_tversky_removed)
        self.lambda_tversky_added = float(lambda_tversky_added)

        self.tversky_alpha_removed = float(tversky_alpha_removed)
        self.tversky_beta_removed = float(tversky_beta_removed)
        self.tversky_alpha_added = float(tversky_alpha_added)
        self.tversky_beta_added = float(tversky_beta_added)
        self.tversky_eps = float(tversky_eps)

        self.lambda_margin = float(lambda_margin)
        self.margin = float(margin)

        # 기존 config와 충돌 방지용. 실제로는 사용하지 않음.
        self.lambda_energy = float(lambda_energy)

        # 공통 pos_weight가 들어오면 removed/added 둘 다에 사용
        if pos_weight_removed is None:
            pos_weight_removed = pos_weight
        if pos_weight_added is None:
            pos_weight_added = pos_weight

        if pos_weight_removed is not None:
            self.register_buffer(
                "pos_weight_removed",
                torch.tensor(float(pos_weight_removed), dtype=torch.float32),
            )
        else:
            self.pos_weight_removed = None

        if pos_weight_added is not None:
            self.register_buffer(
                "pos_weight_added",
                torch.tensor(float(pos_weight_added), dtype=torch.float32),
            )
        else:
            self.pos_weight_added = None

    def _bce_loss(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
        pos_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        label = label.float()

        if pos_weight is not None:
            return F.binary_cross_entropy_with_logits(
                logit,
                label,
                pos_weight=pos_weight.to(logit.device),
            )

        return F.binary_cross_entropy_with_logits(
            logit,
            label,
        )

    def _tversky_loss(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
        alpha: float,
        beta: float,
    ) -> torch.Tensor:
        """
        Binary Tversky loss.

        alpha: FP penalty
        beta : FN penalty

        added에서 FN을 더 줄이고 싶으면 beta를 크게 둔다.
        예: alpha=0.3, beta=0.7
        """
        prob = torch.sigmoid(logit).float().reshape(-1)
        label = label.float().reshape(-1)

        tp = (prob * label).sum()
        fp = (prob * (1.0 - label)).sum()
        fn = ((1.0 - prob) * label).sum()

        numerator = tp + self.tversky_eps
        denominator = tp + alpha * fp + beta * fn + self.tversky_eps

        tversky = numerator / denominator
        return 1.0 - tversky

    def _margin_loss(
        self,
        h0_energy: torch.Tensor,
        h1_energy: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        label=0이면 H0 energy가 H1보다 낮아야 함.
        label=1이면 H1 energy가 H0보다 낮아야 함.

        target_energy + margin < opposite_energy
        를 softplus로 유도한다.
        """
        label = label.float()

        target_energy = torch.where(
            label > 0.5,
            h1_energy,
            h0_energy,
        )

        opposite_energy = torch.where(
            label > 0.5,
            h0_energy,
            h1_energy,
        )

        return F.softplus(
            self.margin + target_energy - opposite_energy
        ).mean()

    def forward(
        self,
        output: dict[str, torch.Tensor],
        removed_label: torch.Tensor,
        added_label: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            output:
                model output dict

            removed_label:
                [B, Nr] or [Nr]
                ref/2016 label.
                removed object point = 1, background = 0

            added_label:
                [B, Nq] or [Nq]
                query/2020 label.
                added object point = 1, background = 0

        Returns:
            loss dict
        """
        if removed_label.dim() == 1:
            removed_label = removed_label.unsqueeze(0)
        if added_label.dim() == 1:
            added_label = added_label.unsqueeze(0)

        removed_label = removed_label.float().to(output["removed_logit"].device)
        added_label = added_label.float().to(output["added_logit"].device)

        # ------------------------------------------------------------
        # BCE
        # ------------------------------------------------------------
        removed_bce = self._bce_loss(
            logit=output["removed_logit"],
            label=removed_label,
            pos_weight=self.pos_weight_removed,
        )

        added_bce = self._bce_loss(
            logit=output["added_logit"],
            label=added_label,
            pos_weight=self.pos_weight_added,
        )

        bce_loss = (
            self.lambda_removed_bce * removed_bce
            + self.lambda_added_bce * added_bce
        )

        # ------------------------------------------------------------
        # Tversky
        # ------------------------------------------------------------
        removed_tversky = self._tversky_loss(
            logit=output["removed_logit"],
            label=removed_label,
            alpha=self.tversky_alpha_removed,
            beta=self.tversky_beta_removed,
        )

        added_tversky = self._tversky_loss(
            logit=output["added_logit"],
            label=added_label,
            alpha=self.tversky_alpha_added,
            beta=self.tversky_beta_added,
        )

        tversky_loss = (
            self.lambda_tversky_removed * removed_tversky
            + self.lambda_tversky_added * added_tversky
        )

        # ------------------------------------------------------------
        # Optional NP margin
        # ------------------------------------------------------------
        margin_loss = torch.zeros(
            (),
            device=output["removed_logit"].device,
            dtype=output["removed_logit"].dtype,
        )

        if self.mode == "separate_np" and self.lambda_margin > 0.0:
            removed_margin = self._margin_loss(
                h0_energy=output["removed_h0_energy"],
                h1_energy=output["removed_h1_energy"],
                label=removed_label,
            )

            added_margin = self._margin_loss(
                h0_energy=output["added_h0_energy"],
                h1_energy=output["added_h1_energy"],
                label=added_label,
            )

            margin_loss = removed_margin + added_margin

        total_loss = (
            bce_loss
            + tversky_loss
            + self.lambda_margin * margin_loss
        )

        loss_dict = {
            "loss": total_loss,

            "bce_loss": bce_loss.detach(),
            "removed_bce": removed_bce.detach(),
            "added_bce": added_bce.detach(),

            "tversky_loss": tversky_loss.detach(),
            "removed_tversky": removed_tversky.detach(),
            "added_tversky": added_tversky.detach(),

            "margin_loss": margin_loss.detach(),

            # 기존 train script 출력 호환용
            "energy_loss": torch.zeros_like(total_loss).detach(),
        }

        return loss_dict


def build_v3_loss_from_config(cfg: dict) -> V3BayesianLoss:
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    return V3BayesianLoss(
        mode=model_cfg.get("mode", "separate_np"),

        # BCE
        pos_weight=train_cfg.get("pos_weight", None),
        pos_weight_removed=train_cfg.get("pos_weight_removed", None),
        pos_weight_added=train_cfg.get("pos_weight_added", None),
        lambda_removed_bce=train_cfg.get("lambda_removed_bce", 1.0),
        lambda_added_bce=train_cfg.get("lambda_added_bce", 1.5),

        # Tversky
        lambda_tversky_removed=train_cfg.get("lambda_tversky_removed", 0.3),
        lambda_tversky_added=train_cfg.get("lambda_tversky_added", 0.7),
        tversky_alpha_removed=train_cfg.get("tversky_alpha_removed", 0.5),
        tversky_beta_removed=train_cfg.get("tversky_beta_removed", 0.5),
        tversky_alpha_added=train_cfg.get("tversky_alpha_added", 0.3),
        tversky_beta_added=train_cfg.get("tversky_beta_added", 0.7),
        tversky_eps=train_cfg.get("tversky_eps", 1e-6),

        # NP margin
        lambda_margin=train_cfg.get("lambda_margin", 0.05),
        margin=train_cfg.get("margin", 1.0),

        # backward compatibility
        lambda_energy=train_cfg.get("lambda_energy", 0.0),
    )