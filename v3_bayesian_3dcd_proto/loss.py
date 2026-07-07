# v3_bayesian_3dcd_proto/loss.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class V3BayesianLoss(nn.Module):
    """
    mode에 따라 loss를 다르게 계산한다.

    direct_lr:
        BCEWithLogitsLoss(log_ratio, label)

    separate_np:
        1. BCEWithLogitsLoss(log_ratio, label)
        2. label에 맞는 hypothesis energy를 낮추는 energy loss
        3. label에 맞는 energy가 반대 energy보다 낮아지도록 margin loss

    Label:
        0 = background / unchanged / H0
        1 = changed / H1
    """

    def __init__(
        self,
        mode: str = "separate_np",
        pos_weight: float | None = None,
        lambda_energy: float = 0.01,
        lambda_margin: float = 0.1,
        margin: float = 1.0,
    ) -> None:
        super().__init__()

        if mode not in ["direct_lr", "separate_np"]:
            raise ValueError(
                f"Unknown mode: {mode}. "
                f"Expected one of ['direct_lr', 'separate_np']."
            )

        self.mode = mode
        self.lambda_energy = lambda_energy
        self.lambda_margin = lambda_margin
        self.margin = margin

        if pos_weight is not None:
            self.register_buffer(
                "pos_weight",
                torch.tensor(float(pos_weight), dtype=torch.float32),
            )
        else:
            self.pos_weight = None

    def _bce_loss(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        label = label.float()

        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(
                logit,
                label,
                pos_weight=self.pos_weight.to(logit.device),
            )

        return F.binary_cross_entropy_with_logits(logit, label)

    def _separate_energy_loss(
        self,
        h0_energy: torch.Tensor,
        h1_energy: torch.Tensor,
        label: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        label=0이면 H0 energy가 낮아야 함.
        label=1이면 H1 energy가 낮아야 함.
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

        # 정답 hypothesis energy를 낮춤
        energy_loss = target_energy.mean()

        # 정답 hypothesis energy가 반대보다 margin만큼 낮도록 유도
        margin_loss = F.softplus(
            self.margin + target_energy - opposite_energy
        ).mean()

        return {
            "energy_loss": energy_loss,
            "margin_loss": margin_loss,
        }

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

        removed_bce = self._bce_loss(
            output["removed_logit"],
            removed_label,
        )

        added_bce = self._bce_loss(
            output["added_logit"],
            added_label,
        )

        bce_loss = removed_bce + added_bce

        total_loss = bce_loss

        loss_dict = {
            "loss": total_loss,
            "bce_loss": bce_loss.detach(),
            "removed_bce": removed_bce.detach(),
            "added_bce": added_bce.detach(),
        }

        if self.mode == "separate_np":
            removed_energy = self._separate_energy_loss(
                h0_energy=output["removed_h0_energy"],
                h1_energy=output["removed_h1_energy"],
                label=removed_label,
            )

            added_energy = self._separate_energy_loss(
                h0_energy=output["added_h0_energy"],
                h1_energy=output["added_h1_energy"],
                label=added_label,
            )

            energy_loss = (
                removed_energy["energy_loss"]
                + added_energy["energy_loss"]
            )

            margin_loss = (
                removed_energy["margin_loss"]
                + added_energy["margin_loss"]
            )

            total_loss = (
                bce_loss
                + self.lambda_energy * energy_loss
                + self.lambda_margin * margin_loss
            )

            loss_dict.update(
                {
                    "loss": total_loss,
                    "energy_loss": energy_loss.detach(),
                    "margin_loss": margin_loss.detach(),
                }
            )

        return loss_dict


def build_v3_loss_from_config(cfg: dict) -> V3BayesianLoss:
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    return V3BayesianLoss(
        mode=model_cfg.get("mode", "separate_np"),
        pos_weight=train_cfg.get("pos_weight", None),
        lambda_energy=train_cfg.get("lambda_energy", 0.01),
        lambda_margin=train_cfg.get("lambda_margin", 0.1),
        margin=train_cfg.get("margin", 1.0),
    )