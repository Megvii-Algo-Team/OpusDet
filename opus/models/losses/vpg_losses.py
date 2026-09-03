# Copyright (c) OpenMMLab. All rights reserved.
"""VPG auxiliary losses built via ``MODELS.build`` (registered under mmdet)."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class VPGContentContrastiveLoss(nn.Module):
    """Paired content<->target alignment: contrastive, L1, or MSE.

    Inputs are stacked pairs ``[P, D]`` with diagonal positives. Used for ICA
    (content<->det query) and RoI distill (content<->RoI slot).
    L1/MSE use per-slot reduction (sum over ``D``, mean over ``P``).
    """

    def __init__(
            self,
            loss_type: str = 'contrastive',
            temperature: float = 0.07,
            loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.temperature = float(temperature)
        self.loss_weight = float(loss_weight)
        if loss_type not in ('contrastive', 'l1', 'mse'):
            raise ValueError(
                f'VPGContentContrastiveLoss loss_type={loss_type!r} not '
                'supported; use contrastive, l1, or mse.')

    @property
    def min_pairs(self) -> int:
        return 2 if self.loss_type == 'contrastive' else 1

    def forward(
            self,
            content: Tensor,
            query: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if query is None:
            raise ValueError(
                'VPGContentContrastiveLoss requires query tensor.')
        if content.size(0) != query.size(0):
            raise ValueError(
                'VPGContentContrastiveLoss expects content/query with the '
                f'same batch size, got {content.size(0)} vs {query.size(0)}.')
        if content.size(0) < self.min_pairs:
            return None
        if self.loss_type == 'contrastive':
            c = F.normalize(content, dim=-1)
            q = F.normalize(query, dim=-1)
            logits = torch.mm(q, c.t()) / self.temperature  # [P, P]
            target = torch.arange(c.size(0), device=c.device)
            loss = 0.5 * (
                F.cross_entropy(logits, target) +
                F.cross_entropy(logits.t(), target))
        elif self.loss_type == 'l1':
            el = F.l1_loss(content, query, reduction='none')
            loss = el.sum(dim=-1).mean()
        else:
            el = F.mse_loss(content, query, reduction='none')
            loss = el.sum(dim=-1).mean()
        return loss * self.loss_weight
