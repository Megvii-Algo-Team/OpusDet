# Copyright (c) OpenMMLab. All rights reserved.
"""TRex2 transformer encoder / decoder (standard Deformable-DETR / DINO style).

Also defines :class:`VisualPromptGenerator`, shared by OPUS and TRex2.
"""
from typing import Optional

import os
import torch
import torch.nn as nn
from torch import Tensor
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList
from mmengine.runner.amp import autocast
from mmdet.utils import ConfigType, OptConfigType
from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoderLayer,
    DeformableDetrTransformerEncoder,
    DeformableDetrTransformerDecoder,
    DeformableDetrTransformerDecoderLayer)
from mmdet.models.layers.transformer.dino_layers import DinoTransformerDecoder
from mmdet.models.layers.transformer.utils import (
    MLP, coordinate_to_encoding, inverse_sigmoid)

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

MSATTN_FORCE_FP32 = os.getenv('MSATTN_FORCE_FP32', 'y').lower() in (
    'y', 'yes', 'true', '1')


class TRex2TransformerEncoder(DeformableDetrTransformerEncoder):
    """Standard deformable transformer encoder (visual features only)."""

    def _init_layers(self) -> None:
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'Install fairscale to use checkpointing: pip install fairscale')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                **kwargs):
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        for layer in self.layers:
            output = layer(
                query=output.to(torch.float32) if MSATTN_FORCE_FP32 else output,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points)
        return output, memory_text


class TRex2TransformerDecoder(DinoTransformerDecoder):
    """Standard DINO decoder without text cross-attention."""

    def _init_layers(self) -> None:
        self.layers = ModuleList([
            TRex2TransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError(f'There is not post_norm in {self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self,
                query: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                self_attn_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                reg_branches: nn.ModuleList,
                early_exit_layer: Optional[int] = None,
                **kwargs):
        intermediate = []
        intermediate_reference_points = [reference_points]
        max_layers = len(self.layers)
        if early_exit_layer is not None:
            max_layers = min(int(early_exit_layer), len(self.layers))
        if max_layers == 0:
            if self.return_intermediate:
                return query.unsqueeze(0), reference_points.unsqueeze(0)
            return query, reference_points

        for lid, layer in enumerate(self.layers[:max_layers]):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None])
            else:
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[:, None])
            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)
            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)
            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return (torch.stack(intermediate),
                    torch.stack(intermediate_reference_points))
        return query, reference_points


class TRex2TransformerDecoderLayer(DeformableDetrTransformerDecoderLayer):
    """Self-attn + image cross-attn + FFN (no text cross-attn)."""

    def _init_layers(self) -> None:
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        self.norms = ModuleList([
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ])

    @autocast(enabled=not MSATTN_FORCE_FP32)
    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        if MSATTN_FORCE_FP32:
            q = k = v = query.to(torch.float32)
            q_pos = (query_pos.to(torch.float32)
                     if query_pos is not None else None)
        else:
            q = k = v = query
            q_pos = query_pos
        query = self.self_attn(
            query=q,
            key=k,
            value=v,
            query_pos=q_pos,
            key_pos=q_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value.to(torch.float32) if MSATTN_FORCE_FP32 else value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[1](query)
        with autocast(enabled=False):
            query = self.ffn(query.to(torch.float32))
            query = self.norms[2](query)
        return query


class VisualPromptGenerator(DeformableDetrTransformerDecoder):
    """Visual prompt encoder with learnable content + class token.

    Shared by OPUS and TRex2. Returns full ``query`` (content slots + class
    token). Caller takes ``query[:, :N]`` for per-box content (e.g.
    distillation), ``query[:, -1]`` for the fused prompt embedding.
    """

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 post_norm_cfg: OptConfigType = None,
                 return_intermediate: bool = True,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(num_layers, layer_cfg, post_norm_cfg, return_intermediate,
                         init_cfg)

    def _init_layers(self) -> None:
        if self.num_layers > 0:
            self.layers = ModuleList([
                DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ])
            self.embed_dims = self.layers[0].embed_dims
            self.content_embedding = nn.Embedding(1, self.embed_dims)
        else:
            self.layers = ModuleList([])
            self.embed_dims = int(self.layer_cfg['self_attn_cfg']['embed_dims'])
            self.content_embedding = None
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.class_embedding = nn.Embedding(1, self.embed_dims)
        self.register_buffer(
            'class_ref_point',
            torch.tensor([[0.5, 0.5]], dtype=torch.float32))
        self.register_buffer(
            'class_ref_box',
            torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float32))
        self.ref_point_head = MLP(
            self.embed_dims, self.embed_dims, self.embed_dims, 2)
        self.ref_box_head = MLP(
            self.embed_dims * 2, self.embed_dims, self.embed_dims, 2)

    def build_decoder_self_attn_mask(
        self,
        self_attn_mask: torch.Tensor,
        num_queries: int,
        num_heads: int,
    ) -> torch.Tensor:
        """Build self-attention mask for decoder (True = masked)."""
        device = self_attn_mask.device
        B, N = self_attn_mask.shape
        assert N == num_queries
        total_q = N + 1
        allow = torch.zeros(B, total_q, total_q, device=device, dtype=torch.bool)
        idx = torch.arange(total_q, device=device)
        allow[:, idx, idx] = True
        for b in range(B):
            valid_idx = self_attn_mask[b].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue
            ii, jj = torch.meshgrid(valid_idx, valid_idx, indexing='ij')
            allow[b, ii, jj] = True
            allow[b, N, valid_idx] = True
        return (~allow).repeat(num_heads, 1, 1)

    def forward(
            self,
            reference_points: Tensor,
            value: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            valid_ratios: Tensor,
            key_padding_mask: Tensor = None,
            self_attn_mask: Tensor = None,
            **kwargs) -> Tensor:
        bs, num_queries, coords_dim = reference_points.shape
        if self.content_embedding is None:
            raise RuntimeError(
                'VisualPromptGenerator requires num_layers>0 with learnable content.')
        learned_content = self.content_embedding.weight.unsqueeze(0).repeat(
            bs, num_queries, 1)
        cls_token = self.class_embedding.weight.unsqueeze(0).repeat(bs, 1, 1)
        query = torch.cat([learned_content, cls_token], dim=1)

        if coords_dim == 4:
            reference_points = torch.cat([
                reference_points,
                self.class_ref_box.unsqueeze(0).repeat(bs, 1, 1)
            ], dim=1)
            reference_points_input = (
                reference_points[:, :, None] * torch.cat(
                    [valid_ratios, valid_ratios], -1)[:, None])
            query_pos = self.ref_box_head(
                coordinate_to_encoding(
                    reference_points_input[:, :, 0, :4],
                    num_feats=self.embed_dims // 2))
        else:
            reference_points = torch.cat([
                reference_points,
                self.class_ref_point.unsqueeze(0).repeat(bs, 1, 1)
            ], dim=1)
            reference_points_input = (
                reference_points[:, :, None] * valid_ratios[:, None])
            query_pos = self.ref_point_head(
                coordinate_to_encoding(
                    reference_points_input[:, :, 0, :],
                    num_feats=self.embed_dims // 2))

        self_attn_mask = self.build_decoder_self_attn_mask(
            self_attn_mask=self_attn_mask,
            num_queries=num_queries,
            num_heads=self.layer_cfg.self_attn_cfg.num_heads)

        for layer in self.layers:
            query = layer(
                query,
                query_pos=query_pos,
                value=value.to(torch.float32) if MSATTN_FORCE_FP32 else value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)
        return query
