"""OPUS transformer layers: deformable encoder, RT-DETR hybrid encoder, and
decoder with optional text cross-attention (``cross_attn_text``).
"""
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.amp import autocast
from mmengine import ConfigDict
from mmdet.utils import ConfigType, OptConfigType
from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoderLayer,
    DeformableDetrTransformerEncoder,
    DeformableDetrTransformerDecoder,
    DeformableDetrTransformerDecoderLayer)
from mmdet.models.layers.transformer.dino_layers import DinoTransformerDecoder
from mmdet.models.layers.transformer.utils import (
    MLP, coordinate_to_encoding, get_text_sine_pos_embed)

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

import os
DEBUG = os.getenv("DEBUG", '').lower() in ('y', 'yes', 'true', '1')
MSATTN_FORCE_FP32 = os.getenv("MSATTN_FORCE_FP32", 'y').lower() in ('y', 'yes', 'true', '1')

def _build_act(act: Optional[str]) -> nn.Module:
    if act is None:
        return nn.Identity()
    return build_activation_layer(dict(type=act))


class ConvNormLayer_fuse(nn.Module):
    """Lightweight conv block used by RT-DETR style hybrid encoder."""

    def __init__(self, ch_in: int, ch_out: int, kernel_size: int,
                 stride: int = 1, g: int = 1, padding: Optional[int] = None,
                 bias: bool = False, act: Optional[str] = None) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=g, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = _build_act(act)

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self) -> None:
        """Fuse Conv+BN into a single Conv2d for inference."""
        if hasattr(self, 'conv_bn_fused'):
            return
        self.conv_bn_fused = nn.Conv2d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            bias=True)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    """Conv+BN block (non-fusable helper for VGGBlock branches)."""

    def __init__(self, ch_in: int, ch_out: int, kernel_size: int,
                 stride: int = 1, g: int = 1, padding: Optional[int] = None,
                 bias: bool = False, act: Optional[str] = None) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=g, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = _build_act(act)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class VGGBlock(nn.Module):
    """RT-DETRv4 VGGBlock-style re-parameterizable block."""

    def __init__(self, ch_in: int, ch_out: int, act: Optional[str] = 'relu') -> None:
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(
            ch_in, ch_out, kernel_size=3, stride=1, padding=1, act=None)
        self.conv2 = ConvNormLayer(
            ch_in, ch_out, kernel_size=1, stride=1, padding=0, act=None)
        self.act = _build_act(act)

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self) -> None:
        if hasattr(self, 'conv'):
            return
        self.conv = nn.Conv2d(
            self.ch_in, self.ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        k3, b3 = self._fuse_bn_tensor(self.conv1)
        k1, b1 = self._fuse_bn_tensor(self.conv2)
        k1 = F.pad(k1, [1, 1, 1, 1])
        self.conv.weight.data = k3 + k1
        self.conv.bias.data = b3 + b1
        self.__delattr__('conv2')
        self.__delattr__('conv1')

    @staticmethod
    def _fuse_bn_tensor(branch: ConvNormLayer) -> Tuple[Tensor, Tensor]:
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class SCDown(nn.Module):
    """RT-DETRv4 SCDown: 1x1 conv then depthwise stride-2 conv."""

    def __init__(self, c1: int, c2: int, k: int, s: int,
                 act: Optional[str] = None) -> None:
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(
            c1, c2, kernel_size=1, stride=1, act=None)
        self.cv2 = ConvNormLayer_fuse(
            c2, c2, kernel_size=k, stride=s, g=c2, act=act)

    def forward(self, x: Tensor) -> Tensor:
        return self.cv2(self.cv1(x))


class CSPLayer(nn.Module):
    """RT-DETRv4 style CSPLayer with VGG bottlenecks."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 3,
                 expansion: float = 1.0,
                 bias: bool = False,
                 act: Optional[str] = 'SiLU',
                 bottletype=VGGBlock) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(
            in_channels, hidden_channels, kernel_size=1, stride=1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(
            in_channels, hidden_channels, kernel_size=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[bottletype(hidden_channels, hidden_channels, act=act)
              for _ in range(max(1, num_blocks))])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(
                hidden_channels, out_channels, kernel_size=1, stride=1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x2 = self.conv2(x)
        x1 = self.bottlenecks(self.conv1(x))
        return self.conv3(x1 + x2)


class RepNCSPELAN4(nn.Module):
    """RT-DETRv4 RepNCSPELAN4 block for FPN/PAN fusion."""

    def __init__(self,
                 c1: int,
                 c2: int,
                 c3: int,
                 c4: int,
                 n: int = 3) -> None:
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, kernel_size=1, stride=1, padding=0)
        self.cv2 = nn.Sequential(
            CSPLayer(self.c, c4, num_blocks=n, expansion=1.0),
            ConvNormLayer_fuse(c4, c4, kernel_size=3, stride=1, padding=1))
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, num_blocks=n, expansion=1.0),
            ConvNormLayer_fuse(c4, c4, kernel_size=3, stride=1, padding=1))
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, dim=1))


class OPUSTransformerEncoder(DeformableDetrTransformerEncoder):
    """OPUS deformable transformer encoder (visual multi-scale features)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

    # @autocast(enabled=not MSATTN_FORCE_FP32)
    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                **kwargs):
        """Forward Transformer encoder (visual features only).

        ``memory_text`` is returned unchanged (no text/fusion layers).
        """
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


class OPUSHybridEncoder(BaseModule):
    """RT-DETR style hybrid encoder for OPUS speedup.

    This encoder keeps the same forward signature as ``OPUSTransformerEncoder``
    and forces the RT-DETR conv+attn hierarchical path on.
    """

    def __init__(self,
                 rt_detr_hybrid_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.rt_detr_hybrid_cfg = rt_detr_hybrid_cfg
        self._encoder_kwargs = kwargs
        self._init_layers()

    def _init_layers(self) -> None:
        # Keep encoder as composition to avoid inheritance coupling.
        self.encoder = OPUSTransformerEncoder(**self._encoder_kwargs)
        self.embed_dims = self.encoder.embed_dims
        cfg = dict(self.rt_detr_hybrid_cfg or {})
        self.hybrid_attn_encode_idx = list(cfg.get('attn_encode_idx', [-1]))
        fusion_depth = int(cfg.get('fusion_depth', 3))
        fusion_expansion = float(cfg.get('fusion_expansion', 1.0))
        num_feature_levels = int(cfg.get('num_feature_levels', 4))
        encode_idx = []
        for idx in self.hybrid_attn_encode_idx:
            ridx = idx if idx >= 0 else num_feature_levels + idx
            if 0 <= ridx < num_feature_levels:
                encode_idx.append(ridx)
        self.hybrid_attn_encode_idx_resolved = sorted(set(encode_idx))
        self.lateral_convs = ModuleList()
        self.fpn_blocks = ModuleList()
        for _ in range(max(0, num_feature_levels - 1), 0, -1):
            self.lateral_convs.append(
                ConvNormLayer_fuse(
                    self.embed_dims, self.embed_dims, kernel_size=1,
                    act='SiLU'))
            self.fpn_blocks.append(
                CSPLayer(
                    in_channels=self.embed_dims * 2,
                    out_channels=self.embed_dims,
                    num_blocks=fusion_depth,
                    expansion=fusion_expansion))
        self.downsample_convs = ModuleList([
            ConvNormLayer_fuse(
                self.embed_dims, self.embed_dims, kernel_size=3,
                stride=2, padding=1)
            for _ in range(max(0, num_feature_levels - 1))
        ])
        self.pan_blocks = ModuleList([
            CSPLayer(
                in_channels=self.embed_dims * 2,
                out_channels=self.embed_dims,
                num_blocks=fusion_depth,
                expansion=fusion_expansion)
            for _ in range(max(0, num_feature_levels - 1))
        ])

    def _split_to_multi_level(self,
                              query: Tensor,
                              spatial_shapes: Tensor,
                              level_start_index: Tensor) -> List[Tensor]:
        bs, _, c = query.shape
        feats = []
        num_lvls = int(spatial_shapes.size(0))
        for lvl in range(num_lvls):
            h, w = spatial_shapes[lvl].tolist()
            hw = int(h * w)
            start = int(level_start_index[lvl].item())
            part = query[:, start:start + hw, :]
            feats.append(part.transpose(1, 2).reshape(bs, c, h, w))
        return feats

    def _merge_multi_level(self, feats: List[Tensor]) -> Tensor:
        return torch.cat(
            [x.flatten(2).transpose(1, 2).contiguous() for x in feats], dim=1)

    def _run_hybrid_attn_on_selected_levels(
            self,
            query: Tensor,
            query_pos: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            valid_ratios: Optional[Tensor],
            memory_text: Optional[Tensor],
            text_attention_mask: Optional[Tensor],
            text_self_attention_masks: Optional[Tensor],
            pos_text: Optional[Tensor],
            position_ids: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        num_lvls = int(spatial_shapes.size(0))
        encode_idx = [
            idx for idx in self.hybrid_attn_encode_idx_resolved
            if 0 <= idx < num_lvls
        ]
        if not encode_idx:
            return query, memory_text
        # Gather selected levels from flattened query using original indexing.
        level_hw = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).to(torch.long)
        seq_chunks = []
        pos_chunks = [] if query_pos is not None else None
        kpm_chunks = [] if key_padding_mask is not None else None
        ranges = []  # (s0, s1) in original flattened query
        for lvl in encode_idx:
            s0 = int(level_start_index[lvl].item())
            s1 = s0 + int(level_hw[lvl].item())
            ranges.append((s0, s1))
            seq_chunks.append(query[:, s0:s1, :])
            if pos_chunks is not None:
                pos_chunks.append(query_pos[:, s0:s1, :])
            if kpm_chunks is not None:
                kpm_chunks.append(key_padding_mask[:, s0:s1])
        query_sel = torch.cat(seq_chunks, dim=1)
        query_pos_sel = torch.cat(pos_chunks, dim=1) if pos_chunks is not None else None
        key_padding_mask_sel = (
            torch.cat(kpm_chunks, dim=1) if kpm_chunks is not None else None)
        spatial_shapes_sel = spatial_shapes[encode_idx]
        level_start_index_sel = torch.cat([
            spatial_shapes_sel.new_zeros((1, ), dtype=torch.long),
            spatial_shapes_sel.prod(1).cumsum(0)[:-1],
        ])
        if valid_ratios is None:
            valid_ratios_sel = query_sel.new_ones(
                (query_sel.size(0), len(encode_idx), 2))
        else:
            valid_ratios_sel = valid_ratios[:, encode_idx, :]

        query_sel_out, memory_text = self.encoder.forward(
            query=query_sel,
            query_pos=query_pos_sel,
            key_padding_mask=key_padding_mask_sel,
            spatial_shapes=spatial_shapes_sel,
            level_start_index=level_start_index_sel,
            valid_ratios=valid_ratios_sel,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
            pos_text=pos_text,
            text_self_attention_masks=text_self_attention_masks,
            position_ids=position_ids)

        # Scatter selected outputs back to original flattened query.
        query_out = query.clone()
        start = 0
        for s0, s1 in ranges:
            span = s1 - s0
            query_out[:, s0:s1, :] = query_sel_out[:, start:start + span, :]
            start += span
        return query_out, memory_text

    def _run_hybrid_fpn_pan(self,
                            query: Tensor,
                            spatial_shapes: Tensor,
                            level_start_index: Tensor) -> Tensor:
        feats = self._split_to_multi_level(query, spatial_shapes, level_start_index)
        num_lvls = len(feats)
        if num_lvls <= 1:
            return query
        # print('[OPUSHybridEncoder] input feat sizes:',
        #       [tuple(x.shape) for x in feats])
        inner_outs = [feats[-1]]
        for i in range(num_lvls - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = feats[i - 1]
            feat_heigh = self.lateral_convs[num_lvls - 1 - i](
                feat_heigh)
            inner_outs[0] = feat_heigh
            up = F.interpolate(
                feat_heigh, scale_factor=2.0, mode='nearest')
            # print(
            #     f'[OPUSHybridEncoder][TD][i={i}] '
            #     f'high={tuple(feat_heigh.shape)} low={tuple(feat_low.shape)} '
            #     f'up={tuple(up.shape)} cat_hw=({up.shape[-2:]}, {feat_low.shape[-2:]})')
            inner_out = self.fpn_blocks[num_lvls - 1 - i](
                torch.cat([up, feat_low], dim=1))
            # print(
            #     f'[OPUSHybridEncoder][TD][i={i}] fused={tuple(fused.shape)}')
            inner_outs.insert(0, inner_out)
        outs = [inner_outs[0]]
        for i in range(num_lvls - 1):
            down = self.downsample_convs[i](outs[-1])
            # print(
            #     f'[OPUSHybridEncoder][BU][i={i}] '
            #     f'out_prev={tuple(outs[-1].shape)} inner_next={tuple(inner_outs[i + 1].shape)} '
            #     f'down={tuple(down.shape)} cat_hw=({down.shape[-2:]}, {inner_outs[i + 1].shape[-2:]})')
            outs.append(self.pan_blocks[i](
                torch.cat([down, inner_outs[i + 1]], dim=1)))
        #     print(
        #         f'[OPUSHybridEncoder][BU][i={i}] out={tuple(outs[-1].shape)}')
        # print('[OPUSHybridEncoder] output feat sizes:',
        #       [tuple(x.shape) for x in outs])
        return self._merge_multi_level(outs)

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
        query, memory_text = self._run_hybrid_attn_on_selected_levels(
            query=query,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
            text_self_attention_masks=text_self_attention_masks,
            pos_text=pos_text,
            position_ids=position_ids)
        output = self._run_hybrid_fpn_pan(
            query=query,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        return output, memory_text


class OPUSTransformerDecoder(DinoTransformerDecoder):
    """OPUS decoder: image deformable cross-attn + optional text cross-attn."""

    def __init__(
            self,
            num_layers: int,
            layer_cfg: ConfigType,
            post_norm_cfg: OptConfigType = dict(type='LN'),
            return_intermediate: bool = True,
            init_cfg: OptConfigType = None,
            layers_without_text_cross: Optional[Sequence[int]] = None,
    ) -> None:
        """Args:
            layers_without_text_cross (Sequence[int], optional): 0-based
                decoder layer indices (``layer_id`` in forward) for which
                ``cross_attn_text`` is not built; each must satisfy
                ``0 <= id < num_layers``. Omitted or empty:
                every layer keeps ``cross_attn_text_cfg`` when set. Unrelated
                to :meth:`forward` ``early_exit_layer`` (runtime truncation).
        """
        if not layers_without_text_cross:
            skip_ids = frozenset()
        else:
            skip_ids = frozenset(int(x) for x in layers_without_text_cross)
            for lid in skip_ids:
                if lid < 0 or lid >= int(num_layers):
                    raise ValueError(
                        f'layers_without_text_cross contains invalid '
                        f'layer_id={lid} for num_layers={num_layers}')
        self.layers_without_text_cross = skip_ids
        super().__init__(
            num_layers=num_layers,
            layer_cfg=layer_cfg,
            post_norm_cfg=post_norm_cfg,
            return_intermediate=return_intermediate,
            init_cfg=init_cfg)

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        layers_list = []
        for i in range(self.num_layers):
            lc = deepcopy(self.layer_cfg)
            if i in self.layers_without_text_cross:
                lc['cross_attn_text_cfg'] = None
            layers_list.append(OPUSTransformerDecoderLayer(**lc))
        self.layers = ModuleList(layers_list)
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                early_exit_layer: Optional[int] = None,
                **kwargs):
        """Override to pass layer_id for debug logging.

        ``early_exit_layer``: if set, run at most this many leading decoder
        layers (count ``N``, indices ``0..N-1``).
        """
        from mmdet.models.layers.transformer.utils import coordinate_to_encoding, inverse_sigmoid
        intermediate = []
        intermediate_reference_points = [reference_points]
        max_layers = len(self.layers)
        if early_exit_layer is not None:
            max_layers = min(int(early_exit_layer), len(self.layers))
        if max_layers == 0:
            # Keep shape contract valid when explicitly requesting 0 decoder
            # layers (e.g. evaluate encoder outputs only).
            if self.return_intermediate:
                return query.unsqueeze(0), reference_points.unsqueeze(0)
            return query, reference_points
        for lid, layer in enumerate(self.layers[:max_layers]):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]
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
                layer_id=lid,
                **{k: v for k, v in kwargs.items() if k != 'layer_id'})
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
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        return query, reference_points

class OPUSTransformerDecoderLayer(
        DeformableDetrTransformerDecoderLayer):

    def __init__(self, **kwargs) -> None:
        """OPUS decoder layer (self-attn + image cross-attn + optional text)."""
        self.cross_attn_text_cfg = kwargs.pop('cross_attn_text_cfg', None)
        if self.cross_attn_text_cfg is not None:
            if 'batch_first' not in self.cross_attn_text_cfg:
                self.cross_attn_text_cfg['batch_first'] = True
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        if self.cross_attn_text_cfg is not None:
            self.cross_attn_text = MultiheadAttention(**self.cross_attn_text_cfg)
            norm_layers = 4
        else:
            self.cross_attn_text = None
            norm_layers = 3
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(norm_layers)
        ]
        self.norms = ModuleList(norms_list)

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
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                **kwargs) -> Tensor:
        """Implements one OPUS decoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_attention_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        # self attention (force fp32 to avoid overflow in Q@K^T under AMP)
        if MSATTN_FORCE_FP32:
            q = k = v = query.to(torch.float32)
        else:
            q = k = v = query
        query = self.self_attn(
            query=q,
            key=k,
            value=v,
            query_pos=query_pos.to(torch.float32) if MSATTN_FORCE_FP32 and query_pos is not None else query_pos,
            key_pos=query_pos.to(torch.float32) if MSATTN_FORCE_FP32 and query_pos is not None else query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        if self.cross_attn_text is not None and memory_text is not None:
            # cross attention between query and text (MultiheadAttention, force fp32 for Q@K^T)
            if MSATTN_FORCE_FP32:
                q_t = query.to(torch.float32)
                k_t = v_t = memory_text.to(torch.float32)
            else:
                q_t, k_t, v_t = query, memory_text, memory_text
            all_masked_rows = (
                text_attention_mask.all(dim=1)
                if text_attention_mask is not None else None)
            query_residual = query
            query = self.cross_attn_text(
                query=q_t,
                query_pos=query_pos.to(torch.float32) if MSATTN_FORCE_FP32 and query_pos is not None else query_pos,
                key=k_t,
                value=v_t,
                key_padding_mask=text_attention_mask)
            if all_masked_rows is not None and all_masked_rows.any():
                query = query.where(~all_masked_rows[:, None, None], query_residual)
            query = self.norms[1](query)
        # cross attention between query and image
        query = self.cross_attn(
            query=query,
            key=key,
            value=value.to(torch.float32) if MSATTN_FORCE_FP32 else value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[-2](query)
        # query = self.ffn(query)
        # query = self.norms[-1](query)
        with autocast(enabled=False):
            query = self.ffn(query.to(torch.float32))
            query = self.norms[-1](query)

        return query
