from typing import Sequence, Union

from mmengine.model import BaseModule

from mmdet.registry import MODELS


@MODELS.register_module()
class HFDINOv3ViTBackbone(BaseModule):
    """MMDet wrapper for transformers.DINOv3ViTBackbone.

    Loading uses Hugging Face ``from_pretrained`` / ``Config.from_pretrained`` as-is.
    Cache and offline behaviour follow the official env vars (``HF_HOME``,
    ``HUGGINGFACE_HUB_CACHE``, ``HF_HUB_OFFLINE``, ``HF_OFFLINE``, etc.); see
    https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables

    ``frozen_stages`` freezes from the front:

    - ``-1``: train all.
    - ``>=0``: freeze patch embeddings (stem).
    - ``N>0``: also freeze the first ``N`` encoder layers.
    - Freeze the whole tower with ``train_backbone=False``.
    """

    def __init__(self,
                 model_name='facebook/dinov3-vits16plus-pretrain-lvd1689m',
                 out_index=-1,
                 use_pretrain=True,
                 frozen_stages=-1,
                 train_backbone=True,
                 init_cfg=None,
                 **hf_hub_kwargs):
        super().__init__(init_cfg=init_cfg)
        try:
            from transformers import DINOv3ViTBackbone, DINOv3ViTConfig
        except ImportError as exc:
            raise ImportError(
                'transformers with DINOv3 support is required. '
                'Please install/upgrade transformers first.') from exc

        self.model_name = model_name
        self.out_index = out_index
        self.use_pretrain = use_pretrain
        self.frozen_stages = frozen_stages
        self.train_backbone = train_backbone
        try:
            if use_pretrain:
                self.backbone = DINOv3ViTBackbone.from_pretrained(
                    model_name, **hf_hub_kwargs)
            else:
                config = DINOv3ViTConfig.from_pretrained(
                    model_name, **hf_hub_kwargs)
                self.backbone = DINOv3ViTBackbone(config)
        except OSError as exc:
            raise OSError(
                f'Failed to load DINOv3 (model_name={model_name!r}). '
                'Set HF_HOME / HUGGINGFACE_HUB_CACHE and ensure the repo is cached, '
                'or use HF_HUB_OFFLINE=1 with a full local cache. '
                f'Original error: {exc}') from exc
        self.patch_size = int(getattr(self.backbone.config, 'patch_size', 16))

        if not self.train_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            self._apply_frozen_stages()

    def _get_encoder_layers(self):
        if not hasattr(self.backbone, 'encoder'):
            return None
        encoder = self.backbone.encoder
        return getattr(encoder, 'layers', getattr(encoder, 'layer', None))

    def _apply_frozen_stages(self) -> None:
        """Freeze stem when ``>=0``, plus the first N encoder layers."""
        if self.frozen_stages == -1:
            return
        if self.frozen_stages < -1:
            raise ValueError(
                f'`frozen_stages` must be >= -1, got {self.frozen_stages}.')

        if hasattr(self.backbone, 'embeddings') and self.backbone.embeddings is not None:
            self.backbone.embeddings.requires_grad_(False)
            self.backbone.embeddings.eval()

        if self.frozen_stages == 0:
            return

        layers = self._get_encoder_layers()
        if layers is None:
            raise AttributeError(
                'Cannot locate encoder layers in DINOv3 backbone; '
                'unable to apply frozen_stages > 0.')

        freeze_n = min(self.frozen_stages, len(layers))
        for i in range(freeze_n):
            layers[i].requires_grad_(False)
            layers[i].eval()

    def train(self, mode: bool = True):
        """Keep frozen modules in eval during detector training."""
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
            return self
        if self.frozen_stages < 0:
            return self

        if hasattr(self.backbone, 'embeddings') and self.backbone.embeddings is not None:
            self.backbone.embeddings.eval()
        if self.frozen_stages > 0:
            layers = self._get_encoder_layers()
            if layers is not None:
                freeze_n = min(self.frozen_stages, len(layers))
                for i in range(freeze_n):
                    layers[i].eval()
        return self

    def _tokens_to_map(self, x, tokens):
        """Convert [B, N, C] tokens into [B, C, H, W] feature map."""
        b, _, c = tokens.shape
        h = x.shape[2] // self.patch_size
        w = x.shape[3] // self.patch_size
        n = h * w
        patch_tokens = tokens[:, -n:, :]
        return patch_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        outputs = self.backbone(pixel_values=x, return_dict=True)
        feat = None

        if hasattr(outputs, 'feature_maps') and outputs.feature_maps:
            feat = outputs.feature_maps[self.out_index]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
            if hidden.ndim == 4:
                feat = hidden
            elif hidden.ndim == 3:
                feat = self._tokens_to_map(x, hidden)
            else:
                raise RuntimeError(
                    f'Unexpected last_hidden_state shape: {tuple(hidden.shape)}')
        else:
            raise RuntimeError('DINOv3 backbone did not return usable features.')

        return feat


@MODELS.register_module()
class HFDINOv3ConvNeXtBackbone(BaseModule):
    """MMDet wrapper for transformers DINOv3 ConvNeXt backbone.

    ``frozen_stages`` freezes from the front:

    - ``-1``: train all.
    - ``>=0``: freeze stem.
    - ``N>0``: also freeze ``stages[0:N]``.
    - Freeze the whole tower with ``train_backbone=False`` (or
      ``frozen_stages`` equal to the number of stages, typically 4).
    """

    def __init__(
            self,
            model_name='facebook/dinov3-convnext-base-pretrain-lvd1689m',
            out_index=-1,
            out_indices: Union[int, Sequence[int], None] = None,
            use_pretrain=True,
            frozen_stages=-1,
            train_backbone=True,
            init_cfg=None,
            **hf_hub_kwargs):
        super().__init__(init_cfg=init_cfg)
        try:
            from transformers import DINOv3ConvNextBackbone
            from transformers import DINOv3ConvNextConfig
        except ImportError as exc:
            raise ImportError(
                'transformers with DINOv3 ConvNeXt backbone support is required. '
                'Please install/upgrade transformers first.') from exc

        self.model_name = model_name
        self.out_index = out_index
        if out_indices is None:
            out_indices = (out_index, )
        elif isinstance(out_indices, int):
            out_indices = (out_indices, )
        else:
            out_indices = tuple(out_indices)
        self.out_indices = out_indices
        self.use_pretrain = use_pretrain
        self.frozen_stages = frozen_stages
        self.train_backbone = train_backbone

        try:
            cfg = DINOv3ConvNextConfig.from_pretrained(model_name, **hf_hub_kwargs)
        except OSError as exc:
            raise OSError(
                f'Failed to load DINOv3 ConvNeXt config (model_name={model_name!r}). '
                'Set HF_HOME / HUGGINGFACE_HUB_CACHE and ensure the repo is cached, '
                'or use HF_HUB_OFFLINE=1 with a full local cache. '
                f'Original error: {exc}') from exc

        num_stages = int(getattr(cfg, 'num_stages', len(getattr(cfg, 'hidden_sizes', [])) or 4))
        # HF DINOv3ConvNextBackbone uses stage_names like:
        # ['stem', 'stage1', ..., f'stage{num_stages}'].
        # Therefore out_indices is indexed over (num_stages + 1) entries.
        stage_cnt = num_stages + 1
        resolved_indices = []
        for idx in self.out_indices:
            ridx = idx if idx >= 0 else stage_cnt + idx
            if ridx < 0 or ridx >= stage_cnt:
                raise ValueError(
                    f'Invalid out_indices={self.out_indices} for ConvNeXt with '
                    f'{stage_cnt} stage_names (stem + {num_stages} stages).')
            resolved_indices.append(ridx)
        self.out_indices = tuple(sorted(set(resolved_indices)))

        model_kwargs = dict(hf_hub_kwargs)
        model_kwargs['out_indices'] = list(self.out_indices)

        try:
            if use_pretrain:
                self.backbone = DINOv3ConvNextBackbone.from_pretrained(
                    model_name, **model_kwargs)
            else:
                config = DINOv3ConvNextConfig.from_pretrained(model_name, **model_kwargs)
                self.backbone = DINOv3ConvNextBackbone(config)
        except OSError as exc:
            raise OSError(
                f'Failed to load DINOv3 ConvNeXt (model_name={model_name!r}). '
                'Set HF_HOME / HUGGINGFACE_HUB_CACHE and ensure the repo is cached, '
                'or use HF_HUB_OFFLINE=1 with a full local cache. '
                f'Original error: {exc}') from exc

        if not self.train_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            self._apply_frozen_stages()

    def _get_stem_and_stages(self):
        """Return ``(stem, stages)`` for HF DINOv3 ConvNeXt layouts."""
        roots = [self.backbone]
        model = getattr(self.backbone, 'model', None)
        if model is not None:
            roots.append(model)
        encoder = getattr(self.backbone, 'encoder', None)
        if encoder is not None:
            roots.append(encoder)
        if model is not None:
            enc = getattr(model, 'encoder', None)
            if enc is not None:
                roots.append(enc)

        stem = None
        stages = None
        for root in roots:
            if stem is None:
                stem = getattr(root, 'embeddings', None)
                if stem is None:
                    downs = getattr(root, 'downsample_layers', None)
                    if downs is not None and len(downs) > 0:
                        stem = downs[0]
            if stages is None:
                stages = getattr(root, 'stages', None)
        return stem, stages

    def _apply_frozen_stages(self) -> None:
        """Freeze stem when ``>=0``, plus ``stages[0:N]``."""
        if self.frozen_stages == -1:
            return
        if self.frozen_stages < -1:
            raise ValueError(
                f'`frozen_stages` must be >= -1, got {self.frozen_stages}.')

        stem, stages = self._get_stem_and_stages()
        if stem is not None:
            stem.requires_grad_(False)
            stem.eval()

        if self.frozen_stages == 0:
            return
        if stages is None:
            raise RuntimeError('ConvNeXt backbone missing stages.')
        freeze_n = min(self.frozen_stages, len(stages))
        for i in range(freeze_n):
            stages[i].requires_grad_(False)
            stages[i].eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
            return self
        if self.frozen_stages < 0:
            return self

        stem, stages = self._get_stem_and_stages()
        if stem is not None:
            stem.eval()
        if self.frozen_stages > 0 and stages is not None:
            freeze_n = min(self.frozen_stages, len(stages))
            for i in range(freeze_n):
                stages[i].eval()
        return self

    def forward(self, x):
        outputs = self.backbone(pixel_values=x, return_dict=True)
        feats = getattr(outputs, 'feature_maps', None)
        if not feats:
            outputs_dict = outputs.to_tuple() if hasattr(outputs, 'to_tuple') else ()
            hidden_states = getattr(outputs, 'hidden_states', None)
            hs_len = len(hidden_states) if hidden_states is not None else 0
            hs_shape_0 = tuple(hidden_states[0].shape) if hs_len > 0 else None
            hs_shape_last = tuple(hidden_states[-1].shape) if hs_len > 0 else None
            print(
                '[HFDINOv3ConvNeXtBackbone] empty feature_maps: '
                f'outputs_fields={list(outputs.keys()) if hasattr(outputs, "keys") else type(outputs)}, '
                f'outputs_tuple_len={len(outputs_dict)}, '
                f'hidden_states_len={hs_len}, '
                f'hidden_states_0={hs_shape_0}, hidden_states_last={hs_shape_last}, '
                f'stage_names={getattr(self.backbone, "stage_names", None)}, '
                f'out_features={getattr(self.backbone, "out_features", None)}, '
                f'out_indices={self.out_indices}')
            stage_names = getattr(self.backbone, 'stage_names', None)
            out_features = getattr(self.backbone, 'out_features', None)
            raise RuntimeError(
                'ConvNeXt backbone returned empty `feature_maps`. '
                f'stage_names={stage_names}, out_features={out_features}, '
                f'out_indices={self.out_indices}.')
        feat_shapes = [tuple(f.shape) for f in feats]
        if len(feats) != len(self.out_indices):
            raise RuntimeError(
                'ConvNeXt returned unexpected number of feature maps. '
                f'len(feature_maps)={len(feats)}, feat_shapes={feat_shapes}, '
                f'out_indices={self.out_indices}, '
                f'out_features={getattr(self.backbone, "out_features", None)}')
        selected = list(feats)
        if len(selected) == 1:
            return selected[0]
        return tuple(selected)
