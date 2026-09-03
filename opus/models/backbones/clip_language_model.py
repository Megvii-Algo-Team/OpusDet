import logging
from collections import OrderedDict
from contextlib import contextmanager
from typing import Sequence

from mmengine.model import BaseModel
from torch import nn

try:
    from transformers import AutoTokenizer, CLIPTextConfig
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
except ImportError:
    AutoTokenizer = None
    CLIPTextModel = None
    CLIPTextModelWithProjection = None

from mmdet.registry import MODELS

import os
DEBUG = os.getenv("DEBUG", '').lower() in ('y', 'yes', 'true', '1')


@contextmanager
def _suppress_hf_load_report():
    """Hide transformers 5.x LOAD REPORT (unexpected keys) when loading text-only from a full CLIP checkpoint.

    Set ``CLIP_VERBOSE_HF_LOAD_REPORT=1`` to keep the report (debug).
    """
    if os.environ.get('CLIP_VERBOSE_HF_LOAD_REPORT', '').lower() in (
            '1', 'true', 'yes', 'y', 'on'):
        yield
        return
    log = logging.getLogger('transformers.modeling_utils')
    prev = log.level
    log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        log.setLevel(prev)


def _tokenize_compat(tokenizer, captions, *, max_tokens, pad_to_max,
                     return_special_tokens_mask):
    """Tokenize captions for both transformers 4.x and 5.x."""
    kwargs = dict(
        max_length=max_tokens,
        padding='max_length' if pad_to_max else 'longest',
        return_special_tokens_mask=return_special_tokens_mask,
        return_tensors='pt',
        truncation=True,
    )
    if hasattr(tokenizer, 'batch_encode_plus'):
        return tokenizer.batch_encode_plus(captions, **kwargs)
    return tokenizer(captions, **kwargs)


@MODELS.register_module(force=True)
class CLIPModel(BaseModel):
    """CLIP text tower from Hugging Face.

    Args:
        frozen_stages (int): Freeze from the front of the text tower.

            - ``-1`` (default): train all parameters.
            - ``>=0``: freeze token/position embeddings (stem).
            - ``N>0``: also freeze the first ``N`` text encoder layers.
            - ``N >= num_layers`` (CLIP-B/32 is 12): freeze the full text
              encoder (embeddings + all layers + final LN). With
              ``with_projection=True``, ``text_projection`` stays trainable.
    """

    def __init__(
            self,
            name: str = "openai/clip_vit_base_patch32",
            max_tokens: int = 77,
            pad_to_max: bool = True,
            use_checkpoint: bool = False,
            with_projection: bool = False,
            use_pretrain: bool = True,
            frozen_stages: int = -1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max
        self.with_projection = with_projection
        self.frozen_stages = frozen_stages

        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if with_projection:
            self.language_backbone = nn.Sequential(
                OrderedDict([('body',
                            CLIPTextProjectionEncoder(
                                name, 
                                use_checkpoint=use_checkpoint,
                                use_pretrain=use_pretrain))]))
        else:         
            self.language_backbone = nn.Sequential(
                OrderedDict([('body',
                            CLIPTextEncoder(
                                name, 
                                use_checkpoint=use_checkpoint,
                                use_pretrain=use_pretrain))]))
        self._apply_frozen_stages()

    @staticmethod
    def _get_text_model(hf_model):
        """``CLIPTextModel`` or inner ``text_model`` of WithProjection."""
        return getattr(hf_model, 'text_model', hf_model)

    @staticmethod
    def _get_encoder_layers(text_model):
        encoder = getattr(text_model, 'encoder', None)
        if encoder is None:
            return None
        return getattr(encoder, 'layers', getattr(encoder, 'layer', None))

    def _iter_frozen_modules(self):
        """Yield modules that should stay in ``eval`` under current freeze."""
        if self.frozen_stages < 0:
            return
        hf = self.language_backbone.body.model
        text = self._get_text_model(hf)
        if hasattr(text, 'embeddings') and text.embeddings is not None:
            yield text.embeddings
        if self.frozen_stages == 0:
            return
        layers = self._get_encoder_layers(text)
        if layers is None:
            return
        freeze_n = min(self.frozen_stages, len(layers))
        for i in range(freeze_n):
            yield layers[i]
        if freeze_n >= len(layers) and hasattr(text, 'final_layer_norm'):
            yield text.final_layer_norm

    def _apply_frozen_stages(self) -> None:
        """Freeze embeddings when ``>=0``, plus the first N encoder layers."""
        if self.frozen_stages == -1:
            return
        if self.frozen_stages < -1:
            raise ValueError(
                f'`frozen_stages` must be >= -1, got {self.frozen_stages}.')

        hf = self.language_backbone.body.model
        text = self._get_text_model(hf)

        if hasattr(text, 'embeddings') and text.embeddings is not None:
            text.embeddings.requires_grad_(False)
            text.embeddings.eval()

        if self.frozen_stages == 0:
            return

        layers = self._get_encoder_layers(text)
        if layers is None:
            raise AttributeError(
                'Cannot locate CLIP text encoder layers; '
                'unable to apply frozen_stages > 0.')

        freeze_n = min(self.frozen_stages, len(layers))
        for i in range(freeze_n):
            layers[i].requires_grad_(False)
            layers[i].eval()

        if freeze_n >= len(layers) and hasattr(text, 'final_layer_norm'):
            text.final_layer_norm.requires_grad_(False)
            text.final_layer_norm.eval()

    def train(self, mode: bool = True):
        """Keep frozen CLIP submodules in ``eval`` when the detector is training."""
        super().train(mode)
        if self.frozen_stages < 0:
            return
        for module in self._iter_frozen_modules():
            module.eval()

    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        """Forward function."""
        device = next(self.language_backbone.parameters()).device
        tokenized = _tokenize_compat(
            self.tokenizer,
            list(captions),
            max_tokens=self.max_tokens,
            pad_to_max=self.pad_to_max,
            return_special_tokens_mask=False)
        if hasattr(tokenized, 'to'):
            tokenized = tokenized.to(device)
        else:
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
        language_dict_features = self.language_backbone(tokenized)
        # if DEBUG:
        #     print('clip:', tokenized, language_dict_features)
        return language_dict_features

class CLIPTextEncoder(nn.Module):
    def __init__(self,
                 name:str,
                 use_checkpoint:bool = False,
                 use_pretrain: bool = True,):
        super().__init__()
        if CLIPTextConfig is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')
        config = CLIPTextConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint
        # if DEBUG:
        #     print(config)
        if use_pretrain:
            with _suppress_hf_load_report():
                self.model = CLIPTextModel.from_pretrained(
                    name, config=config
                )
        else:
            self.model = CLIPTextModel(
                config=config
            )
        self.language_dim = config.hidden_size

    def forward(self, x) -> dict:
        outputs = self.model(**x)
        results = {
            "pooler_output":outputs.pooler_output
        }
        return results
    

class CLIPTextProjectionEncoder(nn.Module):
    def __init__(self,
                 name:str,
                 use_checkpoint:bool = False,
                 use_pretrain: bool = True,):
        super().__init__()
        if CLIPTextConfig is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')
        config = CLIPTextConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint
        # if DEBUG:
        #     print(config)
        if use_pretrain:
            with _suppress_hf_load_report():
                self.model = CLIPTextModelWithProjection.from_pretrained(
                    name, config=config
                )
        else:
            self.model = CLIPTextModelWithProjection(config)
        self.language_dim = config.projection_dim

    def forward(self, x) -> dict:
        outputs = self.model(**x)
        results = {
            "pooler_output":outputs.text_embeds
        }
        return results