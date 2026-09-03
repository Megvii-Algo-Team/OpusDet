# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import re
import warnings
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F

from mmcv.transforms import to_tensor
import numpy as np
import random
from collections import defaultdict

from mmengine.structures import InstanceData
import mmengine
from mmengine.logging import print_log

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import ConfigType
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.models.layers import SinePositionalEncoding
from mmdet.models.detectors.dino import DINO
from mmdet.models.detectors.glip import create_positive_map_label_to_token
from ..layers.trex_layer import (
    TRex2TransformerEncoder,
    TRex2TransformerDecoder,
    VisualPromptGenerator,
)
from ...utils.template import (
    multiple_templates,
    augment_phrase,
    simple_template,
    identity_template,
    apply_caption_prompt_to_entity,
    caption_prompt_from_sample,
)
from ...utils.cache import LRUCache, cache_and_sample_negative_labels
from ...utils.speed_profiler import TreX2SpeedProfiler, trex2_segment

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
import os
DEBUG = os.getenv("DEBUG", '').lower() in ('y', 'yes', 'true', '1')

from ...utils.chunk import chunks
from ...utils.prompt_mode import (
    PROMPT_MODE,
    prompt_mode_base,
    prompt_mode_core,
    prompt_mode_present_only,
    prompt_mode_includes_text,
    prompt_mode_visual_num,
    prompt_mode_chunk_size,
)


def clean_label_name(name: str) -> str:
    while re.search(r'\([^()]*\)', name):
        name = re.sub(r'\([^()]*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = ' '.join(name.split())
    return name


def sample_same_token_column_candidates(
        column_visual_embedded: Tensor,
        column_has_visual: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Pre-sample cross-batch visual candidates at the same token column.

    For each ``(b, t)``, pick a random peer visual from column ``t`` (donors
    with ``column_has_visual[:, t]``, excluding row ``b``). Neg slots naturally
    exclude own image because they have no visual at ``(b, t)`` and thus are
    not in the donor set.

    Returns:
        tuple: ``batch_candidate`` ``[B,M,D]``, ``candidate_valid`` ``[B,M]``.
    """
    B, M, D = column_visual_embedded.shape
    device = column_visual_embedded.device
    dtype = column_visual_embedded.dtype
    batch_candidate = torch.zeros(B, M, D, device=device, dtype=dtype)
    candidate_valid = torch.zeros(B, M, dtype=torch.bool, device=device)

    for t in range(M):
        donor_b = torch.nonzero(
            column_has_visual[:, t], as_tuple=False).squeeze(-1)
        if donor_b.numel() == 0:
            continue

        for b in range(B):
            peers = donor_b[donor_b != b]
            if peers.numel() == 0:
                continue
            pick = peers[torch.randint(0, peers.numel(), (1,), device=device)]
            batch_candidate[b, t] = column_visual_embedded[pick.item(), t]
            candidate_valid[b, t] = True

    return batch_candidate, candidate_valid


@MODELS.register_module(force=True)
class TRex2(DINO):
    """Implementation of `T-Rex2
    <https://github.com/IDEA-Research/T-Rex>`_.
    """

    def __init__(self,
                 language_model,
                 visual_prompt_model,
                 *args,
                 use_cache_label=True,
                 use_visual_pre_encoder=True,
                 mode='text_only',
                 global_neg_max_pos_text_only: int = 20,
                 global_neg_max_pos_text_visual: int = 0,
                 global_neg_sample: str = 'random',
                 **kwargs) -> None:
        if global_neg_sample not in ('random', 'hot'):
            raise ValueError(
                f"global_neg_sample must be 'random' or 'hot', "
                f"got {global_neg_sample!r}")
        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_visual_pre_encoder = use_visual_pre_encoder
        self.mode = mode
        self.visual_prompt_model_cfg = visual_prompt_model
        self.global_neg_max_pos_text_only = global_neg_max_pos_text_only
        self.global_neg_max_pos_text_visual = global_neg_max_pos_text_visual
        self.global_neg_sample = global_neg_sample
        super().__init__(*args, **kwargs)
        self.embed_prompt_memory_global = None
        self.embed_prompt_cache_prev = None
        if self.test_cfg is not None:
            prompt_path = self.test_cfg.get('prompt_path', None)
            if prompt_path is not None:
                self.embed_prompt_memory_global = self.load_embed_prompts(
                    prompt_path)

        if use_cache_label:
            self.label_memory_bank = LRUCache(1e4)
        else:
            self.label_memory_bank = None

    @staticmethod
    def _embed_prompt_source_matches(cached, prompt_path) -> bool:
        """Whether a cached ``prompt_path`` entry matches the current source."""
        if cached is prompt_path:
            return True
        if isinstance(prompt_path, str):
            return isinstance(cached, str) and cached == prompt_path
        if isinstance(prompt_path, dict):
            return isinstance(cached, dict) and cached is prompt_path
        if isinstance(prompt_path, InstanceData):
            return isinstance(cached, InstanceData) and cached is prompt_path
        return False

    def load_embed_prompts(self, prompt_path):
        if self.embed_prompt_memory_global is not None:
            cached_path = self.embed_prompt_memory_global.get("prompt_path")
            if self._embed_prompt_source_matches(cached_path, prompt_path):
                return self.embed_prompt_memory_global
        if self.embed_prompt_cache_prev is not None:
            cached_path = self.embed_prompt_cache_prev.get("prompt_path")
            if self._embed_prompt_source_matches(cached_path, prompt_path):
                return self.embed_prompt_cache_prev
        try:
            embed_prompt_cache_current = None
            embedded = None
            label_names = None
            if isinstance(prompt_path, str):
                print_log(
                    f"{prompt_path}",
                    logger='current')
                prompts = mmengine.load(prompt_path)
                label_names = prompts['label_names']
                embedded = to_tensor(prompts['embedded'])
            elif isinstance(prompt_path, dict):
                prompts = prompt_path
                label_names = prompts['label_names']
                embedded = to_tensor(prompts['embedded'])
            elif isinstance(prompt_path, InstanceData):
                assert hasattr(prompt_path, "label_names")
                assert hasattr(prompt_path, "embedded")
                label_names = prompt_path['label_names']
                embedded = to_tensor(prompt_path['embedded'])
            else:
                raise TypeError(f"Unsupported prompt type: {type(prompt_path)}")

            label_map = defaultdict(list)
            clean_label_map = defaultdict(list)
            for i, name in enumerate(label_names):
                label_map[name].append(i)
                clean_label_map[clean_label_name(name)].append(i)
            unique_label_names = label_map.keys()
            print_log(
                f"loaded embed prompts[{len(unique_label_names)}] "
                f"{unique_label_names}, {embedded.shape}",
                logger='current')
            embed_prompt_cache_current = dict(
                embedded=embedded,
                label_map=label_map,
                clean_label_map=clean_label_map,
                prompt_path=prompt_path
            )
            self.embed_prompt_cache_prev = embed_prompt_cache_current
        except Exception as e:
            print_log(
                f"Failed to load prompts from {type(prompt_path)}: {e.args}",
                logger='current',
                level=logging.WARNING)
        return embed_prompt_cache_current

    def _resolve_prompt_path(self, batch_data_samples):
        """Resolve visual.G embed PKL for inference.

        Priority: per-sample ``prompt_path`` in data meta (e.g. ODinW-35
        pipeline) > ``test_cfg.prompt_path`` (global default for visual.G).
        """
        prompt_path = None
        if batch_data_samples:
            prompt_path = batch_data_samples[0].get('prompt_path', None)
        if prompt_path is None and self.test_cfg is not None:
            prompt_path = self.test_cfg.get('prompt_path', None)
        return prompt_path
    
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        encoder_cfg = dict(self.encoder)
        encoder_type = encoder_cfg.pop('type', 'TRex2TransformerEncoder')
        if encoder_type != 'TRex2TransformerEncoder':
            raise ValueError(f'Unsupported encoder type: {encoder_type}')
        self.encoder = TRex2TransformerEncoder(**encoder_cfg)
        self.decoder = TRex2TransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)
        
        # visual prompt modules
        self.visual_prompt_model = VisualPromptGenerator(**self.visual_prompt_model_cfg)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def get_positive_map(self, tokens_positive):
        max_num_entities = self.bbox_head.max_text_len
        positive_map = torch.zeros((len(tokens_positive), max_num_entities),
                            dtype=torch.float)
        for i, indices in enumerate(tokens_positive):
            positive_map[i, indices] = 1.0
        positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-6)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map
    
    def to_enhance_text_prompts(
            self,
            entities,
            templates,
            adj_aug: bool = False,
            article_aug: bool = False,
            caption_prompt: Optional[Dict] = None):
        """Build per-entity strings for the language model.

        When ``caption_prompt`` is set (same schema as Grounding DINO /
        ``DetDataSample.caption_prompt``), each entity is first expanded with
        optional ``prefix`` / ``name`` / ``suffix`` for that key, then optional
        :func:`~opus.utils.template.augment_phrase` and template formatting.
        ``adj_aug`` / ``article_aug`` control adjective and article variation;
        :meth:`get_label_prompts` passes them from ``enhance_entity`` and
        dataset path (VG / MDETR turn ``adj_aug`` off; eval disables both).
        """
        entity_captions = []
        for entity in entities:
            entity_clean = clean_label_name(entity)
            phrase = apply_caption_prompt_to_entity(entity_clean, caption_prompt)
            phrase = augment_phrase(
                phrase, adj_aug=adj_aug, article_aug=article_aug)
            text = random.choice(templates).format(phrase) if templates else phrase
            entity_captions.append(text)
        return entity_captions

    def get_label_prompts(
        self, 
        batch_data_samples: SampleList, 
        custom_entities: bool = True,
        templates: List[str] = None
    ):
        entity_captions_list = []
        entities_list = []
        caption_prompt_first: Optional[Dict] = None
        text_prompts_all_same = True
        # Phrase augmentation: per-sample adj_aug / article_aug start from enhance_entity
        # (eval off); VG / MDETR only clear adj_aug.
        enhance_entity = bool(self.training)
        for data_samples in batch_data_samples:
            caption_prompt = caption_prompt_from_sample(data_samples)
            text_prompt = data_samples.text
            tokens_positive = data_samples.get('tokens_positive', None)
            adj_aug = article_aug = enhance_entity
            if custom_entities:
                dataset_mode = data_samples.get("dataset_mode", "_unknown")
                if isinstance(text_prompt, str):
                    if dataset_mode == "VG": # odvg keep grounded-noun_phrases only
                        adj_aug = False
                        new_tokens_positive = defaultdict(list)
                        entities = []
                        for label, tokens in tokens_positive.items():
                            tokens = sorted(tokens, key=lambda x: x[0])
                            prev = -10
                            entity = []
                            for (s, e) in tokens:
                                if 0 < prev < s and not text_prompt[prev:s].replace(" ", ""):
                                    entity[-1] += f" {text_prompt[s:e]}"
                                elif 0 < prev <= s and text_prompt[prev:s] in ["", "-"]:
                                    entity[-1] += f"{text_prompt[s:e]}"
                                else:
                                    entity.append(text_prompt[s:e])
                                prev = e
                            # entity = ", ".join(entity)
                            entity = random.choice(entity) if len(entity) > 1 else entity[0]
                            if entity not in entities:
                                entities.append(entity)
                            new_tokens_positive[label].append(entities.index(entity))
                        data_samples.set_metainfo(dict(tokens_positive=new_tokens_positive))
                    else:
                        text_prompt = text_prompt.strip(self._special_tokens)
                        text_prompt = text_prompt.split(self._special_tokens)
                        text_prompt = list(
                            filter(lambda x: len(x), text_prompt)
                        )
                        entities = text_prompt
                        if dataset_mode == "OD": # odvg
                            new_tokens_positive = {lbl_idx:[lbl_idx] for lbl_idx in tokens_positive}
                            data_samples.set_metainfo(dict(tokens_positive=new_tokens_positive))
                else: # coco_like(return_classes=True)
                    entities = list(text_prompt)
                    
            else: # MDETRStyleRefCoco
                adj_aug = False
                entities = [text_prompt]

            entity_captions = self.to_enhance_text_prompts(
                entities,
                templates,
                adj_aug=adj_aug,
                article_aug=article_aug,
                caption_prompt=caption_prompt)
            entity_captions_list.append(entity_captions)
            entities_list.append(entities)
            if caption_prompt_first is None:
                caption_prompt_first = caption_prompt
            text_prompts_all_same &= (
                entities == entities_list[0]
                and caption_prompt == caption_prompt_first)

        if text_prompts_all_same: # share memory
            entity_captions_list = [entity_captions_list[0]] * len(batch_data_samples)
            entities_list = [entities_list[0]] * len(batch_data_samples)
        return entity_captions_list, text_prompts_all_same, entities_list

    def _sample_global_negative_captions(
            self,
            entities: List[str],
            max_neg: int,
    ) -> Tuple[List[str], List[str]]:
        """Sample global negatives from LRU bank.

        Inserts ``entities`` into the bank and excludes them from the draw
        (no duplicate with the current prompt).
        """
        if max_neg <= 0:
            return [], []
        global_neg = cache_and_sample_negative_labels(
            self.label_memory_bank,
            entities,
            num_negatives=random.randint(1, (max_neg+1)//2) if len(entities) <= max_neg else 0, #
            sample=self.global_neg_sample,
        )
        if not global_neg:
            return [], []
        caps = self.to_enhance_text_prompts(
            global_neg, identity_template + multiple_templates,
            adj_aug=False, article_aug=self.training)
        return list(global_neg), caps
    
    def get_visual_targets(self, batch_data_samples):
        targets = []
        for data_sample in batch_data_samples:
            gt_bboxes = data_sample.gt_instances.bboxes
            if len(gt_bboxes) > 0:
                img_meta = data_sample.metainfo
                img_h, img_w = img_meta['img_shape']
                factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                            img_h]).unsqueeze(0)
                gt_bboxes_normalized = gt_bboxes / factor
                gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
                targets.append(gt_bboxes_targets)
            else:
                targets.append(gt_bboxes)
        return targets
    
    def get_training_prompts(self, batch_data_samples, mode):
        # split noun-phrases, and add adjs randomly
        text_prompts, text_prompts_all_same, entities = \
            self.get_label_prompts(
                batch_data_samples,
                custom_entities=True,
                templates=identity_template+multiple_templates)
        if mode == "text_only":
            for sample_id, (data_samples, text_prompt, entity) in enumerate(
                zip(batch_data_samples, text_prompts, entities)):
                tokens_positive = data_samples.get('tokens_positive', None)
                # Global neg: bank \ ``entity``; fills up to global_neg_max_*.
                negative_entities, negative_captions = (
                    self._sample_global_negative_captions(
                        entity,
                        self.global_neg_max_pos_text_only))
                if negative_entities:
                    new_text_prompt = (text_prompt+negative_captions)[:self.bbox_head.max_text_len]
                    new_entity = (entity+negative_entities)[:self.bbox_head.max_text_len]
                    old_to_new = list(range(len(new_entity)))
                    # shuffle
                    random.shuffle(old_to_new)
                    idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(old_to_new)}
                    new_text_prompt = [new_text_prompt[i] for i in old_to_new]
                    new_entity = [new_entity[i] for i in old_to_new]
                    if isinstance(tokens_positive, dict): # odvg
                        new_tokens_positive = {lbl:[idx_map[i] for i in ids] for lbl, ids in tokens_positive.items()}
                        data_samples.set_metainfo(dict(tokens_positive=new_tokens_positive))
                    else: #coco
                        gt_label = data_samples.gt_instances.labels
                        unique_labels = torch.unique(gt_label)
                        remap_gt_label = torch.zeros_like(gt_label, device=gt_label.device)
                        for lbl in unique_labels:
                            remap_gt_label[gt_label==lbl] = idx_map[lbl.item()]
                        data_samples.gt_instances.labels = remap_gt_label
                    text_prompts_all_same = False
                    text_prompts[sample_id] = new_text_prompt
                    entities[sample_id] = new_entity 

            return text_prompts, text_prompts_all_same
        if 'visual' in mode:
            # Batch-shared prompt table: GT pos + local neg + global neg.
            batch_positive_entities = []
            batch_positive_prompts = []
            batch_pos_set = set()
            # 1) GT positives (deduped across batch).
            for sample_id, data_samples in enumerate(batch_data_samples):
                gt_label = data_samples.gt_instances.labels
                if len(gt_label) == 0:
                    continue
                tokens_positive = data_samples.get('tokens_positive', None)
                for lbl in torch.unique(gt_label):
                    if isinstance(tokens_positive, dict):  # odvg
                        token_ids = tokens_positive.get(lbl.item(), [])
                    else:  # coco-like
                        token_ids = [lbl.item()]
                    for token_id in token_ids:
                        e = entities[sample_id][token_id]
                        if e in batch_pos_set:
                            continue
                        batch_pos_set.add(e)
                        batch_positive_entities.append(e)
                        batch_positive_prompts.append(
                            text_prompts[sample_id][token_id])

            text_prompts_all_same = True
            # 2) Local neg: in batch captions but not batch GT.
            neg_pool = {
                e: t for e, t in zip(sum(entities, []), sum(text_prompts, []))
                if e not in batch_pos_set}
            if neg_pool:
                chosen = random.sample(
                    list(neg_pool.items()), random.randint(1, len(neg_pool)))
                batch_neg_entities, batch_neg_text_prompts = map(
                    list, zip(*chosen))
            else:
                batch_neg_entities, batch_neg_text_prompts = [], []

            prompt_entities = batch_positive_entities + batch_neg_entities
            prompt_text = batch_positive_prompts + batch_neg_text_prompts
            # 3) Global neg: LRU bank \ ``prompt_entities``; rest of cap.
            global_neg_entities, global_neg_text_prompts = (
                self._sample_global_negative_captions(
                    prompt_entities,
                    self.global_neg_max_pos_text_visual))

            max_len = self.bbox_head.max_text_len
            batch_share_entities = (
                prompt_entities + global_neg_entities)[:max_len]
            batch_share_text_prompts = (
                prompt_text + global_neg_text_prompts)[:max_len]
            # Shuffle shared table; per-sample GT indices remap via label_map.
            old_to_new = list(range(len(batch_share_entities)))
            random.shuffle(old_to_new)
            label_map = {batch_share_entities[old_idx]: new_idx for new_idx, old_idx in enumerate(old_to_new)}
            batch_share_entities = [batch_share_entities[i] for i in old_to_new]
            batch_share_text_prompts = [batch_share_text_prompts[i] for i in old_to_new]
            
            new_entities = []
            new_text_prompts = []
            targets = self.get_visual_targets(
                batch_data_samples
            )
            use_point = random.random()<0.5 # point or box mode
            visual_prompts = {
                'targets':targets, 
                'selects':[], # group-select by category for each image
                'label_pb':int(not use_point), # 0:point, 1:box, same mode per batch
                'tokens_positive':[] # labels to token positions
            }

            for sample_id, (data_samples, entity, text_prompt) in enumerate(
                zip(batch_data_samples, entities, text_prompts)):
                new_entities.append(batch_share_entities)
                new_text_prompts.append(batch_share_text_prompts)
                # relabel map
                tokens_positive = data_samples.get('tokens_positive', None)
                if isinstance(tokens_positive, dict): # odvg
                    new_tokens_positive = {lbl:[label_map[entity[i]] for i in ids] for lbl, ids in tokens_positive.items()}
                    data_samples.set_metainfo(dict(tokens_positive=new_tokens_positive))
                else:# coco-like
                    gt_label = data_samples.gt_instances.labels
                    unique_labels = torch.unique(gt_label)
                    remap_gt_label = torch.zeros_like(gt_label, device=gt_label.device)
                    for lbl in unique_labels:
                        remap_gt_label[gt_label==lbl] = label_map[entity[lbl.item()]]
                    data_samples.gt_instances.labels = remap_gt_label

                # sample visuals
                selects_per_sample = {}
                gt_label = data_samples.gt_instances.labels
                unique_labels = torch.unique(gt_label)
                tokens_positive = data_samples.get('tokens_positive', None)
                if len(gt_label):
                    for lbl in unique_labels: # per label
                        mask = (gt_label == lbl).nonzero(as_tuple=False).squeeze(1)  # [num]
                        num_select = torch.randint(1, mask.size(0) + 1, (1,), device=gt_label.device).item()
                        rand_idx = torch.randperm(mask.size(0), device=gt_label.device)[:num_select]
                        selected = mask[rand_idx]
                        selects_per_sample[lbl.item()] = selected
                visual_prompts['selects'].append(selects_per_sample)
                visual_prompts['tokens_positive'].append(tokens_positive)

            text_prompts = new_text_prompts
            entities = new_entities

            return text_prompts, text_prompts_all_same, visual_prompts              

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
        visual_prompts: Dict = None,
        mode = "text_only"
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        if self.use_visual_pre_encoder:
            visual_dict = self.encode_visual(
                visual_prompts=visual_prompts,
                memory=encoder_inputs_dict['feat'],
                memory_mask=encoder_inputs_dict['feat_mask'],
                spatial_shapes=encoder_inputs_dict['spatial_shapes'],
                level_start_index=encoder_inputs_dict['level_start_index'],
                valid_ratios=encoder_inputs_dict['valid_ratios'],
            )
        else:
            visual_dict = self.encode_visual(
                visual_prompts=visual_prompts,
                memory=encoder_outputs_dict['memory'],
                memory_mask=encoder_outputs_dict['memory_mask'],
                spatial_shapes=encoder_outputs_dict['spatial_shapes'],
                level_start_index=encoder_inputs_dict['level_start_index'],
                valid_ratios = encoder_inputs_dict['valid_ratios'],  
            )

        text_dict, vpg_aux = self.merge_visual_to_text_dict(
            text_dict, visual_dict, mode)
        encoder_outputs_dict.update({
            'memory_text':text_dict['embedded'],
            'text_token_mask':text_dict['text_token_mask'],
        })
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, 
            batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(
            **decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict, text_dict, vpg_aux

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        """Forward decoder with optional inference early-exit.

        test_cfg option:
            decoder_early_exit_layer (int): use first N decoder layers in
            inference. <=0 or unset means disabled.
        """
        early_exit_layer = None
        if not self.training and self.test_cfg is not None:
            cfg_val = self.test_cfg.get('decoder_early_exit_layer', None)
            if isinstance(cfg_val, int):
                early_exit_layer = cfg_val

        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            early_exit_layer=early_exit_layer,
            **kwargs)

        if len(query) == self.num_queries:
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict
    
    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()
        topk_memory = torch.gather(
            output_memory, 1, 
            topk_indices.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
        if DEBUG:
            print_log(
                f"pre decoder:\n memory: {memory.shape} {memory.dtype} {memory[0][..., :5]}",
                logger='current')
            print_log(
                f"memory_text: {memory_text.shape} {memory_text.dtype} {memory_text[0][:5, :5]}",
                logger='current')
            topk_memory = torch.gather(
                output_memory, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            print_log(
                f"topk_output_memory: {output_memory.shape} {output_memory.dtype} {topk_memory[0]}",
                logger='current')
        assert self.query_embedding is not None
        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords)
        if self.training:
            head_inputs_dict['dn_meta'] = dn_meta
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict
    
    def generate_self_attention_mask(self, text_dict):
        bs, max_num, _ = text_dict['embedded'].shape
        device = text_dict['embedded'].device
        text_dict['masks'] = torch.eye(max_num, device=device, 
                                       dtype=torch.bool).unsqueeze(0).repeat(bs, 1, 1)
        text_dict["position_ids"] = torch.zeros((bs, max_num), device=device)
        return text_dict

    def _placeholder_text_dict(
            self,
            batch_inputs: Tensor,
            text_prompts: List,
    ) -> Dict:
        """占位 ``text_dict``：纯 ``visual`` 推理不走 LM，形状与类别数对齐供 encoder 使用。"""
        bs = len(batch_inputs)
        device = batch_inputs.device
        dtype = batch_inputs.dtype
        if text_prompts and any(len(x) > 0 for x in text_prompts):
            max_num = max(len(x) for x in text_prompts)
        else:
            max_num = 1
        max_num = max(max_num, 1)
        text_dict = {
            'embedded':
            torch.zeros((bs, max_num, self.embed_dims), device=device, dtype=dtype),
            'text_token_mask':
            torch.zeros((bs, max_num), device=device, dtype=torch.bool),
        }
        return self.generate_self_attention_mask(text_dict)

    def encode_text(
        self,
        text_prompts:List,
        text_prompts_all_same:bool,
        device,
        chunked_size = -1
    ) -> Dict:
        if text_prompts is None:
            return None
        if 'embedded' in text_prompts:
            missing_keys = [k for k in ['text_token_mask'] if k not in text_prompts]
            assert not missing_keys, f"text_prompts 缺少必要字段: {missing_keys}"
            return text_prompts
        if text_prompts_all_same:
            flatten_text_prompts = text_prompts[0]
        else:
            ## flatten all
            flatten_text_prompts = [_ for x in text_prompts for _ in x]
        # chunk
        if chunked_size > 0:
            text_prompts_chunked = chunks(flatten_text_prompts, chunked_size)
        else:
            text_prompts_chunked = [flatten_text_prompts]
        text_dict = {}
        for text_prompt in text_prompts_chunked:
            if len(text_prompt) == 0:
                continue
            language_dict_features = self.language_model(text_prompt)
            chunked_text_dict = {
                'embedded':language_dict_features['pooler_output']
            }
            if self.text_feat_map is not None:
                chunked_text_dict['embedded'] = self.text_feat_map(
                    chunked_text_dict['embedded'])
            if len(text_dict) == 0:
                text_dict = chunked_text_dict
            else:
                text_dict = {k:torch.concat([text_dict[k], v], 0) for k, v in
                                chunked_text_dict.items()}
        # padding
        bs = len(text_prompts)
        prompts_per_sample = [len(i) for i in text_prompts]
        max_num = max(prompts_per_sample)
        if text_prompts_all_same:
            # repeat
            text_dict['embedded'] = text_dict['embedded']\
                .unsqueeze(0).repeat(bs, 1, 1) if 'embedded' in text_dict else \
                torch.zeros((bs, max_num, self.embed_dims))
            text_dict['text_token_mask'] = torch.ones(\
                (bs, max_num), device=device, dtype=torch.bool)
        else:
            embedded = text_dict['embedded']
            # zero padding 
            padded_embeds = torch.zeros((bs, max_num, embedded.size(-1)), 
            dtype=embedded.dtype, device=device)
            text_token_mask = torch.zeros((bs, max_num), 
            dtype=torch.bool, device=device)
            ptr = 0
            for i, count in enumerate(prompts_per_sample):
                padded_embeds[i, :count] = embedded[ptr:ptr+count]
                text_token_mask[i, :count] = True
                ptr += count
            text_dict['embedded'] = padded_embeds
            text_dict['text_token_mask'] = text_token_mask
        text_dict = self.generate_self_attention_mask(text_dict)

        if DEBUG:
            print_log(
                f"text encode:\n all same: {text_prompts_all_same} total: {len(flatten_text_prompts)}",
                logger='current')
            print_log(
                f"{text_dict['embedded'].shape} {text_dict['embedded'][0]} "
                f"{torch.nonzero(text_dict['text_token_mask'][0]).flatten()}",
                logger='current')
        return text_dict
    
    def _encode_visual_prepare_interactive(
            self, visual_prompts: Dict,
            memory: Tensor) -> Optional[Dict]:
        """Pack box/point selects into dense tensors (interactive path only)."""
        bs = len(memory)
        targets = visual_prompts['targets']
        label_pb = visual_prompts['label_pb']
        selects = visual_prompts['selects']
        tokens_positive = visual_prompts['tokens_positive']
        device = memory.device
        max_category_num = max([len(t) for t in selects], default=0)
        max_boxes_num = max(
            [len(v) for t in selects for v in t.values()], default=0)
        coords_dim = 2 if label_pb == 0 else 4
        target_coords = torch.zeros(
            [max_category_num, bs, max_boxes_num, coords_dim], device=device)
        target_valid_mask = torch.zeros(
            [max_category_num, bs, max_boxes_num],
            dtype=torch.bool, device=device)
        selected_gt_inds = torch.full(
            [max_category_num, bs, max_boxes_num],
            -1,
            dtype=torch.long,
            device=device)
        visual_token_mask = torch.zeros(
            [bs, max_category_num], device=device, dtype=torch.bool)
        labels = torch.full([bs, max_category_num], -1, device=device)
        for b, (targets_per_sample, selects_per_sample) in enumerate(
                zip(targets, selects)):
            for cat_idx, (label_id, select_ids) in enumerate(
                    selects_per_sample.items()):
                target_coords[cat_idx, b, :len(select_ids)] = \
                    targets_per_sample[select_ids, :coords_dim]
                target_valid_mask[cat_idx, b, :len(select_ids)] = True
                selected_gt_inds[cat_idx, b, :len(select_ids)] = \
                    select_ids.to(device=device, dtype=torch.long)
                visual_token_mask[b, cat_idx] = True
                labels[b, cat_idx] = label_id
        if max_category_num == 0:
            return None
        vpm = self.visual_prompt_model
        if vpm.num_layers == 0 or vpm.content_embedding is None:
            raise RuntimeError(
                'Interactive visual prompts require VPG with learnable '
                'content (num_layers>0).')
        return dict(
            target_coords=target_coords,
            target_valid_mask=target_valid_mask,
            selected_gt_inds=selected_gt_inds,
            visual_token_mask=visual_token_mask,
            labels=labels,
            tokens_positive=tokens_positive,
            num_categories=max_category_num,
        )



    def _get_vpg_category_chunk_size(self) -> int:
        """Max category-index span per VPG loop; ``<=0`` runs all at once.

        Only valid ``(b, cat_idx)`` slots inside each span are encoded.
        Training is always 1. Eval default 16; optional
        ``test_cfg['vpg_category_chunk_size']`` overrides eval only.
        """
        if self.training:
            return 1
        return int(self.test_cfg.get('vpg_category_chunk_size', 16))


    def _encode_visual_assemble_vpg(
            self,
            prep: Dict,
            memory: Tensor,
            memory_mask: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            valid_ratios: Tensor,
    ) -> Optional[Dict]:
        """Batched VPG over valid ``(b, cat_idx)`` slots, chunked by ``test_cfg``.

        ``num_categories`` is the per-batch max; only ``visual_token_mask``
        slots are encoded. Category-axis chunks bound memory; each sub-chunk
        is aligned to MultiScaleDeformableAttention CUDA ``im2col_step``.
        """
        num_categories = prep['num_categories']
        bs = memory.size(0)
        dim = memory.size(-1)
        if num_categories == 0:
            return None
        visual_queries = memory.new_zeros((bs, num_categories, dim))
        num_q = prep['target_coords'].size(2)

        chunk_size = self._get_vpg_category_chunk_size()
        if chunk_size <= 0:
            chunk_size = num_categories

        vpm = self.visual_prompt_model
        if getattr(vpm, 'num_layers', 0) > 0 and len(vpm.layers) > 0:
            im2col_step = int(
                getattr(vpm.layers[0].cross_attn, 'im2col_step', 64))
        else:
            im2col_step = 64

        cat_start = 0
        while cat_start < num_categories:
            cat_end = min(cat_start + chunk_size, num_categories)
            vtm = prep['visual_token_mask'][:, cat_start:cat_end]
            if vtm.any():
                valid_slots = torch.nonzero(vtm, as_tuple=False).clone()
                valid_slots[:, 1] += cat_start
                slot_start = 0
                n_valid = valid_slots.size(0)
                while slot_start < n_valid:
                    remaining = n_valid - slot_start
                    n_chunk = 1
                    for n in range(remaining, 0, -1):
                        if n % min(n, im2col_step) == 0:
                            n_chunk = n
                            break
                    slots_chunk = valid_slots[slot_start:slot_start + n_chunk]
                    b_idx = slots_chunk[:, 0]
                    cat_idx = slots_chunk[:, 1]
                    coords = prep['target_coords'][cat_idx, b_idx]
                    box_mask = prep['target_valid_mask'][cat_idx, b_idx]
                    mem = memory[b_idx]
                    mem_mask = (memory_mask[b_idx]
                                if memory_mask is not None else None)
                    vr = valid_ratios[b_idx]
                    visual_query = self.visual_prompt_model(
                        reference_points=coords,
                        value=mem,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        valid_ratios=vr,
                        key_padding_mask=mem_mask,
                        self_attn_mask=box_mask,
                    )
                    cls_emb = visual_query[:, -1]
                    visual_queries[b_idx, cat_idx] = cls_emb.to(
                        dtype=visual_queries.dtype)
                    slot_start += n_chunk
            cat_start = cat_end
        return {
            'embedded': visual_queries,
            'visual_token_mask': prep['visual_token_mask'],
            'labels': prep['labels'],
            'tokens_positive': prep['tokens_positive'],
        }

    def encode_visual(
        self,
        visual_prompts: Dict,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        chunked_size=-1,
    ) -> Dict:
        if visual_prompts is None:
            return None
        if 'embedded' in visual_prompts:
            missing_keys = [
                k for k in ['labels', 'visual_token_mask']
                if k not in visual_prompts
            ]
            assert not missing_keys, f'visual_prompts 缺少必要字段: {missing_keys}'
            return visual_prompts
        prep = self._encode_visual_prepare_interactive(visual_prompts, memory)
        if prep is None:
            return None
        return self._encode_visual_assemble_vpg(
            prep, memory, memory_mask, spatial_shapes, level_start_index,
            valid_ratios)

    def merge_visual_to_text_dict(self, text_dict, visual_dict, mode):
        """Merge visual prompts into ``text_dict``.

        ``vpg_aux`` is read from ``visual_dict['vpg_aux']`` when present; training
        adds ``vl_align`` into that dict (copied before write).

        Training fills negative visual slots from same-token-column peers;
        ``vl_align`` keeps native positives.

        Returns:
            tuple: (``text_dict``, ``vpg_aux``).
        """
        if visual_dict is None:
            return text_dict, None
        if "visual" not in mode:
            return text_dict, visual_dict.get('vpg_aux')

        if not torch.any(visual_dict['visual_token_mask']):
            return text_dict, visual_dict.get('vpg_aux')

        vpg_aux = visual_dict.get('vpg_aux')

        # ---- 基础信息
        text_emb = text_dict['embedded']
        text_token_mask = text_dict['text_token_mask']
        visual_emb = visual_dict['embedded']
        visual_token_mask = visual_dict['visual_token_mask']

        B, M, D = text_emb.shape
        device = text_emb.device
        dtype = visual_emb.dtype

        # ---- dtype 统一（一次）
        if text_emb.dtype != dtype:
            text_emb = text_emb.to(dtype)

        # ---- clone 一次即可
        new_embedded = text_emb.clone()
        new_token_mask = text_token_mask.clone()

        # ---- 只建一次 buffer
        new_visual_embedded = torch.zeros_like(new_embedded)
        new_visual_token_mask = torch.zeros((B, M), dtype=torch.bool, device=device)

        # ---- 正样本：vectorized 版本
        indices = torch.nonzero(visual_token_mask, as_tuple=False)  # [N, 2]

        for b, c in indices:
            lbl = visual_dict['labels'][b, c]
            if lbl < 0:
                continue
            tokens_positive = visual_dict['tokens_positive'][b]
            if isinstance(tokens_positive, dict):
                tokens = tokens_positive.get(lbl.item(), [])
            else:
                tokens = [lbl.item()]

            # 合法 token
            tokens = [t for t in tokens if 0 <= t < M]
            if not tokens:
                continue

            new_visual_embedded[b, tokens] = visual_emb[b, c]
            new_visual_token_mask[b, tokens] = True

        # =========================
        # ===== 训练阶段 =====
        # =========================
        if self.training:
            # Step 1: same-token-column peer candidates (exclude own batch row).
            batch_candidate, candidate_valid = (
                sample_same_token_column_candidates(
                    new_visual_embedded, new_visual_token_mask))

            # Step 2/3: fill negative slots from same-token-column peers.
            neg_slots_mask = (~new_visual_token_mask) & candidate_valid
            new_visual_embedded = torch.where(
                neg_slots_mask.unsqueeze(-1),
                batch_candidate,
                new_visual_embedded,
            )
            new_visual_token_mask = new_visual_token_mask | neg_slots_mask

            new_token_mask = torch.where(
                new_visual_token_mask, new_visual_token_mask, new_token_mask)
            new_embedded = torch.where(
                new_visual_token_mask.unsqueeze(-1),
                new_visual_embedded,
                new_embedded,
            )
            if DEBUG:
                print_log(
                    f"visual: {visual_dict['embedded'].shape} valid in 1st sample: "
                    f"{torch.nonzero(visual_dict['visual_token_mask'][0], as_tuple=True)[0]} "
                    f"{visual_dict['embedded'][0]}",
                    logger='current')
                print_log(
                    f"visual slots in 1st "
                    f"{torch.nonzero(new_visual_token_mask[0], as_tuple=True)[0]}",
                    logger='current')
                print_log(
                    f"final: {new_embedded.shape} valid in 1st sample: "
                    f"{torch.nonzero(new_token_mask[0], as_tuple=True)[0]} {new_embedded[0]}",
                    logger='current')

            vl_align = dict(
                text_emb=text_emb,
                text_token_mask=text_token_mask,
                new_visual_embedded=new_visual_embedded,
                new_visual_token_mask=new_visual_token_mask,
            )
            if vpg_aux is None:
                vpg_aux = {}
            else:
                vpg_aux = dict(vpg_aux)
            vpg_aux['vl_align'] = vl_align

        # =========================
        # ===== 推理阶段 =====
        # =========================
        else:
            # Use prompt base mode: ``'text' in 'visual'`` is true (substring bug).
            if prompt_mode_includes_text(mode):
                new_embedded = torch.where(new_visual_token_mask.unsqueeze(-1), (new_visual_embedded+new_embedded)/2, new_embedded)
            else:
                new_embedded = torch.where(new_visual_token_mask.unsqueeze(-1), new_visual_embedded, new_embedded)
                new_token_mask = new_visual_token_mask
            if DEBUG:
                align_score = self.bbox_head.vl_align(text_emb, new_visual_embedded)
                valid_mask = text_token_mask.unsqueeze(1)
                align_score = align_score.masked_fill(~valid_mask, float('-inf'))

                print_log(
                    f"visual: {visual_emb.shape} valid in 1st sample: "
                    f"{torch.nonzero(visual_token_mask[0], as_tuple=True)[0]} "
                    f"{visual_emb[0][..., :5]}",
                    logger='current')
                print_log(
                    f"text: {text_emb.shape} valid in 1st sample: "
                    f"{torch.nonzero(text_token_mask[0], as_tuple=True)[0]} "
                    f"{text_emb[0][..., :5]}",
                    logger='current')
                print_log(
                    f"replace in 1st {torch.nonzero(new_visual_token_mask[0], as_tuple=True)[0]}",
                    logger='current')
                print_log(
                    f"final: {new_embedded.shape} valid in 1st sample: "
                    f"{torch.nonzero(new_token_mask[0], as_tuple=True)[0]} {new_embedded[0][..., :5]}",
                    logger='current')

                v_idx = torch.nonzero(visual_token_mask[0], as_tuple=True)[0]
                t_idx = torch.nonzero(text_token_mask[0], as_tuple=True)[0]
                max_val, max_idx = torch.max(align_score[0], dim=-1)
                print_log(
                    f"Align valid pairs in 1st sample: v_{len(v_idx)} @ t_{len(t_idx)} pairs",
                    logger='current')
                print_log(
                    f"Max score of {len(max_val)} v_p in 1st sample: ({max_val}, {max_idx})",
                    logger='current')

                for vi in v_idx[:5]:
                    for ti in t_idx[:5]:
                        print_log(
                            f"align[{vi},{ti}]: align_score: {align_score[0, vi, ti].item()}",
                            logger='current')
        # ---- 回写
        text_dict.update(
            {
                "embedded": new_embedded,
                "text_token_mask": new_token_mask,
            }
        )

        return text_dict, vpg_aux

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        mode = batch_data_samples[0].get("mode", self.mode)
        assert mode.split(".")[0].strip() in PROMPT_MODE, \
        f'mode should be one of {PROMPT_MODE}, but got {mode}'
        # visual features
        visual_features = self.extract_feat(batch_inputs)
        if DEBUG:
            print_log(f"mode {mode}", logger='current')
        prompts = self.get_training_prompts(batch_data_samples, mode) 
        chunked_size = self.bbox_head.max_text_len * 2
    
        # text prompts encoding
        if mode == "text_only":
            text_prompts, text_prompts_all_same = prompts
            text_dict = self.encode_text(
                text_prompts,
                text_prompts_all_same,
                batch_inputs.device,
                chunked_size)
            visual_prompts = None     
        # visual prompt
        elif "visual" in mode:
            text_prompts, text_prompts_all_same, visual_prompts =  prompts
            text_dict = self.encode_text(
                text_prompts,
                text_prompts_all_same,
                batch_inputs.device,
                chunked_size)
        head_inputs_dict, text_dict, vpg_aux = self.forward_transformer(
            visual_features,
            text_dict,
            batch_data_samples,
            visual_prompts,
            mode=mode)
        for i, data_samples in enumerate(batch_data_samples):
            # BinaryFocalLossCost
            tokens_positive = data_samples.get('tokens_positive', None)
            if isinstance(tokens_positive, dict):
                new_tokens_positive = [tokens_positive[lbl.item()] for lbl in data_samples.gt_instances.labels]
            else:
                new_tokens_positive = data_samples.gt_instances.labels
            _, positive_map = self.get_positive_map(new_tokens_positive)
            positive_map = positive_map.to(
                batch_inputs.device).bool().float()
            data_samples.gt_instances.positive_maps = positive_map
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        if DEBUG:
            print_log(f"batch_data_samples {batch_data_samples[0]}", logger='current')
            print_log(
                f"head_inputs_dict "
                f"{ {k: [v.dtype, v.shape] for k, v in head_inputs_dict.items() if isinstance(v, torch.Tensor)} }",
                logger='current')
        losses = self.bbox_head.loss(
            **head_inputs_dict,
            batch_data_samples=batch_data_samples,
            vpg_aux=vpg_aux)
        dummy = losses[list(losses.keys())[0]].new_full((1,), 0)
        # Keep DDP reducer stable for alternating branches by touching all
        # trainable params, not only a subset of submodules.
        encoder_cp = getattr(self.encoder, 'num_cp', 0)
        for name, param in self.named_parameters():
            if not param.requires_grad or param.numel() == 0:
                continue
            # Avoid touching checkpointed encoder params twice in one graph,
            # which may trigger DDP "mark ready only once" with fairscale cp.
            if encoder_cp and name.startswith('encoder.'):
                continue
            # Read one scalar only to keep overhead minimal.
            dummy = dummy + 0 * param.reshape(-1)[0]
        losses['loss_dummy'] = dummy # gradient track
        if DEBUG:
            print_log(f"{ {k: [v.dtype, v] for k, v in losses.items()} }", logger='current')
        return losses

    def load_generic_visual_prompts(
        self,
        batch_data_samples, 
        entities,
        text_prompts_all_same,
        visual_prompts_num,
        device,
    ):
        bs = len(batch_data_samples)
        max_category_num = max([len(entity) for entity in entities])  
        visual_prompts = {
            "labels":torch.full([
                bs, max_category_num],
                -1, device=device),
            "embedded":torch.zeros(
                (bs, max_category_num, self.embed_dims),
                device=device),
            "visual_token_mask":torch.zeros(
                [bs, max_category_num],
                device=device, dtype=torch.bool),
            "tokens_positive":[None]*bs
        }
        embed_prompt_cache_current = self.embed_prompt_memory_global
        prompt_path = self._resolve_prompt_path(batch_data_samples)
        if prompt_path is not None:  # load by path, same in batch
            embed_prompt_cache_current = self.load_embed_prompts(prompt_path)
        embed_label_map = None
        if batch_data_samples:
            embed_label_map = batch_data_samples[0].get('embed_label_map')
        if embed_prompt_cache_current is not None:
            clean_label_map = embed_prompt_cache_current["clean_label_map"]
            label_map = embed_prompt_cache_current["label_map"]
            embedded = embed_prompt_cache_current["embedded"].to(device)
            for b in range(bs):
                entities_per_sample = (
                    entities[0] if text_prompts_all_same else entities[b])
                for cat_idx, target_label_name in enumerate(entities_per_sample):
                    lookup_name = target_label_name
                    if embed_label_map and target_label_name in embed_label_map:
                        lookup_name = embed_label_map[target_label_name]
                    if lookup_name in label_map:
                        indices = label_map[lookup_name]
                    else:
                        _lookup_name = clean_label_name(lookup_name)
                        indices = clean_label_map.get(_lookup_name)
                        if lookup_name != target_label_name:
                            print_log(
                                f"[EMBED] label mapped: {target_label_name} -> "
                                f"{lookup_name} -> {_lookup_name}",
                                logger='current',
                                level=logging.WARNING)
                        elif _lookup_name != lookup_name:
                            print_log(
                                f"[EMBED] label mapped: {lookup_name} -> "
                                f"{_lookup_name}",
                                logger='current',
                                level=logging.WARNING)
                    if not indices:
                        print_log(
                            f"[EMBED] label not found in memory bank: {target_label_name}",
                            logger='current',
                            level=logging.WARNING)
                        continue
                    target_label_embedded = embedded[indices]
                    num_select = visual_prompts_num if visual_prompts_num else len(indices)
                    rand_idx = torch.randperm(len(indices), device=device)[:num_select]
                    cls_emb = target_label_embedded[rand_idx].mean(dim=0)
                    visual_prompts['labels'][b, cat_idx] = cat_idx
                    visual_prompts['embedded'][b, cat_idx] = cls_emb
                    visual_prompts['visual_token_mask'][b, cat_idx] = True
                if text_prompts_all_same:
                    visual_prompts['labels'] = visual_prompts['labels'][0].unsqueeze(0).repeat(bs, 1)
                    visual_prompts['embedded'] = visual_prompts['embedded'][0].unsqueeze(0).repeat(bs, 1, 1)
                    visual_prompts['visual_token_mask'] = visual_prompts['visual_token_mask'][0].unsqueeze(0).repeat(bs, 1)
                    break
        return visual_prompts

    @staticmethod
    def _attach_label_names_to_instances(
            instances: InstanceData, entities_full: List[str]) -> None:
        """Map global class ids in ``instances.labels`` to names via full vocab."""
        if len(instances) == 0:
            return
        names = []
        for lab in instances.labels:
            lid = int(lab.item()) if hasattr(lab, 'item') else int(lab)
            if 0 <= lid < len(entities_full):
                names.append(entities_full[lid])
            else:
                names.append('unobject')
        instances.label_names = names

    @staticmethod
    def _local_to_global_row(
            chunk_local_to_global: Union[List[int], List[List[int]]],
            sample_idx: int,
    ) -> List[int]:
        """One sample's local column index -> global class id map for a chunk."""
        if not chunk_local_to_global:
            return []
        if isinstance(chunk_local_to_global[0], list):
            return chunk_local_to_global[sample_idx]
        return chunk_local_to_global  # type: ignore[return-value]

    @staticmethod
    def _local_to_global_ids_from_gt(
            batch_data_samples: SampleList) -> List[List[int]]:
        """Per-sample sorted unique GT class ids (indices into full entity list)."""
        gt_ids: List[List[int]] = []
        for ds in batch_data_samples:
            gt = ds.gt_instances.labels
            if len(gt) == 0:
                gt_ids.append([])
            else:
                gt_ids.append(sorted(int(x) for x in torch.unique(gt).tolist()))
        return gt_ids

    @staticmethod
    def _build_local_to_global_ids(
            text_prompts: List[List[str]],
            text_prompts_all_same: bool,
            local_to_global_ids_from_gt: Optional[List[List[int]]] = None,
    ) -> List[List[int]]:
        """Full prompt-table map: ``local_to_global_ids[b][local] -> global class id``."""
        if local_to_global_ids_from_gt is not None:
            return local_to_global_ids_from_gt
        bs = len(text_prompts)
        if text_prompts_all_same:
            n = len(text_prompts[0])
            row = list(range(n)) if n else []
            return [row for _ in range(bs)]
        return [
            list(range(len(text_prompts[b])))
            for b in range(bs)
        ]

    @staticmethod
    def _full_table_as_single_chunk(
            local_to_global_ids: List[List[int]],
            text_prompts_all_same: bool,
    ) -> list:
        """Wrap full ``local_to_global_ids`` as the sole chunk remap entry."""
        if not local_to_global_ids:
            return [[]]
        row0 = local_to_global_ids[0]
        if text_prompts_all_same and all(
                row == row0 for row in local_to_global_ids):
            return [row0]
        return [local_to_global_ids]

    @staticmethod
    def empty_pred_instance(
            device: Union[str, torch.device]) -> InstanceData:
        """Build an empty ``pred_instances`` (zero boxes/scores/labels)."""
        inst = InstanceData()
        inst.bboxes = torch.empty([0, 4], device=device)
        inst.scores = torch.empty([0, 1], device=device)
        inst.labels = torch.empty([0, 1], device=device)
        return inst

    def load_interactive_visual_prompts(
            self,
            batch_data_samples,
            entities,
            text_prompts_all_same,
            use_point,
            visual_prompts_num,
            local_to_global_ids: List[List[int]],
    ):
        """Build interactive visual prompts aligned to the prompt table.

        ``selects[b][local_idx]`` uses local column index; GT boxes are
        matched via ``local_to_global_ids[b][local_idx]``.
        """
        bs = len(batch_data_samples)
        targets = self.get_visual_targets(
            batch_data_samples
        )
        
        visual_prompts = {
            'targets':targets, 
            'selects':[], # group-select by category for each image
            'label_pb':int(not use_point), # 0:point, 1:box, same mode per batch
            'tokens_positive':[None]*bs # labels to token positions
        }
        for i, data_samples in enumerate(batch_data_samples):
            gt_label = data_samples.gt_instances.labels
            entities_per_sample = (
                entities[0] if text_prompts_all_same else entities[i])

            selects_per_sample = {}
            if len(gt_label) > 0:
                for cat_idx in range(len(entities_per_sample)):
                    local_to_global_row = (
                        local_to_global_ids[0]
                        if text_prompts_all_same
                        else local_to_global_ids[i])
                    global_id = local_to_global_row[cat_idx]
                    mask = (gt_label == global_id).nonzero(
                        as_tuple=False).squeeze(1)
                    if mask.numel() == 0:
                        continue
                    num_select = (
                        visual_prompts_num
                        if visual_prompts_num else mask.size(0))
                    rand_idx = torch.randperm(
                        mask.size(0), device=gt_label.device)[:num_select]
                    selects_per_sample[cat_idx] = mask[rand_idx]
            visual_prompts['selects'].append(selects_per_sample)
        return visual_prompts

    @staticmethod
    def _finish_predict_empty(
            batch_data_samples: SampleList,
            device: Union[str, torch.device],
            rescale: bool,
    ) -> SampleList:
        """No prompt columns to score: return empty ``pred_instances`` per sample."""
        for data_samples in batch_data_samples:
            data_samples.pred_instances = TRex2.empty_pred_instance(device)
        if rescale:
            for data_samples in batch_data_samples:
                assert data_samples.metainfo.get('scale_factor') is not None
                data_samples.gt_instances.bboxes /= (
                    data_samples.gt_instances.bboxes.new_tensor(
                        data_samples.metainfo['scale_factor']).repeat((1, 2)))
        return batch_data_samples

    @staticmethod
    def _filter_text_entities_present_only(
            text_prompts: List[List[str]],
            entities: List[List[str]],
            text_prompts_all_same: bool,
            local_to_global_ids_from_gt: List[List[int]],
    ) -> Tuple[List[List[str]], List[List[str]], bool]:
        """Keep only image-present classes in text/entity prompt lists (eval: from GT ids)."""
        bs = len(text_prompts)
        new_text: List[List[str]] = []
        new_entities: List[List[str]] = []
        for b in range(bs):
            global_ids = local_to_global_ids_from_gt[b]
            ent_src = entities[0] if text_prompts_all_same else entities[b]
            txt_src = text_prompts[0] if text_prompts_all_same else text_prompts[b]
            if not global_ids:
                new_text.append([])
                new_entities.append([])
                continue
            new_entities.append([ent_src[g] for g in global_ids])
            new_text.append([txt_src[g] for g in global_ids])
        if bs <= 1:
            all_same = True
        else:
            all_same = all(
                new_entities[b] == new_entities[0] for b in range(bs))
        return new_text, new_entities, all_same
    
    def chunk_prompts(
        self,
        chunked_size: int,
        text_prompts: List,
        text_prompts_all_same: bool,
        visual_prompts: Optional[Dict],
        mode: str,
        local_to_global_ids: List[List[int]],
    ) -> Tuple[List, List, List[bool], list]:
        """Split prompts for chunked inference.

        Assumes ``text_prompts`` / ``entities`` are already the final prompt table.
        Chunking slices local column indices; each chunk returns
        ``chunk_local_to_global`` for eval remap via ``local_to_global_ids``.
        """
        bs = len(text_prompts)
        if chunked_size > 0:
            assert text_prompts_all_same, (
                'chunked_size>0 requires text_prompts_all_same: one shared '
                'entity list and batch-aligned class ids. '
                'visual_prompts["selects"][b] holds each sample\'s interactive '
                'prompts separately and must not be interpreted under '
                'per-sample-only label spaces when chunking.')

        if chunked_size <= 0:
            if not any(bool(row) for row in local_to_global_ids):
                return [], [], [], []
            chunk_local_to_global_ids = self._full_table_as_single_chunk(
                local_to_global_ids, text_prompts_all_same)
            return (
                [text_prompts],
                [visual_prompts],
                [False],
                chunk_local_to_global_ids,
            )

        n_ent = len(text_prompts[0])
        slot_chunks = chunks(list(range(n_ent)), chunked_size)
        text_prompts_chunks = []
        visual_prompts_chunks = []
        chunk_skip: List[bool] = []
        chunk_local_to_global_ids = []
        ref = local_to_global_ids[0]
        for chunk_i, slot_chunk in enumerate(slot_chunks):
            if not prompt_mode_includes_text(mode):
                text_prompts_chunks.append(
                    [[""] * len(slot_chunk) for _ in range(bs)])
            elif text_prompts_all_same:
                src = text_prompts[0]
                text_prompts_chunks.append(
                    [[src[i] for i in slot_chunk]] * bs)
            else:
                text_prompts_chunks.append([
                    [text_prompts[b][i] for i in slot_chunk
                     if i < len(text_prompts[b])]
                    for b in range(bs)
                ])
            if visual_prompts is not None:
                vp_once = copy.deepcopy(visual_prompts)
                if 'selects' in visual_prompts:
                    for j in range(bs):
                        sel = visual_prompts['selects'][j]
                        vp_once['selects'][j] = {
                            slot_chunk.index(k): sel[k]
                            for k in slot_chunk if k in sel
                        }
                    skip = not any(
                        len(vp_once['selects'][jj]) for jj in range(bs))
                else:
                    idxs = slot_chunk
                    vp_once['embedded'] = visual_prompts['embedded'][:, idxs]
                    vp_once['visual_token_mask'] = (
                        visual_prompts['visual_token_mask'][:, idxs])
                    chunk_len = len(idxs)
                    vp_once['labels'] = (
                        visual_prompts['labels'].new_full(
                            (bs, chunk_len), -1))
                    for li in range(chunk_len):
                        vp_once['labels'][:, li] = li
                    skip = False
                visual_prompts_chunks.append(vp_once)
                chunk_skip.append(skip)
            else:
                visual_prompts_chunks.append(None)
                chunk_skip.append(False)
            chunk_local_to_global_ids.append(
                [ref[i] for i in slot_chunk])
        return (
            text_prompts_chunks,
            visual_prompts_chunks,
            chunk_skip,
            chunk_local_to_global_ids,
        )

    def _predict_gen_embed_only(
            self,
            batch_inputs: Tensor,
            batch_data_samples: SampleList,
            *,
            mode: str,
            text_prompts,
            text_prompts_all_same: bool,
            entities,
            visual_prompts,
            rescale: bool,
            entities_full,
            local_to_global_ids: List[List[int]],
    ) -> SampleList:
        """Encoder-only ``gen_embed`` path: text/visual embeddings, no transformer or head."""
        bs = len(batch_inputs)
        device = batch_inputs.device
        embed_results_list = [[] for _ in range(bs)]
        if (visual_prompts is not None and 'selects' in visual_prompts):
            visual_feats = self.extract_feat(batch_inputs)
            encoder_inputs_dict, _ = self.pre_transformer(
                visual_feats, batch_data_samples)
            if self.use_visual_pre_encoder:
                visual_dict = self.encode_visual(
                    visual_prompts,
                    encoder_inputs_dict['feat'],
                    encoder_inputs_dict['feat_mask'],
                    encoder_inputs_dict['spatial_shapes'],
                    encoder_inputs_dict['level_start_index'],
                    encoder_inputs_dict['valid_ratios'],
                )
            else:
                text_dict = self._placeholder_text_dict(
                    batch_inputs, text_prompts)
                encoder_outputs_dict = self.forward_encoder(
                    **encoder_inputs_dict, text_dict=text_dict)
                visual_dict = self.encode_visual(
                    visual_prompts,
                    encoder_outputs_dict['memory'],
                    encoder_outputs_dict['memory_mask'],
                    encoder_outputs_dict['spatial_shapes'],
                    encoder_inputs_dict['level_start_index'],
                    encoder_inputs_dict['valid_ratios'],
                )
            if visual_dict is not None:
                embeddings = visual_dict['embedded']
                for j, (embedded, selects_per_sample) in enumerate(
                        zip(embeddings, visual_prompts['selects'])):
                    indices = list(selects_per_sample.keys())
                    if len(indices) > 0:
                        embed_instance = InstanceData()
                        embed_instance.embedded = embedded[indices]
                        row = self._local_to_global_row(local_to_global_ids, j)
                        embed_instance.labels = [row[lbl] for lbl in indices]
                        embed_results_list[j].append(embed_instance)
        else:
            text_dict = self.encode_text(
                text_prompts,
                text_prompts_all_same,
                batch_inputs.device,
                chunked_size=self.bbox_head.max_text_len * 2)
            embeddings = text_dict['embedded']
            for j, (embedded, token_mask) in enumerate(
                    zip(embeddings, text_dict['text_token_mask'])):
                indices = torch.nonzero(
                    token_mask, as_tuple=False).squeeze(1).tolist()
                if len(indices) > 0:
                    embed_instance = InstanceData()
                    embed_instance.embedded = embedded[indices]
                    row = self._local_to_global_row(local_to_global_ids, j)
                    embed_instance.labels = [row[lbl] for lbl in indices]
                    embed_results_list[j].append(embed_instance)
        for data_samples in batch_data_samples:
            data_samples.pred_instances = self.empty_pred_instance(device)
        for data_samples, embed_instances, entity_full in zip(
                batch_data_samples, embed_results_list, entities_full):
            if len(embed_instances) > 0:
                embed_instances = embed_instances[0].cat(embed_instances)
                self._attach_label_names_to_instances(
                    embed_instances, entity_full)
                data_samples.embed_instances = embed_instances
        for data_samples in batch_data_samples:
            if rescale:
                assert data_samples.metainfo.get('scale_factor') is not None
                data_samples.gt_instances.bboxes /= (
                    data_samples.gt_instances.bboxes.new_tensor(
                        data_samples.metainfo['scale_factor']).repeat((1, 2)))
        return batch_data_samples
    
    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        mode = batch_data_samples[0].get("mode", self.mode)
        assert prompt_mode_base(mode) in PROMPT_MODE, \
        f'mode should one of {PROMPT_MODE}, but got {mode}'
        mode_core = prompt_mode_core(mode)
        present_only = prompt_mode_present_only(mode)
        if DEBUG:
            print_log(
                f"mode {mode} core={mode_core} present_only={present_only}",
                logger='current')
            print_log(f"{batch_data_samples}", logger='current')
        bs = len(batch_inputs)
        device = batch_inputs.device
        gen_embed_only = 'gen_embed' in mode_core
        use_lm_text = prompt_mode_includes_text(mode)
        # text prompts
        # Assuming that the `custom_entities` flag
        # inside a batch is always the same. For single image inference
        custom_entities = batch_data_samples[0].get("custom_entities", False)
        text_prompts, text_prompts_all_same, entities = self.get_label_prompts(
            batch_data_samples,
            custom_entities=custom_entities,
            templates=simple_template)
        entities_full = entities
        local_to_global_ids_from_gt = None
        if present_only:
            local_to_global_ids_from_gt = self._local_to_global_ids_from_gt(
                batch_data_samples)
            text_prompts, entities, text_prompts_all_same = (
                self._filter_text_entities_present_only(
                    text_prompts,
                    entities,
                    text_prompts_all_same,
                    local_to_global_ids_from_gt))
            if not any(bool(row) for row in local_to_global_ids_from_gt):
                return self._finish_predict_empty(
                    batch_data_samples, device, rescale)
        local_to_global_ids = self._build_local_to_global_ids(
            text_prompts, text_prompts_all_same, local_to_global_ids_from_gt)
        # visual prompts    
        visual_prompts = None
        if "visual" in mode_core:
            visual_prompts_num = prompt_mode_visual_num(mode)
            if 'G' in mode_core: # generic
                visual_prompts = self.load_generic_visual_prompts(
                    batch_data_samples,
                    entities,
                    text_prompts_all_same,
                    visual_prompts_num,
                    device,
                )
            elif 'I' in mode_core: # interactive
                visual_prompts = self.load_interactive_visual_prompts(
                    batch_data_samples,
                    entities,
                    text_prompts_all_same,
                    use_point = "point" in mode_core, # point or box mode
                    visual_prompts_num=visual_prompts_num,
                    local_to_global_ids=local_to_global_ids,
                )

        if gen_embed_only:
            return self._predict_gen_embed_only(
                batch_inputs,
                batch_data_samples,
                mode=mode,
                text_prompts=text_prompts,
                text_prompts_all_same=text_prompts_all_same,
                entities=entities,
                visual_prompts=visual_prompts,
                rescale=rescale,
                entities_full=entities_full,
                local_to_global_ids=local_to_global_ids,
            )
        # chunks (chunked_size>0 asserts text_prompts_all_same inside chunk_prompts)
        chunked_size = prompt_mode_chunk_size(
            mode, int(self.test_cfg.get('chunked_size', -1)))
        if present_only and chunked_size > 0 and not text_prompts_all_same:
            chunked_size = -1
        (
            text_prompts_chunks,
            visual_prompts_chunks,
            chunk_skip,
            chunk_local_to_global_ids,
        ) = self.chunk_prompts(
            chunked_size,
            text_prompts,
            text_prompts_all_same,
            visual_prompts,
            mode,
            local_to_global_ids,
        )

        if not text_prompts_chunks:
            return self._finish_predict_empty(batch_data_samples, device, rescale)

        # predict
        results_list = [[] for _ in range(bs)]
        embed_results_list = [[] for _ in range(bs)]
        max_num_entities = self.bbox_head.max_text_len
        
        visual_feats = self.extract_feat(batch_inputs)

        for i, (text_prompts_once, visual_prompts_once, skip,
                chunk_local_to_global) in enumerate(
                    zip(text_prompts_chunks, visual_prompts_chunks, chunk_skip,
                        chunk_local_to_global_ids)):
            if skip:
                continue
            if DEBUG:
                print_log(
                    f"chunk_{i} {visual_prompts_once} {text_prompts_once}",
                    logger='current')
            max_num_once = max([len(j) for j in text_prompts_once])
            if max_num_once > max_num_entities:
                warnings.warn('Inputting a text that is too long will result '
                                'in poor prediction performance. '
                                'Please reduce the --chunked-size.')
            if use_lm_text:
                text_dict = self.encode_text(
                    text_prompts_once,
                    text_prompts_all_same,
                    batch_inputs.device)
            else:
                text_dict = self._placeholder_text_dict(
                    batch_inputs, text_prompts_once)
            head_inputs_dict, text_dict, _ = self.forward_transformer(
                copy.deepcopy(visual_feats),
                text_dict,
                batch_data_samples,
                visual_prompts_once,
                mode)
            
            if visual_prompts_once is not None and 'selects' in visual_prompts_once: # interactive mode
                for j, (embedded, selects_per_sample) in enumerate(
                        zip(text_dict['embedded'], visual_prompts_once['selects'])):
                    indices = list(selects_per_sample.keys())
                    if len(indices) > 0:
                        embed_instance = InstanceData()
                        embed_instance.embedded = embedded[indices]
                        row = self._local_to_global_row(
                            chunk_local_to_global, j)
                        embed_instance.labels = [row[lbl] for lbl in indices]
                        embed_results_list[j].append(embed_instance)

            for data_samples, text_prompts_per_sample in zip(
                    batch_data_samples, text_prompts_once):
                labels = np.arange(len(text_prompts_per_sample))
                positive_map_label_to_token, _ = self.get_positive_map(labels)
                data_samples.token_positive_map = positive_map_label_to_token
            results_list_chunk = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)
            for j, pred_instances in enumerate(results_list_chunk):
                if len(pred_instances) > 0:
                    labs = pred_instances.labels
                    local_to_global_row = self._local_to_global_row(
                        chunk_local_to_global, j)
                    gmap = torch.tensor(
                        local_to_global_row,
                        device=labs.device, dtype=labs.dtype)
                    pred_instances.labels = gmap[
                        labs.long().view(-1)].view_as(labs)
                results_list[j].append(pred_instances)
        # postprocess (labels are global ids into entities_full, not filtered entities)
        for data_samples, pred_instances, entity_full in zip(
                batch_data_samples, results_list, entities_full):
            if len(pred_instances) > 0:
                pred_instances = pred_instances[0].cat(pred_instances)
                self._attach_label_names_to_instances(
                    pred_instances, entity_full)
            else:  # fake data
                pred_instances = self.empty_pred_instance(device)
            data_samples.pred_instances = pred_instances
        for data_samples, embed_instances, entity_full in zip(
            batch_data_samples, embed_results_list, entities_full
        ):
            if len(embed_instances) > 0:
                embed_instances = embed_instances[0].cat(embed_instances)
                self._attach_label_names_to_instances(
                    embed_instances, entity_full)
                data_samples.embed_instances = embed_instances
        # bugfix, gt_instances和img同尺度，需要还原
        for data_samples in batch_data_samples:
            if rescale:
                assert data_samples.metainfo.get('scale_factor') is not None
                data_samples.gt_instances.bboxes /= data_samples.gt_instances.bboxes.new_tensor(
                    data_samples.metainfo['scale_factor']).repeat((1, 2))
            if DEBUG:
                print_log(f"{data_samples}", logger='current')
        return batch_data_samples


    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None,
            speed_test: bool = False,
            speed_profile_segments: bool = True,
    ) -> Tuple[List[Tensor]]:
        speed_prof = None
        if speed_test:
            speed_prof = TreX2SpeedProfiler(
                enabled=True,
                profile_segments=bool(speed_profile_segments))

        bs = len(batch_inputs)
        assert bs == 1
        mode = batch_data_samples[0].get("mode", self.mode)
        assert prompt_mode_base(mode) in PROMPT_MODE, \
            f'mode should one of {PROMPT_MODE}, but got {mode}'
        mode_core = prompt_mode_core(mode)
        use_lm_text = prompt_mode_includes_text(mode)

        if speed_prof is not None:
            if batch_inputs.is_cuda:
                torch.cuda.synchronize(batch_inputs.device)
            speed_prof.start_wall()

        with trex2_segment(speed_prof, 'backbone'):
            visual_feats = self.backbone(batch_inputs)
        if self.with_neck:
            with trex2_segment(speed_prof, 'neck'):
                visual_feats = self.neck(visual_feats)

        with trex2_segment(speed_prof, 'prompt_prep'):
            custom_entities = batch_data_samples[0].get("custom_entities", False)
            text_prompts, text_prompts_all_same, entities = self.get_label_prompts(
                batch_data_samples,
                custom_entities=custom_entities,
                templates=simple_template)
            visual_prompts = None
            if "visual" in mode_core:
                visual_prompts_num = prompt_mode_visual_num(mode)
                if visual_prompts_num is None:
                    visual_prompts_num = 1
                if 'G' in mode_core:
                    visual_prompts = self.load_generic_visual_prompts(
                        batch_data_samples,
                        entities,
                        text_prompts_all_same,
                        visual_prompts_num,
                        batch_inputs.device,
                    )
                elif 'I' in mode_core:
                    local_to_global_ids = self._build_local_to_global_ids(
                        text_prompts, text_prompts_all_same)
                    visual_prompts = self.load_interactive_visual_prompts(
                        batch_data_samples,
                        entities,
                        text_prompts_all_same,
                        use_point="point" in mode_core,
                        visual_prompts_num=visual_prompts_num,
                        local_to_global_ids=local_to_global_ids,
                    )

        if DEBUG:
            print_log(f"{visual_prompts} {text_prompts}", logger='current')

        with trex2_segment(speed_prof, 'misc_text_len_check'):
            max_num_once = max([len(i) for i in text_prompts])
            max_num_entities = self.bbox_head.max_text_len
            if max_num_once > max_num_entities:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')

        if use_lm_text:
            with trex2_segment(speed_prof, 'text_encoder'):
                text_dict = self.encode_text(
                    text_prompts,
                    text_prompts_all_same,
                    batch_inputs.device)
        else:
            text_dict = self._placeholder_text_dict(
                batch_inputs, text_prompts)

        with trex2_segment(speed_prof, 'pre_transformer'):
            encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
                visual_feats, batch_data_samples)
        with trex2_segment(speed_prof, 'encoder'):
            encoder_outputs_dict = self.forward_encoder(
                **encoder_inputs_dict, text_dict=text_dict)
        if self.use_visual_pre_encoder:
            mem = encoder_inputs_dict['feat']
            mem_mask = encoder_inputs_dict['feat_mask']
            sp = encoder_inputs_dict['spatial_shapes']
            lsi = encoder_inputs_dict['level_start_index']
            vr = encoder_inputs_dict['valid_ratios']
        else:
            mem = encoder_outputs_dict['memory']
            mem_mask = encoder_outputs_dict['memory_mask']
            sp = encoder_outputs_dict['spatial_shapes']
            lsi = encoder_inputs_dict['level_start_index']
            vr = encoder_inputs_dict['valid_ratios']
        prof_seg = speed_prof is not None and speed_prof.profile_segments
        use_split_vpg = (
            prof_seg and visual_prompts is not None
            and 'embedded' not in visual_prompts)
        if use_split_vpg:
            with trex2_segment(speed_prof, 'vpg_prepare'):
                prep = self._encode_visual_prepare_interactive(
                    visual_prompts, mem)
            if prep is None:
                visual_dict = None
            else:
                with trex2_segment(speed_prof, 'visual_prompt_generator'):
                    visual_dict = self._encode_visual_assemble_vpg(
                        prep, mem, mem_mask, sp, lsi, vr)
        else:
            with trex2_segment(speed_prof, 'vpg'):
                visual_dict = self.encode_visual(
                    visual_prompts=visual_prompts,
                    memory=mem,
                    memory_mask=mem_mask,
                    spatial_shapes=sp,
                    level_start_index=lsi,
                    valid_ratios=vr,
                )
        text_dict, vpg_aux = self.merge_visual_to_text_dict(
            text_dict, visual_dict, mode)
        encoder_outputs_dict.update({
            'memory_text':text_dict['embedded'],
            'text_token_mask':text_dict['text_token_mask'],
        })
        with trex2_segment(speed_prof, 'pre_decoder'):
            tmp_dec_in, head_inputs_dict = self.pre_decoder(
                **encoder_outputs_dict,
                batch_data_samples=batch_data_samples)
            decoder_inputs_dict.update(tmp_dec_in)
        with trex2_segment(speed_prof, 'decoder'):
            decoder_outputs_dict = self.forward_decoder(
                **decoder_inputs_dict)
        with trex2_segment(speed_prof, 'misc_merge_head_inputs'):
            head_inputs_dict.update(decoder_outputs_dict)
        with trex2_segment(speed_prof, 'head'):
            results = self.bbox_head.forward(
                **head_inputs_dict)

        if speed_prof is not None:
            self.latest_speed_stats = speed_prof.finalize()
        return results

    def deploy(self):
        """Switch model to deploy mode by fusing supported blocks."""
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
