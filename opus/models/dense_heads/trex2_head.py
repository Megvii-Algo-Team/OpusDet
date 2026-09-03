import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import constant_init
from mmengine.structures import InstanceData
from mmengine.runner.amp import autocast
from torch import Tensor
import torch.nn.functional as F
from mmdet.models.losses import QualityFocalLoss
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import InstanceList, reduce_mean
from mmdet.models.utils import multi_apply
from mmdet.models.layers import inverse_sigmoid
from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores
from mmdet.models.dense_heads.dino_head import DINOHead

import os
DEBUG = os.getenv("DEBUG", '').lower() in ('y', 'yes', 'true', '1')




class ContrastiveEmbed(nn.Module):
    """Text/visual contrastive embedding used by the detection head."""

    def __init__(self,
                 max_text_len: int = 256,
                 log_scale: Optional[Union[str, float]] = None,
                 bias: bool = False):
        super().__init__()
        self.max_text_len = max_text_len
        self.log_scale = log_scale
        if isinstance(log_scale, float):
            self.log_scale = nn.Parameter(
                torch.Tensor([float(log_scale)]), requires_grad=True)
        elif log_scale not in ['auto', 'none', None]:
            raise ValueError(f'log_scale should be one of '
                             f'"auto", "none", None, but got {log_scale}')
        self.bias = None
        if bias:
            bias_value = -math.log((1 - 0.01) / 0.01)
            self.bias = nn.Parameter(
                torch.Tensor([bias_value]), requires_grad=True)

    def forward(self, visual_feat: Tensor, text_feat: Tensor,
                text_token_mask: Tensor) -> Tensor:
        res = visual_feat @ text_feat.transpose(-1, -2)
        if isinstance(self.log_scale, nn.Parameter):
            res = res * self.log_scale.exp()
        elif self.log_scale == 'auto':
            res = res / math.sqrt(visual_feat.shape[-1])
        if self.bias is not None:
            res = res + self.bias
        res.masked_fill_(~text_token_mask[:, None, :], float('-inf'))
        new_res = torch.full((*res.shape[:-1], self.max_text_len),
                             float('-inf'),
                             device=res.device)
        new_res[..., :res.shape[-1]] = res
        return new_res


@MODELS.register_module(force=True)
class TRex2Head(DINOHead):
    """Head of TRex2."""

    def __init__(self,
                 contrastive_cfg=dict(max_text_len=256),
                 vpg_loss_cfg: Optional[Dict] = None,
                 **kwargs):
        self.contrastive_cfg = contrastive_cfg
        self.max_text_len = contrastive_cfg.get('max_text_len', 256)
        super().__init__(**kwargs)
        self.vpg_loss_cfg = dict(vpg_loss_cfg) if vpg_loss_cfg else {}
        self._init_vpg_loss_modules()

    def _init_vpg_loss_modules(self) -> None:
        cfg = self.vpg_loss_cfg
        if 'vl_align' in cfg:
            vl = dict(cfg.get('vl_align') or {})
            vl.pop('log_scale', None)
            vl.pop('fixed_T', None)
            vl.pop('align_norm_type', None)
            vl.pop('use_semantic_downweight', False)
            vl.setdefault('type', 'CrossEntropyLoss')
            vl.setdefault('loss_weight', 1.0)
            self.loss_vl_align_type = vl['type']
            assert self.loss_vl_align_type in ["CrossEntropyLoss", "FocalLoss"], (
                f'{self.loss_vl_align_type} not supported')
            self.vl_align_loss_module = MODELS.build(vl)
            self.vl_align_log_scale = nn.Parameter(
                torch.tensor([math.log(1.0/0.07)], dtype=torch.float32), requires_grad=True)
        else:
            self.vl_align_loss_module = None
            self.loss_vl_align_type = None
            self.vl_align_log_scale = None

    def vl_align(self, text, visual):
        """L2-normalized text/visual similarity, scaled by ``exp(align_log_scale)``."""
        sim = F.normalize(visual, p=2, dim=-1) @ F.normalize(
            text, p=2, dim=-1).transpose(-1, -2)
        if self.vl_align_log_scale is None:
            return sim
        return sim * self.vl_align_log_scale.exp()

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = ContrastiveEmbed(**self.contrastive_cfg)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        # NOTE: due to the fc_cls is a contrastive embedding and don't
        # have any trainable parameters,we do not need to copy it.
        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        gt_bboxes = gt_instances.bboxes

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        labels = gt_bboxes.new_full((num_bboxes, self.max_text_len),
                                    0,
                                    dtype=torch.float32)
        labels[pos_inds] = gt_instances.positive_maps[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        pos_gt_inds = pos_assigned_gt_inds.long()
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, pos_gt_inds)

    def get_targets(self, cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict]) -> tuple:
        """Same as :meth:`DETRHead.get_targets` plus ``pos_gt_inds_list``."""
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list, pos_gt_inds_list) = multi_apply(
            self._get_targets_single,
            cls_scores_list,
            bbox_preds_list,
            batch_gt_instances,
            batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg, pos_inds_list,
                pos_gt_inds_list)

    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
        enc_outputs_class: Optional[Tensor] = None,
        enc_outputs_coord: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        if not self.training:
            test_cfg = getattr(self, 'test_cfg', None)
            decoder_early_exit_layer = (
                int(test_cfg.get('decoder_early_exit_layer', -1))
                if test_cfg is not None and test_cfg.get('decoder_early_exit_layer', None) is not None
                else -1)
            if decoder_early_exit_layer == 0 \
                    and enc_outputs_class is not None \
                    and enc_outputs_coord is not None:
                all_layers_cls_scores = (
                    enc_outputs_class.unsqueeze(0)
                    if enc_outputs_class.dim() == 3 else enc_outputs_class)
                all_layers_bbox_preds = (
                    enc_outputs_coord.unsqueeze(0)
                    if enc_outputs_coord.dim() == 3 else enc_outputs_coord)
                return all_layers_cls_scores, all_layers_bbox_preds

        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        layer_ids = range(hidden_states.shape[0])
        if not self.training:
            test_cfg = getattr(self, 'test_cfg', None)
            inference_head_last_layer_only = bool(
                test_cfg.get('inference_head_last_layer_only', False)
            ) if test_cfg is not None else False
            if inference_head_last_layer_only and hidden_states.shape[0] > 0:
                layer_ids = [int(hidden_states.shape[0] - 1)]

        for layer_id in layer_ids:
            _ref_in = references[layer_id]
            reference = inverse_sigmoid(_ref_in)
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state,
                                                        memory_text,
                                                        text_token_mask)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                tmp_reg_preds += reference
            else:
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_text: Tensor,
                text_token_mask: Tensor,
                batch_data_samples: SampleList,
                enc_outputs_class: Optional[Tensor] = None,
                enc_outputs_coord: Optional[Tensor] = None,
                rescale: bool = True) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_samples.token_positive_map
            for data_samples in batch_data_samples
        ]

        outs = self(
            hidden_states,
            references,
            memory_text,
            text_token_mask,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        batch_token_positive_maps: Optional[List[dict]] = None,
                        rescale: bool = False) -> InstanceList:
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            token_positive_maps = batch_token_positive_maps[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   token_positive_maps,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                token_positive_maps: dict,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']

        if token_positive_maps is not None:
            cls_score = convert_grounding_to_cls_scores(
                logits=cls_score.sigmoid()[None],
                positive_maps=[token_positive_maps])[0]
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            if DEBUG:
                print(f"{scores, indexes}")
            num_classes = cls_score.shape[-1]
            det_labels = indexes % num_classes
            bbox_index = indexes // num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            cls_score = cls_score.sigmoid()
            scores, _ = cls_score.max(-1)
            scores, indexes = scores.topk(max_per_img)
            bbox_pred = bbox_pred[indexes]
            det_labels = scores.new_zeros(scores.shape, dtype=torch.long)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))
        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        return results

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_text: Tensor, text_token_mask: Tensor,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int],
             vpg_aux: Optional[Dict] = None) -> dict:
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references, memory_text, text_token_mask)
        self.text_masks = text_token_mask
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(
            *loss_inputs, hidden_states=hidden_states)
        losses.update(self._loss_vl_align(vpg_aux))
        return losses

    def loss_by_feat(
            self,
            all_layers_cls_scores: Tensor,
            all_layers_bbox_preds: Tensor,
            enc_cls_scores: Tensor,
            enc_bbox_preds: Tensor,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            dn_meta: Dict[str, int],
            hidden_states: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)
        all_layers_matching_hs = [None] * len(all_layers_matching_cls_scores)
        if hidden_states is not None:
            if dn_meta is not None \
                and dn_meta.get('num_denoising_queries', 0) > 0:
                n_dn = dn_meta['num_denoising_queries']
                all_layers_matching_hs = hidden_states[:, :, n_dn:, :]
                # all_layers_matching_hs[-1] = hidden_states[-1, :, n_dn:, :] # last layer only
            else:
                all_layers_matching_hs = hidden_states
                # all_layers_matching_hs[-1] = hidden_states[-1, :, :, :] # last layer only

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_by_feat_single,
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            all_layers_matching_hs,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        num_dec_layers = len(losses_cls)
        layer_weights = [1.0] * num_dec_layers

        loss_dict: Dict[str, Tensor] = dict()
        loss_dict['loss_cls'] = losses_cls[-1] * layer_weights[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1] * layer_weights[-1]
        loss_dict['loss_iou'] = losses_iou[-1] * layer_weights[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            w = layer_weights[num_dec_layer]
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i * w
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i * w
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i * w
            num_dec_layer += 1

        if enc_cls_scores is not None:
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
        return loss_dict

    def _loss_vl_align(
            self, vpg_aux: Optional[Dict] = None) -> Dict[str, Tensor]:
        """VL align loss from ``vpg_aux['vl_align']``."""
        out: Dict[str, Tensor] = {}
        if not self.training:
            return out
        vl_align_bundle = (
            vpg_aux.get('vl_align') if vpg_aux is not None else None)
        if vl_align_bundle is None or self.vl_align_loss_module is None:
            return out
        text_emb = vl_align_bundle.get('text_emb')
        visual_emb = vl_align_bundle.get('new_visual_embedded')
        text_token_mask = vl_align_bundle.get('text_token_mask')
        visual_token_mask = vl_align_bundle.get('new_visual_token_mask')
        if text_emb is None or visual_emb is None:
            out['loss_align'] = next(self.parameters()).sum() * 0
            return out
        B, M, _ = text_emb.shape
        device = text_emb.device
        align_score = self.vl_align(text_emb, visual_emb)
        valid_mask = (
            visual_token_mask.unsqueeze(-1) &
            text_token_mask.unsqueeze(1)
        )
        align_score = align_score.masked_fill(~valid_mask, float('-inf'))
        if self.loss_vl_align_type == "CrossEntropyLoss":
            token_ids = torch.arange(
                M, device=device).unsqueeze(0).expand(B, M)

            def _ce_on_selected_rows(score_mat, row_pos_mask):
                return self.vl_align_loss_module(
                    score_mat[row_pos_mask],
                    token_ids[row_pos_mask],
                    avg_factor=None)

            row_diag = visual_token_mask & text_token_mask
            loss_out = 0.5*(
                _ce_on_selected_rows(align_score, row_diag) +
                _ce_on_selected_rows(
                    align_score.transpose(1, 2), row_diag))
            if DEBUG and self.vl_align_log_scale is not None:
                with torch.no_grad():
                    _sc = float(self.vl_align_log_scale.exp().item())
                    print(
                        "[VL-Align] "
                        f"scale(exp(align_log_scale))={_sc:.4f}, "
                        f"align_score max={align_score.max().item():.2f}"
                    )
        elif self.loss_vl_align_type == "FocalLoss":
            batch_idx, token_idx = torch.where(visual_token_mask)
            align_label = torch.zeros((B, M, M), device=device)
            align_label[batch_idx, token_idx, token_idx] = 1.0
            align_score_flat = align_score.masked_select(valid_mask)
            align_label_flat = align_label.masked_select(valid_mask)
            avg_factor = torch.clamp(
                align_label_flat.sum(), min=1).item()
            loss_out = self.vl_align_loss_module(
                align_score_flat, align_label_flat,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError(
                f'Loss type {self.loss_vl_align_type} is not implemented')
        out['loss_align'] = loss_out
        return out


    def loss_by_feat_single(
            self,
            cls_scores: Tensor,
            bbox_preds: Tensor,
            hidden_states_layer: Optional[Tensor] = None,
            *,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        with torch.no_grad():
            cls_reg_targets = self.get_targets(cls_scores_list,
                                               bbox_preds_list,
                                               batch_gt_instances,
                                               batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, pos_inds_list,
         pos_gt_inds_list) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert (self.text_masks.dim() == 2)
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_scores.size(1), 1)
        cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()

        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        # classification loss
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            raise NotImplementedError(
                'QualityFocalLoss for TRex2Head is not supported yet.')
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        return loss_cls, loss_bbox, loss_iou

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        # ===== this change =====
        assert (self.text_masks.dim() == 2)
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, dn_cls_scores.size(1), 1)
        cls_scores = torch.masked_select(dn_cls_scores, text_mask).contiguous()
        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)
        # =======================

        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, QualityFocalLoss):
                raise NotImplementedError('QualityFocalLoss is not supported')
            else:
                loss_cls = self.loss_cls(
                    cls_scores,
                    labels,
                    label_weights,
                    avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> tuple:
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2
        labels = gt_bboxes.new_full((num_denoising_queries, self.max_text_len),
                                    0,
                                    dtype=torch.float32)
        labels[pos_inds] = gt_instances.positive_maps[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']

        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)
