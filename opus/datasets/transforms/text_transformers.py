# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np

from mmdet.datasets.transforms.text_transformers import (
    RandomSamplingNegPos as _Base, clean_name,
    generate_senetence_given_labels)
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes


def check_for_positive_overflow(gt_bboxes, gt_labels, text, tokenizer,
                                max_tokens):
    positive_label_list = np.unique(gt_labels).tolist()
    random.shuffle(positive_label_list)

    kept_lables = []
    length = 0

    for label in positive_label_list:
        label_text = clean_name(text[str(label)]) + '. '
        tokenized = tokenizer.tokenize(label_text)
        length += len(tokenized)
        if length > max_tokens:
            break
        kept_lables.append(label)

    keep_box_index = []
    keep_gt_labels = []
    for i in range(len(gt_labels)):
        if gt_labels[i] in kept_lables:
            keep_box_index.append(i)
            keep_gt_labels.append(gt_labels[i])

    return gt_bboxes[keep_box_index], np.array(
        keep_gt_labels, dtype=np.int64), length


@TRANSFORMS.register_module(force=True)
class RandomSamplingNegPos(_Base):
    """TreX2 training transform: ``rstrip`` text and ``int64`` label dtype."""

    def vg_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        text = results['text'].lower().rstrip()
        if not text.endswith('.'):
            text = text + '. '

        phrases = results['phrases']
        positive_label_list = np.unique(gt_labels).tolist()
        label_to_positions = {
            label: phrases[label]['tokens_positive']
            for label in positive_label_list
        }

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels
        results['text'] = text
        results['tokens_positive'] = label_to_positions
        return results

    def od_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']

        if 'text' not in results:
            assert self.label_map is not None
            text = self.label_map
        else:
            text = results['text']

        original_box_num = len(gt_labels)
        for key, value in text.items():
            if '/' in value:
                text[key] = random.choice(value.split('/')).strip()

        gt_bboxes, gt_labels, positive_caption_length = \
            check_for_positive_overflow(gt_bboxes, gt_labels,
                                        text, self.tokenizer, self.max_tokens)

        if len(gt_bboxes) < original_box_num:
            print('WARNING: removed {} boxes due to positive caption overflow'
                  .format(original_box_num - len(gt_bboxes)))

        valid_negative_indexes = list(text.keys())
        positive_label_list = np.unique(gt_labels).tolist()
        full_negative = self.num_sample_negative

        if full_negative > len(valid_negative_indexes):
            full_negative = len(valid_negative_indexes)

        outer_prob = random.random()

        if outer_prob < self.full_sampling_prob:
            num_negatives = full_negative
        else:
            if random.random() < 1.0:
                num_negatives = np.random.choice(max(1, full_negative)) + 1
            else:
                num_negatives = full_negative

        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)

            for i in np.random.choice(
                    valid_negative_indexes, size=num_negatives, replace=False):
                if int(i) not in positive_label_list:
                    negative_label_list.add(i)

        random.shuffle(positive_label_list)

        negative_label_list = list(negative_label_list)
        random.shuffle(negative_label_list)

        negative_max_length = self.max_tokens - positive_caption_length
        screened_negative_label_list = []

        for negative_label in negative_label_list:
            label_text = clean_name(text[str(negative_label)]) + '. '
            tokenized = self.tokenizer.tokenize(label_text)
            negative_max_length -= len(tokenized)
            if negative_max_length > 0:
                screened_negative_label_list.append(negative_label)
            else:
                break
        negative_label_list = screened_negative_label_list
        label_to_positions, pheso_caption, label_remap_dict = \
            generate_senetence_given_labels(positive_label_list,
                                            negative_label_list, text)

        if len(gt_labels) > 0:
            gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels
        results['text'] = pheso_caption
        results['tokens_positive'] = label_to_positions
        return results
