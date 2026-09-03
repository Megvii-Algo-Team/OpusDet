# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import sys
from collections import defaultdict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from mmcv.image import imwrite
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmdet.registry import DATASETS, VISUALIZERS
from mmdet.structures.bbox import BaseBoxes

def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        '-o',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument('--show-num', '-n', type=int, default=30)
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=0,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def draw_all_character(visualizer, characters, w):
    start_index = 2
    y_index = 5
    for char in characters:
        if isinstance(char, str):
            visualizer.draw_texts(
                str(char),
                positions=np.array([start_index, y_index]),
                colors=(0, 0, 0),
                font_families='monospace')
            start_index += len(char) * 8
        else:
            visualizer.draw_texts(
                str(char[0]),
                positions=np.array([start_index, y_index]),
                colors=char[1],
                font_families='monospace')
            start_index += len(char[0]) * 8

        if start_index > w - 10:
            start_index = 2
            y_index += 15

    drawn_text = visualizer.get_image()
    return drawn_text


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    assert args.show_num > 0

    # register all modules in mmdet into the registries
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    dataset_index = list(range(len(dataset)))
    if args.shuffle:
        import random
        random.shuffle(dataset_index)

    progress_bar = ProgressBar(len(dataset))
    for i in dataset_index[:args.show_num]:
        progress_bar.update()
        item = dataset[i]
        img = item['inputs'].permute(1, 2, 0).numpy()
        data_sample = item['data_samples'].numpy()
        gt_instances = data_sample.gt_instances
        tokens_positive = data_sample.tokens_positive

        gt_labels = gt_instances.labels

        base_name = osp.basename(item['data_samples'].img_path)
        name, extension = osp.splitext(base_name)

        out_file = osp.join(args.output_dir, name + '_' + str(i) +
                            extension) if args.output_dir is not None else None

        img = img[..., [2, 1, 0]]  # bgr to rgb
        gt_bboxes = gt_instances.get('bboxes', None)
        if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
            gt_instances.bboxes = gt_bboxes.tensor

        print(data_sample.img_path, data_sample.text)

        dataset_mode = data_sample.dataset_mode
        if dataset_mode == 'VG':
            max_label = int(max(gt_labels) if len(gt_labels) > 0 else 0)
            palette = np.random.randint(0, 256, size=(max_label + 1, 3))
            bbox_palette = [tuple(c) for c in palette]
            # bbox_palette = get_palette('random', max_label + 1)
            colors = [bbox_palette[label] for label in gt_labels]

            visualizer.set_image(img)

            for label, bbox, color in zip(gt_labels, gt_bboxes, colors):
                visualizer.draw_bboxes(
                    bbox, edge_colors=color, face_colors=color, alpha=0.3)
                visualizer.draw_bboxes(bbox, edge_colors=color, alpha=1)
                # texts = ". ".join([data_sample.text[s:e] for s, e in tokens_positive[label]])
                entities = []
                text_prompt = data_sample.text
                tokens = tokens_positive[label]
                tokens = sorted(tokens, key=lambda x: x[0])
                prev = -10
                for (s, e) in tokens:
                    if prev+1 == s and text_prompt[prev] == ' ':
                        entities[-1] += f" {text_prompt[s:e]}"
                    else:
                        entities.append(text_prompt[s:e])
                    prev = e
                texts = ". ".join(entities)
                visualizer.draw_texts(
                    texts, np.array([int(bbox[0]), int(bbox[1])]),
                    colors=color)

            drawn_img = visualizer.get_image()

            new_image = np.ones((100, img.shape[1], 3), dtype=np.uint8) * 255
            visualizer.set_image(new_image)

            gt_tokens_positive = [
                tokens_positive[label] for label in gt_labels
            ]
            # print(gt_labels, gt_tokens_positive)
            split_by_character = [char for char in data_sample.text]
            characters = []
            start_index = 0
            end_index = 0
            for w in split_by_character:
                end_index += len(w)
                is_find = False
                for i, positive in enumerate(gt_tokens_positive):
                    for p in positive:
                        if start_index >= p[0] and end_index <= p[1]:
                            characters.append([w, colors[i]])
                            is_find = True
                            break
                    if is_find:
                        break
                if not is_find:
                    characters.append([w, (0, 0, 0)])
                start_index = end_index

            drawn_text = draw_all_character(visualizer, characters,
                                            img.shape[1])
            drawn_img = np.concatenate((drawn_img, drawn_text), axis=0)
        else:
            gt_labels = gt_instances.labels
            text = data_sample.text
            label_names = []
            for label in gt_labels:
                label_names.append(text[
                    tokens_positive[label][0][0]:tokens_positive[label][0][1]])
            gt_instances.label_names = label_names
            data_sample.gt_instances = gt_instances

            visualizer.add_datasample(
                base_name,
                img,
                data_sample,
                draw_pred=False,
                show=False,
                wait_time=0,
                out_file=None)
            drawn_img = visualizer.get_image()

            new_image = np.ones((100, img.shape[1], 3), dtype=np.uint8) * 255
            visualizer.set_image(new_image)

            # characters = [char for char in text]
            gt_tokens_positive = list(tokens_positive.values())
            # print(gt_labels, gt_tokens_positive)
            split_by_character = [char for char in data_sample.text]
            characters = []
            start_index = 0
            end_index = 0
            for w in split_by_character:
                end_index += len(w)
                is_find = False
                for positive in gt_tokens_positive:
                    for p in positive:
                        if start_index >= p[0] and end_index <= p[1]:
                            characters.append([w, (255, 0, 0)])
                            is_find = True
                            break
                    if is_find:
                        break
                if not is_find:
                    characters.append([w, (0, 0, 0)])
                start_index = end_index
            drawn_text = draw_all_character(visualizer, characters,
                                            img.shape[1])
            drawn_img = np.concatenate((drawn_img, drawn_text), axis=0)

        if not args.not_show:
            visualizer.show(
                drawn_img, win_name=base_name, wait_time=args.show_interval)

        if out_file is not None:
            imwrite(drawn_img[..., ::-1], out_file)

if __name__ == '__main__':
    main()
