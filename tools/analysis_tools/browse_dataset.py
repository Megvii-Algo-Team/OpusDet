# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmdet.models.utils import mask2ndarray
from mmdet.registry import DATASETS, VISUALIZERS
from mmdet.structures.bbox import BaseBoxes


def get_dataloader_cfg(cfg, phase: str):
    """Return dataloader config for ``train`` or ``test`` split."""
    if phase == 'train':
        if not hasattr(cfg, 'train_dataloader'):
            raise ValueError('Config has no train_dataloader.')
        return cfg.train_dataloader
    if phase == 'test':
        if hasattr(cfg, 'test_dataloader'):
            return cfg.test_dataloader
        if hasattr(cfg, 'val_dataloader'):
            return cfg.val_dataloader
        raise ValueError(
            'Config has no test_dataloader or val_dataloader for --phase test.')
    raise ValueError(f'Unsupported phase: {phase!r}')


def resolve_dataset_meta(dataset, index: int):
    """Per-sample metainfo; ConcatDataset may expose ``metainfo`` as a list."""
    metainfo = dataset.metainfo
    if not isinstance(metainfo, list):
        return metainfo
    if hasattr(dataset, 'get_dataset_source'):
        return metainfo[dataset.get_dataset_source(index)]
    return metainfo[0]


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '-o',
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument('--show-num', '-n', type=int, default=30)
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument(
        '--phase',
        choices=('train', 'test'),
        default='train',
        help='which dataloader to browse: train_dataloader (default) or '
        'test_dataloader (falls back to val_dataloader)')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
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


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    dataloader_cfg = get_dataloader_cfg(cfg, args.phase)
    dataset = DATASETS.build(dataloader_cfg.dataset)
    visualizer_cfg = copy.deepcopy(cfg.visualizer)
    if args.output_dir is not None:
        visualizer_cfg['save_dir'] = args.output_dir
    visualizer = VISUALIZERS.build(visualizer_cfg)
    visualizer.dataset_meta = resolve_dataset_meta(dataset, 0)

    dataset_index = list(range(len(dataset)))
    if args.shuffle:
        import random
        random.shuffle(dataset_index)

    show_indices = dataset_index[:args.show_num]
    progress_bar = ProgressBar(len(show_indices))
    for i in show_indices:
        visualizer.dataset_meta = resolve_dataset_meta(dataset, i)
        item = dataset[i]
        img = item['inputs'].permute(1, 2, 0).numpy()
        data_sample = item['data_samples'].numpy()
        gt_instances = data_sample.gt_instances
        img_path = osp.basename(item['data_samples'].img_path)

        out_file = osp.join(
            args.output_dir,
            osp.basename(img_path)) if args.output_dir is not None else None

        img = img[..., [2, 1, 0]]  # bgr to rgb
        gt_bboxes = gt_instances.get('bboxes', None)
        if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
            gt_instances.bboxes = gt_bboxes.tensor
        gt_masks = gt_instances.get('masks', None)
        if gt_masks is not None:
            masks = mask2ndarray(gt_masks)
            gt_instances.masks = masks.astype(bool)
        data_sample.gt_instances = gt_instances

        visualizer.add_datasample(
            osp.basename(img_path),
            img,
            data_sample,
            draw_pred=False,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file)

        progress_bar.update()


if __name__ == '__main__':
    main()
