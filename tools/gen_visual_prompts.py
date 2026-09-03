#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Repo root so ``opus`` imports work without setting PYTHONPATH.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import json
import random
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Lock
from functools import partial
from argparse import ArgumentParser
from tqdm import tqdm
from mmengine.logging import print_log
import mmengine
from collections import defaultdict
import traceback

from opus.apis import PromptDetInferencer
from mmengine.fileio import get as get_
from mmcv.image import imfrombytes



def load_data_from_json(path):
    """Load a JSON object, JSON list, or JSONL (one object per line)."""
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            rows = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
            return rows


def prompt_output_subdir(weights, default='clip_b'):
    """``{work_dir_basename}_{ckpt_stem}`` under visual_prompts / text_prompts."""
    if not weights:
        return default
    norm = os.path.normpath(weights)
    ckpt_stem = os.path.splitext(os.path.basename(norm))[0]
    parts = norm.split(os.sep)
    if len(parts) >= 2:
        return f'{parts[-2]}_{ckpt_stem}'
    return ckpt_stem


def list_files(path, file_ext=None):
    res = []
    if isinstance(path, str):
        path = [path]
    for p in path:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for file in files:
                    if file_ext is None or file.endswith(file_ext):
                        res.append(os.path.join(root, file))
        elif os.path.isfile(p) and (file_ext is None or p.endswith(file_ext)):
            res.append(p)
    return res


def load_items_auto(json_file: str, min_size: int = 1):
    """
    自动加载 COCO / OD / VG 格式的数据，并转换为每张图片一条 item
    每条 item 结构：
        {
            'img_id': ...,
            'file_name': ...,
            'instances': [ {'bbox': [x1,y1,x2,y2], 'category_id': int, 'category': str, 'ignore_flag': 0}, ...],
            'text': [类别名称1, 类别名称2, ...]
        }
    返回：
        items: List[item]
        label_map: dict, 类名 -> category_id
    """
    try:
        data = load_data_from_json(json_file)
    except Exception as e:
        raise RuntimeError(f'Failed to load {json_file}: {e}') from e

    items = []
    label_set = set()
    if not data:
        return items, {}

    # Standard COCO is a single dict; OD/VG dumps are lists (or JSONL -> list).
    if isinstance(data, dict):
        if 'images' in data and 'annotations' in data:
            data = [data]
        else:
            raise ValueError(
                f'Unsupported dict JSON (need COCO images+annotations): {json_file}')

    item0 = data[0]

    # ---------- 1) COCO / OD ----------
    if isinstance(item0, dict) and "images" in item0 and "annotations" in item0:
        print("[INFO] Detected COCO/OD format")
        imgs = item0.get("images", [])
        anns = item0.get("annotations", [])
        categories = {i["id"]: i["name"] for i in item0.get("categories", [])}
        # 建立 image_id -> annotations 映射
        img_id2anns = defaultdict(list)
        for ann in anns:
            if ann.get('ignore', False) or ann.get('iscrowd', 0):
                continue
            try:
                x, y, w, h = ann['bbox']
            except Exception:
                continue
            if ann.get('area', w * h) <= 0 or w < min_size or h < min_size:
                continue
            ann['bbox'] = [x, y, x + w, y + h]
            img_id2anns[ann['image_id']].append(ann)
        for img_info in imgs:
            img_id = img_info.get("id")
            img_anns = img_id2anns.get(img_id, [])
            insts = []
            cat_names = []
            for ann in img_anns:
                cat_name = categories.get(ann.get("category_id"), "__unknown__")
                insts.append({
                    'bbox': ann['bbox'],
                    'category': cat_name,
                    'ignore_flag': 0
                })
                cat_names.append(cat_name)
                label_set.add(cat_name)
            if not insts:
                continue
            items.append({
                'img_id': img_id,
                'file_name': img_info.get("file_name", img_info.get("filename")),
                'instances': insts,
                'text': list(set(cat_names))
            })

    # ---------- 2) OD 格式 ----------
    elif isinstance(item0, dict) and "detection" in item0:
        print("[INFO] Detected OD format")
        for idx, it in enumerate(data):
            img_id = it.get("image_id", idx)
            file_name = it.get("filename", it.get("file_name", f"{img_id}.jpg"))

            det = it.get("detection", {})
            instances_raw = det.get("instances", [])
            insts = []
            cat_names = []

            for ann in instances_raw:
                cat_name = ann.get("category") or ann.get("label") or "__unknown__"
                cat_names.append(str(cat_name))
                label_set.add(str(cat_name))
                if not isinstance(ann, dict):
                    continue
                bbox = ann.get('bbox', [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    continue
                insts.append({
                    'bbox': [x1, y1, x2, y2],
                    'category': str(cat_name),
                    'ignore_flag': 0
                })

            if not insts:
                continue

            items.append({
                'img_id': img_id,
                'file_name': file_name,
                'instances': insts,
                'text': list(set(cat_names))
            })

    # ---------- 3) VG 格式 ----------
    elif isinstance(item0, dict) and "grounding" in item0:  # VG
        print("[INFO] Detected VG format")
        for idx, it in enumerate(data):
            grounding = it.get("grounding", {})
            caption = grounding.get("caption", "")
            img_id = it.get("image_id", idx)
            file_name = it.get("filename", it.get("file_name", f"{img_id}.jpg"))

            insts = []
            entity_set = set()

            for region in grounding.get("regions", []):
                tokens = region.get("tokens_positive", [])
                sorted_tokens = sorted(tokens, key=lambda x: x[0]) if isinstance(tokens, list) else []
                prev = -10
                entities = []
                for (s, e) in sorted_tokens:
                    if prev > 0 and prev < s and not caption[prev:s].replace(" ", ""):
                        entities[-1] += f" {caption[s:e]}"
                    elif prev == s:
                        entities[-1] += f"{caption[s:e]}"
                    else:
                        entities.append(caption[s:e])
                    prev = e

                label = ", ".join(entities)
                if not label:
                    label = "__unknown__"
                entity_set.add(label)
                label_set.add(label)

                bbox = region.get("bbox", [0, 0, 0, 0])
                bbox = region.get("bbox", [0,0,0,0])
                if not isinstance(bbox[0], list):
                    bbox = [bbox]
                for box in bbox:
                    if len(box) != 4:
                        continue
                    x1, y1, x2, y2 = box
                    if (x2 - x1) < min_size or (y2 - y1) < min_size:
                        continue
                    # 每个 region 作为一个 instance
                    insts.append({
                        "bbox": box, 
                        "category": label,
                        'ignore_flag':0
                    })

            if insts:
                items.append({
                    "img_id": img_id,
                    "file_name": file_name,
                    "instances": insts,
                    "text": list(entity_set)
                })
    else:
        raise ValueError(f"Unknown dataset format: {json_file}")

    # 构建 label_map
    label_list = sorted(label_set)
    label_map = {name: idx for idx, name in enumerate(label_list)}

    print(f"[INFO] Loaded {len(items)} items with {len(label_map)} categories from {json_file}")
    return items, label_map


# -----------------------------
# run_batch_infer: 执行单个 batch 的推理并返回 embeddings dict
# -----------------------------
def run_batch_infer(task_param, inferencer, data_prefix=''):
    """
    task_param:
      {
        'json_file', 'mode', 'text',
        'items' (list of item dicts for this batch),
        'batch_id'
      }
    Returns:
      dict: {embedding_key: [list of embeddings]}
    """
    mode = task_param['mode']
    text = task_param['text']
    items = task_param['items']
    pred_score_thr = task_param.get('pred_score_thr', 0.2)
    save_pred = task_param.get('save_pred', False)
    save_vis = task_param.get('save_vis', False)
    print_result = task_param.get('print_result', False)
    out_dir = task_param.get('out_dir', None)
    batch_id = task_param['batch_id']
    batch_size = task_param['batch_size']
    inferencer.num_visualized_imgs = batch_id*batch_size
    sample_inputs, sample_texts, sample_instances = [], [], []

    for item in items:
        insts = []
        for inst in item.get("instances", []):
            if inst.get("category") == text:
                inst_copy = inst.copy()
                inst_copy['bbox_label'] = 0
                insts.append(inst_copy)
        if not insts:
            continue

        img_path = os.path.join(data_prefix, item['file_name'])
        try:
            img_bytes = get_(img_path)  # local path via mmengine.fileio
            img = imfrombytes(img_bytes, flag='color', backend='pillow')
        except Exception as e:
            print(f"[WARN] Failed to read {img_path}: {e}")
            continue

        sample_inputs.append(img)
        sample_texts.append(text)
        sample_instances.append(insts)

    if not sample_inputs:
        return {}

    call_args = {
        'inputs': sample_inputs,
        'texts': sample_texts,
        'mode': mode,
        'instances': sample_instances,
        'out_dir': out_dir,
        'pred_score_thr': pred_score_thr,
        'batch_size': len(sample_inputs),
        'no_save_vis': not save_vis,
        'no_save_pred': not save_pred,
        'print_result': print_result,
        'custom_entities': True,
        'tokens_positive': None,
        'show': False,
    }

    results = inferencer(**call_args)
    batch_results = defaultdict(list)
    for res in results.get('predictions', []):
        if "embeddings" in res:
            for k, v in res['embeddings'].items():
                batch_results[k].extend(v)

    return dict(batch_results)


# -----------------------------
# gpu_worker: 每个 GPU 上跑一个 worker，从队列拉任务 -> 推理 -> 写回 result_queue
# -----------------------------
def gpu_worker(task_queue: Queue,
               result_queue: Queue,
               progress_counter: Value,
               progress_lock: Lock,
               worker_id: int,
               model_cfg: str,
               weights_path: str,
               data_prefix: str = ''):
    """
    每个进程启动一个 PromptDetInferencer 并循环拉任务。
    result_queue: 用于将 (task_id, json_file, mode, batch_results) 发送回主进程
    progress_counter: multiprocessing.Value (int) - 已完成 batch 计数
    progress_lock: Lock protecting increment of progress_counter
    """
    try:
        device = f'cuda:{worker_id}' if torch.cuda.is_available() else None
        inferencer = PromptDetInferencer(model=model_cfg, weights=weights_path, device=device, show_progress=False)
    except Exception as e:
        print(f"[ERROR] worker {worker_id} failed to init inferencer: {e}")
        traceback.print_exc()
        inferencer = None

    while True:
        try:
            task_param = task_queue.get(timeout=5)  # 超时退出
        except Exception:
            break

        try:
            if inferencer is None:
                batch_results = {}
            else:
                batch_results = run_batch_infer(task_param, inferencer, data_prefix=data_prefix)
        except Exception as e:
            print(f"[ERROR][worker {worker_id}] Exception while inferencing: {e}")
            traceback.print_exc()
            batch_results = {}

        # 发送到结果队列：包含 task_id, json_file, mode, batch_results
        try:
            res_obj = task_param
            res_obj['batch_results'] = batch_results
            result_queue.put(res_obj)
            # debug print
            if batch_results:
                total = len(list(batch_results.values())[0])
            else:
                total = 0
            print(f"[worker {worker_id}] finished task {res_obj['task_id']} cat='{res_obj['text']}' count={total}")
        except Exception as e:
            print(f"[ERROR][worker {worker_id}] Failed to put result into queue: {e}")
            traceback.print_exc()

        # 更新进度计数（在 progress_lock 下）
        if progress_counter is not None:
            with progress_lock:
                progress_counter.value += 1

    # 结束
    print(f"[worker {worker_id}] exiting.")


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-json-path", type=str, nargs='+', help="List of COCO annotation JSON files or a directory.")
    parser.add_argument("--file-ext", type=str, default='.json', help="File extension to look for when traversing directories, e.g., '.json' or '.jsonl'")
    parser.add_argument('--file-patten', type=str, default='.json')
    parser.add_argument('--data-prefix', type=str, default='')
    parser.add_argument('--out-dir', type=str, default='./work_dirs')
    parser.add_argument('--mode', type=str, nargs='+', default=['visual.I', 'text_only'], choices=['visual.I', 'text_only'])
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel threads.")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--sample-num-limit', type=int, default=32)
    parser.add_argument('--save-vis', action='store_true')
    parser.add_argument('--save-pred-single', action='store_true')
    parser.add_argument('--print-result', action='store_true')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='PromptDetInferencer config .py (default: opus_dinov3_convnext-b pretrain_o365).')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Checkpoint .pth (default: paired with default model).')
    args = parser.parse_args()

    _default_model = 'configs/opus_dinov3_convnext-b/opus_dinov3_convnext-b_pretrain_o365.py'
    _default_weights = (
        'work_dirs/opus_dinov3_convnext-b_text_visual_pretrain_o365_goldg_amp/'
        'iter_100000.pth')
    model = args.model or _default_model
    weights = args.weights or _default_weights

    # 列出输入 json 文件
    json_files = list_files(args.input_json_path, args.file_ext)
    json_files = [file for file in json_files if args.file_patten in file]
    if not json_files:
        print_log("No JSON files found.")
        return

    # workers 数量（每个 GPU 一个 worker）
    args.start_worker = 0
    if torch.cuda.is_available():
        args.num_workers = torch.cuda.device_count()
        print(f"Start processing with {args.num_workers} workers across {args.num_workers} GPUs.")
    else:
        print(f"Start processing with {args.num_workers} workers, CPU only.")

    # 任务队列与结果队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # 生成任务：按 json_file -> mode -> category -> 切 batch，并为每个任务分配唯一 task_id
    total_batches = 0
    task_id = 0
    for json_file in json_files:
        items, _ = load_items_auto(json_file)
        # collect categories
        cat_to_item_ids = defaultdict(list)
        for i, item in enumerate(items):
            for cat_name in item.get("text", []):
                cat_to_item_ids[cat_name].append(i)
        for mode in args.mode:
            mode += ".gen_embed"
            subdir_1 = mode
            if mode == 'text_only.gen_embed':
                subdir_1 = 'text_prompts'
            elif mode == 'visual.I.gen_embed':
                subdir_1 = 'visual_prompts'
            subdir_2 = prompt_output_subdir(weights)
            out_dir = os.path.join(
                args.out_dir,
                subdir_1,
                subdir_2
            )
            preds_dir = os.path.join(out_dir, 'preds')
            os.makedirs(preds_dir, exist_ok=True)
            filename = os.path.splitext(os.path.basename(json_file))[0]
            out_pickle_path = os.path.join(preds_dir, f'{filename}.pkl')
            out_sub_dir = os.path.join(out_dir, "samples", filename)
            if args.save_vis or args.save_pred_single:
                os.makedirs(out_sub_dir, exist_ok=True)

            if 'text_only' in mode:
                sample_num_limit = 1
                batch_size = 1
            else:
                sample_num_limit = args.sample_num_limit
                batch_size = args.batch_size
            # 按类别采样
            for idx, (cat_name, cat_item_ids) in enumerate(cat_to_item_ids.items()):
                print(f"[INFO] Sampling category {idx} '{cat_name}' with {len(cat_item_ids)} images")
                random.shuffle(cat_item_ids)

                # optional sample limit across category
                sample_limit = sample_num_limit if sample_num_limit > 0 else len(cat_item_ids)
                cat_item_ids = cat_item_ids[:sample_limit]

                # split batches
                for i in range(0, len(cat_item_ids), batch_size):
                    batch_items = [items[item_id] for item_id in cat_item_ids[i:i + batch_size]]
                    task = {
                        'task_id': task_id,
                        'batch_id': i // batch_size,
                        'batch_size':batch_size,
                        'items': batch_items,
                        'out_pickle_path':out_pickle_path,
                        'items': batch_items,
                        # detinferencer params
                        'mode': mode,
                        'text': cat_name,
                        'save_vis':args.save_vis,
                        'save_pred': args.save_pred_single,
                        'print_result': args.print_result,
                        'out_dir':out_sub_dir
                    }
                    task_queue.put(task)
                    total_batches += 1
                    task_id += 1

    print(f"[INFO] Total batches queued: {total_batches}")

    # 多进程构件：不使用 Manager.dict，使用 result_queue 汇总
    progress_counter = Value('i', 0)  # 已完成 batch 计数
    progress_lock = Lock()

    # 启动 GPU workers（每个 GPU 启动一个 process）
    workers = []
    for wid in range(args.start_worker, args.start_worker+args.num_workers):
        p = Process(target=gpu_worker, args=(
            task_queue, result_queue, progress_counter, progress_lock,
            wid, model, weights, args.data_prefix))
        p.start()
        workers.append(p)

    # 主进程收集结果并合并
    merged_results = {}  # key = f"{json_file}||{mode}" -> {embed_key: [..]}
    processed = 0

    with tqdm(total=total_batches, desc="Batches processed") as pbar:
        last = 0
        while processed < total_batches:
            try:
                res = result_queue.get(timeout=1.0)
            except Exception:
                # 可能没有新结果，继续等待
                continue

            key = res.get('out_pickle_path')
            batch_results = res.get('batch_results', {})

            if key not in merged_results:
                merged_results[key] = {}

            for k, v in batch_results.items():
                merged_results[key].setdefault(k, [])
                merged_results[key][k].extend(v)

            processed += 1
            pbar.update(1)

            # keep progress_counter in sync for compatibility (optional)
            with progress_lock:
                progress_counter.value = processed

    # 等待 worker 结束
    for p in workers:
        p.join()

    # 主程序统一保存：按 json_file + mode 保存一个 pickle 文件
    for out_pickle_path, results in merged_results.items():
        if not results:
            print_log(
                f"[WARN] No embeddings for {out_pickle_path}. "
                "Check logs for '[WARN] Failed to read' (missing local files) "
                "or '[ERROR][worker]' (inferencer). "
                f"Images are read from data-prefix='{args.data_prefix}' + file_name "
                "(LVIS OD uses train2017/…; local data/coco often has no train2017).",
                logger='current',
                level='WARNING')
        mmengine.dump(dict(results), out_pickle_path)
        total_counts = [(k, len(v)) for k, v in results.items()]
        print(f"[INFO] Saved merged results to {out_pickle_path}")
        print(f"[INFO] Total embeddings: {total_counts}")

    print("[INFO] All done.")
    # clean exit
    try:
        # close queues
        task_queue.close()
        result_queue.close()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    # Use spawn to be safe with CUDA + multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
