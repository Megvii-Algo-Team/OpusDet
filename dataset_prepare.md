# OPUS Dataset Preparation

Put datasets under `data/` (or set `ROOT_DATA_DIR=/path/to/datasets`). You can also edit `root_data_dir` in dataset configs under [`configs/datasets/`](configs/datasets/).

Download / convert steps for shared datasets follow MM-GDINO docs:

- [dataset_prepare.md](third_party/mmdetection/configs/mm_grounding_dino/dataset_prepare.md)
- [dataset_prepare_zh-CN.md](third_party/mmdetection/configs/mm_grounding_dino/dataset_prepare_zh-CN.md)

Legend: **raw** = original download, **mid** = converter output, **used** = path read by configs.

---

# Train

## 1. Objects365 v1

Follow MM-GDINO § Objects365v1 (`coco2odvg.py`).

```bash
python tools/dataset_converters/coco2odvg.py data/objects365v1/objects365_train.json -d o365v1
# → objects365_train_od.json + o365v1_label_map.json
```

```text
data/objects365v1/
├── objects365_train.json       # raw
├── objects365_val.json         # raw (optional)
├── objects365_train_od.json    # used (coco2odvg)
├── o365v1_label_map.json       # used (coco2odvg)
├── train/                      # used images
│   └── *.jpg
├── val/
└── test/
```

---

## 2. Flickr30k (GoldG)

Follow MM-GDINO § GoldG for download + `goldg2odvg.py`, then **merge same-image** (OPUS).

```bash
python tools/dataset_converters/goldg2odvg.py \
  data/flickr30k_entities/final_flickr_separateGT_train.json

python tools/dataset_converters/merge_odvg_same_image.py --dataset flickr30k
# → flickr_30k_vg.json
```

```text
data/flickr30k_entities/
├── final_flickr_separateGT_train.json       # raw (mdetr ann)
├── final_flickr_separateGT_train_vg.json    # mid (goldg2odvg)
├── flickr_30k_vg.json                       # used (merge_odvg_same_image)
└── images/                                  # used (MM-GDINO: flickr30k_images/)
    └── *.jpg
```

---

## 3. GQA (GoldG)

Same as MM-GDINO § GoldG, then merge same-image.

```bash
python tools/dataset_converters/goldg2odvg.py \
  data/gqa/final_mixed_train_no_coco.json

python tools/dataset_converters/merge_odvg_same_image.py --dataset gqa
# → gqa_46k_vg.json
```

```text
data/gqa/
├── final_mixed_train_no_coco.json       # raw (mdetr ann)
├── final_mixed_train_no_coco_vg.json    # mid (goldg2odvg)
├── gqa_46k_vg.json                      # used (merge_odvg_same_image)
└── images/                              # used
    └── *.jpg
```

---

## 4. CrowdHuman

Convert ODGT → COCO with [`crowdhuman2coco.py`](tools/dataset_converters/crowdhuman2coco.py) (from MMDet crowdhuman2coco).

Download CrowdHuman, place raw anns and split image folders under `data/crowdhuman/`, then:

```bash
python tools/dataset_converters/crowdhuman2coco.py \
  -i data/crowdhuman \
  -o data/crowdhuman/annotations
# → annotations/crowdhuman_train.json, crowdhuman_val.json
```

The converter expects raw layout `train/Images/{ID}.jpg` and writes `file_name`
as `{ID}.jpg` (basename). OPUS `data_prefix` is `Images/`, so used images are
under `data/crowdhuman/Images/` (raw: `train/Images/`).

```text
data/crowdhuman/
├── annotation_train.odgt              # raw
├── annotation_val.odgt                # raw
├── train/
│   └── Images/                        # raw train images
│       └── *.jpg
├── val/
│   └── Images/                        # raw val images
├── annotations/
│   ├── crowdhuman_train.json          # used (crowdhuman2coco)
│   └── crowdhuman_val.json            # mid / optional
└── Images/                            # used (raw: train/Images/)
    └── *.jpg
```

---

## 5. HierText

Official `train.jsonl` is hierarchical (paragraphs/lines/words), **not** COCO. Convert word-level boxes with [`hiertext2coco.py`](tools/dataset_converters/hiertext2coco.py).

```bash
# raw from https://github.com/google-research-datasets/hiertext (gt/*.jsonl.gz)
gzip -d -c data/hiertext/annotations/train.jsonl.gz \
  > data/hiertext/annotations/train.jsonl

python tools/dataset_converters/hiertext2coco.py \
  -i data/hiertext/annotations/train.jsonl \
  -o data/hiertext/annotations/train.json
# → annotations/train.json (COCO, category=text)
```

```text
data/hiertext/
├── annotations/
│   ├── train.jsonl.gz                 # raw
│   ├── train.jsonl                    # mid (gunzip)
│   ├── validation.jsonl.gz            # raw (optional)
│   └── train.json                     # used (hiertext2coco, word-level)
└── train/                             # used images
    └── *.jpg
```

---

## 6. OpenImages v6 (OIv6)

Follow MM-GDINO § OpenImages v6 (`openimages2odvg.py`).

```bash
python tools/dataset_converters/openimages2odvg.py data/open_image/annotations
# → oidv6-train-annotations_od.json + openimages_label_map.json
```

```text
data/open_image/                             # (MM-GDINO: data/OpenImages/)
├── annotations/
│   ├── oidv6-train-annotations-bbox.csv     # raw
│   ├── class-descriptions-boxable.csv       # raw
│   ├── oidv6-train-annotations_od.json      # used (openimages2odvg)
│   └── openimages_label_map.json            # used
└── images/                                  # used (MM-GDINO: OpenImages/train/)
    └── *.jpg
```

---

## 7. V3Det

Follow MM-GDINO § V3Det (`coco2odvg.py -d v3det`).

```bash
python tools/dataset_converters/coco2odvg.py \
  data/v3det/annotations/v3det_2023_v1_train.json -d v3det
# → annotations/v3det_2023_v1_train_od.json + v3det_2023_v1_label_map.json
```

OPUS `data_prefix` is `''` (image paths relative to `data/v3det/`, typically under `images/`).

```text
data/v3det/
├── annotations/
│   ├── v3det_2023_v1_train.json           # raw
│   ├── v3det_2023_v1_train_od.json        # used (coco2odvg)
│   └── v3det_2023_v1_label_map.json       # used
└── images/                                # used
    └── */
        └── *.jpg
```

---

# Eval

## 8. COCO 2017

Follow MM-GDINO § COCO 2017. Used for training-time COCO val (`coco2017_val_dataset`) and **visual.G** prompt generation (`instances_train2017.json` + `train2017/`).

```text
data/coco/
├── annotations/
│   ├── instances_train2017.json    # raw / used (visual prompt embed)
│   └── instances_val2017.json      # used (val)
├── train2017/                      # used (visual prompt + LVIS train images)
│   └── *.jpg
└── val2017/                        # used
    └── *.jpg
```

---

## 9. LVIS 1.0

Follow MM-GDINO § LVIS. Place JSONs under **`data/coco/annotations/`** (same image roots as COCO). Eval configs: each product's `configs/*/lvis/`.

**Eval (used by `BMK=lvis` / `lvis-mv`):**

```bash
# download into data/coco/annotations/ (MM-GDINO LVIS section):
#   lvis_od_val.json
#   lvis_v1_minival_inserted_image_name.json
```

**Train OD (used by visual.G / text_visual.G embeddings for LVIS & LVIS-mini):**

Download official `lvis_v1_train.json` from [LVIS](https://www.lvisdataset.org/dataset) into `data/coco/annotations/`, then:

```bash
python tools/dataset_converters/lvis2odvg.py data/coco/annotations/lvis_v1_train.json
# → annotations/lvis_v1_train_od.json + lvis_v1_label_map.json
```

`lvis-mv` and full `lvis` share the same train embed pkl (`lvis_v1_train_od.pkl`); only the eval JSON differs.

```text
data/coco/
├── annotations/
│   ├── instances_train2017.json                     # COCO raw / visual prompt
│   ├── instances_val2017.json                       # COCO used
│   ├── lvis_v1_train.json                           # LVIS raw
│   ├── lvis_v1_train_od.json                        # used (lvis2odvg; visual prompt)
│   ├── lvis_v1_label_map.json                       # used (lvis2odvg)
│   ├── lvis_od_val.json                             # used (LVIS eval)
│   └── lvis_v1_minival_inserted_image_name.json     # used (mini-LVIS eval)
├── train2017/                                       # used (COCO + LVIS train images)
└── val2017/
```

---

## 10. ODinW

Follow MM-GDINO ODinW. Images + anns colocated under `data/odinw/`. Configs: `configs/opus_dinov3_convnext-b/odinw/`.

```text
data/odinw/
├── AerialMaritimeDrone/
│   └── ...
└── ...
```

---

## 11. Additional pretrain datasets

- Bamboo-1M (`bamboo1M_dataset`)
- BambooCLS (`BambooCLS_3m_sampled_dataset`)
- CC3M (`cc3m_dataset`)
- SA-1B (`sa_1b_3m_sampled_dataset`)

---

## Full layout

```text
OPUS/
├── data
│   # ---- train ----
│   ├── objects365v1/
│   │   ├── objects365_train.json              # raw
│   │   ├── objects365_val.json                # raw
│   │   ├── objects365_train_od.json           # used (coco2odvg)
│   │   ├── o365v1_label_map.json              # used
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── flickr30k_entities/
│   │   ├── final_flickr_separateGT_train.json      # raw
│   │   ├── final_flickr_separateGT_train_vg.json   # mid
│   │   ├── flickr_30k_vg.json                      # used
│   │   └── images/                                 # used (MM-GDINO: flickr30k_images/)
│   ├── gqa/
│   │   ├── final_mixed_train_no_coco.json          # raw
│   │   ├── final_mixed_train_no_coco_vg.json       # mid
│   │   ├── gqa_46k_vg.json                         # used
│   │   └── images/
│   ├── crowdhuman/
│   │   ├── annotation_train.odgt                   # raw
│   │   ├── annotation_val.odgt                     # raw
│   │   ├── train/Images/
│   │   ├── val/Images/
│   │   ├── annotations/
│   │   │   ├── crowdhuman_train.json               # used
│   │   │   └── crowdhuman_val.json
│   │   └── Images/                                 # used (raw: train/Images/)
│   ├── hiertext/
│   │   ├── annotations/
│   │   │   ├── train.jsonl.gz                      # raw
│   │   │   ├── train.jsonl                         # mid
│   │   │   └── train.json                          # used (hiertext2coco)
│   │   └── train/
│   ├── open_image/                                 # (MM-GDINO: OpenImages/)
│   │   ├── annotations/
│   │   │   ├── oidv6-train-annotations-bbox.csv     # raw
│   │   │   ├── class-descriptions-boxable.csv       # raw
│   │   │   ├── oidv6-train-annotations_od.json      # used
│   │   │   └── openimages_label_map.json            # used
│   │   └── images/                                  # used (MM-GDINO: OpenImages/train/)
│   ├── v3det/
│   │   ├── annotations/
│   │   │   ├── v3det_2023_v1_train.json             # raw
│   │   │   ├── v3det_2023_v1_train_od.json          # used
│   │   │   └── v3det_2023_v1_label_map.json         # used
│   │   └── images/
│   ├── bamboo-1M/
│   ├── bamboo-cls/
│   ├── cc3m/
│   ├── sa-1b/
│   # ---- eval ----
│   ├── coco/
│   │   ├── annotations/
│   │   │   ├── instances_train2017.json       # raw / used (visual prompt)
│   │   │   ├── instances_val2017.json         # used
│   │   │   ├── lvis_v1_train.json             # raw (LVIS)
│   │   │   ├── lvis_v1_train_od.json          # used (lvis2odvg; visual prompt)
│   │   │   ├── lvis_v1_label_map.json         # used (lvis2odvg)
│   │   │   ├── lvis_od_val.json               # used (LVIS eval)
│   │   │   └── lvis_v1_minival_inserted_image_name.json  # used (mini-LVIS eval)
│   │   ├── train2017/
│   │   └── val2017/
│   └── odinw/
```
