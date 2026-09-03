_base_ = '../opus_dinov3_convnext-b_pretrain_o365.py'
root_data_dir = _base_.root_data_dir

env_cfg = dict(
    dist_cfg=dict(
        backend='nccl',
        timeout=7200,
    ),
)

model = dict(
    test_cfg=dict(
        max_per_img=300,
    ),
)

dataset_type = 'CocoDataset'
# Same layout as configs/mm_grounding_dino/odinw (images + anns colocated)
data_root = f'{root_data_dir}/odinw/'
odinw_img_root = data_root
odinw_ann_root = data_root

base_test_pipeline = _base_.coco_od_test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')

# ---------------------1 AerialMaritimeDrone---------------------#
class_name = ('boat', 'car', 'dock', 'jetski', 'lift')
metainfo = dict(classes=class_name)
dataset_AerialMaritimeDrone = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}AerialMaritimeDrone/large/valid/'
    'annotations_without_background.json',
    data_prefix=dict(
        img=f'{odinw_img_root}AerialMaritimeDrone/large/valid/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator_AerialMaritimeDrone = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}AerialMaritimeDrone/large/valid/'
    'annotations_without_background.json',
    metric='bbox')

# ---------------------2 Aquarium---------------------#
class_name = ('fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish',
              'stingray')
metainfo = dict(classes=class_name)

caption_prompt = None
# caption_prompt = {
#     'penguin': {
#         'suffix': ', which is black and white'
#     },
#     'puffin': {
#         'suffix': ' with orange beaks'
#     },
#     'stingray': {
#         'suffix': ' which is flat and round'
#     },
# }
dataset_Aquarium = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}Aquarium/Aquarium Combined.v2-raw-1024.coco/'
    'valid/annotations_without_background.json',
    data_prefix=dict(
        img=f'{odinw_img_root}Aquarium/Aquarium Combined.v2-raw-1024.coco/'
        'valid/'),
    pipeline=base_test_pipeline,
    caption_prompt=caption_prompt,
    test_mode=True,
    return_classes=True)
val_evaluator_Aquarium = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}Aquarium/Aquarium Combined.v2-raw-1024.coco/'
    'valid/annotations_without_background.json',
    metric='bbox')

# ---------------------3 CottontailRabbits---------------------#
class_name = ('Cottontail-Rabbit', )
metainfo = dict(classes=class_name)

# caption_prompt = None
caption_prompt = {'Cottontail-Rabbit': {'name': 'rabbit'}}

dataset_CottontailRabbits = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}CottontailRabbits/valid/'
    'annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}CottontailRabbits/valid/'),
    pipeline=base_test_pipeline,
    caption_prompt=caption_prompt,
    test_mode=True,
    return_classes=True)
val_evaluator_CottontailRabbits = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}CottontailRabbits/valid/'
    'annotations_without_background.json',
    metric='bbox')

# ---------------------4 EgoHands---------------------#
class_name = ('hand', )
metainfo = dict(classes=class_name)

# caption_prompt = None
caption_prompt = {'hand': {'suffix': ' of a person'}}

dataset_EgoHands = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}EgoHands/generic/valid/'
    'annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}EgoHands/generic/valid/'),
    pipeline=base_test_pipeline,
    caption_prompt=caption_prompt,
    test_mode=True,
    return_classes=True)
val_evaluator_EgoHands = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}EgoHands/generic/valid/'
    'annotations_without_background.json',
    metric='bbox')

# ---------------------5 NorthAmericaMushrooms---------------------#
class_name = ('CoW', 'chanterelle')
metainfo = dict(classes=class_name)

# caption_prompt = None
caption_prompt = {
    'CoW': {
        'name': 'flat mushroom'
    },
    'chanterelle': {
        'name': 'yellow mushroom'
    }
}

dataset_NorthAmericaMushrooms = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}NorthAmericaMushrooms/'
    'North American Mushrooms.v1-416x416.coco/valid/'
    'annotations_without_background.json',
    data_prefix=dict(
        img=f'{odinw_img_root}NorthAmericaMushrooms/'
        'North American Mushrooms.v1-416x416.coco/valid/'),
    pipeline=base_test_pipeline,
    caption_prompt=caption_prompt,
    test_mode=True,
    return_classes=True)
val_evaluator_NorthAmericaMushrooms = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}NorthAmericaMushrooms/'
    'North American Mushrooms.v1-416x416.coco/valid/'
    'annotations_without_background.json',
    metric='bbox')

# ---------------------6 Packages---------------------#
class_name = ('package', )
metainfo = dict(classes=class_name)

# caption_prompt = None
caption_prompt = {
    'package': {
        'prefix': 'there is a ',
        'suffix': ' on the porch'
    }
}

dataset_Packages = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}Packages/Raw/valid/'
    'annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}Packages/Raw/valid/'),
    pipeline=base_test_pipeline,
    caption_prompt=caption_prompt,
    test_mode=True,
    return_classes=True)
val_evaluator_Packages = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}Packages/Raw/valid/'
    'annotations_without_background.json',
    metric='bbox')

# ---------------------7 PascalVOC---------------------#
class_name = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
              'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor')
metainfo = dict(classes=class_name)
dataset_PascalVOC = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}PascalVOC/valid/'
    'annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}PascalVOC/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_PascalVOC = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}PascalVOC/valid/'
    'annotations_without_background.json',
    metric='bbox')

# ---------------------8 pistols---------------------#
class_name = ('pistol', )
metainfo = dict(classes=class_name)
dataset_pistols = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}pistols/export/'
    'val_annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}pistols/export/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_pistols = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}pistols/export/'
    'val_annotations_without_background.json',
    metric='bbox')

# ---------------------9 pothole---------------------#
class_name = ('pothole', )
metainfo = dict(classes=class_name)

# caption_prompt = None
caption_prompt = {
    'pothole': {
        'prefix': 'there are some ',
        'name': 'holes',
        'suffix': ' on the road'
    }
}

dataset_pothole = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}pothole/valid/'
    'annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}pothole/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_pothole = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}pothole/valid/'
    'annotations_without_background.json',
    metric='bbox')

# ---------------------10 Raccoon---------------------#
class_name = ('raccoon', )
metainfo = dict(classes=class_name)
dataset_Raccoon = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}Raccoon/Raccoon.v2-raw.coco/valid/'
    'annotations_without_background.json',
    data_prefix=dict(
        img=f'{odinw_img_root}Raccoon/Raccoon.v2-raw.coco/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_Raccoon = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}Raccoon/Raccoon.v2-raw.coco/valid/'
    'annotations_without_background.json',
    metric='bbox')

# ---------------------11 ShellfishOpenImages---------------------#
class_name = ('Crab', 'Lobster', 'Shrimp')
metainfo = dict(classes=class_name)
dataset_ShellfishOpenImages = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}ShellfishOpenImages/raw/valid/'
    'annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}ShellfishOpenImages/raw/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_ShellfishOpenImages = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}ShellfishOpenImages/raw/valid/'
    'annotations_without_background.json',
    metric='bbox')

# ---------------------12 thermalDogsAndPeople---------------------#
class_name = ('dog', 'person')
metainfo = dict(classes=class_name)
dataset_thermalDogsAndPeople = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}thermalDogsAndPeople/valid/'
    'annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}thermalDogsAndPeople/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_thermalDogsAndPeople = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}thermalDogsAndPeople/valid/'
    'annotations_without_background.json',
    metric='bbox')

# ---------------------13 VehiclesOpenImages---------------------#
class_name = ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')
metainfo = dict(classes=class_name)
dataset_VehiclesOpenImages = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}VehiclesOpenImages/416x416/valid/'
    'annotations_without_background.json',
    data_prefix=dict(
        img=f'{odinw_img_root}VehiclesOpenImages/416x416/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_VehiclesOpenImages = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}VehiclesOpenImages/416x416/valid/'
    'annotations_without_background.json',
    metric='bbox')

# --------------------- Config---------------------#
dataset_prefixes = [
    'AerialMaritimeDrone', 'Aquarium', 'CottontailRabbits', 'EgoHands',
    'NorthAmericaMushrooms', 'Packages', 'PascalVOC', 'pistols', 'pothole',
    'Raccoon', 'ShellfishOpenImages', 'thermalDogsAndPeople',
    'VehiclesOpenImages'
]
datasets = [
    dataset_AerialMaritimeDrone, dataset_Aquarium, dataset_CottontailRabbits,
    dataset_EgoHands, dataset_NorthAmericaMushrooms, dataset_Packages,
    dataset_PascalVOC, dataset_pistols, dataset_pothole, dataset_Raccoon,
    dataset_ShellfishOpenImages, dataset_thermalDogsAndPeople,
    dataset_VehiclesOpenImages
]
metrics = [
    val_evaluator_AerialMaritimeDrone, val_evaluator_Aquarium,
    val_evaluator_CottontailRabbits, val_evaluator_EgoHands,
    val_evaluator_NorthAmericaMushrooms, val_evaluator_Packages,
    val_evaluator_PascalVOC, val_evaluator_pistols, val_evaluator_pothole,
    val_evaluator_Raccoon, val_evaluator_ShellfishOpenImages,
    val_evaluator_thermalDogsAndPeople, val_evaluator_VehiclesOpenImages
]

# -------------------------------------------------#
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator
