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

# ---------------------1 AerialMaritimeDrone_large---------------------#
class_name = ('boat', 'car', 'dock', 'jetski', 'lift')
metainfo = dict(classes=class_name)
dataset_AerialMaritimeDrone_large = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}AerialMaritimeDrone/large/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}AerialMaritimeDrone/large/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_AerialMaritimeDrone_large = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}AerialMaritimeDrone/large/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------2 AerialMaritimeDrone_tiled---------------------#
class_name = ('boat', 'car', 'dock', 'jetski', 'lift')
metainfo = dict(classes=class_name)
dataset_AerialMaritimeDrone_tiled = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}AerialMaritimeDrone/tiled/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}AerialMaritimeDrone/tiled/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_AerialMaritimeDrone_tiled = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}AerialMaritimeDrone/tiled/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------3 AmericanSignLanguageLetters---------------------#
class_name = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
metainfo = dict(classes=class_name)
dataset_AmericanSignLanguageLetters = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_AmericanSignLanguageLetters = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------4 Aquarium---------------------#
class_name = ('fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish',
              'stingray')
metainfo = dict(classes=class_name)
dataset_Aquarium = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_Aquarium = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------5 BCCD---------------------#
class_name = ('Platelets', 'RBC', 'WBC')
metainfo = dict(classes=class_name)
dataset_BCCD = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}BCCD/BCCD.v3-raw.coco/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}BCCD/BCCD.v3-raw.coco/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_BCCD = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}BCCD/BCCD.v3-raw.coco/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------6 boggleBoards---------------------#
class_name = ('Q', 'a', 'an', 'b', 'c', 'd', 'e', 'er', 'f', 'g', 'h', 'he',
              'i', 'in', 'j', 'k', 'l', 'm', 'n', 'o', 'o ', 'p', 'q', 'qu',
              'r', 's', 't', 't\\', 'th', 'u', 'v', 'w', 'wild', 'x', 'y', 'z')
metainfo = dict(classes=class_name)
dataset_boggleBoards = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}boggleBoards/416x416AutoOrient/export/val_annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}boggleBoards/416x416AutoOrient/export/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_boggleBoards = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}boggleBoards/416x416AutoOrient/export/val_annotations_without_background.json',
    metric='bbox')

# ---------------------7 brackishUnderwater---------------------#
class_name = ('crab', 'fish', 'jellyfish', 'shrimp', 'small_fish', 'starfish')
metainfo = dict(classes=class_name)
dataset_brackishUnderwater = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}brackishUnderwater/960x540/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}brackishUnderwater/960x540/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_brackishUnderwater = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}brackishUnderwater/960x540/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------8 ChessPieces---------------------#
class_name = ('  ', 'black bishop', 'black king', 'black knight', 'black pawn',
              'black queen', 'black rook', 'white bishop', 'white king',
              'white knight', 'white pawn', 'white queen', 'white rook')
metainfo = dict(classes=class_name)
dataset_ChessPieces = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}ChessPieces/Chess Pieces.v23-raw.coco/valid/new_annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}ChessPieces/Chess Pieces.v23-raw.coco/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_ChessPieces = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}ChessPieces/Chess Pieces.v23-raw.coco/valid/new_annotations_without_background.json',
    metric='bbox')

# ---------------------9 CottontailRabbits---------------------#
class_name = ('rabbit', )
metainfo = dict(classes=class_name)
dataset_CottontailRabbits = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}CottontailRabbits/valid/new_annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}CottontailRabbits/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_CottontailRabbits = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}CottontailRabbits/valid/new_annotations_without_background.json',
    metric='bbox')

# ---------------------10 dice---------------------#
class_name = ('1', '2', '3', '4', '5', '6')
metainfo = dict(classes=class_name)
dataset_dice = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}dice/mediumColor/export/val_annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}dice/mediumColor/export/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_dice = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}dice/mediumColor/export/val_annotations_without_background.json',
    metric='bbox')

# ---------------------11 DroneControl---------------------#
class_name = ('follow', 'follow_hand', 'land', 'land_hand', 'null', 'object',
              'takeoff', 'takeoff-hand')
metainfo = dict(classes=class_name)
dataset_DroneControl = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}DroneControl/Drone Control.v3-raw.coco/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}DroneControl/Drone Control.v3-raw.coco/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_DroneControl = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}DroneControl/Drone Control.v3-raw.coco/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------12 EgoHands_generic---------------------#
class_name = ('hand', )
metainfo = dict(classes=class_name)
caption_prompt = {'hand': {'suffix': ' of a person'}}
dataset_EgoHands_generic = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}EgoHands/generic/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}EgoHands/generic/valid/'),
    pipeline=base_test_pipeline,
    caption_prompt=caption_prompt,
    test_mode=True,
    return_classes=True)
val_evaluator_EgoHands_generic = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}EgoHands/generic/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------13 EgoHands_specific---------------------#
class_name = ('myleft', 'myright', 'yourleft', 'yourright')
metainfo = dict(classes=class_name)
dataset_EgoHands_specific = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}EgoHands/specific/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}EgoHands/specific/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_EgoHands_specific = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}EgoHands/specific/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------14 HardHatWorkers---------------------#
class_name = ('head', 'helmet', 'person')
metainfo = dict(classes=class_name)
dataset_HardHatWorkers = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}HardHatWorkers/raw/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}HardHatWorkers/raw/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_HardHatWorkers = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}HardHatWorkers/raw/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------15 MaskWearing---------------------#
class_name = ('mask', 'no-mask')
metainfo = dict(classes=class_name)
dataset_MaskWearing = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}MaskWearing/raw/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}MaskWearing/raw/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_MaskWearing = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}MaskWearing/raw/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------16 MountainDewCommercial---------------------#
class_name = ('bottle', )
metainfo = dict(classes=class_name)
dataset_MountainDewCommercial = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}MountainDewCommercial/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}MountainDewCommercial/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_MountainDewCommercial = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}MountainDewCommercial/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------17 NorthAmericaMushrooms---------------------#
class_name = ('flat mushroom', 'yellow mushroom')
metainfo = dict(classes=class_name)
dataset_NorthAmericaMushrooms = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/new_annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_NorthAmericaMushrooms = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/new_annotations_without_background.json',
    metric='bbox')

# ---------------------18 openPoetryVision---------------------#
class_name = ('American Typewriter', 'Andale Mono', 'Apple Chancery', 'Arial',
              'Avenir', 'Baskerville', 'Big Caslon', 'Bradley Hand',
              'Brush Script MT', 'Chalkboard', 'Comic Sans MS', 'Copperplate',
              'Courier', 'Didot', 'Futura', 'Geneva', 'Georgia', 'Gill Sans',
              'Helvetica', 'Herculanum', 'Impact', 'Kefa', 'Lucida Grande',
              'Luminari', 'Marker Felt', 'Menlo', 'Monaco', 'Noteworthy',
              'Optima', 'PT Sans', 'PT Serif', 'Palatino', 'Papyrus',
              'Phosphate', 'Rockwell', 'SF Pro', 'SignPainter', 'Skia',
              'Snell Roundhand', 'Tahoma', 'Times New Roman', 'Trebuchet MS',
              'Verdana')
metainfo = dict(classes=class_name)
dataset_openPoetryVision = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}openPoetryVision/512x512/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}openPoetryVision/512x512/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_openPoetryVision = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}openPoetryVision/512x512/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------19 OxfordPets_by_breed---------------------#
class_name = ('cat-Abyssinian', 'cat-Bengal', 'cat-Birman', 'cat-Bombay',
              'cat-British_Shorthair', 'cat-Egyptian_Mau', 'cat-Maine_Coon',
              'cat-Persian', 'cat-Ragdoll', 'cat-Russian_Blue', 'cat-Siamese',
              'cat-Sphynx', 'dog-american_bulldog',
              'dog-american_pit_bull_terrier', 'dog-basset_hound',
              'dog-beagle', 'dog-boxer', 'dog-chihuahua',
              'dog-english_cocker_spaniel', 'dog-english_setter',
              'dog-german_shorthaired', 'dog-great_pyrenees', 'dog-havanese',
              'dog-japanese_chin', 'dog-keeshond', 'dog-leonberger',
              'dog-miniature_pinscher', 'dog-newfoundland', 'dog-pomeranian',
              'dog-pug', 'dog-saint_bernard', 'dog-samoyed',
              'dog-scottish_terrier', 'dog-shiba_inu',
              'dog-staffordshire_bull_terrier', 'dog-wheaten_terrier',
              'dog-yorkshire_terrier')
metainfo = dict(classes=class_name)
dataset_OxfordPets_by_breed = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}OxfordPets/by-breed/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}OxfordPets/by-breed/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_OxfordPets_by_breed = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}OxfordPets/by-breed/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------20 OxfordPets_by_species---------------------#
class_name = ('cat', 'dog')
metainfo = dict(classes=class_name)
dataset_OxfordPets_by_species = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}OxfordPets/by-species/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}OxfordPets/by-species/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_OxfordPets_by_species = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}OxfordPets/by-species/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------21 PKLot---------------------#
class_name = ('space-empty', 'space-occupied')
metainfo = dict(classes=class_name)
dataset_PKLot = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}PKLot/640/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}PKLot/640/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_PKLot = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}PKLot/640/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------22 Packages---------------------#
class_name = ('package', )
metainfo = dict(classes=class_name)
caption_prompt = {
    'package': {
        'prefix': 'there is a ',
        'suffix': ' on the porch'
    }
}
dataset_Packages = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}Packages/Raw/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}Packages/Raw/valid/'),
    pipeline=base_test_pipeline,
    caption_prompt=caption_prompt,
    test_mode=True,
    return_classes=True)
val_evaluator_Packages = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}Packages/Raw/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------23 PascalVOC---------------------#
class_name = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
              'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor')
metainfo = dict(classes=class_name)
dataset_PascalVOC = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}PascalVOC/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}PascalVOC/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_PascalVOC = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}PascalVOC/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------24 pistols---------------------#
class_name = ('pistol', )
metainfo = dict(classes=class_name)
dataset_pistols = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}pistols/export/val_annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}pistols/export/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_pistols = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}pistols/export/val_annotations_without_background.json',
    metric='bbox')

# ---------------------25 plantdoc---------------------#
class_name = ('Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf',
              'Bell_pepper leaf', 'Bell_pepper leaf spot', 'Blueberry leaf',
              'Cherry leaf', 'Corn Gray leaf spot', 'Corn leaf blight',
              'Corn rust leaf', 'Peach leaf', 'Potato leaf',
              'Potato leaf early blight', 'Potato leaf late blight',
              'Raspberry leaf', 'Soyabean leaf', 'Soybean leaf',
              'Squash Powdery mildew leaf', 'Strawberry leaf',
              'Tomato Early blight leaf', 'Tomato Septoria leaf spot',
              'Tomato leaf', 'Tomato leaf bacterial spot',
              'Tomato leaf late blight', 'Tomato leaf mosaic virus',
              'Tomato leaf yellow virus', 'Tomato mold leaf',
              'Tomato two spotted spider mites leaf', 'grape leaf',
              'grape leaf black rot')
metainfo = dict(classes=class_name)
dataset_plantdoc = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}plantdoc/416x416/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}plantdoc/416x416/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_plantdoc = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}plantdoc/416x416/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------26 pothole---------------------#
class_name = ('pothole', )
metainfo = dict(classes=class_name)
caption_prompt = {
    'pothole': {
        'name': 'holes',
        'prefix': 'there are some ',
        'suffix': ' on the road'
    }
}
dataset_pothole = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}pothole/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}pothole/valid/'),
    caption_prompt=caption_prompt,
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_pothole = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}pothole/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------27 Raccoon---------------------#
class_name = ('raccoon', )
metainfo = dict(classes=class_name)
dataset_Raccoon = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}Raccoon/Raccoon.v2-raw.coco/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}Raccoon/Raccoon.v2-raw.coco/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_Raccoon = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}Raccoon/Raccoon.v2-raw.coco/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------28 selfdrivingCar---------------------#
class_name = ('biker', 'car', 'pedestrian', 'trafficLight',
              'trafficLight-Green', 'trafficLight-GreenLeft',
              'trafficLight-Red', 'trafficLight-RedLeft',
              'trafficLight-Yellow', 'trafficLight-YellowLeft', 'truck')
metainfo = dict(classes=class_name)
dataset_selfdrivingCar = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}selfdrivingCar/fixedLarge/export/val_annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}selfdrivingCar/fixedLarge/export/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_selfdrivingCar = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}selfdrivingCar/fixedLarge/export/val_annotations_without_background.json',
    metric='bbox')

# ---------------------29 ShellfishOpenImages---------------------#
class_name = ('Crab', 'Lobster', 'Shrimp')
metainfo = dict(classes=class_name)
dataset_ShellfishOpenImages = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}ShellfishOpenImages/raw/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}ShellfishOpenImages/raw/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_ShellfishOpenImages = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}ShellfishOpenImages/raw/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------30 ThermalCheetah---------------------#
class_name = ('cheetah', 'human')
metainfo = dict(classes=class_name)
dataset_ThermalCheetah = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}ThermalCheetah/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}ThermalCheetah/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_ThermalCheetah = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}ThermalCheetah/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------31 thermalDogsAndPeople---------------------#
class_name = ('dog', 'person')
metainfo = dict(classes=class_name)
dataset_thermalDogsAndPeople = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}thermalDogsAndPeople/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}thermalDogsAndPeople/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_thermalDogsAndPeople = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}thermalDogsAndPeople/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------32 UnoCards---------------------#
class_name = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
              '12', '13', '14')
metainfo = dict(classes=class_name)
dataset_UnoCards = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}UnoCards/raw/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}UnoCards/raw/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_UnoCards = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}UnoCards/raw/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------33 VehiclesOpenImages---------------------#
class_name = ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')
metainfo = dict(classes=class_name)
dataset_VehiclesOpenImages = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}VehiclesOpenImages/416x416/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}VehiclesOpenImages/416x416/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_VehiclesOpenImages = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}VehiclesOpenImages/416x416/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------34 WildfireSmoke---------------------#
class_name = ('smoke', )
metainfo = dict(classes=class_name)
dataset_WildfireSmoke = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}WildfireSmoke/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}WildfireSmoke/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_WildfireSmoke = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}WildfireSmoke/valid/annotations_without_background.json',
    metric='bbox')

# ---------------------35 websiteScreenshots---------------------#
class_name = ('button', 'field', 'heading', 'iframe', 'image', 'label', 'link',
              'text')
metainfo = dict(classes=class_name)
dataset_websiteScreenshots = dict(
    type=dataset_type,
    metainfo=metainfo,
    ann_file=f'{odinw_ann_root}websiteScreenshots/valid/annotations_without_background.json',
    data_prefix=dict(img=f'{odinw_img_root}websiteScreenshots/valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_websiteScreenshots = dict(
    type='CocoMetric',
    ann_file=f'{odinw_ann_root}websiteScreenshots/valid/annotations_without_background.json',
    metric='bbox')

from configs.datasets.pipeline_builders import inject_odinw35_visual_prompt_pipeline

# --------------------- Config---------------------#

dataset_prefixes = [
    'AerialMaritimeDrone_large',
    'AerialMaritimeDrone_tiled',
    'AmericanSignLanguageLetters',
    'Aquarium',
    'BCCD',
    'boggleBoards',
    'brackishUnderwater',
    'ChessPieces',
    'CottontailRabbits',
    'dice',
    'DroneControl',
    'EgoHands_generic',
    'EgoHands_specific',
    'HardHatWorkers',
    'MaskWearing',
    'MountainDewCommercial',
    'NorthAmericaMushrooms',
    'openPoetryVision',
    'OxfordPets_by_breed',
    'OxfordPets_by_species',
    'PKLot',
    'Packages',
    'PascalVOC',
    'pistols',
    'plantdoc',
    'pothole',
    'Raccoons',
    'selfdrivingCar',
    'ShellfishOpenImages',
    'ThermalCheetah',
    'thermalDogsAndPeople',
    'UnoCards',
    'VehiclesOpenImages',
    'WildfireSmoke',
    'websiteScreenshots',
]

_odinw_raw_datasets = [
    dataset_AerialMaritimeDrone_large, dataset_AerialMaritimeDrone_tiled,
    dataset_AmericanSignLanguageLetters, dataset_Aquarium, dataset_BCCD,
    dataset_boggleBoards, dataset_brackishUnderwater, dataset_ChessPieces,
    dataset_CottontailRabbits, dataset_dice, dataset_DroneControl,
    dataset_EgoHands_generic, dataset_EgoHands_specific,
    dataset_HardHatWorkers, dataset_MaskWearing, dataset_MountainDewCommercial,
    dataset_NorthAmericaMushrooms, dataset_openPoetryVision,
    dataset_OxfordPets_by_breed, dataset_OxfordPets_by_species, dataset_PKLot,
    dataset_Packages, dataset_PascalVOC, dataset_pistols, dataset_plantdoc,
    dataset_pothole, dataset_Raccoon, dataset_selfdrivingCar,
    dataset_ShellfishOpenImages, dataset_ThermalCheetah,
    dataset_thermalDogsAndPeople, dataset_UnoCards, dataset_VehiclesOpenImages,
    dataset_WildfireSmoke, dataset_websiteScreenshots
]


datasets = inject_odinw35_visual_prompt_pipeline(
    _odinw_raw_datasets, dataset_prefixes)

metrics = [
    val_evaluator_AerialMaritimeDrone_large,
    val_evaluator_AerialMaritimeDrone_tiled,
    val_evaluator_AmericanSignLanguageLetters, val_evaluator_Aquarium,
    val_evaluator_BCCD, val_evaluator_boggleBoards,
    val_evaluator_brackishUnderwater, val_evaluator_ChessPieces,
    val_evaluator_CottontailRabbits, val_evaluator_dice,
    val_evaluator_DroneControl, val_evaluator_EgoHands_generic,
    val_evaluator_EgoHands_specific, val_evaluator_HardHatWorkers,
    val_evaluator_MaskWearing, val_evaluator_MountainDewCommercial,
    val_evaluator_NorthAmericaMushrooms, val_evaluator_openPoetryVision,
    val_evaluator_OxfordPets_by_breed, val_evaluator_OxfordPets_by_species,
    val_evaluator_PKLot, val_evaluator_Packages, val_evaluator_PascalVOC,
    val_evaluator_pistols, val_evaluator_plantdoc, val_evaluator_pothole,
    val_evaluator_Raccoon, val_evaluator_selfdrivingCar,
    val_evaluator_ShellfishOpenImages, val_evaluator_ThermalCheetah,
    val_evaluator_thermalDogsAndPeople, val_evaluator_UnoCards,
    val_evaluator_VehiclesOpenImages, val_evaluator_WildfireSmoke,
    val_evaluator_websiteScreenshots
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
