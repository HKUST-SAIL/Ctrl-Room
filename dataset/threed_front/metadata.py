#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

TDFRONT_COLOR_TO_ADEK_LABEL = {
    (0, 0, 0): "unknown",
    (120, 120, 120): "wall",
    (80, 50, 50): "floor",
    (224, 5, 255): "cabinet",
    (223, 5, 255): "children_cabinet",
    (225, 5, 255): "wine_cabinet",
    (0, 255, 112): "coffee_table",
    (221, 5, 255): "console_table",
    (222, 5, 255): "corner_side_table",
    (220, 5, 255): "dining_table",
    (219, 5, 255): "round_end_table",
    (204, 5, 255): "bed",
    (203, 5, 255): "double_bed",
    (202, 5, 255): "kids_bed",
    (201, 5, 255): "single_bed",
    (204, 70, 3): "chair",
    (8, 255, 21): "armchair",
    (9, 255, 21): "dressing_chair",
    (10, 255, 21): "chinese_chair",
    (11, 255, 21): "dining_chair",
    (12, 255, 21): "lounge_chair",
    (11, 102, 255): "sofa",
    (10, 102, 255): "chaise_longue_sofa",
    (9, 102, 255): "l_shaped_sofa",
    (8, 102, 255): "lazy_sofa",
    (7, 102, 255): "loveseat_sofa",
    (6, 102, 255): "multi_seat_sofa",
    (255, 6, 82): "table",
    (255, 6, 83): "dressing_table",
    (8, 255, 51): "door",
    (230, 230, 230): "window",
    (0, 255, 245): "bookshelf",
    (255, 6, 51): "picture",
    (235, 12, 255): "counter",
    (0, 61, 255): "blinds",
    (10, 255, 71): "desk",
    (255, 7, 71): "shelf",
    (255, 51, 8): "curtain",
    (6, 51, 255): "dresser",
    (0, 235, 255): "pillow",
    (220, 220, 220): "mirror",
    (255, 9, 92): "floor mat",
    (0, 112, 255): "clothes",
    (120, 120, 80): "ceiling",
    (255, 163, 0): "books",
    (20, 255, 0): "fridge",
    (0, 255, 194): "television",
    (153, 98, 156): "paper",
    (255, 0, 102): "towel",
    (255, 51, 7): "shower curtain",
    (0, 255, 20): "box",
    (184, 255, 0): "whiteboard",
    (150, 5, 61): "person",
    (146, 111, 194): "nightstand",
    (0, 255, 133): "toilet",
    (0, 255, 134): "stool",
    (0, 163, 255): "sink",
    (0, 31, 255): "ceiling_lamp",
    (0, 32, 255): "pendant_lamp",
    (0, 133, 255): "bathtub",
    (70, 184, 160): "bag",
    (94, 106, 211): "tv_stand",
    (7, 255, 255): "wardrobe",
    (100, 85, 144): "prop"
}

THREED_FRONT_BEDROOM_FURNITURE_MAP = {
    "desk": "desk",
    "nightstand": "nightstand",
    "king-size bed": "double_bed",
    "single bed": "single_bed",
    "kids bed": "kids_bed",
    "ceiling lamp": "ceiling_lamp",
    "pendant lamp": "pendant_lamp",
    "bookcase/jewelry armoire": "bookshelf",
    "tv stand": "tv_stand",
    "wardrobe": "wardrobe",
    "lounge chair/cafe chair/office chair": "chair",
    "dining chair": "chair",
    "classic chinese chair": "chair",
    "armchair": "armchair",
    "dressing table": "dressing_table",
    "dressing chair": "dressing_chair",
    "corner/side table": "table",
    "dining table": "table",
    "round end table": "table",
    "drawer chest/corner cabinet": "cabinet",
    "sideboard/side cabinet/console table": "cabinet",
    "children cabinet": "children_cabinet",
    "shelf": "shelf",
    "footstool/sofastool/bed end stool/stool": "stool",
    "coffee table": "coffee_table",
    "loveseat sofa": "sofa",
    "three-seat/multi-seat sofa": "sofa",
    "l-shaped sofa": "sofa",
    "lazy sofa": "sofa",
    "chaise longue sofa": "sofa",
}

THREED_FRONT_BEDROOM_FURNITURE = [
    'armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chair', 'children_cabinet', 'coffee_table', 'desk', 'door',
    'double_bed', 'dressing_chair', 'dressing_table', 'kids_bed', 'nightstand', 'pendant_lamp', 'shelf', 'single_bed',
    'sofa', 'stool', 'table', 'tv_stand', 'wall', 'wardrobe', 'window', 'empty'
]
THREED_FRONT_BEDROOM_WO_DOOR_WINDOW_WALL_FURNITURE = [
    "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chair", "children_cabinet", "coffee_table", "desk",
    "double_bed", "dressing_chair", "dressing_table", "kids_bed", "nightstand", "pendant_lamp", "shelf", "single_bed",
    "sofa", "stool", "table", "tv_stand", "wardrobe", "end"
]

THREED_FRONT_BEDROOM_FURNITURE_CNTS = {
    "wall": 11072,
    "nightstand": 2931,
    "door": 2644,
    "double_bed": 1910,
    "window": 1798,
    "wardrobe": 1749,
    "pendant_lamp": 1379,
    "ceiling_lamp": 731,
    "tv_stand": 333,
    "chair": 271,
    "dressing_table": 239,
    "single_bed": 225,
    "table": 218,
    "cabinet": 214,
    "desk": 201,
    "stool": 127,
    "kids_bed": 93,
    "bookshelf": 90,
    "children_cabinet": 88,
    "shelf": 86,
    "dressing_chair": 64,
    "armchair": 40,
    "sofa": 21,
    "coffee_table": 17
}

THREED_FRONT_BEDROOM_MIN_FURNITURE_NUM = 3
THREED_FRONT_BEDROOM_MAX_FURNITURE_NUM = 13
THREED_FRONT_BEDROOM_MIN_WALL_NUM = 4
THREED_FRONT_BEDROOM_MAX_WALL_NUM = 10

THREED_FRONT_LIBRARY_FURNITURE_MAP = {
    "bookcase/jewelry armoire": "bookshelf",
    "desk": "desk",
    "pendant lamp": "pendant_lamp",
    "ceiling lamp": "ceiling_lamp",
    "lounge chair/cafe chair/office chair": "lounge_chair",
    "dining chair": "dining_chair",
    "dining table": "dining_table",
    "corner/side table": "corner_side_table",
    "classic chinese chair": "chinese_chair",
    "armchair": "armchair",
    "shelf": "shelf",
    "sideboard/side cabinet/console table": "console_table",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool": "stool",
    "round end table": "round_end_table",
    "loveseat sofa": "loveseat_sofa",
    "drawer chest/corner cabinet": "cabinet",
    "wardrobe": "wardrobe",
    "three-seat/multi-seat sofa": "multi_seat_sofa",
    "wine cabinet": "wine_cabinet",
    "coffee table": "coffee_table",
    "lazy sofa": "lazy_sofa",
    "children cabinet": "cabinet",
    "chaise longue sofa": "chaise_longue_sofa",
    "l-shaped sofa": "l_shaped_sofa",
    "dressing table": "dressing_table",
    "dressing chair": "dressing_chair",
}

THREED_FRONT_LIBRARY_FURNITURE = [
    'armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chair', 'children_cabinet', 'coffee_table', 'desk', 'door',
    'double_bed', 'dressing_chair', 'dressing_table', 'kids_bed', 'nightstand', 'pendant_lamp', 'shelf', 'single_bed',
    'sofa', 'stool', 'table', 'tv_stand', 'wall', 'wardrobe', 'window', 'empty'
]
THREED_FRONT_LIBRARY_FURNITURE_CNTS = {
    "wall": 11072,
    "nightstand": 2931,
    "door": 2644,
    "double_bed": 1910,
    "window": 1798,
    "wardrobe": 1749,
    "pendant_lamp": 1379,
    "ceiling_lamp": 731,
    "tv_stand": 333,
    "chair": 271,
    "dressing_table": 239,
    "single_bed": 225,
    "table": 218,
    "cabinet": 214,
    "desk": 201,
    "stool": 127,
    "kids_bed": 93,
    "bookshelf": 90,
    "children_cabinet": 88,
    "shelf": 86,
    "dressing_chair": 64,
    "armchair": 40,
    "sofa": 21,
    "coffee_table": 17
}

THREED_FRONT_LIBRARY_MIN_FURNITURE_NUM = 3
THREED_FRONT_LIBRARY_MAX_FURNITURE_NUM = 10
THREED_FRONT_LIBRARY_MIN_WALL_NUM = 4
THREED_FRONT_LIBRARY_MAX_WALL_NUM = 10

THREED_FRONT_LIVINGROOM_FURNITURE_MAP = {
    "bookcase/jewelry armoire": "bookshelf",
    "desk": "desk",
    "pendant lamp": "pendant_lamp",
    "ceiling lamp": "ceiling_lamp",
    "lounge chair/cafe chair/office chair": "lounge_chair",
    "dining chair": "dining_chair",
    "dining table": "dining_table",
    "corner/side table": "corner_side_table",
    "classic chinese chair": "chinese_chair",
    "armchair": "armchair",
    "shelf": "shelf",
    "sideboard/side cabinet/console table": "console_table",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool": "stool",
    "round end table": "round_end_table",
    "loveseat sofa": "loveseat_sofa",
    "drawer chest/corner cabinet": "cabinet",
    "wardrobe": "wardrobe",
    "three-seat/multi-seat sofa": "multi_seat_sofa",
    "wine cabinet": "wine_cabinet",
    "coffee table": "coffee_table",
    "lazy sofa": "lazy_sofa",
    "children cabinet": "cabinet",
    "chaise longue sofa": "chaise_longue_sofa",
    "l-shaped sofa": "l_shaped_sofa",
    "tv stand": "tv_stand"
}
THREED_FRONT_LIVINGROOM_FURNITURE = [
    "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table",
    "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "door", "l_shaped_sofa", "lazy_sofa",
    "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand",
    "wardrobe", "window", "wine_cabinet", "wall", 'empty'
]
THREED_FRONT_LIVINGROOM_WO_DOOR_WINDOW_WALL_FURNITURE = [
    "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table",
    "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa",
    "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand",
    "wardrobe", "wine_cabinet", "end"
]

THREED_FRONT_LIVINGROOM_FURNITURE_CNTS = {
    "wall": 4292,
    "dining_chair": 1698,
    "door": 1209,
    "pendant_lamp": 861,
    "coffee_table": 565,
    "corner_side_table": 476,
    "dining_table": 457,
    "tv_stand": 397,
    "multi_seat_sofa": 342,
    "armchair": 319,
    "console_table": 234,
    "lounge_chair": 195,
    "window": 183,
    "stool": 163,
    "cabinet": 152,
    "bookshelf": 144,
    "loveseat_sofa": 140,
    "ceiling_lamp": 112,
    "wine_cabinet": 67,
    "l_shaped_sofa": 61,
    "round_end_table": 34,
    "shelf": 25,
    "chinese_chair": 18,
    "wardrobe": 17,
    "chaise_longue_sofa": 8,
    "desk": 6,
    "lazy_sofa": 5
}

THREED_FRONT_LIVINGROOM_MIN_FURNITURE_NUM = 3
THREED_FRONT_LIVINGROOM_MAX_FURNITURE_NUM = 21
THREED_FRONT_LIVINGROOM_MIN_WALL_NUM = 4
THREED_FRONT_LIVINGROOM_MAX_WALL_NUM = 20

THREED_FRONT_DININGROOM_FURNITURE = [
    "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table",
    "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "door", "l_shaped_sofa", "lazy_sofa",
    "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand",
    "wardrobe", "window", "wine_cabinet", "wall", 'empty'
]
THREED_FRONT_DININGROOM_FURNITURE_WO_DOOR_WINDOW_WALL = [
    "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table",
    "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa",
    "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand",
    "wardrobe", "wine_cabinet", "end"
]
THREED_FRONT_DININGROOM_FURNITURE_CNTS = {
    "wall": 4858,
    "dining_chair": 2487,
    "door": 1262,
    "pendant_lamp": 931,
    "dining_table": 631,
    "coffee_table": 424,
    "corner_side_table": 373,
    "tv_stand": 309,
    "console_table": 259,
    "armchair": 254,
    "multi_seat_sofa": 249,
    "lounge_chair": 203,
    "window": 194,
    "cabinet": 158,
    "stool": 139,
    "bookshelf": 129,
    "ceiling_lamp": 111,
    "wine_cabinet": 107,
    "loveseat_sofa": 101,
    "l_shaped_sofa": 51,
    "shelf": 24,
    "round_end_table": 21,
    "wardrobe": 19,
    "chinese_chair": 17,
    "desk": 15,
    "chaise_longue_sofa": 7,
    "lazy_sofa": 5
}
