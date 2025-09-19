import os
import sys
sys.path.append('.')
sys.path.append('..')

import glob
import cv2
import numpy as np
from tqdm import tqdm

COLOR_TO_LABEL1 = {
    (0, 0, 0): "unknown",
    (174, 199, 232): "wall",
    (152, 223, 138): "floor",
    (31, 119, 180): "cabinet",
    (255, 187, 120): "bed",
    (188, 189, 34): "chair",
    (140, 86, 75): "sofa",
    (255, 152, 150): "table",
    (214, 39, 40): "door",
    (197, 176, 213): "window",
    (148, 103, 189): "bookshelf",
    (196, 156, 148): "picture",
    (23, 190, 207): "counter",
    (178, 76, 76): "blinds",
    (247, 182, 210): "desk",
    (66, 188, 102): "shelves",
    (219, 219, 141): "curtain",
    (140, 57, 197): "dresser",
    (202, 185, 52): "pillow",
    (51, 176, 203): "mirror",
    (200, 54, 131): "floor mat",
    (92, 193, 61): "clothes",
    (78, 71, 183): "ceiling",
    (172, 114, 82): "books",
    (255, 127, 14): "fridge",
    (91, 163, 138): "television",
    (153, 98, 156): "paper",
    (140, 153, 101): "towel",
    (158, 218, 229): "shower curtain",
    (100, 125, 154): "box",
    (178, 127, 135): "whiteboard",
    (120, 185, 128): "person",
    (146, 111, 194): "night stand",
    (44, 160, 44): "toilet",
    (112, 128, 144): "sink",
    (96, 207, 209): "lamp",
    (227, 119, 194): "bathtub",
    (213, 92, 176): "bag",
    (94, 106, 211): "structure",
    (82, 84, 163): "furniture",
    (100, 85, 144): "prop"
}

LABEL1_TO_LABEL2 = {
    "unknown":"background",
    "wall":"wall",
    "floor":"floor",
    "cabinet":"cabinet",
    "bed":"bed",
    "chair":"chair",
    "table":"table",
    "sofa":"sofa",
    "door":"door",
    "window":"window",
    "bookshelf":"bookcase",
    "picture":"painting, picture",
    "counter":"counter",
    "blinds":"blind",
    "desk":"desk",
    "shelves":"shelf",
    "curtain":"curtain",
    "dresser":"dresser",
    "pillow":"pillow",
    "mirror":"mirror",
    "floor mat":"rug",
    "clothes":"clothes",
    "ceiling":"ceiling",
    "books":"book",
    "fridge": "refrigerator, icebox",
    "television":"tv",
    "paper":"background",#1
    "towel":"towel",
    "shower curtain":"curtain",
    "box":"box",
    "whiteboard":"bulletin board",
    "person":"person",
    "night stand":"night stand",#2
    "toilet":"toilet, can, commode, crapper, pot, potty, stool, throne",
    "sink":"sink",
    "lamp":"chandelier",
    "bathtub":'shower',
    "bag":"bag",
    "structure":"background",#3
    "furniture":"background",#4
    "prop":"background",#5
}


from annotator.oneformer.oneformer.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES

if __name__=='__main__':
    # 2023.8.4 11:44 已知 test和train的rgb和bgr是反过来的。label是rgb的，但是test和train把label弄反了。
    # 2023.8.4 11:44 以下为处理test和train的代码。
    # 最容易的检查方法，床是橙色的，墙是蓝色的，这是structure3d的gt。
    names, labels = [], []
    root_path_train = '/mnt/nas_3dv/hdd1/datasets/datasets/Structured3D/new_sem_layout/train/bedroom/sem_layout_img'
    root_path_test = '/mnt/nas_3dv/hdd1/datasets/datasets/Structured3D/new_sem_layout/test/bedroom/sem_layout_img'

    ade_labels = [label_dict["name"] for label_dict in ADE20K_150_CATEGORIES]
    ade_colors = [label_dict["color"] for label_dict in ADE20K_150_CATEGORIES]
    img_path_train_list = sorted(glob.glob(os.path.join(root_path_train, '*.png')))
    # print(img_path_train_list)

    img_path_test_list = sorted(glob.glob(os.path.join(root_path_test, '*.png')))

    for i in tqdm(img_path_test_list):
        img_path_train = i
        # print(img_path_train)
        img_train = cv2.imread(img_path_train) # 本来读进来是bgr，但是由于fangchuan反了一次，所以是rgb
        img_train = cv2.cvtColor(img_train, cv2.COLOR_BGR2RGB)
        H, W, C = img_train.shape
        print(i)
        
        for color, cls_label in COLOR_TO_LABEL1.items():
            if cls_label in ['unknown', 'paper', 'night stand','structure', 'furniture', 'prop']:
                continue
            img_roi = img_train[img_train == color]
            coords_x, coords_y = np.where(np.all(img_train == color, axis=2))
            # print(f'len(img): {len(img_roi)}, color:{color}, cls_label: {cls_label}, coords_x: {coords_x}, coords_y: {coords_y}')
            if len(img_roi) >0:
                trans_cls_label = LABEL1_TO_LABEL2[cls_label]
                if trans_cls_label in ade_labels:
                    trans_color = ade_colors[ade_labels.index(trans_cls_label)]
                    img_train[coords_x, coords_y] = trans_color[::-1]
        cv2.imwrite(os.path.join('/mnt/nas_3dv/hdd1/datasets/datasets/Structured3D/new_sem_layout/test/bedroom/sem_layout_img_fix', i.split('/')[-1]), img_train)

    # for i in tqdm(img_path_test_list):
    #     img_path_test = i
    #     img_test = cv2.imread(img_path_test) # 本来读进来是bgr，但是由于fangchuan反了一次，所以是rgb
    #     H, W, C = img_test.shape
    #     print(i)
    #     for j in range(H):
    #         for k in range(W):
    #             color = tuple(img_test[j,k])
    #             assert color in COLOR_TO_LABEL1.keys()
    #             class_name = COLOR_TO_LABEL1[color]
    #             # print(class_name)
    #             assert class_name in LABEL1_TO_LABEL2.keys()
    #             trans_class_name = LABEL1_TO_LABEL2[class_name]
    #             index = names.index(trans_class_name)
    #             assert names[index] == trans_class_name
    #             trans_color = labels[index]#rgb
    #             img_test[j,k] = list(trans_color)[::-1]
    #     cv2.imwrite(os.path.join('/mnt/nas_3dv/hdd1/datasets/datasets/Structured3D/new_sem_layout/test/bedroom/sem_layout_img', i.split('/')[-1]), img_test)
    