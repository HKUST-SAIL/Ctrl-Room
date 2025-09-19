import os
import sys
import numpy as np
from PIL import Image
from shapely.geometry import LineString
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.signal import correlate2d
from scipy.ndimage import shift, map_coordinates
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import Polygon

from typing import Any, List, Dict, Tuple
# import imghdr
import json

import trimesh

import torch.utils.data as data

from .threed_front.metadata import THREED_FRONT_BEDROOM_FURNITURE, THREED_FRONT_BEDROOM_FURNITURE_CNTS, \
    THREED_FRONT_LIVINGROOM_FURNITURE, THREED_FRONT_DININGROOM_FURNITURE, THREED_FRONT_LIVINGROOM_FURNITURE_CNTS, \
    THREED_FRONT_BEDROOM_MIN_FURNITURE_NUM, THREED_FRONT_BEDROOM_MAX_FURNITURE_NUM, \
    THREED_FRONT_LIVINGROOM_MIN_FURNITURE_NUM, THREED_FRONT_LIVINGROOM_MAX_FURNITURE_NUM, \
    THREED_FRONT_BEDROOM_MAX_WALL_NUM, THREED_FRONT_LIVINGROOM_MAX_WALL_NUM
from .text_utils import complete_stop_in_sentence
# room types
ROOM_TYPE_DICT = {
    'living room': 0,
    'kitchen': 1,
    'bedroom': 2,
    'bathroom': 3,
    'balcony': 4,
    'corridor': 5,
    'dining room': 6,
    'study': 7,
    'studio': 8,
    'store room': 9,
    'garden': 10,
    'laundry room': 11,
    'office': 12,
    'basement': 13,
    'garage': 14,
    'undefined': 15
}


# ROOM_CLASS_LST = [10, 8, 3, 2, 0, 4, 5, 14, 13, 12, 7, 9, 11, 1, 6, 15]
def get_room_type(room_type_v: int) -> str:
    for k, v in ROOM_TYPE_DICT.items():
        if v == room_type_v:
            return k
    return 'undefined'


def ClassLabelsEncode(room_type: int, obj_bbox_label: str) -> np.array:
    """Implement the encoding for the class labels."""
    # Make a local copy of the class labels
    classes = None
    if room_type == ROOM_TYPE_DICT['bedroom']:
        classes = THREED_FRONT_BEDROOM_FURNITURE
    elif room_type == ROOM_TYPE_DICT['living room']:
        classes = THREED_FRONT_LIVINGROOM_FURNITURE
    elif room_type == ROOM_TYPE_DICT['dining room']:
        classes = THREED_FRONT_DININGROOM_FURNITURE

    def one_hot_label(all_labels, current_label):
        return np.eye(len(all_labels))[all_labels.index(current_label)]

    C = len(classes)  # number of classes
    class_label = np.zeros(C, dtype=np.float32)
    class_label = one_hot_label(classes, obj_bbox_label)
    return class_label


def TranslationEncode(obj_bbox_centroid: np.array) -> np.array:
    """Implement the encoding for the object centroid."""
    # Make a local copy of the class labels
    box_centroid = obj_bbox_centroid
    return box_centroid


def SizeEncode(obj_bbox_size: np.array) -> np.array:
    """Implement the encoding for the object size."""
    # Make a local copy of the class labels
    box_size = obj_bbox_size
    return box_size


def RotationEncode(obj_bbox_angle: np.array) -> np.array:
    """Implement the encoding for the object rotation."""
    # Make a local copy of the class labels
    box_a_angle_rad = obj_bbox_angle
    return box_a_angle_rad


def NormalEncode(obj_bbox_normal: np.array) -> np.array:
    """Implement the encoding for the object normal."""
    # Make a local copy of the class labels
    box_normal = obj_bbox_normal
    return box_normal


def ordered_bboxes_with_class_frequencies(room_type: int, object_bbox_lst: List[Dict]) -> List[Dict]:
    if room_type == ROOM_TYPE_DICT['bedroom']:
        class_freq_dict = THREED_FRONT_BEDROOM_FURNITURE_CNTS
    elif room_type == ROOM_TYPE_DICT['living room']:
        class_freq_dict = THREED_FRONT_LIVINGROOM_FURNITURE_CNTS
    elif room_type == ROOM_TYPE_DICT['dining room']:
        class_freq_dict = THREED_FRONT_LIVINGROOM_FURNITURE_CNTS

    bbox_size_lst = np.array([np.array(bbox['size']) for bbox in object_bbox_lst])
    # print(f'bbox_size_lst: {bbox_size_lst}')
    class_freqs_lst = np.array([[class_freq_dict[bbox['class']]] for bbox in object_bbox_lst])
    # print('class_labels_lst: {}, class_freqs_lst: {}'.format([bbox['class'] for bbox in object_bbox_lst], class_freqs_lst))
    # first sort by class frequency, then by size
    ordering = np.lexsort(np.hstack([bbox_size_lst, class_freqs_lst]).T)
    # print(f'sort by {np.hstack([bbox_size_lst, class_freqs_lst]).T}')
    # print(f'ordering: {ordering}')

    ordered_bboxes = [object_bbox_lst[i] for i in ordering[::-1]]
    return ordered_bboxes


def padding_and_reshape_object_bbox(room_type: int, object_bbox_lst: np.array, bbox_dim: int) -> List:
    """Implement the padding for the object bounding boxes."""
    L = len(object_bbox_lst)
    max_len = L
    if room_type == ROOM_TYPE_DICT['bedroom']:
        max_len = THREED_FRONT_BEDROOM_MAX_FURNITURE_NUM
        class_num = len(THREED_FRONT_BEDROOM_FURNITURE)
    elif room_type == ROOM_TYPE_DICT['living room']:
        max_len = THREED_FRONT_LIVINGROOM_MAX_FURNITURE_NUM
        class_num = len(THREED_FRONT_LIVINGROOM_FURNITURE)
    elif room_type == ROOM_TYPE_DICT['dining room']:
        max_len = THREED_FRONT_LIVINGROOM_MAX_FURNITURE_NUM
        class_num = len(THREED_FRONT_DININGROOM_FURNITURE)

    # Pad the end label in the end of each sequence, and convert the class labels to -1, 1
    if L < max_len:
        empty_label = np.eye(class_num)[-1]
        padding = np.concatenate([empty_label, np.zeros(bbox_dim - class_num, dtype=np.float32)], axis=0)
        object_bbox_lst = np.vstack([object_bbox_lst, np.tile(padding, [max_len - L, 1])])
    elif L >= max_len:
        object_bbox_lst = object_bbox_lst[:max_len]

    ret_lst = object_bbox_lst
    return ret_lst


def padding_and_reshape_wall_bbox(room_type: int, wall_bbox_lst: np.array, bbox_dim: int) -> List:
    """ Implement the padding for the quad wall boxes.
    Args:
        room_type (int): The room type.
        wall_bbox_lst (np.array): The quadwall bounding box list.
        bbox_dim (int): The dimension of the quadwallbounding box.
    Returns:
        _type_: _description_
    """
    L = len(wall_bbox_lst)
    if room_type == ROOM_TYPE_DICT['bedroom']:
        class_num = len(THREED_FRONT_BEDROOM_FURNITURE)
        max_len = THREED_FRONT_BEDROOM_MAX_WALL_NUM
    elif room_type == ROOM_TYPE_DICT['living room']:
        class_num = len(THREED_FRONT_LIVINGROOM_FURNITURE)
        max_len = THREED_FRONT_LIVINGROOM_MAX_WALL_NUM
    elif room_type == ROOM_TYPE_DICT['dining room']:
        class_num = len(THREED_FRONT_DININGROOM_FURNITURE)
        max_len = THREED_FRONT_LIVINGROOM_MAX_WALL_NUM

    assert L <= max_len, f'The length of the wall bbox list should be less than {max_len}.'

    # Pad the end label in the end of each sequence, and convert the class labels to -1, 1
    empty_label = np.eye(class_num)[-1]
    padding = np.concatenate([empty_label, np.zeros(bbox_dim - class_num, dtype=np.float32)], axis=0)
    wall_bbox_lst = np.vstack([wall_bbox_lst, np.tile(padding, [max_len - L, 1])])

    # print(f'wall_bbox_lst: {wall_bbox_lst}')
    return wall_bbox_lst


class ThreedFrontDataset(data.Dataset):
    '''
    dataset for ThreeD-Front
    '''

    def __init__(
            self,
            root_dir,
            room_type='bedroom',
            is_train=True,
            is_test=False,
            max_text_sentences=4,  #  max number of text_prompt sentences
            shard=0,  #  support parallel training
            num_shards=1):
        self.img_dir = os.path.join(root_dir, 'img')

        # quad walls folder
        self.quad_wall_dir = os.path.join(root_dir, 'quad_walls')
        # room centroid
        self.cam_pos_dir = os.path.join(root_dir, 'cam_pose')
        # object bbox folder
        self.bbox_3d_dir = os.path.join(root_dir, 'bbox_3d')
        # text descritpion folder
        self.text_desc_dir = os.path.join(root_dir, 'text_desc')
        self.text_emb_dir = os.path.join(root_dir, 'text_desc_emb')

        # total image file names and text file names
        self.img_fnames = sorted(
            [fname for fname in os.listdir(self.img_dir) if fname.endswith('.jpg') or fname.endswith('.png')])
        self.txt_fnames = ['%s.txt' % fname[:-4] for fname in self.img_fnames]
        self.json_fnames = ['%s.json' % fname[:-4] for fname in self.img_fnames]
        self.npy_fnames = ['%s.npy' % fname[:-4] for fname in self.img_fnames]
        #  image file names and text file names on local_rank machine
        self.local_img_fnames = self.img_fnames[shard::num_shards]
        self.local_txt_fnames = self.txt_fnames[shard::num_shards]
        self.local_json_fnames = self.json_fnames[shard::num_shards]
        self.local_npy_fnames = self.npy_fnames[shard::num_shards]

        self.max_text_sentences = max_text_sentences

        self._check_dataset()

        self.local_img_lst = []
        self.local_text_lst = []
        self.local_text_emb_lst = []
        self.local_room_type_lst = []
        self.local_obj_bbox_lst = []
        self.local_wall_bbox_lst = []

        self.room_type = room_type
        self.is_train = is_train
        self.is_test = is_test

        self._preload_()

    def _check_dataset(self):
        for fname in self.json_fnames:
            assert os.path.isfile(os.path.join(self.bbox_3d_dir,
                                               fname)), '%s not found' % os.path.join(self.bbox_3d_dir, fname)

    def __len__(self):
        return len(self.local_img_fnames)

    def _preload_(self):
        """pre-load all data
        """
        for idx in range(len(self)):
            # self.local_img_lst.append(self._load_image(idx))

            # room_type_filepath = os.path.join(self.room_type_dir, self.local_txt_fnames[idx])
            # with open(room_type_filepath) as f:
            #     room_type = f.readline().strip()
            #     assert room_type in ROOM_TYPE_DICT.keys(), room_type_filepath
            #     room_type = ROOM_TYPE_DICT[room_type]
            room_type = ROOM_TYPE_DICT[self.room_type]
            self.local_room_type_lst.append(room_type)

            self.local_text_lst.append(self._load_text(idx))

            # read text embedding file
            self.local_text_emb_lst.append(self._load_text_embedding(idx))

            # read object bbox file
            self.local_obj_bbox_lst.append(self._load_object_bbox(idx, room_type=room_type))
            self.local_wall_bbox_lst.append(self._load_wall_bbox(idx, room_type=room_type))

    def _load_text(self, idx: int) -> str:
        text_desc_lst = []
        text_desc_filepath = os.path.join(self.text_desc_dir, self.local_txt_fnames[idx])
        with open(text_desc_filepath) as f:
            text_desc = f.readline()
            text_desc_lst = text_desc.strip().split('. ')
            text_desc_lst = [complete_stop_in_sentence(sen) for sen in text_desc_lst if len(sen)]
        text_desc_len = len(text_desc_lst)
        if text_desc_len:
            if text_desc_len > self.max_text_sentences:
                text_desc_lst = text_desc_lst[:self.max_text_sentences]
            text_prompt = ''.join(text_desc_lst)
        return text_prompt if text_desc_len else 'The room is empty.'

    def _load_text_embedding(self, idx: int) -> np.ndarray:
        # read text embedding file
        text_emb_filepath = os.path.join(self.text_emb_dir, self.local_npy_fnames[idx])
        text_emb = np.load(text_emb_filepath).astype(np.float32)
        return np.squeeze(text_emb, axis=0)

    def _load_wall_bbox(self, idx: int, room_type: int = 2) -> np.ndarray:
        # read wall bbox
        wall_bbox_filepath = os.path.join(self.quad_wall_dir, self.local_json_fnames[idx])
        wall_bbox_lst = []
        with open(wall_bbox_filepath, 'r') as f:
            wall_bbox_dicts = json.load(f)
            wall_bbox_dicts = wall_bbox_dicts['walls']
        for wall_bbox in wall_bbox_dicts:
            wall_class_label = wall_bbox['class'].lower()
            wall_class = ClassLabelsEncode(room_type=room_type, obj_bbox_label=wall_class_label)
            wall_centroid = np.array(wall_bbox['center'], np.float32)
            wall_centroid = TranslationEncode(wall_centroid)
            wall_size = np.array([wall_bbox['width'], 0.01, wall_bbox['height']], np.float32)
            wall_size = SizeEncode(wall_size)
            wall_angle = np.array(wall_bbox['angles'], np.float32)
            wall_angle = RotationEncode(wall_angle)
            wall_property_encode = np.concatenate([wall_class, wall_centroid, wall_size, wall_angle], axis=-1)
            # print(f'wall_property_encode: {wall_property_encode}')
            wall_property_encode_dim = wall_property_encode.shape[-1]
            wall_bbox_lst.append(wall_property_encode)
        wall_bbox_lst = padding_and_reshape_wall_bbox(room_type=room_type,
                                                      wall_bbox_lst=np.array(wall_bbox_lst),
                                                      bbox_dim=wall_property_encode_dim)
        return wall_bbox_lst

    def _load_object_bbox(self, idx: int, room_type: int = 2) -> np.ndarray:
        # read object bbox file
        object_bbox_filepath = os.path.join(self.bbox_3d_dir, self.local_json_fnames[idx])
        object_bbox_lst = []
        with open(object_bbox_filepath) as f:
            object_bbox_dicts = json.load(f)
            object_bbox_dicts = object_bbox_dicts['objects']
        # sort object bbox by class frequency and bbox size
        object_bbox_dicts = ordered_bboxes_with_class_frequencies(room_type=room_type,
                                                                  object_bbox_lst=object_bbox_dicts)

        for obj_bbox in object_bbox_dicts:
            bbox_class_label = obj_bbox['class'].lower()
            bbox_class = ClassLabelsEncode(room_type=room_type, obj_bbox_label=bbox_class_label)
            bbox_centroid = np.array(obj_bbox['center'], np.float32)
            bbox_centroid = TranslationEncode(bbox_centroid)
            bbox_size = np.array(obj_bbox['size'], np.float32)
            bbox_size = SizeEncode(bbox_size)
            # only use Z angle
            bbox_angle = np.array(obj_bbox['angles'], np.float32)
            bbox_angle = RotationEncode(bbox_angle)
            bbox_property_encode = np.concatenate([bbox_class, bbox_centroid, bbox_size, bbox_angle], axis=-1)
            # print(f'bbox_property_encode: {bbox_property_encode}')
            bbox_property_encode_dim = bbox_property_encode.shape[-1]
            object_bbox_lst.append(bbox_property_encode)
        object_bbox_lst = padding_and_reshape_object_bbox(room_type=room_type,
                                                          object_bbox_lst=np.array(object_bbox_lst),
                                                          bbox_dim=bbox_property_encode_dim)
        return object_bbox_lst

    def __getitem__(self, idx: int) -> List:
        """retrieve scene data

        Args:
            idx (int): panorama/room idx

        Returns:
            List: 
        """

        # read camera position file
        # cam_pos_lst = []
        # cam_pos_filepath = os.path.join(self.cam_pos_dir, self.local_txt_fnames[idx])
        # with open(cam_pos_filepath) as f:
        #     cam_pos_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)
        # assert len(cam_pos_lst) == 1, cam_pos_filepath

        # # read room type file
        # room_type = None
        # room_type_filepath = os.path.join(self.room_type_dir, self.local_txt_fnames[idx])
        # with open(room_type_filepath) as f:
        #     room_type = f.readline().strip()
        #     assert room_type in ROOM_TYPE_DICT.keys(), room_type_filepath
        #     room_type = ROOM_TYPE_DICT[room_type]
        # room_type = ROOM_TYPE_DICT['bedroom']

        # # read room textual description file
        # text_desc_lst = []
        # text_emb = np.array([])
        # text_desc_filepath = os.path.join(self.text_desc_dir, self.local_txt_fnames[idx])
        # with open(text_desc_filepath) as f:
        #     text_desc = f.readline()
        #     text_desc_lst = text_desc.strip().split('. ')
        #     text_desc_lst = [complete_stop_in_sentence(sen) for sen in text_desc_lst if len(sen)]
        # # print(text_desc_lst)
        # # read text embedding file
        # text_emb_filepath = os.path.join(self.text_emb_dir, self.local_npy_fnames[idx])
        # text_emb = np.load(text_emb_filepath).astype(np.float32)
        # # print(f'text_emb.shape: {text_emb.shape}')

        # # read object bbox file
        # object_bbox_filepath = os.path.join(self.bbox_3d_dir, self.local_json_fnames[idx])
        # object_bbox_lst = []
        # with open(object_bbox_filepath) as f:
        #     object_bbox_dicts = json.load(f)
        #     object_bbox_dicts = object_bbox_dicts['objects']
        # # sort object bbox by class frequency and bbox size
        # object_bbox_dicts = ordered_bboxes_with_class_frequencies(room_type=room_type,
        #                                                           object_bbox_lst=object_bbox_dicts)

        # for obj_bbox in object_bbox_dicts:
        #     bbox_class_label = obj_bbox['class'].lower()
        #     bbox_class = ClassLabelsEncode(room_type=room_type, obj_bbox_label=bbox_class_label)
        #     bbox_centroid = np.array(obj_bbox['center'], np.float32)
        #     bbox_centroid = TranslationEncode(bbox_centroid)
        #     bbox_size = np.array(obj_bbox['size'], np.float32)
        #     bbox_size = SizeEncode(bbox_size)
        #     # only use Z angle
        #     bbox_angle = np.array(obj_bbox['angles'], np.float32)
        #     bbox_angle = RotationEncode(bbox_angle)
        #     bbox_property_encode = np.concatenate([bbox_class, bbox_centroid, bbox_size, bbox_angle], axis=-1)
        #     # print(f'bbox_property_encode: {bbox_property_encode}')
        #     bbox_property_encode_dim = bbox_property_encode.shape[-1]
        #     object_bbox_lst.append(bbox_property_encode)
        # object_bbox_lst = padding_and_reshape_object_bbox(room_type=room_type,
        #                                                   object_bbox_lst=np.array(object_bbox_lst),
        #                                                   bbox_dim=bbox_property_encode_dim)

        # # read wall bbox
        # wall_bbox_filepath = os.path.join(self.quad_wall_dir, self.local_json_fnames[idx])
        # wall_bbox_lst = []
        # with open(wall_bbox_filepath, 'r') as f:
        #     wall_bbox_dicts = json.load(f)
        #     wall_bbox_dicts = wall_bbox_dicts['walls']
        # for wall_bbox in wall_bbox_dicts:
        #     wall_class_label = wall_bbox['class'].lower()
        #     wall_class = ClassLabelsEncode(room_type=room_type, obj_bbox_label=wall_class_label)
        #     wall_centroid = np.array(wall_bbox['center'], np.float32)
        #     wall_centroid = TranslationEncode(wall_centroid)
        #     wall_size = np.array([wall_bbox['width'], 0.01, wall_bbox['height']], np.float32)
        #     wall_size = SizeEncode(wall_size)
        #     wall_angle = np.array(wall_bbox['angles'], np.float32)
        #     wall_angle = RotationEncode(wall_angle)
        #     wall_property_encode = np.concatenate([wall_class, wall_centroid, wall_size, wall_angle], axis=-1)
        #     # print(f'wall_property_encode: {wall_property_encode}')
        #     wall_property_encode_dim = wall_property_encode.shape[-1]
        #     wall_bbox_lst.append(wall_property_encode)
        # wall_bbox_lst = padding_and_reshape_wall_bbox(room_type=room_type,
        #                                               wall_bbox_lst=np.array(wall_bbox_lst),
        #                                               bbox_dim=wall_property_encode_dim)

        room_type = self.local_room_type_lst[idx]
        wall_bbox_lst = self.local_wall_bbox_lst[idx]
        object_bbox_lst = self.local_obj_bbox_lst[idx]
        text_prompt = self.local_text_lst[idx]
        text_emb = self.local_text_emb_lst[idx]

        assert wall_bbox_lst.shape[-1] == object_bbox_lst.shape[-1]
        out_lst = np.concatenate([wall_bbox_lst, object_bbox_lst], axis=0)
        out_lst = out_lst.transpose(1, 0)
        
        cond_dict = {}
        if room_type is not None:
            cond_dict["y"] = np.array(room_type, dtype=np.int64)

        # text_desc_len = len(text_desc_lst)
        # if text_desc_len:
        #     if text_desc_len > self.max_text_sentences:
        #         text_desc_lst = text_desc_lst[:self.max_text_sentences]
        cond_dict["text"] = text_prompt
        cond_dict["context"] = text_emb
        if self.is_test:
            return out_lst, cond_dict, self.local_img_fnames[idx][:-4]
        else:
            return out_lst, cond_dict
