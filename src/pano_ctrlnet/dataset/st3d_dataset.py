import os
import sys

sys.path.append('.')
sys.path.append('..')

import json
import cv2
import numpy as np

from typing import Dict, List, Tuple
from torch.utils.data import Dataset

from utils import panostrech 

def ClassLabelsEncode(room_type: int, obj_bbox_label: str) -> np.array:
    """Implement the encoding for the class labels."""
    # Make a local copy of the class labels
    classes = None
    if room_type == ROOM_TYPE_DICT['bedroom']:
        classes = ST3D_BEDROOM_FURNITURE
    elif room_type == ROOM_TYPE_DICT['living room']:
        classes = ST3D_LIVINGROOM_FURNITURE
    elif room_type == ROOM_TYPE_DICT['dining room']:
        classes = ST3D_DININGROOM_FURNITURE

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
        class_freq_dict = ST3D_BEDROOM_FURNITURE_CNTS
    elif room_type == ROOM_TYPE_DICT['living room']:
        class_freq_dict = ST3D_LIVINGROOM_FURNITURE_CNTS
    elif room_type == ROOM_TYPE_DICT['dining room']:
        class_freq_dict = ST3D_DININGROOM_FURNITURE_CNTS

    bbox_size_lst = np.array([(np.array(bbox['size']) + 1) * 0.5 for bbox in object_bbox_lst])
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
        max_len = ST3D_BEDROOM_MAX_LEN
        class_num = len(ST3D_BEDROOM_FURNITURE)
    elif room_type == ROOM_TYPE_DICT['living room']:
        max_len = ST3D_LIVINGROOM_MAX_LEN
        class_num = len(ST3D_LIVINGROOM_FURNITURE)
    elif room_type == ROOM_TYPE_DICT['dining room']:
        max_len = ST3D_DININGROOM_MAX_LEN
        class_num = len(ST3D_DININGROOM_FURNITURE)

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
    max_len = ST3D_ROOM_QUAD_WALL_MAX_LEN
    if room_type == ROOM_TYPE_DICT['bedroom']:
        class_num = len(ST3D_BEDROOM_FURNITURE)
    elif room_type == ROOM_TYPE_DICT['living room']:
        class_num = len(ST3D_LIVINGROOM_FURNITURE)
    elif room_type == ROOM_TYPE_DICT['dining room']:
        class_num = len(ST3D_DININGROOM_FURNITURE)

    assert L <= max_len, 'The length of the wall bbox list should be less than 10.'

    # Pad the end label in the end of each sequence, and convert the class labels to -1, 1
    empty_label = np.eye(class_num)[-1]
    padding = np.concatenate([empty_label, np.zeros(bbox_dim - class_num, dtype=np.float32)], axis=0)
    wall_bbox_lst = np.vstack([wall_bbox_lst, np.tile(padding, [max_len - L, 1])])

    # print(f'wall_bbox_lst: {wall_bbox_lst}')
    return wall_bbox_lst


def complete_stop_in_sentence(sentence: str) -> str:
    """Implement the stop in the end of the sentence."""
    if sentence[-1] != '.':
        sentence += '.'
    return sentence

def cor2xybound(cor):
    ''' Helper function to clip max/min stretch factor '''
    corU = cor[0::2]
    corB = cor[1::2]
    zU = -50
    u = panostrech.coorx2u(corU[:, 0])
    vU = panostrech.coory2v(corU[:, 1])
    vB = panostrech.coory2v(corB[:, 1])

    x, y = panostrech.uv2xy(u, vU, z=zU)
    c = np.sqrt(x**2 + y**2)
    zB = c * np.tan(vB)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    S = 3 / abs(zB.mean() - zU)
    dx = [abs(xmin * S), abs(xmax * S)]
    dy = [abs(ymin * S), abs(ymax * S)]

    return min(dx), min(dy), max(dx), max(dy)

class ST3DDataset(Dataset):
    '''
    dataset for layout: PanoCoordinatesBoundary
    '''

    def __init__(
            self,
            root_dir,
            split='train',
            room_type='bedroom',
            flip=False,
            rotate=False,
            gamma=False,
            stretch=False,
            p_base=0.96,
            max_stretch=2.0,
            use_raw_img=False,
            max_text_sentences=4,  #  max number of text_prompt sentences
            shard=0,  #  support parallel training
            num_shards=1,
            downsample_scale = 2):
        self.root_dir = os.path.join(root_dir, split, room_type)
        self.img_dir = os.path.join(self.root_dir, 'img')
        self.cor_dir = os.path.join(self.root_dir, 'label_cor')
        self.quad_wall_dir = os.path.join(self.root_dir, 'quad_walls')
        self.cam_pos_dir = os.path.join(self.root_dir, 'cam_pos')
        self.room_type_dir = os.path.join(self.root_dir, 'room_type')
        # object bbox folder
        self.bbox_3d_dir = os.path.join(self.root_dir, 'bbox_3d')
        # text descritpion folder
        self.text_desc_dir = os.path.join(self.root_dir, 'text_desc')
        self.text_emb_dir = os.path.join(self.root_dir, 'text_desc_emb')
        # semantic bbox image folder
        self.sem_bbox_img_dir = os.path.join(self.root_dir, 'sem_bbox_img')
        # semantic layout image folder
        self.sem_layout_img_dir = os.path.join(self.root_dir, 'sem_layout_img')

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

        self.flip = flip
        self.rotate = rotate
        self.gamma = gamma
        self.stretch = stretch
        self.p_base = p_base
        self.max_stretch = max_stretch
        self.use_raw_img = use_raw_img
        self.max_text_sentences = max_text_sentences
        self.downsample_scale = downsample_scale

        # The direction of all camera is always along the negative y-axis.
        self.cam_R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], np.float32)

        self._check_dataset()

    def _check_dataset(self):
        for fname in self.txt_fnames:
            assert os.path.isfile(os.path.join(self.text_desc_dir,
                                               fname)), '%s not found' % os.path.join(self.text_desc_dir, fname)

    def __len__(self):
        return len(self.local_img_fnames)

    def __getitem__(self, idx: int) -> Dict:
        """retrieve scene data

        Args:
            idx (int): panorama/room idx

        Returns:
            List: [Image, [boundary_x:1x1024, boundary_y:1x1024], boundary_wall_probability:1x-024]
        """
        # Read panorama image
        img_path = os.path.join(self.img_dir, self.local_img_fnames[idx])
        # pano_img = Image.open(img_path)
        pano_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)

        # read semantic bbox image
        sem_bbox_img_path = os.path.join(self.sem_layout_img_dir, self.local_img_fnames[idx])
        # sem_bbox_img = Image.open(sem_bbox_img_path)
        sem_bbox_img = cv2.imread(sem_bbox_img_path, cv2.IMREAD_UNCHANGED)
        sem_bbox_img = cv2.cvtColor(sem_bbox_img, cv2.COLOR_BGR2RGB)

        # Stretch augmentation
        if self.stretch:
            # Read ground truth corners
            with open(os.path.join(self.cor_dir,
                                self.local_txt_fnames[idx])) as f:
                cor = np.array([line.strip().split() for line in f if line.strip()], np.float32)
                # Corner with minimum x should at the beginning
                cor = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)

            xmin, ymin, xmax, ymax = cor2xybound(cor)
            kx = np.random.uniform(1.0, self.max_stretch)
            ky = np.random.uniform(1.0, self.max_stretch)
            if np.random.randint(2) == 0:
                kx = max(1 / kx, min(0.5 / xmin, 1.0))
            else:
                kx = min(kx, max(10.0 / xmax, 1.0))
            if np.random.randint(2) == 0:
                ky = max(1 / ky, min(0.5 / ymin, 1.0))
            else:
                ky = min(ky, max(10.0 / ymax, 1.0))
            pano_img, _ = panostrech.pano_stretch(pano_img, cor, kx, ky)
            sem_bbox_img, _ = panostrech.pano_stretch(sem_bbox_img, cor, kx, ky)

        if not self.use_raw_img:
            # Normalize target images to [-1, 1].
            pano_img = (pano_img.astype(np.float32) / 127.5) - 1.0
            img_h, img_w = pano_img.shape[:2]
            pano_img = cv2.resize(pano_img, (img_w//self.downsample_scale, img_h//self.downsample_scale))
            H, W = pano_img.shape[:2]
            # Normalize source img to [0, 1]
            sem_bbox_img = np.array(sem_bbox_img, np.float32) / 255.
            sem_bbox_img = cv2.resize(sem_bbox_img, (img_w//self.downsample_scale, img_h//self.downsample_scale))


        # print(f'pano_img.shape: {pano_img.shape}')


        # Random flip
        if self.flip and np.random.randint(2) == 0:
            pano_img = np.flip(pano_img, axis=1)
            sem_bbox_img = np.flip(sem_bbox_img, axis=1)

        # Random horizontal rotate
        if self.rotate:
            delta_x = np.random.randint(W)
            pano_img = np.roll(pano_img, delta_x, axis=1)
            sem_bbox_img = np.roll(sem_bbox_img, delta_x, axis=1)

        # Random gamma augmentation
        if self.gamma:
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            pano_img = pano_img**p
            sem_bbox_img = sem_bbox_img**p


        # read room textual description file
        text_desc_lst = []
        text_desc_filepath = os.path.join(self.text_desc_dir, self.local_txt_fnames[idx])
        with open(text_desc_filepath) as f:
            text_desc = f.readline()
            text_desc_lst = text_desc.strip().split('. ')
            text_desc_lst = [complete_stop_in_sentence(sen) for sen in text_desc_lst if len(sen)]
        text_prompt = ''.join(text_desc_lst)



        if self.use_raw_img:
            # read object bbox file
            object_bbox_filepath = os.path.join(self.bbox_3d_dir, self.local_json_fnames[idx])
            with open(object_bbox_filepath) as f:
                object_bbox_dicts = json.load(f)
                object_bbox_dicts = object_bbox_dicts['objects']
            for obj_bbox in object_bbox_dicts:
                bbox_class_label = obj_bbox['class'].lower()
                bbox_centroid = np.array(obj_bbox['center'], np.float32)
                # bbox_size = (np.array(obj_bbox['size'], np.float32) + 1) * 0.5
                bbox_size = np.array(obj_bbox['size'], np.float32)
                obj_bbox['size'] = bbox_size.tolist()
                # only use Z angle
                bbox_angles = np.array(obj_bbox['angles'], np.float32)
                # z_angle = np.arcsin(bbox_angles[1]) if abs(bbox_angles[0]) < 5e-3 else np.arccos(bbox_angles[0])
                # obj_bbox['angles'] = [0, 0, z_angle]
                obj_bbox['angles'] = bbox_angles.tolist()

            # read wall bbox
            wall_bbox_filepath = os.path.join(self.quad_wall_dir, self.local_json_fnames[idx])
            with open(wall_bbox_filepath, 'r') as f:
                wall_bbox_dicts = json.load(f)
                wall_bbox_dicts = wall_bbox_dicts['walls']
            for wall_bbox in wall_bbox_dicts:
                # wall_class = wall_bbox['class']
                wall_bbox['class'] = 'wall'
                wall_centroid = np.array(wall_bbox['center'], np.float32)
                wall_bbox['size'] = [wall_bbox['width'], 0.01, wall_bbox['height']]
                wall_angles = np.array(wall_bbox['angles'], np.float32)
                z_angle = np.arcsin(wall_angles[1]) if abs(wall_angles[0]) < 5e-3 else np.arccos(wall_angles[0])
                wall_bbox['angles'] = [0, 0, z_angle]

            # merge wall bbox and object bbox
            object_bbox_dicts.extend(wall_bbox_dicts)
            return dict(jpg=pano_img, txt=text_prompt, hint=sem_bbox_img), object_bbox_dicts
        else:
            return dict(jpg=pano_img, txt=text_prompt, hint=sem_bbox_img)