import os
import sys
import numpy as np
from PIL import Image

from typing import Any, List, Dict, Tuple
import json, random
from collections import defaultdict, Counter

import torch.utils.data as data

from .threed_front.metadata import THREED_FRONT_BEDROOM_WO_DOOR_WINDOW_WALL_FURNITURE, THREED_FRONT_BEDROOM_FURNITURE_CNTS, \
    THREED_FRONT_LIVINGROOM_WO_DOOR_WINDOW_WALL_FURNITURE, THREED_FRONT_DININGROOM_FURNITURE_WO_DOOR_WINDOW_WALL, THREED_FRONT_LIVINGROOM_FURNITURE_CNTS, \
    THREED_FRONT_BEDROOM_MIN_FURNITURE_NUM, THREED_FRONT_BEDROOM_MAX_FURNITURE_NUM, \
    THREED_FRONT_LIVINGROOM_MIN_FURNITURE_NUM, THREED_FRONT_LIVINGROOM_MAX_FURNITURE_NUM, \
    THREED_FRONT_BEDROOM_MAX_WALL_NUM, THREED_FRONT_LIVINGROOM_MAX_WALL_NUM
from .text_utils import compute_rel_3dfront, get_article
from num2words import num2words
from src.diffu_layout.improved_diffusion.clip_util import FrozenCLIPEmbedder

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

from .threed_front.utils import CSVSplitsBuilder


def padding_and_reshape_bbox(room_type: int, bbox_lst: np.array, bbox_dim: int) -> List:
    """ Implement the padding for the quad wall boxes.
    Args:
        room_type (int): The room type.
        bbox_lst (np.array): The quadwall bounding box list.
        bbox_dim (int): The dimension of the quadwallbounding box.
    Returns:
        _type_: _description_
    """
    L = len(bbox_lst)
    if room_type == ROOM_TYPE_DICT['bedroom']:
        class_num = len(THREED_FRONT_BEDROOM_WO_DOOR_WINDOW_WALL_FURNITURE)
        max_len = THREED_FRONT_BEDROOM_MAX_FURNITURE_NUM
    elif room_type == ROOM_TYPE_DICT['living room']:
        class_num = len(THREED_FRONT_LIVINGROOM_WO_DOOR_WINDOW_WALL_FURNITURE)
        max_len = THREED_FRONT_LIVINGROOM_MAX_FURNITURE_NUM
    elif room_type == ROOM_TYPE_DICT['dining room']:
        class_num = len(THREED_FRONT_DININGROOM_FURNITURE_WO_DOOR_WINDOW_WALL)
        max_len = THREED_FRONT_LIVINGROOM_MAX_FURNITURE_NUM

    assert L <= max_len, 'The length of the wall bbox list should be less than 10.'

    # Pad the end label in the end of each sequence, and convert the class labels to -1, 1
    empty_label = np.eye(class_num)[-1] * 2 - 1
    padding = np.concatenate([empty_label, np.zeros(bbox_dim - class_num, dtype=np.float32)], axis=0)
    bbox_lst = np.vstack([bbox_lst, np.tile(padding, [max_len - L, 1])])

    return bbox_lst


def dict_bbox_to_vec(dict_box):
    '''
    input: {'min': [1,2,3], 'max': [4,5,6]}
    output: [1,2,3,4,5,6]
    '''
    return dict_box['min'] + dict_box['max']


def clean_obj_name(name):
    return name.replace('_', ' ')


def Add_Text(samples_dict: Dict, clip_encoder:FrozenCLIPEmbedder, class_labels: List, eval=False, max_sentences=3, max_token_length=50):

    def add_relation(sample):
        '''
            Add relations to sample['relations']
        '''
        relations = []
        num_objs = len(sample['translations'])

        for ndx in range(num_objs):
            this_box_trans = sample['translations'][ndx, :]
            this_box_sizes = sample['sizes'][ndx, :]
            this_box = {'min': list(this_box_trans - this_box_sizes), 'max': list(this_box_trans + this_box_sizes)}

            # only backward relations
            choices = [other for other in range(num_objs) if other < ndx]
            for other_ndx in choices:
                prev_box_trans = sample['translations'][other_ndx, :]
                prev_box_sizes = sample['sizes'][other_ndx, :]
                prev_box = {'min': list(prev_box_trans - prev_box_sizes), 'max': list(prev_box_trans + prev_box_sizes)}
                box1 = dict_bbox_to_vec(this_box)
                box2 = dict_bbox_to_vec(prev_box)

                relation_str, distance = compute_rel_3dfront(box1, box2)
                if relation_str is not None:
                    relation = (ndx, relation_str, other_ndx, distance)
                    relations.append(relation)

        sample['relations'] = relations

        return sample

    def add_description(sample, class_labels, is_eval=False):
        '''
            Add text descriptions to each scene
            sample['description'] = str is a sentence
            eg: 'The room contains a bed, a table and a chair. The chair is next to the window'
        '''
        sentences = []
        # clean object names once
        classes = class_labels
        class_index = sample['class_labels'].argmax(-1)
        obj_names = list(map(clean_obj_name, [classes[ind] for ind in class_index]))
        # objects that can be referred to
        refs = []
        # TODO: handle commas, use "and"
        # TODO: don't repeat, get counts and pluralize
        # describe the first 2 or 3 objects
        if is_eval:
            first_n = 3
        else:
            first_n = random.choice([2, 3])
        # first_n = len(obj_names)
        first_n_names = obj_names[:first_n]
        first_n_counts = Counter(first_n_names)

        s = 'The room has '
        for ndx, name in enumerate(sorted(set(first_n_names), key=first_n_names.index)):
            if ndx == len(set(first_n_names)) - 1 and len(set(first_n_names)) >= 2:
                s += "and "
            if first_n_counts[name] > 1:
                s += f'{num2words(first_n_counts[name])} {name}s '
            else:
                s += f'{get_article(name)} {name} '
            if ndx == len(set(first_n_names)) - 1:
                s += ". "
            if ndx < len(set(first_n_names)) - 2:
                s += ', '
        sentences.append(s)
        refs = set(range(first_n))

        # for each object, the "position" of that object within its class
        # eg: sofa table table sofa
        #   -> 1    1    2      1
        # use this to get "first", "second"

        seen_counts = defaultdict(int)
        in_cls_pos = [0 for _ in obj_names]
        for ndx, name in enumerate(first_n_names):
            seen_counts[name] += 1
            in_cls_pos[ndx] = seen_counts[name]

        for ndx in range(1, len(obj_names)):
            # higher prob of describing the 2nd object
            prob_thresh = 0.3

            if is_eval:
                random_num = 1.0
            else:
                random_num = random.random()
            if random_num > prob_thresh:
                # possible backward references for this object
                possible_relations = [r for r in sample['relations'] \
                                        if r[0] == ndx \
                                        and r[2] in refs \
                                        and r[3] < 1.5]
                if len(possible_relations) == 0:
                    continue
                # now future objects can refer to this object
                refs.add(ndx)

                # if we haven't seen this object already
                if in_cls_pos[ndx] == 0:
                    # update the number of objects of this class which have been seen
                    seen_counts[obj_names[ndx]] += 1
                    # update the in class position of this object = first, second ..
                    in_cls_pos[ndx] = seen_counts[obj_names[ndx]]

                # pick any one
                if is_eval:
                    (n1, rel, n2, dist) = possible_relations[0]
                else:
                    (n1, rel, n2, dist) = random.choice(possible_relations)
                o1 = obj_names[n1]
                o2 = obj_names[n2]

                # prepend "second", "third" for repeated objects
                if seen_counts[o1] > 1:
                    o1 = f'{num2words(in_cls_pos[n1], ordinal=True)} {o1}'
                if seen_counts[o2] > 1:
                    o2 = f'{num2words(in_cls_pos[n2], ordinal=True)} {o2}'

                # dont relate objects of the same kind
                if o1 == o2:
                    continue

                a1 = get_article(o1)

                if 'touching' in rel:
                    if ndx in (1, 2):
                        s = F'The {o1} is next to the {o2}'
                    else:
                        s = F'There is {a1} {o1} next to the {o2}'
                elif rel in ('left of', 'right of'):
                    if ndx in (1, 2):
                        s = f'The {o1} is to the {rel} the {o2}'
                    else:
                        s = f'There is {a1} {o1} to the {rel} the {o2}'
                elif rel in ('surrounding', 'inside', 'behind', 'in front of', 'on', 'above'):
                    if ndx in (1, 2):
                        s = F'The {o1} is {rel} the {o2}'
                    else:
                        s = F'There is {a1} {o1} {rel} the {o2}'
                s += ' . '
                sentences.append(s)

        # set back into the sample
        sample['description'] = sentences

        # delete sample['relations']
        del sample['relations']
        return sample

    # def add_glove_embeddings(sample):
    #     sentence = ''.join(sample['description'][:self.max_sentences])
    #     sample['description'] = sentence
    #     tokens = list(word_tokenize(sentence))
    #     # pad to maximum length
    #     tokens += ['<pad>'] * (self.max_token_length - len(tokens))

    #     # embed words
    #     sample['desc_emb'] = torch.cat([self.glove[token].unsqueeze(0) for token in tokens]).numpy()

    #     return sample

    def add_clip_embeddings(sample, clip_encoder, max_sentences=3):
        sentence = ''.join(sample['description'][:max_sentences])
        sample['description'] = sentence
        # print(f'sentence: {sentence}')
        # embed words
        sample['desc_emb'] = clip_encoder.get_text_embeds(sample["description"]).cpu().numpy()
        return sample

    # Add relationship between objects
    sample = add_relation(samples_dict)

    # Add description
    sample = add_description(sample, class_labels=class_labels, is_eval=eval)

    # sample = add_glove_embeddings(sample)
    sample = add_clip_embeddings(sample, clip_encoder, max_sentences=max_sentences)
    return sample


def Permutate_Scene(samples_dict: Dict,
                    permutation_keys: List = ['translations', 'sizes', 'angles'],
                    permutation_axis=0):
    shapes = samples_dict["class_labels"].shape
    # print(f'Permutation: {shapes}')
    ordering = np.random.permutation(shapes[permutation_axis])
    # print(f'Permutation new orders: {ordering}')

    for k in permutation_keys:
        samples_dict[k] = samples_dict[k][ordering]

    return samples_dict


class ThreedFrontDataset(data.Dataset):
    '''
    dataset for ThreeD-Front
    '''

    def __init__(
            self,
            config: Dict[str, Any],
            room_type='bedroom',
            is_train=True,
            is_test=False,
            rot_augmentation=True,  # data augmentation by random rotation
            random_text_desc=True,  # use random text description
            permutation=True,  # permute objects
            shard=0,  #  support parallel training
            num_shards=1):

        self._base_dir = config["data"]["dataset_directory"]
        self.config = config
        self.room_type = room_type
        self.is_train = is_train
        self.is_test = is_test
        self.rot_augmentation = rot_augmentation
        self.text_encoder = FrozenCLIPEmbedder()
        self.random_text_desc = random_text_desc
        self.permutation = permutation

        self._parse_train_stats(config["data"]["train_stats"])

        # Make the train/test/validation splits
        splits_builder = CSVSplitsBuilder(config["data"]["annotation_file"])
        split_scene_ids = splits_builder.get_splits(
            keep_splits=config["training"].get("splits", ["train", "val"])) if is_train \
                  else splits_builder.get_splits(keep_splits=config["validation"].get("splits", [ "test"]))

        # rooms liist
        self._tags_lst = sorted([
            oi for oi in os.listdir(self._base_dir) if oi.split("_")[1] in split_scene_ids
            if os.path.exists(os.path.join(self._base_dir, oi))
        ])
        # bbox npz files
        self._path_to_rooms_lst = sorted([
            os.path.join(self._base_dir, pi, "boxes.npz")
            for pi in self._tags_lst
            if os.path.exists(os.path.join(self._base_dir, pi, "boxes.npz"))
        ])
        # # text prompt files
        # self._path_to_rooms_text_lst = sorted([
        #     os.path.join(self._base_dir, pi, "text_prompt.txt")
        #     for pi in self._tags_lst
        #     if os.path.exists(os.path.join(self._base_dir, pi, "text_prompt.txt"))
        # ])
        assert len(self._tags_lst) == len(self._path_to_rooms_lst), "Number of scenes and boxes.npz files do not match"
        # assert len(self._tags_lst) == len(
        #     self._path_to_rooms_text_lst), "Number of scenes and text_prompt.txt files do not match"

        rendered_scene = "rendered_scene_256.png"
        path_to_rendered_scene = os.path.join(self._base_dir, self._tags_lst[0], rendered_scene)
        if not os.path.isfile(path_to_rendered_scene):
            rendered_scene = "rendered_scene_256_no_lamps.png"

        self._path_to_renders = sorted([os.path.join(self._base_dir, pi, rendered_scene) for pi in self._tags_lst])

        self.local_tags_lst = self._tags_lst[shard:][::num_shards]
        self.local_path_to_rooms_lst = self._path_to_rooms_lst[shard:][::num_shards]
        self.local_path_to_renders = self._path_to_renders[shard:][::num_shards]
        # self.local_path_to_rooms_text_lst = self._path_to_rooms_text_lst[shard:][::num_shards]

        # pre-load all data
        self.local_rooms_dict_lst = []
        # self.local_rooms_text_lst = []
        self._preload_()

    def _get_room_layout(self, room_layout):
        # Resize the room_layout if needed
        img = Image.fromarray(room_layout[:, :, 0])
        img = img.resize(tuple(map(int, self.config["data"]["room_layout_size"].split(","))), resample=Image.BILINEAR)
        D = np.asarray(img).astype(np.float32) / np.float32(255)
        return D

    def __len__(self):
        return len(self.local_path_to_rooms_lst)

    def _parse_train_stats(self, train_stats):
        with open(os.path.join(self._base_dir, train_stats), "r") as f:
            train_stats = json.load(f)
        self._centroids = train_stats["bounds_translations"]
        self._centroids = (np.array(self._centroids[:3]), np.array(self._centroids[3:]))
        self._sizes = train_stats["bounds_sizes"]
        self._sizes = (np.array(self._sizes[:3]), np.array(self._sizes[3:]))
        self._angles = train_stats["bounds_angles"]
        self._angles = (np.array(self._angles[0]), np.array(self._angles[1]))

        self._class_labels = train_stats["class_labels"]
        self._object_types = train_stats["object_types"]
        self._class_frequencies = train_stats["class_frequencies"]
        self._class_order = train_stats["class_order"]
        self._count_furniture = train_stats["count_furniture"]

    def _preload_(self):
        """pre-load all data
        """
        for idx in range(len(self)):
            # load npz data
            npz_data = np.load(self.local_path_to_rooms_lst[idx])
            self.local_rooms_dict_lst.append(npz_data)
            # # load text prompt file
            # with open(self.local_path_to_rooms_text_lst[idx], "r") as f:
            #     text = f.read().strip()
            #     self.local_rooms_text_lst.append(text)

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def object_types(self):
        return self._object_types

    @property
    def class_frequencies(self):
        return self._class_frequencies

    @property
    def class_order(self):
        return self._class_order

    @property
    def count_furniture(self):
        return self._count_furniture

    def _check_dataset(self):
        for scene_tag in self._tags_lst:
            assert os.path.isfile(os.path.join(self._base_dir, scene_tag, "boxes.npz")), \
                "boxes.npz not found in {}".format(os.path.join(self._base_dir, scene_tag))

    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def post_process(self, samples: np.ndarray):
        """ post process the samples in the room

        Args:
            samples (np.ndarray): sampled furniture and walls in the room
        """
        N, C = samples.shape
        center_bounds = self._centroids
        size_bounds = self._sizes

        center_dim = 3
        size_dim = 3
        angle_dim = 2
        class_label_dim = len(self._class_labels)

        new_samples = []
        for i in range(N):
            # descale class labels
            class_labels = samples[i, :class_label_dim]
            descaled_class_labels = (class_labels + 1) / 2
            class_label_prob = np.where(descaled_class_labels > 0.5, 1, 0)
            class_label = self.class_labels[class_label_prob.argmax()]
            if class_label == 'end':
                continue

            # descale center
            center = samples[i, class_label_dim:class_label_dim + center_dim]
            descaled_centers = self.descale(center, *center_bounds)

            # descale size
            size = samples[i, class_label_dim + center_dim:class_label_dim + center_dim + size_dim]
            descaled_sizes = self.descale(size, *size_bounds)

            # cvt cos,sin to angle
            cos_sin_angle = samples[i, class_label_dim + center_dim + size_dim:class_label_dim + center_dim + size_dim +
                                    angle_dim]
            angles = np.arctan2(cos_sin_angle[1:2], cos_sin_angle[0:1])

            # concatenate
            descaled_samples = np.concatenate([descaled_class_labels, descaled_centers, descaled_sizes, angles],
                                              axis=-1)
            new_samples.append(descaled_samples)
        return np.array(new_samples)

    @staticmethod
    def rotation_matrix_around_y(theta):
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        return R

    def __getitem__(self, idx: int):
        """retrieve scene data

        Args:
            idx (int): panorama/room idx

        Returns:
            List: 
        """
        # D = np.load(self.local_path_to_rooms_lst[idx])
        D = self.local_rooms_dict_lst[idx]
        # text_prompt = self.local_rooms_text_lst[idx]
        # print('text_prompt: ', text_prompt)

        # room_layout_img = self._get_room_layout(D["room_layout"])
        # room_layout_img = np.transpose(room_layout_img[:, :, None], (2, 0, 1))
        room_text_emb = D["desc_emb"].squeeze(0).astype(np.float32)
        # print(f'room_text_emb: {room_text_emb.shape}')

        # "room_text_emb": room_text_emb,
        # "class_labels": D["class_labels"],
        # "translations": D["translations"],
        # "sizes": D["sizes"],
        # "angles": D["angles"]
        bbox_onehot_class_labels = D["class_labels"]
        # print(f'bbox_onehot_class_labels: {bbox_onehot_class_labels.shape}')
        bbox_trans = D["translations"]
        # print(f'bbox_translations: {bbox_trans.shape}')
        bbox_sizes = D["sizes"]
        # print(f'bbox_sizes: {bbox_sizes.shape}')
        bbox_angles = D["angles"]
        # print(f'bbox_angles: {bbox_angles.shape}')

        # data augmentation
        if self.rot_augmentation:
            rot_angle = np.random.uniform(0.0, np.pi)
            if np.random.rand() < 0.25:
                rot_angle = np.pi * 1.5
            elif np.random.rand() < 0.50:
                rot_angle = np.pi
            elif np.random.rand() < 0.75:
                rot_angle = np.pi * 0.5
            else:
                rot_angle = 0.0
            R = self.rotation_matrix_around_y(rot_angle)
            # rotate translations
            bbox_trans = bbox_trans.dot(R)
            # rotate angles
            angle_min, angle_max = self._angles
            bbox_angles = (bbox_angles + rot_angle - angle_min) % (2 * np.pi) + angle_min

        # add text description
        if self.random_text_desc:
            samples_dict = {
                'class_labels': bbox_onehot_class_labels,
                'translations': bbox_trans,
                'sizes': bbox_sizes,
                'angles': bbox_angles
            }
            samples = Add_Text(samples_dict=samples_dict,
                               clip_encoder=self.text_encoder,
                               class_labels=self.class_labels,
                               eval=(self.is_train == False),
                               max_sentences=3)
            room_text = samples["description"]
            room_text_emb = samples['desc_emb'].squeeze(0).astype(np.float32)

        # encode angles as [cosin, sin]
        bbox_cos_sin_angles = np.concatenate([np.cos(bbox_angles), np.sin(bbox_angles)], axis=-1)
        # scale properties to -1 ~ 1
        scaled_class_labels = bbox_onehot_class_labels * 2 - 1
        scaled_trans = self.scale(bbox_trans, self._centroids[0], self._centroids[1])
        scaled_size = self.scale(bbox_sizes, self._sizes[0], self._sizes[1])

        # perrmutate objects
        if self.permutation:
            samples_dict = {
                'class_labels': scaled_class_labels,
                'translations': scaled_trans,
                'sizes': scaled_size,
                'angles': bbox_cos_sin_angles
            }
            samples = Permutate_Scene(
                samples_dict=samples_dict,
                permutation_keys=['class_labels', 'translations', 'sizes', 'angles'],
            )
            scaled_class_labels = samples['class_labels']
            scaled_trans = samples['translations']
            scaled_size = samples['sizes']
            bbox_cos_sin_angles = samples['angles']

        # concatenate
        out_lst = np.concatenate([scaled_class_labels, scaled_trans, scaled_size, bbox_cos_sin_angles], axis=-1)
        # print(f'concatenated out_lst: {out_lst.shape}')

        # pad to max_length
        bbox_dim = out_lst.shape[-1]
        out_lst = padding_and_reshape_bbox(room_type=ROOM_TYPE_DICT[self.room_type],
                                           bbox_lst=out_lst,
                                           bbox_dim=bbox_dim)
        # print(f'padding out_lst: {out_lst.shape}')
        # print(f'out_lst: {out_lst}')
        out_lst = out_lst.transpose(1, 0)

        cond_dict = {}
        # if self.room_type is not None:
        #     cond_dict["y"] = np.array(ROOM_TYPE_DICT[self.room_type], dtype=np.int64)

        cond_dict["text"] = room_text
        cond_dict["text_condition"] = room_text_emb
        if not self.is_train:
            return out_lst, cond_dict, self.local_tags_lst[idx]
        else:
            return out_lst, cond_dict
