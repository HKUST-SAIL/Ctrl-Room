import os

import numpy as np
import trimesh
import copy

from typing import List, Dict, Tuple
from functools import cached_property, reduce, lru_cache
from collections import Counter, OrderedDict
from torch.utils.data import Dataset

from dataset.st3d.utils import create_spatial_quad_polygen

class ST3dFutureModel(object):

    def __init__(self, 
                 centroid:np.array, 
                 rotation:np.array, 
                 size:np.array, 
                 class_name:np.array,
                 normal:np.array=None,
                 corners:np.array=None):
        self._centroid = centroid
        self._angle = rotation
        self._size = size
        self._label = class_name
        self._normal = normal
        self._corners = corners

    def __str__(self) -> str:
        return "ST3dFutureModel: {} with centroid: {} and size: {}, z_angle: {}".format(self.label, self.centroid, self.size, self.z_angle)
    
    def centroid(self, offset=[[0, 0, 0]]):
        return self._centroid + offset

    @property
    def z_angle(self):
        return self._angle
    @property
    def normal(self):
        if self._normal is None:
            self._normal = [0, 0, 0]
        return self._normal
    @property
    def size(self):
        return self._size
    
    @property
    def label(self):
        if self._label is None:
            self._label = 'unknown'
        return self._label

    @label.setter
    def label(self, _label):
        self._label = _label

    def corners(self, offset=[[0, 0, 0]]):
        if self._corners is not None:
            return self._corners + offset
        
        # calculate the corners of the bounding box
        # size is in the order of x, y, z
        size = self._size
        corners = np.array([
            [-size[0] / 2, -size[1] / 2, -size[2] / 2],
            [size[0] / 2, -size[1] / 2, -size[2] / 2],
            [size[0] / 2, size[1] / 2, -size[2] / 2],
            [-size[0] / 2, size[1] / 2, -size[2] / 2],
            [-size[0] / 2, -size[1] / 2, size[2] / 2],
            [size[0] / 2, -size[1] / 2, size[2] / 2],
            [size[0] / 2, size[1] / 2, size[2] / 2],
            [-size[0] / 2, size[1] / 2, size[2] / 2]
        ])
        corners = self._centroid + corners
        return corners + offset

    def one_hot_label(self, all_labels):
        return np.eye(len(all_labels))[self.int_label(all_labels)]

    def int_label(self, all_labels):
        return all_labels.index(self.label)

class BaseScene(object):
    """Contains all the information for a scene."""

    def __init__(self, scene_id:str, 
                 scene_type:str, 
                 walls: List[ST3dFutureModel], 
                 bboxes: List[ST3dFutureModel]):
        self.walls = walls
        self.bboxes = bboxes
        self.scene_id = scene_id
        self.scene_type = scene_type

    def __str__(self):
        return "Scene: {} of type: {} contains {} bboxes".format(self.scene_id, self.scene_type, self.nobjects)

    @property
    def nobjects(self):
        """Number of bounding boxes / objects in a Scene."""
        return len(self.bboxes) + len(self.walls)

    @property
    def object_types(self):
        """The set of object types in this scene."""
        return sorted(set([b.label for b in self.bboxes] + [w.label for w in self.walls]))

    @property
    def n_object_types(self):
        """Number of distinct objects in a Scene."""
        return len(self.object_types)

    def ordered_bboxes_with_centroid(self):
        centroids = np.array([b.centroid for b in self.bboxes])
        ordering = np.lexsort(centroids.T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes

    def ordered_bboxes_with_class_labels(self, all_labels):
        centroids = np.array([b.centroid for b in self.bboxes])
        int_labels = np.array([[b.int_label(all_labels)] for b in self.bboxes])
        ordering = np.lexsort(np.hstack([centroids, int_labels]).T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes

class BaseDataset(Dataset):
    """Implements the interface for all datasets that consist of scenes."""

    def __init__(self, scenes):
        assert len(scenes) > 0
        self.scenes = scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        return self.scenes[idx]

    @property
    def class_labels(self):
        raise NotImplementedError()

    @property
    def n_classes(self):
        return len(self.class_labels)

    @property
    def object_types(self):
        raise NotImplementedError()

    @property
    def n_object_types(self):
        """The number of distinct objects contained in the scenes."""
        return len(self.object_types)

    @property
    def room_types(self):
        return set([si.scene_type for si in self.scenes])

    @property
    def count_objects_in_rooms(self):
        return Counter([len(si.bboxes) for si in self.scenes])

    def post_process(self, s):
        return s

    @staticmethod
    def with_valid_scene_ids(invalid_scene_ids):

        def inner(scene):
            return scene if scene.scene_id not in invalid_scene_ids else False

        return inner

    @staticmethod
    def with_scene_ids(scene_ids):

        def inner(scene):
            return scene if scene.scene_id in scene_ids else False

        return inner

    @staticmethod
    def with_room(scene_type):

        def inner(scene):
            return scene if scene_type in scene.scene_type else False

        return inner

    @staticmethod
    def room_smaller_than_along_axis(max_size, axis=1):

        def inner(scene):
            return scene if scene.scene_bbox[1][axis] <= max_size else False

        return inner

    @staticmethod
    def room_larger_than_along_axis(min_size, axis=1):

        def inner(scene):
            return scene if scene.scene_bbox[0][axis] >= min_size else False

        return inner

    @staticmethod
    def walls_num_with_limits(min_wall_num=4, max_wall_num=10):

        def inner(scene):
            wall_num = len(scene.quad_walls)
            if min_wall_num <= wall_num and wall_num <= max_wall_num:
                return scene
            else:
                False

        return inner

    @staticmethod
    def with_valid_boxes(box_types):

        def inner(scene):
            for i in range(len(scene.bboxes) - 1, -1, -1):
                if scene.bboxes[i].label not in box_types:
                    scene.bboxes.pop(i)
            return scene

        return inner

    @staticmethod
    def without_box_types(box_types):

        def inner(scene):
            for i in range(len(scene.bboxes) - 1, -1, -1):
                if scene.bboxes[i].label in box_types:
                    scene.bboxes.pop(i)
            return scene

        return inner

    @staticmethod
    def with_generic_classes(box_types_map):

        def inner(scene):
            for box in scene.bboxes:
                # Update the box label based on the box_types_map
                box.label = box_types_map[box.label]
            return scene

        return inner

    @staticmethod
    def with_valid_bbox_jids(invalid_bbox_jds):

        def inner(scene):
            return (False if any(b.model_jid in invalid_bbox_jds for b in scene.bboxes) else scene)

        return inner

    @staticmethod
    def at_most_boxes(n):

        def inner(scene):
            return scene if len(scene.bboxes) <= n else False

        return inner

    @staticmethod
    def at_least_boxes(n):

        def inner(scene):
            return scene if len(scene.bboxes) >= n else False

        return inner

    @staticmethod
    def with_object_types(objects):

        def inner(scene):
            return (scene if all(b.label in objects for b in scene.bboxes) else False)

        return inner

    @staticmethod
    def contains_object_types(objects):

        def inner(scene):
            return (scene if any(b.label in objects for b in scene.bboxes) else False)

        return inner

    @staticmethod
    def contains_doors():

        def inner(scene):
            return (scene if len(scene.doors) else False)

        return inner

    @staticmethod
    def without_object_types(objects):

        def inner(scene):
            return (False if any(b.label in objects for b in scene.bboxes) else scene)

        return inner

    @staticmethod
    def filter_compose(*filters):

        def inner(scene):
            s = scene
            fs = iter(filters)
            try:
                while s:
                    s = next(fs)(s)
            except StopIteration:
                pass
            return s

        return inner


class St3dRoom(BaseScene):
    def __init__(self, scene_id: str, scene_type: str, 
                 walls_lst: List[Dict], 
                 bboxes_lst: List[Dict]):
        walls = []
        for wall_dct in walls_lst:
            wall = ST3dFutureModel(centroid=wall_dct['center'], 
                                   rotation=wall_dct['angles'][-1], 
                                   size=[wall_dct['width'], 0.01, wall_dct['height']], 
                                   class_name=wall_dct['class'],
                                   normal=wall_dct['normal'],
                                   corners=wall_dct['corners'])
            walls.append(wall)
        bboxes = []
        for bbox_dct in bboxes_lst:
            bbox = ST3dFutureModel(centroid=bbox_dct['center'], 
                                   rotation=bbox_dct['angles'][-1], 
                                   size=bbox_dct['size'], 
                                   class_name=bbox_dct['class'])
            bboxes.append(bbox)
        super().__init__(scene_id, scene_type, walls, bboxes)
        
    @cached_property
    def wall_meshes(self) -> trimesh.Trimesh:
        wall_meshes_lst = []
        for w in self.walls:
            wall_corners = w.corners()
            wall_normal = w.normal
            quad_wall_mesh, quad_wall_ply = create_spatial_quad_polygen(wall_corners, wall_normal, None)
            wall_meshes_lst.append(copy.deepcopy(quad_wall_mesh))
        return trimesh.util.concatenate(wall_meshes_lst)
    
    @cached_property
    def bbox_meshes(self) -> trimesh.Trimesh:
        bbox_meshes_lst = []
        for b in self.bboxes:
            bbox_corners = b.corners()
            bbox_mesh = trimesh.primitives.Box(b.size, b.centroid)
            bbox_meshes_lst.append(copy.deepcopy(bbox_mesh))
        return bbox_meshes_lst
    
    @property
    @lru_cache(maxsize=512)
    def scene_bbox(self):
        """get the room bounding box """
        assert self.wall_meshes is not None
        corners = trimesh.bounds.corners(self.wall_meshes.bounding_box_oriented.bounds)
        return corners.min(axis=0), corners.max(axis=0)

    @cached_property
    def scene_bbox_centroid(self):
        a, b = self.scene_bbox
        return (a + b) / 2

    @cached_property
    def scene_bbox_size(self):
        a, b = self.scene_bbox
        return b - a

    @property
    def furniture_in_room(self):
        furniture_label_lst = [f.label for f in self.bboxes]
        wall_label_lst = ['wall' for w in self.walls]
        # print(f'furniture_label_lst: {furniture_label_lst + door_label_lst + window_label_lst + wall_label_lst}')
        return furniture_label_lst + wall_label_lst


    @cached_property
    def centroid(self):
        return self.scene_bbox_centroid

    @property
    def count_furniture_in_room(self):
        return Counter(self.furniture_in_room)


    def category_counts(self, class_labels):
        """List of category counts in the room
        """
        print(class_labels)
        if "start" in class_labels and "end" in class_labels:
            class_labels = class_labels[:-2]
        category_counts = [0] * len(class_labels)

        for di in self.furniture_in_room:
            category_counts[class_labels.index(di)] += 1
        return category_counts

    def ordered_bboxes_with_centroid(self):
        furniture_centroids = np.array([f.centroid(-self.centroid) for f in self.bboxes])
        wall_centroids = np.array([w['center'] - self.centroid for w in self.walls])
        centroids = np.vstack([furniture_centroids, wall_centroids])
        print(f'all furniture centroids: {centroids.shape}')
        ordering = np.lexsort(centroids.T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes

    def ordered_bboxes_with_class_frequencies(self, class_order):
        centroids = np.array([f.centroid(-self.centroid) for f in self.bboxes])
        # wall_centroids = np.array([w['center'] - self.centroid for w in self.quad_walls])
        # door_centroids = np.array([d['center'] - self.centroid for d in self.doors])
        # window_centroids = np.array([w['center'] - self.centroid for w in self.windows])
        # centroids = np.vstack([furniture_centroids, wall_centroids, door_centroids, window_centroids])
        label_order = np.array([[class_order[f.label]] for f in self.bboxes])
        ordering = np.lexsort(np.hstack([centroids, label_order]).T)
        ordered_bboxes = [self.bboxes[i] for i in ordering[::-1]]

        return ordered_bboxes


class ST3DDataset(BaseDataset):
    """Container for the scenes in the ST3D dataset.

        Arguments
        ---------
        scenes: list of St3dRoom objects for all scenes in ST3D dataset
    """

    def __init__(self, scenes: List[St3dRoom]):
        super().__init__(scenes)
        # verify that all scenes are of type St3dRoom
        assert isinstance(self.scenes[0], St3dRoom)
        self._object_types = None
        self._room_types = None
        self._count_furniture = None
        self._bbox = None

        self._sizes = self._centroids = self._angles = None

        self._max_wall_length = None
        self._max_furniture_length = None

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(len(self.scenes), self.n_object_types)

    @property
    def bbox(self):
        """The bbox for the entire dataset is simply computed based on the
        bounding boxes of all scenes in the dataset.
        """
        if self._bbox is None:
            _bbox_min = np.array([1000, 1000, 1000])
            _bbox_max = np.array([-1000, -1000, -1000])
            for s in self.scenes:
                # furniture/door/window bounding boxes
                bbox_min, bbox_max = s.bbox
                _bbox_min = np.minimum(bbox_min, _bbox_min)
                _bbox_max = np.maximum(bbox_max, _bbox_max)
            self._bbox = (_bbox_min, _bbox_max)
        return self._bbox

    def _centroid(self, box: ST3dFutureModel, offset: np.array):
        tr_centroid = box.centroid(offset)
        if np.any(np.isnan(tr_centroid)):
            print(f'ThreedFront: _centroid: box_centroid: {tr_centroid}, scene_centroid: {offset}')
        return tr_centroid

    def _size(self, box: ST3dFutureModel):
        return box.size

    def _compute_bounds(self):
        _size_min = np.array([10000000] * 3)
        _size_max = np.array([-10000000] * 3)
        _centroid_min = np.array([10000000] * 3)
        _centroid_max = np.array([-10000000] * 3)
        _angle_min = np.array([10000000000])
        _angle_max = np.array([-10000000000])
        for s in self.scenes:
            s: St3dRoom
            # print(f'********************* s.scene_id: {s.scene_id} *********************')

            # normalize furnitures
            for f in s.bboxes:
                f: ST3dFutureModel
                # if np.any(f.size > 5):
                #     print(s.scene_id, f.size, f.model_uid, f.scale)

                # print(f'********************* f.label: {f.label} *********************')

                # normalize the centroid and size by the room size
                centroid = self._centroid(f, -s.centroid)
                _centroid_min = np.minimum(centroid, _centroid_min)
                _centroid_max = np.maximum(centroid, _centroid_max)
                _size_min = np.minimum(self._size(f), _size_min)
                _size_max = np.maximum(self._size(f), _size_max)
                _angle_min = np.minimum(f.z_angle, _angle_min)
                _angle_max = np.maximum(f.z_angle, _angle_max)
            # normalize door/window and walls
            for w in s.walls:
                w: ST3dFutureModel
                # print(f'********************* e.label: {w.label} *********************')

                centroid = self._centroid(w, -s.centroid)
                _centroid_min = np.minimum(centroid, _centroid_min)
                _centroid_max = np.maximum(centroid, _centroid_max)
                _size_min = np.minimum(self._size(w), _size_min)
                _size_max = np.maximum(self._size(w), _size_max)
                _angle_min = np.minimum(w.z_angle, _angle_min)
                _angle_max = np.maximum(w.z_angle, _angle_max)

        self._sizes = (_size_min, _size_max)
        self._centroids = (_centroid_min, _centroid_max)
        self._angles = (_angle_min, _angle_max)

    @property
    def bounds(self):
        return {"translations": self.centroids, "sizes": self.sizes, "angles": self.angles}

    @property
    def sizes(self):
        if self._sizes is None:
            self._compute_bounds()
        return self._sizes

    @property
    def centroids(self):
        if self._centroids is None:
            self._compute_bounds()
        return self._centroids

    @property
    def angles(self):
        if self._angles is None:
            self._compute_bounds()
        return self._angles

    @property
    def count_furniture(self):
        if self._count_furniture is None:
            counts = []
            for s in self.scenes:
                s: St3dRoom
                counts.append(s.furniture_in_room)
            counts = Counter(sum(counts, []))
            counts = OrderedDict(sorted(counts.items(), key=lambda x: -x[1]))
            self._count_furniture = counts
        return self._count_furniture

    @property
    def class_order(self):
        return dict(zip(self.count_furniture.keys(), range(len(self.count_furniture))))

    @property
    def class_frequencies(self):
        object_counts = self.count_furniture
        class_freq = {}
        n_objects_in_dataset = sum([object_counts[k] for k, v in object_counts.items()])
        for k, v in object_counts.items():
            class_freq[k] = object_counts[k] / n_objects_in_dataset
        return class_freq

    @property
    def object_types(self):
        if self._object_types is None:
            self._object_types = set()
            for s in self.scenes:
                s: St3dRoom
                self._object_types |= set(s.object_types)
            self._object_types = sorted(self._object_types)
        return self._object_types

    @property
    def room_types(self):
        if self._room_types is None:
            self._room_types = set([s.scene_type for s in self.scenes])
        return self._room_types

    @property
    def class_labels(self):
        # return self.object_types
        # add empty class, to be able to use the same class_code for all datasets
        return self.object_types + ["empty"]

    # compute max_lenght for diffusion models
    @property
    def max_wall_length(self):
        if self._max_wall_length is None:
            _room_types = set([str(s.scene_type) for s in self.scenes])
            if 'bed' in _room_types:
                self._max_wall_length = 10
            elif 'living' in _room_types:
                self._max_wall_length = 20
            elif 'dining' in _room_types:
                self._max_wall_length = 20
            elif 'kitchen' in _room_types:
                self._max_wall_length = 10

        return self._max_wall_length

    @property
    def max_furniture_length(self):
        if self._max_furniture_length is None:
            _room_types = set([str(s.scene_type) for s in self.scenes])
            if 'bed' in _room_types:
                self._max_furniture_length = 13
            elif 'living' in _room_types:
                self._max_furniture_length = 24
            elif 'dining' in _room_types:
                self._max_furniture_length = 24
            elif 'kitchen' in _room_types:
                self._max_furniture_length = 24

        return self._max_furniture_length

    @classmethod
    def from_dataset_directory(cls,
                               scenes: List[St3dRoom],
                               filter_fn=lambda s: s):
        filtered_scenes = []
        for s in map(filter_fn, scenes):
            if not s:
                continue
            filtered_scenes.append(s)

        return cls(filtered_scenes)
