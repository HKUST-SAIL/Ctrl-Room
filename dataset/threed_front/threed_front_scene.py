#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

from collections import Counter
from dataclasses import dataclass
from functools import cached_property, reduce, lru_cache
import json
import os
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

import trimesh
import open3d as o3d

from .base import BaseScene
from utils.utils import check_mesh_attachment

# from simple_3dviz import Lines, Mesh, Spherecloud
# from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
# from simple_3dviz.behaviours.keyboard import SnapshotOnKey
# from simple_3dviz.behaviours.misc import LightToCamera
# try:
#     from simple_3dviz.window import show
# except ImportError:
#     import sys
#     print("No GUI library found. Simple-3dviz will be running headless only.", file=sys.stderr)


def matrix_to_euler_angles(R_w_i: np.array):
    """
    Convert rotation matrix to euler angles in Z-X-Y order
    """
    roll = np.arcsin(R_w_i[2, 1])
    pitch = np.arctan2(-R_w_i[2, 0] / np.cos(roll), R_w_i[2, 2] / np.cos(roll))
    yaw = np.arctan2(-R_w_i[0, 1] / np.cos(roll), R_w_i[1, 1] / np.cos(roll))

    return np.array([roll, pitch, yaw])


def rotation_matrix(axis, theta):
    """Axis-angle rotation matrix from 3D-Front-Toolbox."""
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad),
                      2 * (bd - ac)], [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def get_floor_vertice(vertices: np.array):
    """get the vertices on floor of a mesh
    """
    # we assume that the floor is on y=0
    floor_vertices = vertices[vertices[:, 1] == 0.0]
    # if floor vertices is empty
    if floor_vertices.shape[0] == 0:
        return None

    # if x is not equal ,find the vertice that has minimum x
    if not np.allclose(floor_vertices[:, 0], floor_vertices[0, 0]):
        x_min = np.min(floor_vertices[:, 0])
        min_vertice = floor_vertices[floor_vertices[:, 0] == x_min][0]
        return min_vertice
    z_min = np.min(floor_vertices[:, 2])
    min_vertice = floor_vertices[floor_vertices[:, 2] == z_min][0]
    # distances = np.linalg.norm(floor_vertices, axis=1)
    # min_idx = np.argmin(distances)
    # min_vertice = floor_vertices[min_idx]
    return min_vertice


def get_ceiling_vertice(vertices: np.array):
    """get the max vertice of a mesh
    """
    ceiling_height = np.max(vertices[:, 1])
    ceiling_vertices = vertices[vertices[:, 1] == ceiling_height]
    if ceiling_vertices.shape[0] == 0:
        return None

    # if x is not equal ,find the vertice that has maximum x
    if not np.allclose(ceiling_vertices[:, 0], ceiling_vertices[0, 0]):
        x_max = np.max(ceiling_vertices[:, 0])
        max_vertice = ceiling_vertices[ceiling_vertices[:, 0] == x_max][0]
        return max_vertice
    z_max = np.max(ceiling_vertices[:, 2])
    max_vertice = ceiling_vertices[ceiling_vertices[:, 2] == z_max][0]
    return max_vertice


def create_spatial_quad_polygen(quad_vertices: np.array, normal: np.array, camera_center: np.array):
    """create a quad polygen for spatial mesh
    """
    if camera_center is None:
        camera_center = np.array([0, 0, 0])
    quad_vertices = (quad_vertices - camera_center)
    quad_triangles = []
    triangle = np.array([[0, 2, 1], [2, 0, 3]])
    quad_triangles.append(triangle)

    quad_triangles = np.concatenate(quad_triangles, axis=0)

    mesh = trimesh.Trimesh(vertices=quad_vertices,
                           faces=quad_triangles,
                           vertex_normals=np.tile(normal, (4, 1)),
                           process=False)

    centroid = np.mean(quad_vertices, axis=0)
    # print(f'centroid: {centroid}')
    normal_point = centroid + np.array(normal) * 0.5
    # print(f'normal_point: {normal_point}')

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    pcd_o3d.points.append(normal_point)
    pcd_o3d.points.append(centroid)

    # pcd = trimesh.PointCloud(np.asarray(pcd.points))
    return mesh, pcd_o3d


def cat_mesh(m1, m2):
    v1, f1 = m1
    v2, f2 = m2
    v = np.vstack([v1, v2])
    f = np.vstack([f1, f2 + len(v1)])
    return v, f


@dataclass
class Asset:
    """Contains the information for each 3D-FUTURE model."""
    super_category: str
    category: str
    style: str
    theme: str
    material: str

    @property
    def label(self):
        return self.category


class ModelInfo(object):
    """Contains all the information for all 3D-FUTURE models.

        Arguments
        ---------
        model_info_data: list of dictionaries containing the information
                         regarding the 3D-FUTURE models.
    """

    def __init__(self, model_info_data):
        self.model_info_data = model_info_data
        self._model_info = None
        # List to keep track of the different styles, themes
        self._styles = []
        self._themes = []
        self._categories = []
        self._super_categories = []
        self._materials = []

    @property
    def model_info(self):
        if self._model_info is None:
            self._model_info = {}
            # Create a dictionary of all models/assets in the dataset
            for m in self.model_info_data:
                # Keep track of the different styles
                if m["style"] not in self._styles and m["style"] is not None:
                    self._styles.append(m["style"])
                # Keep track of the different themes
                if m["theme"] not in self._themes and m["theme"] is not None:
                    self._themes.append(m["theme"])
                # Keep track of the different super-categories
                if m["super-category"] not in self._super_categories and m["super-category"] is not None:
                    self._super_categories.append(m["super-category"])
                # Keep track of the different categories
                if m["category"] not in self._categories and m["category"] is not None:
                    self._categories.append(m["category"])
                # Keep track of the different categories
                if m["material"] not in self._materials and m["material"] is not None:
                    self._materials.append(m["material"])

                super_cat = "unknown_super-category"
                cat = "unknown_category"

                if m["super-category"] is not None:
                    super_cat = m["super-category"].lower().replace(" / ", "/")

                if m["category"] is not None:
                    cat = m["category"].lower().replace(" / ", "/")

                self._model_info[m["model_id"]] = Asset(super_cat, cat, m["style"], m["theme"], m["material"])

        return self._model_info

    @property
    def styles(self):
        return self._styles

    @property
    def themes(self):
        return self._themes

    @property
    def materials(self):
        return self._materials

    @property
    def categories(self):
        return set([s.lower().replace(" / ", "/") for s in self._categories])

    @property
    def super_categories(self):
        return set([s.lower().replace(" / ", "/") for s in self._super_categories])

    @classmethod
    def from_file(cls, path_to_model_info):
        with open(path_to_model_info, "rb") as f:
            model_info = json.load(f)

        return cls(model_info)


class BaseThreedFutureModel(object):

    def __init__(self, model_uid, model_jid, position, rotation, scale):
        self.model_uid = model_uid
        self.model_jid = model_jid
        self.position = position
        self.rotation = rotation
        self.scale = scale

    def _transform(self, vertices):
        # the following code is adapted and slightly simplified from the
        # 3D-Front toolbox (json2obj.py). It basically scales, rotates and
        # translates the model based on the model info.
        ref = [0, 0, 1]
        # use rotation[1:] ?
        axis = np.cross(ref, self.rotation[1:])
        theta = np.arccos(np.dot(ref, self.rotation[1:])) * 2
        vertices = vertices * self.scale
        if np.sum(axis) != 0 and not np.isnan(theta):
            R = rotation_matrix(axis, theta)
            vertices = vertices.dot(R.T)
        vertices += self.position

        return vertices

    def mesh_renderable(self, colors=(0.5, 0.5, 0.5, 1.0), offset=[[0, 0, 0]], with_texture=False):
        if not with_texture:
            m = self.raw_model_transformed(offset)
            return Mesh.from_faces(m.vertices, m.faces, colors=colors)
        else:
            m = TexturedMesh.from_file(self.raw_model_path)
            m.scale(self.scale)
            # Extract the predicted affine transformation to position the
            # mesh
            theta = self.z_angle
            R = np.zeros((3, 3))
            R[0, 0] = np.cos(theta)
            R[0, 2] = -np.sin(theta)
            R[2, 0] = np.sin(theta)
            R[2, 2] = np.cos(theta)
            R[1, 1] = 1.

            # Apply the transformations in order to correctly position the mesh
            m.affine_transform(R=R, t=self.position)
            m.affine_transform(t=offset)
            return m


class ThreedFutureModel(BaseThreedFutureModel):

    def __init__(self, model_uid, model_jid, model_info, position, rotation, scale, path_to_models):
        super().__init__(model_uid, model_jid, position, rotation, scale)
        self.model_info = model_info
        self.path_to_models = path_to_models
        self._label = None

    @property
    def raw_model_path(self):
        return os.path.join(self.path_to_models, self.model_jid, "raw_model.obj")

    @property
    def texture_image_path(self):
        return os.path.join(self.path_to_models, self.model_jid, "texture.png")

    @property
    def path_to_bbox_vertices(self):
        return os.path.join(self.path_to_models, self.model_jid, "bbox_vertices.npy")

    def raw_model(self):
        try:
            return trimesh.load(self.raw_model_path,
                                process=False,
                                force="mesh",
                                skip_materials=True,
                                skip_texture=True)
        except:
            import pdb
            pdb.set_trace()
            print(f"Loading model {self.raw_model_path} failed", flush=True)
            # print(self.raw_model_path, flush=True)
            raise

    def raw_model_transformed(self, offset=[[0, 0, 0]]):
        model = self.raw_model()
        faces = np.array(model.faces)
        vertices = self._transform(np.array(model.vertices)) + offset

        return trimesh.Trimesh(vertices, faces)

    def centroid(self, offset=[[0, 0, 0]]):
        return self.corners(offset).mean(axis=0)

    @cached_property
    def size(self):
        corners = self.corners()
        # TODO: This is not the correct way to compute the size of the object.
        return np.array([
            np.sqrt(np.sum((corners[4] - corners[0])**2)) / 2,
            np.sqrt(np.sum((corners[2] - corners[0])**2)) / 2,
            np.sqrt(np.sum((corners[1] - corners[0])**2)) / 2
        ])

    def bottom_center(self, offset=[[0, 0, 0]]):
        centroid = self.centroid(offset)
        size = self.size
        return np.array([centroid[0], centroid[1] - size[1], centroid[2]])

    @cached_property
    def bottom_size(self):
        return self.size * [1, 2, 1]

    @cached_property
    def z_angle(self):
        # See BaseThreedFutureModel._transform for the origin of the following
        # code.
        ref = [0, 0, 1]
        axis = np.cross(ref, self.rotation[1:])
        theta = np.arccos(np.dot(ref, self.rotation[1:])) * 2

        if np.sum(axis) == 0 or np.isnan(theta):
            return 0

        assert np.dot(axis, [1, 0, 1]) == 0
        assert 0 <= theta <= 2 * np.pi

        if theta >= np.pi:
            theta = theta - 2 * np.pi

        return np.sign(axis[1]) * theta

    @property
    def label(self):
        if self._label is None:
            self._label = self.model_info.label
        return self._label

    @label.setter
    def label(self, _label):
        self._label = _label

    def corners(self, offset=[[0, 0, 0]]):
        try:
            bbox_vertices = np.load(self.path_to_bbox_vertices, mmap_mode="r")
        except:
            bbox_vertices = np.array(self.raw_model().bounding_box.vertices)
            if np.isnan(bbox_vertices).any():
                print(f"model {self.raw_model_path} contains NaN vertices", flush=True)
            np.save(self.path_to_bbox_vertices, bbox_vertices)
        c = self._transform(bbox_vertices)
        if np.isnan(c).any():
            print(f"model {self.raw_model_path} contains NaN vertices", flush=True)

        return c + offset

    def origin_renderable(self, offset=[[0, 0, 0]]):
        corners = self.corners(offset)
        return Lines([corners[0], corners[4], corners[0], corners[2], corners[0], corners[1]],
                     colors=np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0],
                                      [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
                     width=0.02)

    def bbox_corners_renderable(self, sizes=0.1, colors=(1, 0, 0), offset=[[0, 0, 0]]):
        return Spherecloud(self.corners(offset), sizes=sizes, colors=colors)

    def bbox_renderable(self, colors=(0.00392157, 0., 0.40392157, 1.), offset=[[0, 0, 0]]):
        alpha = np.array(self.size)[None]
        epsilon = np.ones((1, 2)) * 0.1
        translation = np.array(self.centroid(offset))[None]
        R = np.zeros((1, 3, 3))
        theta = np.array(self.z_angle)
        R[:, 0, 0] = np.cos(theta)
        R[:, 0, 2] = -np.sin(theta)
        R[:, 2, 0] = np.sin(theta)
        R[:, 2, 2] = np.cos(theta)
        R[:, 1, 1] = 1.

        return Mesh.from_superquadrics(alpha, epsilon, translation, R, colors)

    # def show(self, behaviours=[LightToCamera()], with_bbox_corners=False, offset=[[0, 0, 0]]):
    #     renderables = self.mesh_renderable(offset=offset)
    #     if with_bbox_corners:
    #         renderables += [self.bbox_corners_renderable(offset=offset)]
    #     show(renderables, behaviours=behaviours)

    def one_hot_label(self, all_labels):
        return np.eye(len(all_labels))[self.int_label(all_labels)]

    def int_label(self, all_labels):
        return all_labels.index(self.label)

    def copy_from_other_model(self, other_model):
        model = ThreedFutureModel(model_uid=other_model.model_uid,
                                  model_jid=other_model.model_jid,
                                  model_info=other_model.model_info,
                                  position=self.position,
                                  rotation=self.rotation,
                                  scale=other_model.scale,
                                  path_to_models=self.path_to_models)
        model.label = self.label
        return model


class ThreedFutureExtra(BaseThreedFutureModel):

    def __init__(self, model_uid, model_jid, xyz, faces, normals, model_type, position, rotation, scale):
        super().__init__(model_uid, model_jid, position, rotation, scale)
        self.xyz = xyz
        self.faces = faces
        self.normals = normals
        self.model_type = model_type
        self._label = None
        self._z_angle = None

    def raw_model_transformed(self, offset=[[0, 0, 0]]):
        vertices = self._transform(np.array(self.xyz)) + offset
        faces = np.array(self.faces)
        return trimesh.Trimesh(vertices, faces)

    # def show(self, behaviours=[LightToCamera(), SnapshotOnKey()], offset=[[0, 0, 0]]):
    #     renderables = self.mesh_renderable(offset=offset)
    #     show(renderables, behaviours)

    def centroid(self, offset=[[0, 0, 0]]):
        if self.label == 'wall':
            return np.mean(self.xyz, axis=0) + offset
        else:
            return self.corners(offset).mean(axis=0)

    @cached_property
    def size(self):
        # half size
        if self.label == 'wall':
            return np.array([
                (np.max(self.xyz[:, 0]) - np.min(self.xyz[:, 0])),  # X
                (np.max(self.xyz[:, 1]) - np.min(self.xyz[:, 1])),  # Y
                (np.max(self.xyz[:, 2]) - np.min(self.xyz[:, 2]))
            ])  # Z
        else:
            corners = self.corners()
            # TODO: This is not the correct way to compute the size of the object.
            return np.array([
                np.sqrt(np.sum((corners[2] - corners[0])**2)),  # X
                np.sqrt(np.sum((corners[1] - corners[0])**2)),  # Y
                np.sqrt(np.sum((corners[4] - corners[0])**2))  # Z
            ])

    @property
    def z_angle(self):
        if self._z_angle is None:
            ref = [0, 0, 1]
            axis = np.cross(ref, self.rotation[1:])
            theta = np.arccos(np.dot(ref, self.rotation[1:])) * 2

            if np.sum(axis) == 0 or np.isnan(theta):
                return 0

            assert np.dot(axis, [1, 0, 1]) == 0
            assert 0 <= theta <= 2 * np.pi

            if theta >= np.pi:
                theta = theta - 2 * np.pi
            self._z_angle = np.sign(axis[1]) * theta
        return self._z_angle

    @z_angle.setter
    def z_angle(self, _z_angle):
        self._z_angle = _z_angle

    @property
    def label(self):
        if self._label is None:
            self._label = self.model_type
        return self._label

    @label.setter
    def label(self, _label):
        self._label = _label

    def corners(self, offset=[[0, 0, 0]]):
        if self.label == 'wall':
            bbox_vertices = self.xyz
        else:
            bbox_vertices = trimesh.Trimesh(self.xyz, self.faces).bounding_box_oriented.vertices
        c = self._transform(bbox_vertices)
        return c + offset

    def origin_renderable(self, offset=[[0, 0, 0]]):
        corners = self.corners(offset)
        return Lines([corners[0], corners[4], corners[0], corners[2], corners[0], corners[1]],
                     colors=np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0],
                                      [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
                     width=0.02)

    def bbox_corners_renderable(self, sizes=0.1, colors=(1, 0, 0), offset=[[0, 0, 0]]):
        return Spherecloud(self.corners(offset), sizes=sizes, colors=colors)

    def bbox_renderable(self, colors=(0.00392157, 0., 0.40392157, 1.), offset=[[0, 0, 0]]):
        alpha = np.array(self.size)[None]
        epsilon = np.ones((1, 2)) * 0.1
        translation = np.array(self.centroid(offset))[None]
        R = np.zeros((1, 3, 3))
        theta = np.array(self.z_angle)
        R[:, 0, 0] = np.cos(theta)
        R[:, 0, 2] = -np.sin(theta)
        R[:, 2, 0] = np.sin(theta)
        R[:, 2, 2] = np.cos(theta)
        R[:, 1, 1] = 1.

        return Mesh.from_superquadrics(alpha, epsilon, translation, R, colors)

    # def show(self, behaviours=[LightToCamera()], with_bbox_corners=False, offset=[[0, 0, 0]]):
    #     renderables = self.mesh_renderable(offset=offset)
    #     if with_bbox_corners:
    #         renderables += [self.bbox_corners_renderable(offset=offset)]
    #     show(renderables, behaviours=behaviours)

    def one_hot_label(self, all_labels):
        return np.eye(len(all_labels))[self.int_label(all_labels)]

    def int_label(self, all_labels):
        return all_labels.index(self.label)


import copy


class Room(BaseScene):

    def __init__(self,
                 scene_id: str,
                 scene_type: str,
                 bboxes: List[ThreedFutureModel],
                 extras: List[ThreedFutureExtra],
                 json_path: str,
                 path_to_room_masks_dir: str = None,
                 doors_dict: Dict = None,
                 windows_dict: Dict = None):
        """_summary_

        Args:
            scene_id (str): room id
            scene_type (str): room type[bedroom, livingroom, bathroom, kitchen, diningroom, library]
            bboxes (List[ThreedFutureModel]): oobject bounding boxes
            extras (List[ThreedFutureExtra]): extra meshes, e.g. walls
            json_path (str): scene json description path
            path_to_room_masks_dir (str, optional): room mask image directory. Defaults to None.
            doors_dict (Dict, optional): doors in the house, key is door uid, value is mesh vertices and faces. Defaults to None.
            windows_dict (Dict, optional): windows in the house. Defaults to None.
        """
        super().__init__(scene_id, scene_type, bboxes)
        self.json_path = json_path
        self.extras = extras
        self.doors_dict = doors_dict
        self.windows_dict = windows_dict

        self.uid = "_".join([self.json_path, scene_id])
        self.path_to_room_masks_dir = path_to_room_masks_dir
        if path_to_room_masks_dir is not None:
            self.path_to_room_mask = os.path.join(self.path_to_room_masks_dir, self.uid, "room_mask.png")
        else:
            self.path_to_room_mask = None

        self._original_walls = None
        self._quad_walls = None
        self._windows = None
        self._doors = None

    @property
    def floor(self):
        return [ei for ei in self.extras if ei.model_type == "Floor"][0]

    @cached_property
    def wall_meshes(self) -> trimesh.Trimesh:
        if self._original_walls is None:
            wall_lst = [ei for ei in self.extras if ei.model_type == "WallInner"]
            if len(wall_lst) == 0:
                print(f"room {self.uid} has no wall!!!!")
            # walls_mesh = o3d.geometry.TriangleMesh()
            walls_mesh_lst = []
            for wall in wall_lst:
                # print(f'wall_id: {wall.model_uid}, wall_faces:{wall.faces.shape}, wall_vertices:{wall.xyz.shape}')
                mesh = trimesh.Trimesh(vertices=wall.xyz, faces=wall.faces, vertex_normals=wall.normals)
                walls_mesh_lst.append(mesh)
            if len(walls_mesh_lst):
                self._original_walls = trimesh.util.concatenate(walls_mesh_lst)
        return self._original_walls

    @property
    def quad_walls(self):

        def is_vertical_wall(wall_normal):
            return (abs(abs(wall_normal[0]) - 1) < 1e-6 or abs(abs(wall_normal[2]) - 1) < 1e-6) and abs(
                wall_normal[1]) < 1e-6

        if self._quad_walls is None:
            # calculate wall height
            scene_bbox_min, scene_bbox_max = self.scene_bbox
            if scene_bbox_min is None or scene_bbox_max is None:
                return list()

            scene_bbox_centroid = (scene_bbox_min + scene_bbox_max) / 2
            wall_height = scene_bbox_max[1] - scene_bbox_min[1]

            floor_plan_vertices, floor_plan_faces = self.floor_plan
            if floor_plan_faces is None or floor_plan_vertices is None:
                return list()

            floor_mesh = trimesh.Trimesh(vertices=floor_plan_vertices, faces=floor_plan_faces)
            # detect if vertice lay on the contour of the floor plan: https://github.com/mikedh/trimesh/issues/1060
            # an edge which occurs only once is on the boundary
            unique_edges = floor_mesh.edges[trimesh.grouping.group_rows(floor_mesh.edges_sorted, require_count=1)]
            if len(unique_edges) == 0:
                return list()

            quad_wall_lst = []
            for i, edge in enumerate(unique_edges):
                v0 = floor_mesh.vertices[edge[0]]
                v1 = floor_mesh.vertices[edge[1]]
                quad_wall_corners = np.array([v0, v0 + [0, wall_height, 0], v1 + [0, wall_height, 0], v1])
                quad_wall_faces = np.array([[0, 2, 1], [2, 0, 3]])
                normals = np.cross(quad_wall_corners[1] - quad_wall_corners[0],
                                   quad_wall_corners[2] - quad_wall_corners[0])
                normals = normals / np.linalg.norm(normals)

                quad_wall_mesh = trimesh.Trimesh(vertices=quad_wall_corners,
                                                 faces=quad_wall_faces,
                                                 vertex_normals=np.tile(normals, (4, 1)))
                # if self.uid == '01805656-e66f-44b1-8bc1-5e722fff3fff_Bedroom-8715':
                #     print(f'wall {i} normal: {normals}')
                #     quad_wall_mesh.export(
                #         os.path.join(
                #             '/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/01805656-e66f-44b1-8bc1-5e722fff3fff_Bedroom-8715/',
                #             f"quad_wall_from_floor_{i}.ply"))
                quad_wall_lst.append(quad_wall_mesh)

            merged_wall_ids = []
            for i in range(len(quad_wall_lst)):
                if i in merged_wall_ids:
                    continue

                mesh_0 = quad_wall_lst[i]
                normal_0 = mesh_0.vertex_normals[0, :]

                for j in range(i + 1, len(quad_wall_lst)):
                    mesh_1 = quad_wall_lst[j]
                    normal_1 = mesh_1.vertex_normals[0, :]
                    # if self.uid == '01805656-e66f-44b1-8bc1-5e722fff3fff_Bedroom-8715':
                    #     print(f'wall {i} normal: {normal_0}, wall {j} normal: {normal_1}')
                    # if two walls have common corners and normals are the same
                    if np.allclose(np.abs(normal_0), np.abs(normal_1), atol=1e-2) and check_mesh_attachment(
                            mesh_0, mesh_1):
                        if self.uid == '01805656-e66f-44b1-8bc1-5e722fff3fff_Bedroom-8715':
                            print(f'wall {i} and wall {j} are merged, normal: {normal_0}')

                        # merge two wall
                        mesh_vertices, mesh_faces = cat_mesh((mesh_0.vertices, mesh_0.faces),
                                                             (mesh_1.vertices, mesh_1.faces))
                        # mesh_0 = trimesh.util.concatenate([mesh_0, mesh_1])
                        mesh_0 = trimesh.Trimesh(vertices=mesh_vertices,
                                                 faces=mesh_faces,
                                                 vertex_normals=np.tile(normal_0, (len(mesh_vertices), 1)))
                        quad_wall_lst[i] = mesh_0
                        merged_wall_ids.append(j)

                if self.uid == '01805656-e66f-44b1-8bc1-5e722fff3fff_Bedroom-8715':
                    print(f'merged_wall_ids:\n {merged_wall_ids}')
            walls_lst = []
            idx = 0
            for id in range(len(quad_wall_lst)):
                if id in merged_wall_ids:
                    continue

                mesh = quad_wall_lst[id]
                if is_vertical_wall(mesh.vertex_normals[0, :]):
                    min_corner = np.min(mesh.vertices, axis=0)
                    max_corner = np.max(mesh.vertices, axis=0)
                else:
                    min_corner = get_floor_vertice(mesh.vertices)
                    max_corner = get_ceiling_vertice(mesh.vertices)
                    if min_corner is None or max_corner is None:
                        print(f'room {self.uid} has invalid wall {id}')
                        return list()

                wall = {}
                wall["ID"] = 'wall_' + str(idx)
                idx += 1
                wall["class"] = 'wall'
                wall["corners"] = np.array(
                    [min_corner, min_corner + [0, wall_height, 0], max_corner, max_corner - [0, wall_height, 0]])
                wall["center"] = np.mean(wall["corners"], axis=0)
                wall["width"] = np.linalg.norm(wall["corners"][3] - wall["corners"][0])
                wall["height"] = np.linalg.norm(wall["corners"][1] - wall["corners"][0])
                wall["size"] = np.array([wall["width"], wall["height"], 0])
                normal = mesh.vertex_normals[0, :]
                sign = np.dot(normal, wall["center"] - scene_bbox_centroid)
                wall["normal"] = -normal if sign > 0 else normal
                angle = np.arccos(np.dot(wall["normal"], np.array([0, 0, 1])))
                if abs(wall["normal"][0]) == 1:
                    angle = np.pi / 2 if wall["normal"][0] == 1 else -np.pi / 2
                wall["angles"] = [0, angle, 0]
                # print(f'wall {wall["ID"]} center: {wall["center"]}, size: {wall["size"]},  angle: {wall["angles"]}')

                walls_lst.append(wall)

            self._quad_walls = walls_lst
        return self._quad_walls

    @cached_property
    def quad_wall_meshes(self) -> List[trimesh.Trimesh]:

        # print(f'room {self.uid} has {len(self.quad_walls)} quad walls')
        # wall_pcd_o3d = o3d.geometry.PointCloud()
        # for w in self.quad_walls:
        #     wall_corners = w['corners']
        #     wall_normal = w['normal']
        #     quad_wall_mesh, quad_wall_ply_o3d = create_spatial_quad_polygen(wall_corners, wall_normal, None)
        #     wall_pcd_o3d += quad_wall_ply_o3d
        # if self.uid == '01805656-e66f-44b1-8bc1-5e722fff3fff_Bedroom-8715':
        #     o3d.io.write_point_cloud(
        #         '/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/01805656-e66f-44b1-8bc1-5e722fff3fff_Bedroom-8715/quad_wall_pcl.ply',
        #         wall_pcd_o3d)

        wall_meshes_lst = []
        for w in self.quad_walls:
            wall_corners = w['corners']
            wall_normal = w['normal']
            quad_wall_mesh, quad_wall_ply = create_spatial_quad_polygen(wall_corners, wall_normal, None)
            wall_meshes_lst.append(copy.deepcopy(quad_wall_mesh))
        return wall_meshes_lst

    @property
    @lru_cache(maxsize=512)
    def windows(self):
        """get windows in the room

        Returns:
            List: a list of windows' mesh
        """
        if self._windows is None:
            # no windows in the room
            if self.windows_mesh is None:
                return list()

            window_mesh_lst = self.windows_mesh
            windows_lst = []
            for i, mesh in enumerate(window_mesh_lst):
                mesh_bbox = mesh.bounding_box_oriented
                mesh_bbos_corners = np.array(mesh_bbox.vertices)
                mesh_bbox_size = mesh_bbox.extents
                mesh_bbox_center = mesh_bbox.primitive.transform[:3, 3]
                # if np.any(np.isnan(mesh_bbox_size)) or np.any(np.isnan(mesh_bbox_center)):
                #     print(f'room {self.uid} has invalid window {i}')
                #     continue

                # rotation derived from bounding box is weird, we use the rotation from the mesh
                # R = mesh_bbox.primitive.transform[:3, :3]
                # y_angle = matrix_to_euler_angles(R)[1]
                # use X-Y plane corners to compute the normal
                normal = np.cross(mesh_bbos_corners[2] - mesh_bbos_corners[0],
                                  mesh_bbos_corners[1] - mesh_bbos_corners[0])
                normal = normal / np.linalg.norm(normal)
                sign = np.dot(normal, mesh_bbox_center - self.scene_bbox_centroid)
                normal = -normal if sign > 0 else normal
                w = {}
                w["ID"] = 'window_' + str(i)
                w['class'] = 'window'
                w["corners"] = mesh.bounding_box_oriented.vertices.tolist()
                w["center"] = mesh_bbox_center.tolist()
                w["size"] = mesh_bbox_size.tolist()
                w["normal"] = normal.tolist()
                y_angle = np.arccos(np.dot(normal, np.array([0, 0, 1])))
                if abs(normal[0]) == 1:
                    y_angle = np.pi / 2 if normal[0] == 1 else -np.pi / 2
                w["angles"] = [0, y_angle, 0]
                # print(f'window {w["ID"]} center: {w["center"]}, size: {w["size"]},  angle: {w["angles"]}')

                windows_lst.append(w)
            self._windows = windows_lst
        return self._windows

    @cached_property
    def windows_mesh(self) -> List[trimesh.Trimesh]:
        """get windows in the room

        Returns:
            List: a list of windows' mesh
        """
        window_lst = [v for k, v in self.windows_dict.items()]
        if len(window_lst) == 0:
            return list()

        room_walls = self.wall_meshes
        if room_walls is None:
            return None

        windows_mesh_lst = []
        for window in window_lst:
            mesh = trimesh.Trimesh(vertices=window["mesh_xyz"],
                                   faces=window["mesh_faces"],
                                   vertex_normals=window["mesh_normals"])

            # check if the window is in the room
            if check_mesh_attachment(mesh, room_walls):
                windows_mesh_lst.append(mesh)

        # return windows_mesh_lst if len(windows_mesh_lst) else None
        return windows_mesh_lst

    @property
    @lru_cache(maxsize=512)
    def doors(self) -> List[Dict]:
        """get doors in the room

        Returns:
            List: a list of dooors' mesh
        """
        if self._doors is None:
            if self.doors_mesh is None:
                return list()

            door_mesh_lst = self.doors_mesh
            new_doors_lst = []
            for i, mesh in enumerate(door_mesh_lst):
                mesh_bbox = mesh.bounding_box_oriented
                mesh_bbos_corners = np.array(mesh_bbox.vertices)
                mesh_bbox_size = np.array(mesh_bbox.extents)
                mesh_bbox_center = np.array(mesh_bbox.primitive.transform[:3, 3])
                if np.any(np.isnan(mesh_bbox_size)) or np.any(np.isnan(mesh_bbox_center)):
                    print(f'room {self.uid} has invalid door {i}')
                    continue

                # use X-Y plane corners to compute the normal
                normal = np.cross(mesh_bbos_corners[2] - mesh_bbos_corners[0],
                                  mesh_bbos_corners[1] - mesh_bbos_corners[0])
                normal = normal / np.linalg.norm(normal)
                sign = np.dot(normal, mesh_bbox_center - self.scene_bbox_centroid)
                normal = -normal if sign > 0 else normal
                d = {}
                d["ID"] = 'door_' + str(i)
                d['class'] = 'door'
                corners = np.array(mesh.bounding_box_oriented.vertices)
                if np.any(np.isnan(corners)):
                    print(f'room {self.uid} has invalid door {i}')
                    continue
                d["corners"] = corners.tolist()
                d["center"] = mesh_bbox_center.tolist()
                d["size"] = mesh_bbox_size.tolist()
                d["normal"] = normal.tolist()
                y_angle = np.arccos(np.dot(normal, np.array([0, 0, 1])))
                if abs(normal[0]) == 1:
                    y_angle = np.pi / 2 if normal[0] == 1 else -np.pi / 2
                d["angles"] = [0, y_angle, 0]
                # print(f'room {self.uid} door {d["ID"]} center: {d["center"]}, size: {d["size"]},  angle: {d["angles"]}')
                new_doors_lst.append(d)

            self._doors = new_doors_lst
        return self._doors

    @cached_property
    def doors_mesh(self) -> List[trimesh.Trimesh]:
        """get doors in the room

        Returns:
            List: a list of dooors' mesh
        """
        door_lst = [v for k, v in self.doors_dict.items()]
        if len(door_lst) == 0:
            return list()

        room_walls = self.wall_meshes
        if room_walls is None:
            print(f'room {self.uid} has no wall mesh')
            return list()
        doors_mesh_lst = []
        for door in door_lst:
            mesh = trimesh.Trimesh(vertices=door["mesh_xyz"], faces=door["mesh_faces"])

            # check if the window is in the room
            if check_mesh_attachment(mesh, room_walls):
                doors_mesh_lst.append(mesh)

        # in some special case, the door is splitted into two parts, we need to merge them
        new_doors_mesh_lst = []
        merged_doors_ids = []
        for i in range(len(doors_mesh_lst)):
            if i in merged_doors_ids:
                continue
            door_1 = doors_mesh_lst[i]
            for j in range(i + 1, len(doors_mesh_lst)):
                door_2 = doors_mesh_lst[j]
                if check_mesh_attachment(door_1, door_2):
                    door_1 = trimesh.util.concatenate([door_1, door_2])
                    doors_mesh_lst[i] = door_1
                    merged_doors_ids.append(j)

            new_doors_mesh_lst.append(doors_mesh_lst[i])

        # return new_doors_mesh_lst if len(new_doors_mesh_lst) else None
        return new_doors_mesh_lst

    @property
    def object_types(self):
        """The set of object types in this scene."""
        return sorted(
            set([b.label for b in self.bboxes] +
                [e.label for e in self.extras if e.label in ['wall', 'door', 'window']]))

    @property
    @lru_cache(maxsize=512)
    def scene_bbox(self):
        """get the room bounding box """
        # corners = np.empty((0, 3))
        # # furniture
        # for f in self.bboxes:
        #     corners = np.vstack([corners, f.corners()])
        # return np.min(corners, axis=0), np.max(corners, axis=0)
        if self.wall_meshes is None:
            return None, None
        else:
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
        door_label_lst = ['door' for d in self.doors]
        window_label_lst = ['window' for w in self.windows]
        wall_label_lst = ['wall' for w in self.quad_walls]
        # print(f'furniture_label_lst: {furniture_label_lst + door_label_lst + window_label_lst + wall_label_lst}')
        return furniture_label_lst + door_label_lst + window_label_lst + wall_label_lst

    @property
    def floor_plan(self):

        floor_v_f = [(ei.xyz, ei.faces) for ei in self.extras if ei.model_type == "Floor"]
        # assert len(floor_v_f), "No floor found in the scene {}".format(self.uid)
        if len(floor_v_f) == 0:
            return None, None
        # Compute the full floor plan
        vertices, faces = reduce(cat_mesh, floor_v_f)
        return np.copy(vertices), np.copy(faces)

    @cached_property
    def floor_plan_bbox(self):
        vertices, faces = self.floor_plan
        return np.min(vertices, axis=0), np.max(vertices, axis=0)

    @cached_property
    def floor_plan_centroid(self):
        a, b = self.floor_plan_bbox
        return (a + b) / 2

    @cached_property
    def centroid(self):
        # use quad walls to calculate the scene centroid
        # wall_center_lst = []
        # for quad_wall in self.quad_walls:
        #     print(f'Room::centroid: quad_wall: {quad_wall["center"]}')
        #     wall_center_lst.append(quad_wall["center"])
        # quad_wall_centroid = np.mean(wall_center_lst, axis=0)
        # print(f'Room::centroid: quad_wall_centroid: {quad_wall_centroid}')
        return self.scene_bbox_centroid
        # return self.floor_plan_centroid

    @property
    def count_furniture_in_room(self):
        return Counter(self.furniture_in_room)

    @property
    def room_mask(self):
        return self.room_mask_rotated(0)

    def room_mask_rotated(self, angle=0):
        # The angle is in rad
        im = Image.open(self.path_to_room_mask).convert("RGB")
        # Downsample the room_mask image by applying bilinear interpolation
        im = im.rotate(angle * 180 / np.pi, resample=Image.BICUBIC)

        return np.asarray(im).astype(np.float32) / np.float32(255)

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
        wall_centroids = np.array([w['center'] - self.centroid for w in self.quad_walls])
        door_centroids = np.array([d['center'] - self.centroid for d in self.doors])
        window_centroids = np.array([w['center'] - self.centroid for w in self.windows])
        centroids = np.vstack([furniture_centroids, wall_centroids, door_centroids, window_centroids])
        print(f'all furniture centroids: {centroids.shape}')
        ordering = np.lexsort(centroids.T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes

    def ordered_bboxes_with_class_labels(self, all_labels):
        furniture_centroids = np.array([f.centroid(-self.centroid) for f in self.bboxes])
        furniture_int_labels = np.array([[f.int_label(all_labels)] for f in self.bboxes])
        wall_centroids = np.array([w['center'] - self.centroid for w in self.quad_walls])
        wall_int_labels = np.array([[all_labels.index('wall')]] * len(self.quad_walls))
        door_centroids = np.array([d['center'] - self.centroid for d in self.doors])
        door_int_labels = np.array([[all_labels.index('door')]] * len(self.doors))
        window_centroids = np.array([w['center'] - self.centroid for w in self.windows])
        window_int_labels = np.array([[all_labels.index('window')]] * len(self.windows))
        centroids = np.vstack([furniture_centroids, wall_centroids, door_centroids, window_centroids])
        int_labels = np.vstack([furniture_int_labels, wall_int_labels, door_int_labels, window_int_labels])
        ordering = np.lexsort(np.hstack([centroids, int_labels]).T)
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

    def furniture_renderables(self,
                              colors=(0.5, 0.5, 0.5),
                              with_bbox_corners=False,
                              with_origin=False,
                              with_bboxes=False,
                              with_objects_offset=False,
                              with_floor_plan_offset=False,
                              with_floor_plan=False,
                              with_texture=False):
        if with_objects_offset:
            offset = -self.scene_bbox_centroid
        elif with_floor_plan_offset:
            offset = -self.floor_plan_centroid
        else:
            offset = [[0, 0, 0]]

        renderables = [f.mesh_renderable(colors=colors, offset=offset, with_texture=with_texture) for f in self.bboxes]
        renderables += [f.mesh_renderable(offset) for f in self.extras if f.model_type in ["wall", "door", "window"]]
        if with_origin:
            renderables += [f.origin_renderable(offset) for f in self.bboxes]
        if with_bbox_corners:
            for f in self.bboxes:
                renderables += [f.bbox_corners_renderable(offset=offset)]
        if with_bboxes:
            for f in self.bboxes:
                renderables += [f.bbox_renderable(offset=offset)]
        if with_floor_plan:
            vertices, faces = self.floor_plan
            vertices = vertices + offset
            renderables += [Mesh.from_faces(vertices, faces, colors=(0.8, 0.8, 0.8, 0.6))]
        return renderables

    # def show(self,
    #          behaviours=[LightToCamera(), SnapshotOnKey()],
    #          with_bbox_corners=False,
    #          with_bboxes=False,
    #          with_objects_offset=False,
    #          with_floor_plan_offset=False,
    #          with_floor_plan=False,
    #          background=(1.0, 1.0, 1.0, 1.0),
    #          camera_target=(0, 0, 0),
    #          camera_position=(-2, -2, -2),
    #          up_vector=(0, 0, 1),
    #          window_size=(512, 512)):
    #     renderables = self.furniture_renderables(with_bbox_corners=with_bbox_corners,
    #                                              with_bboxes=with_bboxes,
    #                                              with_objects_offset=with_objects_offset,
    #                                              with_floor_plan_offset=with_floor_plan_offset,
    #                                              with_floor_plan=with_floor_plan)
    #     show(renderables,
    #          behaviours=behaviours,
    #          size=window_size,
    #          camera_position=camera_position,
    #          camera_target=camera_target,
    #          up_vector=up_vector,
    #          background=background)

    def merge_extras_with_walls_doors_windows(self):

        quad_wall_lst = self.quad_walls
        quad_wall_mesh_lst = self.quad_wall_meshes
        door_lst = self.doors
        door_mesh_lst = self.doors_mesh
        window_lst = self.windows
        window_mesh_lst = self.windows_mesh
        bboxes = self.bboxes
        extras = self.extras

        for id, (quad_wall, mesh) in enumerate(zip(quad_wall_lst, quad_wall_mesh_lst)):
            wall = ThreedFutureExtra(model_uid="wall_" + str(id),
                                     model_jid="",
                                     xyz=mesh.vertices,
                                     faces=mesh.faces,
                                     normals=mesh.vertex_normals,
                                     model_type="wall",
                                     position=[0, 0, 0],
                                     rotation=[0, 0, 0, 1],
                                     scale=[1, 1, 1])
            wall.z_angle = quad_wall["angles"][1]
            extras.append(wall)

        for id, (door, mesh) in enumerate(zip(door_lst, door_mesh_lst)):
            # door_mesh = convert_oriented_box_to_trimesh_fmt(door)
            # if np.any(np.isnan(door_mesh.vertices)):
            #     print(f'Room::merge_extras_with_walls_doors_windows: room {self.uid} has invalid door {door["ID"]}')
            #     continue

            door_f = ThreedFutureExtra(
                model_uid="door_" + str(id),
                model_jid="",
                xyz=mesh.vertices,
                faces=mesh.faces,
                normals=mesh.vertex_normals,
                model_type="door",
                position=[0, 0, 0],
                rotation=[0, 0, 0, 1],  # use [0,0,1] to represent 0 degree rotation
                scale=[1, 1, 1])
            door_f.z_angle = door['angles'][1]
            extras.append(door_f)

        if window_mesh_lst is not None:
            for id, (window, mesh) in enumerate(zip(window_lst, window_mesh_lst)):
                # window_mesh = convert_oriented_box_to_trimesh_fmt(window)
                window_f = ThreedFutureExtra(
                    model_uid="window_" + str(id),
                    model_jid="",
                    xyz=mesh.vertices,
                    faces=mesh.faces,
                    normals=mesh.vertex_normals,
                    model_type="window",
                    position=[0, 0, 0],
                    rotation=[0, 0, 0, 1],  # use [0,0,1] to represent 0 degree rotation
                    scale=[1, 1, 1])
                window_f.z_angle = window['angles'][1]
                extras.append(window_f)

        return extras

    def augment_room(self, objects_dataset):
        bboxes = self.bboxes
        # Randomly pick an asset to be augmented
        bi = np.random.choice(self.bboxes)
        query_label = bi.label
        query_size = bi.size + np.random.normal(0, 0.02)
        # Retrieve the new asset based on the size of the picked asset
        furniture = objects_dataset.get_closest_furniture_to_box(query_label, query_size)
        bi_retrieved = bi.copy_from_other_model(furniture)

        new_bboxes = [box for box in bboxes if not box == bi] + [bi_retrieved]

        return Room(scene_id=self.scene_id + "_augm",
                    scene_type=self.scene_type,
                    bboxes=new_bboxes,
                    extras=self.extras,
                    json_path=self.json_path,
                    path_to_room_masks_dir=self.path_to_room_masks_dir)
