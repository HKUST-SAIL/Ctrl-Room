import sys
sys.path.append('.')
sys.path.append('..')

import copy
from typing import List, Dict, Tuple

import numpy as np
import trimesh

from dataset.st3d_dataset import find_occlusion, corners_to_1d_boundary
from utils.equirect_projection import vis_objs3d, vis_floor_ceiling
from dataset.metadata import COLOR_TO_ADEK_LABEL

def heading2rotmat(heading_angle_rad):
    rotmat = np.eye(3)
    cosval = np.cos(heading_angle_rad)
    sinval = np.sin(heading_angle_rad)
    # rot around z axis
    rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])

    return rotmat


def convert_oriented_box_to_trimesh_fmt(box: Dict, color_to_labels: Dict = None):
    box_center = box['center']
    box_lengths = box['size']
    transform_matrix = np.eye(4)
    transform_matrix[0:3, 3] = box_center
    # only use z angle, rad
    transform_matrix[0:3, 0:3] = heading2rotmat(box['angles'][-1])
    box_trimesh_fmt = trimesh.creation.box(box_lengths, transform_matrix)
    if color_to_labels is not None:
        labels_lst = list(color_to_labels.values())
        colors_lst = list(color_to_labels.keys())
        color = colors_lst[labels_lst.index(box['class'])]
    else:
        color = (np.random.random(3) * 255).astype(np.uint8).tolist()
        # pass
    box_trimesh_fmt.visual.face_colors = color
    return box_trimesh_fmt

def vis_scene_mesh(room_layout_mesh: trimesh.Trimesh,
                   obj_bbox_lst: List[Dict],
                   color_to_labels: Dict = None,
                   room_layout_bbox=None) -> trimesh.Trimesh:

    def create_oriented_bbox(scene_bbox: List[Dict]) -> trimesh.Trimesh:
        """Export oriented (around Z axis) scene bbox to meshes
        Args:
            scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
                and heading angle around Z axis.
                Y forward, X right, Z upward. heading angle of positive X is 0,
                heading angle of positive Y is 90 degrees.
            out_filename: (string) filename
        """
        scene = trimesh.scene.Scene()
        for box in scene_bbox:
            scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box, color_to_labels))

        mesh_list = trimesh.util.concatenate(scene.dump())
        return mesh_list

    v_object_meshes = create_oriented_bbox(obj_bbox_lst)
    if room_layout_bbox is not None:
        scene_mesh = trimesh.util.concatenate([room_layout_mesh, v_object_meshes, room_layout_bbox])
    elif room_layout_mesh is not None:
        scene_mesh = trimesh.util.concatenate([room_layout_mesh, v_object_meshes])
    else:
        scene_mesh = trimesh.util.concatenate([v_object_meshes])
    return scene_mesh

def save_visualization_and_mesh(
    objects_lst: List[Dict],
    quad_walls_lst: List[Dict],
    source_cor_path: str,
    source_img_path: str,
) -> Tuple[np.ndarray, trimesh.Trimesh]:
    """ get visualization of the layout and mesh

    Args:
        objects_lst (List[Dict]): objects bbox list
        quad_walls_lst (List[Dict]): quad walls list
        source_cor_path (str): 2d wall corners filepath
        source_img_path (str): source panorama filepath

    Returns:
        _type_: semantic image and mesh
    """
    # visualize quad wall if exists
    assert len(objects_lst)
    assert len(quad_walls_lst)

    quad_walls_lst_cp = copy.deepcopy(quad_walls_lst)
    objects_lst_cp = copy.deepcopy(objects_lst)
    for quad_wall in quad_walls_lst_cp:
        wall_dict = {}

        wall_dict['center'] = quad_wall['center']
        wall_dict['size'] = [quad_wall['width'], 0.01, quad_wall['height']]
        normal = quad_wall['normal']
        # The direction of all camera is always along the negative y-axis.
        cos_angle = np.array(normal).dot(np.array([0, -1, 0]))
        angle = np.arccos(cos_angle)
        if abs(cos_angle) < 1e-6:
            angle = np.pi / 2 if normal[0] > 0 else -np.pi / 2

        wall_dict['angles'] = [0, 0, angle]
        wall_dict['class'] = 'wall'
        objects_lst_cp.append(wall_dict)

    # visualize room layout and object bbox in panorama
    # rgb_img = cv2.imread(source_img_path, cv2.IMREAD_UNCHANGED)
    # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    # bg_img = rgb_img
    img_H, img_W = 512, 1024
    bg_img = np.zeros((img_H, img_W, 3), dtype=np.uint8)
    # Read ground truth corners
    with open(source_cor_path, 'r') as f:
        corners_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)
        # Corner with minimum x should at the beginning
        corners_lst = np.roll(corners_lst[:, :2], -2 * np.argmin(corners_lst[::2, 0]), 0)
        # Detect occlusion
        occlusion = find_occlusion(corners_lst[::2].copy()).repeat(2)
        # corners correspondenses' x coordinate should be identical
        assert (np.abs(corners_lst[0::2, 0] - corners_lst[1::2, 0]) > img_W / 100).sum() == 0, source_img_path
        # corners correspondenses' y coordinate should be y_floor < y_ceiling
        assert (corners_lst[0::2, 1] > corners_lst[1::2, 1]).sum() == 0, source_img_path
    # 1d ceiling-wall/floor-wall boundary, [-pi/2, pi/2]
    boundary_lst = corners_to_1d_boundary(corners_lst, img_H, img_W)
    sem_bbox_img = vis_floor_ceiling(image=bg_img, coords_2d=boundary_lst, color_to_labels=COLOR_TO_ADEK_LABEL)
    sem_bbox_img = vis_objs3d(image=bg_img,
                              v_bbox3d=objects_lst_cp,
                              camera_position=np.array([0, 0, 0]),
                              color_to_labels=COLOR_TO_ADEK_LABEL,
                              b_show_axes=False,
                              b_show_centroid=False,
                              b_show_bbox3d=False,
                              b_show_info=False,
                              b_show_polygen=True,
                              thickness=2)
    # convert BGR to RGB
    debug_bbox_img = sem_bbox_img

    debug_bbox_trimesh = vis_scene_mesh(room_layout_mesh=None,
                                        obj_bbox_lst=objects_lst_cp,
                                        color_to_labels=COLOR_TO_ADEK_LABEL,
                                        room_layout_bbox=None)

    return debug_bbox_img, debug_bbox_trimesh