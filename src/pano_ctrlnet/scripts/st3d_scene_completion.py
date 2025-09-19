### given scene layout json file, reconstruct the 3D layout 
# and render its semanntic segmentation map, depth map, and normal map , instance map
import os
import sys
sys.path.append('.')
sys.path.append('..')
import math
from typing import Dict, List, Tuple, Union
import copy
import argparse

import numpy as np
import cv2
import trimesh
import json
import shapely
from shapely.geometry import Polygon, LineString, MultiLineString
import matplotlib.pyplot as plt


import pyrender
from tqdm import tqdm
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import open3d as o3d
from icecream import ic


from panoformer.network.model import Panoformer as PanoBiT
from egformer.egformer import EGDepthModel
from scripts.st3d_panorama_recons import load_module_weights
from scipy.spatial.transform import Rotation as R

COLOR_TO_ADEK_LABEL = {
    (0, 0, 0): "unknown",
    (120, 120, 120): "wall",
    (80, 50, 50): "floor",
    (224, 5, 255): "cabinet",
    (204, 5, 255): "bed",
    (204, 70, 3): "chair",
    (11, 102, 255): "sofa",
    (255, 6, 82): "table",
    (8, 255, 51): "door",
    (230, 230, 230): "window",
    (0, 255, 245): "bookshelf",
    (255, 6, 51): "picture",
    (235, 12, 255): "counter",
    (0, 61, 255): "blinds",
    (10, 255, 71): "desk",
    (255, 7, 71): "shelves",
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
    (146, 111, 194): "night stand",
    (0, 255, 133): "toilet",
    (0, 163, 255): "sink",
    (0, 31, 255): "lamp",
    (0, 133, 255): "bathtub",
    (70, 184, 160): "bag",
    (94, 106, 211): "structure",
    (82, 84, 163): "furniture",
    (100, 85, 144): "prop",
    (0, 153, 255): "hood",
    (51, 255, 0): "stove",
    (255, 51, 7): "shower",
}

# TODO: this is only for scene_03110_207, nneed to fix this!!!
# ROOM_CENTROID = np.array([-0.44564146, -0.40997267, -0.22152991])
ROOM_CENTROID = np.array([0,0,0])

def heading2rotmat(heading_angle_rad: float) -> np.array:
    """
    Convert z_angle to rotation matrix
    """
    rotmat = np.eye(3)
    cosval = np.cos(heading_angle_rad)
    sinval = np.sin(heading_angle_rad)
    rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
    return rotmat


def convert_oriented_box_to_trimesh_fmt(box: Dict, color_to_labels: Dict = None) -> trimesh.Trimesh:
    """
    Convert oriented box dict to mesh
    """
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
                   room_layout_bbox:trimesh.Trimesh = None) -> trimesh.Trimesh:
    """ visualize scene bbox as mesh

    Args:
        room_layout_mesh (trimesh.Trimesh): closed mesh of room layout
        obj_bbox_lst (List[Dict]): object bounding box list
        color_to_labels (Dict, optional): color for object categories. Defaults to None.
        room_layout_bbox (trimesh.Trimesh): _description_. Defaults to None.

    Returns:
        trimesh.Trimesh: _description_
    """

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

class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram("shaders/mesh.vert", "shaders/mesh.frag", defines=defines)
        return self.program

    def clear(self):
        del self.program

class CubemapCamera:
    Fovx: float = np.pi / 2
    Fovy:float = np.pi / 2
    image_height: int = 512
    image_width: int = 512
    R: np.ndarray = np.eye(3)
    T: np.ndarray = np.zeros(3)
    
class PyRenderScene:
    def __init__(self, viewpoint_camera: CubemapCamera):
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.Fovx * 0.5)
        tanfovy = math.tan(viewpoint_camera.Fovy * 0.5)
        image_height = int(viewpoint_camera.image_height)
        image_width = int(viewpoint_camera.image_width)
        fx = image_width * 0.5 / tanfovx
        fy = image_height * 0.5 / tanfovy
        cx = image_width / 2.0
        cy = image_height / 2.0

        self.R_gl_cv = np.asarray([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])
        self.R_cv_gl = self.R_gl_cv.T

        self.scene = pyrender.Scene(
            ambient_light=np.array([0.35]*3 + [1.0]),
        )
        self.cam_node = None
        self.direc_l = None
        self.spot_l = None
        self.point_l = None

        self.intrinsic_cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        self.set_camera_pose(np.eye(4))

        r = pyrender.OffscreenRenderer(
            viewport_width=image_width, viewport_height=image_height
        )

        self.r = r

    def __del__(self):
        self.r.delete()

    def render(self, **kwargs):
        return self.r.render(self.scene, **kwargs)

    def load_mesh(self, mesh_list: List[trimesh.Trimesh]):
        for mesh in mesh_list:
            # print(f'mesh is_wateright: {mesh.is_watertight}')        
            # rotate mesh to match the GL coordinate
            R_world_gl = np.asarray([1, 0, 0, 0, 
                                    0, 0, -1, 0, 
                                    0, 1, 0, 0,
                                    0, 0, 0, 1]).reshape(4, 4)
            mesh.apply_transform(R_world_gl)

            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            mesh_node = self.scene.add(pyrender_mesh)
        
    def set_camera_pose(self, T_c2w: np.ndarray):
        T_c2w[:3, :3] = T_c2w[:3, :3] @ self.R_cv_gl

        attrs_to_remove = ["cam_node", "direc_l", "spot_l", "point_l"]
        for attr in attrs_to_remove:
            node = getattr(self, attr)
            if node is not None:
                self.scene.remove_node(node)

        self.cam_node = self.scene.add(self.intrinsic_cam, pose=T_c2w)

        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        spot_l = pyrender.SpotLight(color=np.ones(3), intensity=5.0,
                           innerConeAngle=np.pi/16*0.1, outerConeAngle=np.pi/6*0.1)
        point_l = pyrender.PointLight(color=np.ones(3), intensity=10.0)

        self.direc_l = self.scene.add(direc_l, pose=T_c2w)
        self.spot_l = self.scene.add(spot_l, pose=T_c2w)
        self.point_l = self.scene.add(point_l, pose=T_c2w)


class Perspective2Panorama:
    """convert perspective image to panorama image
    """
    def __init__(
        self, image:np.array, FOV:float, PHI:float, THETA:float, channel=3,
        interpolation=cv2.INTER_NEAREST
    ):
        self._img = image
        [self._height, self._width, c] = self._img.shape
        self.wFOV = FOV
        self.PHI = PHI
        self.THETA = THETA
        self.hFOV = float(self._height) / self._width * FOV

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))
        self.channel = channel
        self.interpolation = interpolation
        assert self.channel == c

    def GetEquirec(self, height, width):
        #
        # PHI is left/right angle, THETA is up/down angle, both in degree
        #
        x,y = np.meshgrid(np.linspace(-180, 180,width),np.linspace(90,-90,height))
        
        x_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map,y_map,z_map),axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.PHI))
        [R2, _] = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(self.THETA))  # +y axis forward

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T

        xyz = xyz.reshape([height , width, 3])
        inverse_mask = np.where(xyz[:,:,1]>0, 1, 0) # +y axis forward
        whatsit = np.repeat(xyz[:,:,1][:, :, np.newaxis], 3, axis=2) # +y axis forward
        xyz[:,:] = xyz[:,:]/whatsit
        
        # +y axis forward
        lon_map = np.where((-self.w_len < xyz[:,:,0]) & (xyz[:,:,0] < self.w_len) & 
                        (-self.h_len < xyz[:,:,2]) & (xyz[:,:,2] < self.h_len),
                        (xyz[:,:,0]+self.w_len)/2/self.w_len*self._width, 0)
        lat_map = np.where((-self.w_len<xyz[:,:,0]) & (xyz[:,:,0] < self.w_len) &
                        (-self.h_len<xyz[:,:,2]) & (xyz[:,:,2] < self.h_len),
                        (-xyz[:,:,2]+self.h_len)/2/self.h_len*self._height, 0)
        mask = np.where((-self.w_len < xyz[:,:,0]) & (xyz[:,:,0] < self.w_len) &
                        (-self.h_len < xyz[:,:,2]) & (xyz[:,:,2] < self.h_len), 1, 0)

        # INTER_NEAREST needed to avoid interpolation for depth, semantic, and instance map
        # otherwise it will average nearby pixels
        persp = cv2.remap(self._img, lon_map.astype(np.float32), lat_map.astype(np.float32), self.interpolation, borderMode=cv2.BORDER_REPLICATE)
        
        mask = mask * inverse_mask
        mask = np.repeat(mask[:, :, np.newaxis], self.channel, axis=2)
        while len(persp.shape) != len(mask.shape):
            persp = persp[..., np.newaxis]
        persp = persp * mask
        
        
        return persp , mask
        
class MultiPers2Panorama:
    def __init__(
        self, img_array , F_P_T_array, channel=3,
        interpolation=cv2.INTER_NEAREST, average=True
    ):
        
        assert len(img_array)==len(F_P_T_array)
        
        self.img_array = img_array
        self.F_P_T_array = F_P_T_array
        self.channel = channel
        self.interpolation = interpolation
        self.average = average

    def GetEquirec(self, height:int=512, width:int=1024):
        #
        # PHI is left/right angle, THETA is up/down angle, both in degree
        #
        merge_image = np.zeros((height, width, self.channel))
        merge_mask = np.zeros((height, width, self.channel))

        for img, [F,P,T] in zip (self.img_array, self.F_P_T_array):
            per = Perspective2Panorama(img, F, P, T, channel=self.channel, interpolation=self.interpolation)        # Load equirectangular image
            img , mask = per.GetEquirec(height,width)   # Specify parameters(FOV, theta, phi, height, width)
            if self.average:
                merge_image += img
            else:
                merge_image = np.where(merge_image==0, img, merge_image)
            merge_mask +=mask

        if self.average:
            merge_mask = np.where(merge_mask==0,1,merge_mask)
            merge_image = (np.divide(merge_image,merge_mask))
        else:
            merge_mask = np.where(merge_mask>0,1,0)

        return merge_image, merge_mask

def convert_z_to_distannce(depth_image: np.array, height: int, width: int):
    xs, ys = np.meshgrid(np.linspace(-1,1,width), np.linspace(-1,1,height))
    depth_map = depth_image.reshape(1, height, width)
    xs = xs.reshape(1,height,width)
    ys = ys.reshape(1,height,width)
    
    # convert distannce map to depth
    depth_cos = np.cos(np.arctan2(np.sqrt(xs*xs + ys*ys),1))
    distance_map = depth_map.astype(float) / depth_cos
    return distance_map.reshape(height, width)

def convert_distannce_to_z(distance_image: np.array, height: int, width: int):
    xs, ys = np.meshgrid(np.linspace(-1,1,width), np.linspace(-1,1,height))
    distance_map = distance_image.reshape(1, height, width)
    xs = xs.reshape(1,height,width)
    ys = ys.reshape(1,height,width)
    
    # convert distannce map to depth
    depth_cos = np.cos(np.arctan2(np.sqrt(xs*xs + ys*ys),1))
    depth_map = distance_map.astype(float) * depth_cos
    return depth_map.reshape(height, width)

def render_semantic_depth(scene_mesh_path: str, 
               output_path: str,
               cam_pose:np.ndarray =np.eye(4), 
               cam_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """ render a set of images from different viewpoints

    Args:
        scene_mesh_path (str): mesh file path
        output_path (str): output path
        cam_pose (np.ndarray, optional): camera pose c2w. Defaults to np.eye(4).
        cam_idx (int, optional): camera index. Defaults to 0.
    """
    
    camera_center = cam_pose[:3, 3]
    cubemap_cam_lst: List[CubemapCamera] = []
    for i in range(6):
        # rotate referenncee view to subview
        R_ref_subview = np.eye(3)
        # up, left, front, right, back, down
        # phi_angle_lst = [-90, 270, 0, 90, 180, -90]
        # theta_angle_lst = [90, 0, 0, 0, 0, -90]
        if i == 3:
            R_ref_subview = R.from_rotvec(np.pi/2 * np.array([0, 1, 0])).as_matrix()
        elif i == 4:
            R_ref_subview = R.from_rotvec(np.pi * np.array([0, 1, 0])).as_matrix()
        elif i == 1:
            R_ref_subview = R.from_rotvec(-np.pi/2 * np.array([0, 1, 0])).as_matrix()
        elif i == 0:
            R_ref_subview = R.from_rotvec(-np.pi/2 * np.array([0, 1, 0])).as_matrix() @ R.from_rotvec(np.pi/2 * np.array([1, 0, 0])).as_matrix()
        elif i == 5:
            R_ref_subview = R.from_rotvec(-np.pi/2 * np.array([0, 1, 0])).as_matrix() @ R.from_rotvec(-np.pi/2 * np.array([1, 0, 0])).as_matrix()
        cubemap_cam = CubemapCamera()
        cubemap_cam.R = R_ref_subview
        cubemap_cam.T = camera_center
        cubemap_cam.image_height = 512
        cubemap_cam.image_width = 512
        cubemap_cam.Fovx = np.pi/2
        cubemap_cam.Fovy = np.pi/2
        cubemap_cam_lst.append(cubemap_cam)


    renderer = PyRenderScene(cubemap_cam_lst[0])
    scene_layout_mesh = trimesh.load(scene_mesh_path, process=True)
    renderer.load_mesh(mesh_list=[scene_layout_mesh])

    os.makedirs(output_path, exist_ok=True)
    rgb_rendering_output_path = os.path.join(output_path, 'rgb')
    os.makedirs(rgb_rendering_output_path, exist_ok=True)
    depth_rendering_output_path = os.path.join(output_path, 'depth')
    os.makedirs(depth_rendering_output_path, exist_ok=True)
    normal_rendering_output_path = os.path.join(output_path, 'normal')
    os.makedirs(normal_rendering_output_path, exist_ok=True)
    semantic_faces_img_lst = []
    depth_faces_img_lst = [] 
    for idx, view in enumerate(tqdm(cubemap_cam_lst, desc="Rendering progress")):
        view : CubemapCamera
        R_c2w = view.R
        t_c2w = view.T
        T_c2w = np.eye(4)
        T_c2w[:3, :3] = R_c2w
        T_c2w[:3, 3] = t_c2w
        renderer.set_camera_pose(T_c2w)

        # render color and depth
        rendered_color, rendered_depth = renderer.render(flags= pyrender.RenderFlags.FLAT | pyrender.RenderFlags.SKIP_CULL_FACES)
        Image.fromarray(rendered_color).save(os.path.join(rgb_rendering_output_path, f"{idx:05d}.png"))
        semantic_faces_img_lst.append(rendered_color)
        
        # print(f'renndered_depth: {rendered_depth.shape}')
        if len(rendered_depth.shape) == 3:
            depth_img = (rendered_depth[:,:,0] * 4000.0).astype(np.int32)
        if len(rendered_depth.shape) == 2:
            depth_img = (rendered_depth * 4000.0).astype(np.int32)
        depth_img = convert_z_to_distannce(depth_img, 512, 512).astype(np.int32)
        Image.fromarray(depth_img).save(os.path.join(depth_rendering_output_path, f"{idx:05d}.png"))
        depth_faces_img_lst.append(depth_img[:,:,np.newaxis])

    # given color and depth cubemap, convert to panoramic image
    pano_height, pano_width = 512, 1024
    # cubemap fovs, phi and theta angles    
    F_P_T_lst = [[90, 90, 90],  # up
                [90, 90, 0], # left
                [90, 0, 0], # front
                [90, -90, 0], # right
                [90, -180, 0], # back
                [90, 90, -90]] # down
    convert_keys = ['semantic', 'depth']
    convert_images_lst = [semantic_faces_img_lst, depth_faces_img_lst]
    semantic_pano_img = None
    depth_pano_img = None
    for img_type, faces_img_lst in zip(convert_keys, convert_images_lst):
        img_channel = faces_img_lst[0].shape[-1]

        kwargs = {}
        if img_type in ['semantic', 'instance', 'depth']:
            kwargs['interpolation'] = cv2.INTER_NEAREST
            kwargs['average'] = False
        else:
            kwargs['interpolation'] = cv2.INTER_NEAREST
            kwargs['average'] = True

        per = MultiPers2Panorama(faces_img_lst, F_P_T_lst, channel=img_channel, **kwargs)
        img, mask = per.GetEquirec(pano_height, pano_width)
        if img_type == 'depth':
            img = img.astype(np.int32)
            img = np.where(mask > 0, img, 0)
            img = img.reshape((pano_height, pano_width))
        elif img_type == 'albedo' or img_type == 'normal':
            img = img.astype(np.uint8)
        elif img_type == 'instance':
            img = img.astype(np.int32)
            img = np.where(mask > 0, img, -1)
            img = img.reshape((pano_height, pano_width))
        elif img_type == 'semantic':
            img = img.astype(np.uint8)
        
        if img_type == 'semantic':
            semantic_pano_img = img
        elif img_type == 'depth':
            depth_pano_img = img
            
        img = Image.fromarray(img)
        img.save(os.path.join(output_path, f'{img_type}_{cam_idx}.png'))

    
    return semantic_pano_img, depth_pano_img
    # pano_ply = test_panoramic_depth(raw_rgb_img=semantic_pano_img, raw_depth_img=depth_pano_img, depth_scale=4000.0)
    # o3d.io.write_point_cloud(os.path.join(output_path, f'pano_ply_{cam_idx}.ply'), pano_ply)
        
def test_cubemap_img_depth():
    # up, left, front, right, back, down
    rgb_img_path_lst = [
        '/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/our_livingrooms/0/rendering/rgb/00000.png',
        '/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/our_livingrooms/0/rendering/rgb/00001.png',
        '/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/our_livingrooms/0/rendering/rgb/00002.png',
        '/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/our_livingrooms/0/rendering/rgb/00003.png',
        '/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/our_livingrooms/0/rendering/rgb/00004.png',
        '/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/our_livingrooms/0/rendering/rgb/00005.png',
    ]
    depth_img_path_lst = [
        '/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/our_livingrooms/0/rendering/depth/00000.png',
        '/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/our_livingrooms/0/rendering/depth/00001.png',
        '/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/our_livingrooms/0/rendering/depth/00002.png',
        '/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/our_livingrooms/0/rendering/depth/00003.png',
        '/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/our_livingrooms/0/rendering/depth/00004.png',
        '/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/our_livingrooms/0/rendering/depth/00005.png',
    ]
    total_pointcloud = o3d.geometry.PointCloud()
    for i, (rgb_img_filepath, depth_img_filepath) in enumerate(zip(rgb_img_path_lst, depth_img_path_lst)):

        rgb_img = np.array(Image.open(rgb_img_filepath).convert('RGB'))
        depth_img = np.array(Image.open(depth_img_filepath))
        H, W, C = rgb_img.shape

        # Get camera intrinsic
        hfov = 90. * np.pi / 180.
        K = np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])

        depth_img = (depth_img/4000.0).astype(np.float32)
        depth_img = np.expand_dims((depth_img).astype(np.float32),axis=2)
        # Now get an approximation for the true world coordinates -- see if they make sense
        # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
        xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(-1,1,H))
        distance_map = depth_img.reshape(1,H,W)
        xs = xs.reshape(1,H,W)
        ys = ys.reshape(1,H,W)
        
        # convert distannce map to depth
        depth_cos = np.cos(np.arctan2(np.sqrt(xs*xs + ys*ys),1))
        # print(f'depth_cos: {depth_cos.shape}')
        # depth = distance_map * depth_cos
        depth = distance_map
        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack((xs * depth , ys * depth, depth, np.ones(depth.shape)))
        xys = xys.reshape(4, -1)
        xy_c0 = np.matmul(np.linalg.inv(K), xys)

        if rgb_img.shape[2] == 4:
            rgb_img = rgb_img[:, :, :3]
        if np.isnan(rgb_img.any()) or len(rgb_img[rgb_img > 0]) == 0:
            print('empyt rgb image')
            exit(-1)
        color = np.clip(rgb_img, 0.0, 255.0) / 255.0

        # chose front as reference view
        T_ref_subview = np.eye(4)
        # up, left, front, right, back, down
        # theta_angle_lst = [-90, 270, 0, 90, 180, -90]
        # phi_angle_lst = [90, 0, 0, 0, 0, -90]
        if i == 3:
            T_ref_subview[:3, :3] = R.from_rotvec(np.pi/2 * np.array([0, 1, 0])).as_matrix()
        elif i == 4:
            T_ref_subview[:3, :3] = R.from_rotvec(np.pi * np.array([0, 1, 0])).as_matrix()
        elif i == 1:
            T_ref_subview[:3, :3] = R.from_rotvec(-np.pi/2 * np.array([0, 1, 0])).as_matrix()
        elif i == 0:
            T_ref_subview[:3, :3] = R.from_rotvec(-np.pi/2 * np.array([0, 1, 0])).as_matrix() @ R.from_rotvec(np.pi/2 * np.array([1, 0, 0])).as_matrix()
        elif i == 5:
            T_ref_subview[:3, :3] = R.from_rotvec(-np.pi/2 * np.array([0, 1, 0])).as_matrix() @ R.from_rotvec(-np.pi/2 * np.array([1, 0, 0])).as_matrix()
        subview_pointcloud = T_ref_subview @ xy_c0
        subview_pointcloud_T = np.transpose(subview_pointcloud)
        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(subview_pointcloud_T[:,:3])
        o3d_pointcloud.colors = o3d.utility.Vector3dVector(color.reshape(-1,3))
        o3d.io.write_point_cloud(f'./scene_03110_207_full_cubemap{i}.ply', o3d_pointcloud)
        total_pointcloud += o3d_pointcloud

    o3d.io.write_point_cloud('./scene_03110_207_full.ply', total_pointcloud)
    
def test_panoramic_depth(raw_rgb_img: np.ndarray, raw_depth_img: np.ndarray, depth_scale: float = 1000.0):
    def get_unit_spherical_map():
        h = 512
        w = 1024
        Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
        Theta = np.repeat(Theta, w, axis=1)
        Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        # do not flip horizontal
        Phi = np.repeat(Phi, h, axis=0)

        X = np.expand_dims(np.sin(Theta) * np.sin(Phi), 2)
        Y = np.expand_dims(np.cos(Theta), 2)
        Z = np.expand_dims(np.sin(Theta) * np.cos(Phi), 2)
        unit_map = np.concatenate([X, Z, Y], axis=2)

        return unit_map

    color = np.clip(raw_rgb_img, 0.0, 255.0) / 255.0
    depth_img = np.asarray(raw_depth_img)
    if np.isnan(depth_img.any()) or len(depth_img[depth_img > 0]) == 0:
        print('empyt depth image')
        exit(-1)

    if len(depth_img.shape) == 3:
        depth_img = depth_img[:, :, 0]
    depth_img = np.expand_dims((depth_img / depth_scale), axis=2)
    pointcloud = depth_img * get_unit_spherical_map()

    o3d_pointcloud = o3d.geometry.PointCloud()
    o3d_pointcloud.points = o3d.utility.Vector3dVector(pointcloud.reshape(-1, 3))
    o3d_pointcloud.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
    return o3d_pointcloud
    
def complete_scene_layout(scene_layout_json: str,
                          output_scene_mesh_path: str,
                          room_centroid: np.ndarray):
    """ complete scene layout with floor and ceiling

    Args:
        scene_layout (Dict): scene layout dict
    """
    with open(scene_layout_json, 'r') as f:
        scene_layout = json.load(f)
    
    wall_dict_lst = scene_layout['walls']
    obj_bbox_dict_lst = scene_layout['objects']
    # save object bboxes as ply
    scene_mesh = vis_scene_mesh(room_layout_mesh=None,
                                obj_bbox_lst=obj_bbox_dict_lst,
                                color_to_labels=COLOR_TO_ADEK_LABEL,
                                room_layout_bbox=None)
    # find floor and ceiling lines
    floor_lines = []
    ceiling_lines = []
    avg_floor_h = 0
    avg_ceiling_h = 0
    wall_mesh_lst = []
    for wall in wall_dict_lst:
        # corners: [left-bottom, left-top, right-top, right-bottom]
        corners = np.around(wall['corners'], 3)
        floor_lines.append([corners[0][:2], corners[3][:2]])
        avg_floor_h += corners[0][2]
        ceiling_lines.append([corners[1][:2], corners[2][:2]])
        avg_ceiling_h += corners[1][2]
        
        wall_vertices = np.array([corners[0], corners[1], corners[2], corners[3]])
        wall_faces = np.array([[0, 1, 2], [0, 2, 3]])
        # rotation_matrix = euler_angle_to_matrix(wall['angles'])
        # wall_normal = rotation_matrix.dot(np.array([0, -1, 0]))
        wall_normal = np.array(wall['normal'])
        wall_mesh = trimesh.Trimesh(vertices=wall_vertices, faces=wall_faces, face_normals=np.array([wall_normal]).repeat(2, axis=0))
        labels_lst = list(COLOR_TO_ADEK_LABEL.values())
        colors_lst = list(COLOR_TO_ADEK_LABEL.keys())
        color = colors_lst[labels_lst.index('wall')]
        wall_mesh.face_colors = np.array([color]).repeat(2, axis=0)
        wall_mesh_lst.append(wall_mesh)
    
    avg_floor_h /= len(wall_dict_lst)
    avg_ceiling_h /= len(wall_dict_lst)
    wall_mesh = trimesh.util.concatenate(wall_mesh_lst)
    
    # import matplotlib.pyplot as plt
    # import shapely.plotting
    # shapely.plotting.plot_line(MultiLineString(floor_lines))
    # print(f'floor lines: {floor_lines}')
    # plt.show()
    # sort floor and ceiling lines by connection
    floor_linestrs = shapely.ops.linemerge(MultiLineString(floor_lines))
    # print(f'is valid floor linestr: {floor_linestrs.is_valid}')
    ceiling_linestrs = shapely.ops.linemerge(MultiLineString(ceiling_lines))
    # print(f'is valid ceiling linestr: {ceiling_linestrs.is_valid}')
    
    # construct floor and ceiling polygons
    floor_polygon = Polygon(floor_linestrs)
    # print(f'is valid floor polygon: {floor_polygon.is_valid}')
    # ceiling_polygon = Polygon(ceiling_linestrs)
    # print(f'is valid ceiling polygon: {ceiling_polygon.is_valid}')
    # convert polygon to mesh
    floor_vertices, floor_faces = trimesh.creation.triangulate_polygon(floor_polygon)
    # print(f'floor vertices: {floor_vertices.shape}, faces: {floor_faces.shape}')
    floor_vertices_3d = np.concatenate([floor_vertices, avg_floor_h * np.ones((floor_vertices.shape[0], 1))], axis=1)
    labels_lst = list(COLOR_TO_ADEK_LABEL.values())
    colors_lst = list(COLOR_TO_ADEK_LABEL.keys())
    floor_face_color = colors_lst[labels_lst.index('floor')]
    floor_mesh = trimesh.Trimesh(vertices=floor_vertices_3d, faces=floor_faces, 
                                 face_normals=np.array([[0, 0, 1]]).repeat(floor_faces.shape[0], axis=0),
                                 face_colors=np.array([floor_face_color]).repeat(floor_faces.shape[0], axis=0))
    # ceiling_vertices, ceiling_faces = trimesh.creation.triangulate_polygon(ceiling_polygon)
    # ceiling_vertices_3d = np.concatenate([ceiling_vertices, ceiling_h * np.ones((ceiling_vertices.shape[0], 1))], axis=1)
    ceiling_faces = floor_faces
    ceiling_vertices_3d = np.concatenate([floor_vertices, avg_ceiling_h * np.ones((floor_vertices.shape[0], 1))], axis=1)
    ceiling_face_color = colors_lst[labels_lst.index('ceiling')]
    ceiling_mesh = trimesh.Trimesh(vertices=ceiling_vertices_3d, faces=ceiling_faces, 
                                   face_normals=np.array([[0, 0, -1]]).repeat(ceiling_faces.shape[0], axis=0),
                                   face_colors=np.array([ceiling_face_color]).repeat(ceiling_faces.shape[0], axis=0))
    # save scene layout as ply
    # closed_scene_mesh = trimesh.util.concatenate([scene_mesh, floor_mesh, ceiling_mesh, wall_mesh])
    closed_scene_mesh = trimesh.util.concatenate([scene_mesh, floor_mesh, wall_mesh])
    closed_scene_mesh.apply_translation(room_centroid)
    closed_scene_mesh.export(output_scene_mesh_path)
    
    
def render_instance(scene_layout_json: str,
                    output_path: str,
                    cam_pose:np.ndarray =np.eye(4),
                    cam_idx: int = 0,
                    room_centroid: np.ndarray = np.zeros(3),  # translate to the original room_centroid, due to some historical reason
                    ):  
    with open(scene_layout_json, 'r') as f:
        scene_layout = json.load(f)
    
    objects_dict_lst = scene_layout['objects']
    objects_bbox_mesh_lst = []
    for box in objects_dict_lst:
        if box['class'] in ['curtain']:
            continue
        obj_bbox_mesh = convert_oriented_box_to_trimesh_fmt(box, color_to_labels=COLOR_TO_ADEK_LABEL)
        obj_bbox_mesh.apply_translation(room_centroid)
        objects_bbox_mesh_lst.append(obj_bbox_mesh)
        
    
    camera_center = cam_pose[:3, 3]
    cubemap_cam_lst: List[CubemapCamera] = []
    for i in range(6):
        # rotate referenncee view to subview
        R_ref_subview = np.eye(3)
        # up, left, front, right, back, down
        # phi_angle_lst = [-90, 270, 0, 90, 180, -90]
        # theta_angle_lst = [90, 0, 0, 0, 0, -90]
        if i == 3:
            R_ref_subview = R.from_rotvec(np.pi/2 * np.array([0, 1, 0])).as_matrix()
        elif i == 4:
            R_ref_subview = R.from_rotvec(np.pi * np.array([0, 1, 0])).as_matrix()
        elif i == 1:
            R_ref_subview = R.from_rotvec(-np.pi/2 * np.array([0, 1, 0])).as_matrix()
        elif i == 0:
            R_ref_subview = R.from_rotvec(-np.pi/2 * np.array([0, 1, 0])).as_matrix() @ R.from_rotvec(np.pi/2 * np.array([1, 0, 0])).as_matrix()
        elif i == 5:
            R_ref_subview = R.from_rotvec(-np.pi/2 * np.array([0, 1, 0])).as_matrix() @ R.from_rotvec(-np.pi/2 * np.array([1, 0, 0])).as_matrix()
        cubemap_cam = CubemapCamera()
        cubemap_cam.R = R_ref_subview
        cubemap_cam.T = camera_center
        cubemap_cam.image_height = 512
        cubemap_cam.image_width = 512
        cubemap_cam.Fovx = np.pi/2
        cubemap_cam.Fovy = np.pi/2
        cubemap_cam_lst.append(cubemap_cam)

    # setup renderer
    renderer = PyRenderScene(cubemap_cam_lst[0])
    renderer.load_mesh(objects_bbox_mesh_lst)
    # TODO: the mesh node is limited to 25, need to fix this
    # mesh_node -> segment_id map
    nm = {node: 10*(idx+1) for idx, node in enumerate(renderer.scene.mesh_nodes)}
    os.makedirs(output_path, exist_ok=True)
    ins_rendering_output_path = os.path.join(output_path, 'instance')
    os.makedirs(ins_rendering_output_path, exist_ok=True)
    
    instance_faces_img_lst = []
    for idx, view in enumerate(tqdm(cubemap_cam_lst, desc="Instance Rendering progress")):
        view : CubemapCamera
        R_c2w = view.R
        t_c2w = view.T
        T_c2w = np.eye(4)
        T_c2w[:3, :3] = R_c2w
        T_c2w[:3, 3] = t_c2w
        renderer.set_camera_pose(T_c2w)

        # render instance segmentation
        instance_seg = renderer.render(flags=pyrender.RenderFlags.SEG | pyrender.RenderFlags.SKIP_CULL_FACES, seg_node_map=nm)[0]
        # print(f'instance_seg: {instance_seg.shape}')
        Image.fromarray(instance_seg).save(os.path.join(ins_rendering_output_path, f"{idx:05d}.png"))
        instance_faces_img_lst.append(instance_seg)
        
        # get fine-grained instance segmentation with SAM
        

    # given instance cubemap, convert to panoramic image
    pano_height, pano_width = 512, 1024
    # cubemap fovs, phi and theta angles    
    F_P_T_lst = [[90, 90, 90],  # up
                [90, 90, 0], # left
                [90, 0, 0], # front
                [90, -90, 0], # right
                [90, -180, 0], # back
                [90, 90, -90]] # down
    convert_keys = ['instance']
    convert_images_lst = [instance_faces_img_lst]
    instance_pano_img = None
    for img_type, faces_img_lst in zip(convert_keys, convert_images_lst):
        img_channel = faces_img_lst[0].shape[-1]

        kwargs = {}
        if img_type in ['semantic', 'instance', 'depth']:
            kwargs['interpolation'] = cv2.INTER_NEAREST
            kwargs['average'] = False
        else:
            kwargs['interpolation'] = cv2.INTER_NEAREST
            kwargs['average'] = True

        per = MultiPers2Panorama(faces_img_lst, F_P_T_lst, channel=img_channel, **kwargs)
        img, mask = per.GetEquirec(pano_height, pano_width)
        # instance_img = np.where(mask > 0, img, -1)
        if len(img.shape) == 3:
            instance_img = img.astype(np.uint8)[:,:,0]
        elif len(img.shape) == 2:
            instance_img = img.astype(np.uint8)
        
        instance_pano_img = instance_img[:,:,np.newaxis]
        
        instance_img = Image.fromarray(instance_img)
        instance_img.save(os.path.join(output_path, f'{img_type}_{cam_idx}.png'))
                
    return instance_pano_img  
        

def get_unit_spherical_map():
    h = 512
    w = 1024
    Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
    Theta = np.repeat(Theta, w, axis=1)
    Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
    # do not flip horizontal
    Phi = np.repeat(Phi, h, axis=0)

    X = np.expand_dims(np.sin(Theta) * np.sin(Phi), 2)
    Y = np.expand_dims(np.cos(Theta), 2)
    Z = np.expand_dims(np.sin(Theta) * np.cos(Phi), 2)
    unit_map = np.concatenate([X, Z, Y], axis=2)
    return unit_map

def run_depth_estimation(rgb: np.ndarray, 
                         device: str, 
                         depth_estimation_model: Union[PanoBiT, EGDepthModel],
                         output_folderpath:str = None,
                         save_ply=False):
    def gen_pointcloud_from_panorama(raw_rgb_img:np.ndarray, raw_depth_img: np.ndarray, depth_scale: float = 1000.0):

        color = np.clip(raw_rgb_img, 0.0, 255.0) / 255.0
        depth_img = np.asarray(raw_depth_img)

        if len(depth_img.shape) == 3:
            depth_img = depth_img[:, :, 0]
        depth_img = np.expand_dims((depth_img / depth_scale), axis=2)
        pointcloud = depth_img * get_unit_spherical_map()

        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(pointcloud.reshape(-1, 3))
        # o3d_pointcloud.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
        # rotate point cloud
        R_x_90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        o3d_pointcloud.rotate(R_x_90, center=(0, 0, 0))
        o3d_pointcloud.translate((0, 1.6, 0))
        o3d_pointcloud.estimate_normals()
        o3d_pointcloud.orient_normals_towards_camera_location(camera_location=(0, 1.6, 0))
        return o3d_pointcloud
    
    to_tensor_fn = transforms.ToTensor()
    normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    assert rgb.shape == (512, 1024, 3), 'rgb shape is not 512x1024x3!!!'

    input_rgb = to_tensor_fn(rgb.copy()).float()
    inputs = {}
    inputs["rgb"] = input_rgb.unsqueeze(0).to(device)
    inputs["normalized_rgb"] = normalize_fn(input_rgb).unsqueeze(0).to(device)
    # print(f'inputs["normalized_rgb"].shape: {inputs["normalized_rgb"].shape}')

    # test on my pano image
    if isinstance(depth_estimation_model, PanoBiT):
        outputs = depth_estimation_model(inputs["normalized_rgb"])
        pred_depth = outputs["pred_depth"]
        pred_depth = pred_depth.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        
    elif isinstance(depth_estimation_model, EGDepthModel):
        outputs = depth_estimation_model(inputs["rgb"])
        # pred_depth = outputs.detach().cpu().numpy().squeeze()
        # scale up the predicted depth
        # pred_depth *= 10.0
        pred_depth = outputs.squeeze()
        
    # print(f'max pred_depth: {pred_depth.max()}, min pred_depth: {pred_depth.min()}')

    # save pred_depth
    if save_ply and output_folderpath is not None:
        o3d_pointcloud = gen_pointcloud_from_panorama(raw_rgb_img=rgb,
                                                    raw_depth_img=pred_depth.detach().cpu().numpy()*10,
                                                    depth_scale=1.0)
        o3d.io.write_point_cloud(os.path.join(output_folderpath, 'est_pano_ply.ply'), o3d_pointcloud)
    return pred_depth*10

def get_scheduler(optimizer, lr_steps, lr_min):
    scheduler = CosineAnnealingLR(optimizer=optimizer,
                                  eta_min=lr_min,
                                  T_max=lr_steps)

    return scheduler

def depth_to_normal(pano_depth: torch.Tensor, unit_spherical_points: torch.Tensor):
    """
        depth: depthmap 
    """
    if len(pano_depth.shape) < 3:
        pano_depth = pano_depth[..., None]
    points = pano_depth * unit_spherical_points # [H, W, 3]
    
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)  # [H-2, W-2, 3]
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)  # [H-2, W-2, 3]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def update_egformer(in_rgb_img: np.ndarray, 
                    egformer_model: EGDepthModel,
                    gt_bg_depth: np.ndarray,
                    max_iters:int=25,
                    init_learning_rate:float=0.0001,
                    min_learning_rate:float=0.00005,
                    lambda_depth:float=0.6,
                    lambda_normal:float=0.4,
                    output_folder:str=None,
                    device:str='cuda',
                    b_debug:bool=True) -> np.ndarray:
    """ update egformer model with the given ground truth depth map

    Args:
        in_rgb_img (np.ndarray): input rgb panorama image
        egformer_model (EGDepthModel): model
        gt_bg_depth (np.ndarray): ground truth background depth map, only floor, ceiling and walls are connsidered
        max_iters (int, optional): maximuim iterations. Defaults to 200.
        init_learning_rate (float, optional): learning rate. Defaults to 0.0001.
        min_learning_rate (float, optional): learning rate. Defaults to 0.00005.
        lambda_depth (float, optional): depth loss coefficient. Defaults to 0.6.
        lambda_normal (float, optional): normal loss coefficient. Defaults to 0.4.
        output_folder (str, optional): output directory. Defaults to None.
        device (str, optional): device. Defaults to 'cuda'.
        b_debug (bool, optional): _description_. Defaults to True.

    Returns:
        np.ndarray: finetuned depth map
    """
    
    # prepare optimizable param
    optimizer = torch.optim.Adam(egformer_model.parameters(), lr=init_learning_rate)
    scheduler = get_scheduler(optimizer, max_iters, min_learning_rate)

    # loss function
    l1_loss_fn = nn.SmoothL1Loss()
    
    def dilate(x, k=7):
        x = torch.nn.functional.conv2d(
            x.float()[None, None, ...],
            torch.ones(1, 1, k, k).to(x.device),
            padding="same"
        )
        return x.clamp(min=0, max=1).to(x.dtype)

    gt_bg_depth = torch.from_numpy(gt_bg_depth).float().to(device)
    mask = torch.where(gt_bg_depth > 0, 0, 1).float().to(device)
    mask = (dilate(mask, k=7).squeeze()).to(bool)
    depth_mask = (~mask).to(float)
    normal_mask = mask.to(float)
    # if b_debug:
    #     Image.fromarray((depth_mask*255).cpu().numpy().astype(np.uint8)).save(os.path.join(output_folder, 'dilated_mask.png'))

    unit_sphere = torch.tensor(get_unit_spherical_map(), device=device)
    unit_sphere.requires_grad = False
    init_normal = None

    for step_idx in tqdm(range(max_iters), desc="Optimization progress"):
        optimizer.zero_grad()
        
        # permute feature
        pred_depth = run_depth_estimation(rgb=in_rgb_img,
                             device=device,
                             depth_estimation_model=egformer_model,
                             output_folderpath=output_folder,
                             save_ply=(step_idx==max_iters-1))
        # calculate normal from points        
        pred_normal = depth_to_normal(pred_depth, unit_sphere)
        
        if step_idx == 0:
            init_normal = depth_to_normal(pred_depth, unit_sphere).detach()
        
        if b_debug:
            # plt.figure(frameon=False)
            # plt.imshow((-pred_normal.detach().cpu().numpy()+1)/2)
            # plt.savefig(os.path.join(output_folder, f'pred_normal_{step_idx}.png'), dpi=200)
            # plt.close()
            vis_depth_img = colorize_single_channel_image(pred_depth.detach().cpu().numpy())
            Image.fromarray(vis_depth_img).save(os.path.join(output_folder, f'pred_depth_{step_idx}.png'))
            vis_normal_img = (-pred_normal.detach().cpu().numpy()+1)/2 * 255.
            Image.fromarray(vis_normal_img.astype(np.uint8)).save(os.path.join(output_folder, f'pred_normal_{step_idx}.png'))
            
        depth_loss = l1_loss_fn(pred_depth*depth_mask, gt_bg_depth)
        normal_loss = (normal_mask[:,:,None] * (init_normal - pred_normal)).abs().mean() 
        loss = lambda_depth * depth_loss + lambda_normal * normal_loss
        
        tqdm.write(f'L1 loss: {loss.item()}, depth loss: {depth_loss.item()}, normal loss: {normal_loss.item()}')
        
        loss.backward()
        optimizer.step()
        scheduler.step()
            
    return pred_depth.detach().cpu().numpy()

# load egformer model
def load_depth_estimator_egformer_model(device: str = 'cuda'):
    model_path = '../egformer/ckpts/EGformer_pretrained.pkl'
    print("loading model from folder {}".format(model_path))
    
    egformer_model = EGDepthModel(hybrid=False)
    state_dict = load_module_weights(path=model_path, 
                                    map_location="cpu")
    egformer_model.load_state_dict(state_dict, strict=False)
            
    egformer_model.to(device)
    egformer_model.eval()
    
    return egformer_model

from PeRF.utils.utils import colorize_single_channel_image
def render_scene_layout(layout_json_path: str,
                        layout_mesh_path: str,
                        output_folder: str,
                        cam_pose: np.ndarray,
                        cam_idx: int,
                        room_centroid: np.ndarray = np.zeros(3)):
    
    # rennder depth and semantic map
    semantic_pano_img, depth_pano_img = render_semantic_depth(scene_mesh_path=layout_mesh_path, 
               output_path=output_folder,
               cam_pose=cam_pose, 
               cam_idx=cam_idx)
    inst_pano_img = render_instance(scene_layout_json=layout_json_path,
                    output_path=output_folder,
                    cam_pose=cam_pose,
                    cam_idx=cam_idx,
                    room_centroid=room_centroid)

    # extract background depth map
    if len(inst_pano_img.shape) == 3:
        inst_pano_img = inst_pano_img[:,:,0]
    bg_depth_map = np.zeros_like(depth_pano_img)
    bg_depth_map = np.where(inst_pano_img == 0, depth_pano_img/4000, 0)
    saved_bg_depth_filepath = os.path.join(output_folder, 'bg_depth.png')
    Image.fromarray(colorize_single_channel_image(bg_depth_map)).save(saved_bg_depth_filepath)
    # Image.fromarray((bg_depth_map*4000).astype(np.int32)).save(saved_bg_depth_filepath)
    
    pano_depth_map = depth_pano_img.astype(float)/4000.0
    Image.fromarray(colorize_single_channel_image(pano_depth_map)).save(os.path.join(output_folder, f'depth_{cam_idx}.png'))
    
    return semantic_pano_img, bg_depth_map, pano_depth_map

def geometry_align(scene_pano_img_filepath: str,
                   scene_layout_json_filepath: str,
                   scene_layout_mesh_filepath: str,
                   output_dir: str,
                   device: str,):
    if not os.path.exists(scene_pano_img_filepath):
        raise FileExistsError(f'{scene_pano_img_filepath} does not exist!!!')
        return None, None, None
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    init_rgb_pano = np.array(Image.open(in_scene_pano_img_path).convert('RGB'))
    
    # load MDE model
    depth_estimation_model = load_depth_estimator_egformer_model(device)

    # perf camera trajectory
    R_world_gl = np.asarray([1, 0, 0, 0, 
                                0, 0, -1, 0, 
                                0, 1, 0, 0,
                                0, 0, 0, 1]).reshape(4, 4)
    # camera 0 c2w pose
    cam0_pose = np.eye(4)
    # render new semantic map
    cam_pose_gl = R_world_gl @ cam0_pose
    
    debug_out_folder = os.path.join(output_dir, 'geo_align')
    os.makedirs(debug_out_folder, exist_ok=True)
    semantic_map, bg_depth_map, _ = render_scene_layout(layout_json_path=scene_layout_json_filepath,
                                                layout_mesh_path=scene_layout_mesh_filepath,
                                                output_folder=debug_out_folder,
                                                cam_pose=cam_pose_gl,
                                                cam_idx=0,
                                                room_centroid=ROOM_CENTROID)
              
    # align the predicted depth map with the background depth map
    updated_depth = update_egformer(in_rgb_img=init_rgb_pano,
                    egformer_model=depth_estimation_model,
                    gt_bg_depth=bg_depth_map,
                    device=device,
                    output_folder=debug_out_folder,
                    b_debug=True)
    Image.fromarray((updated_depth*1000).astype(np.int32)).save(os.path.join(debug_out_folder, 'depth.png'))
    # trannsform the point cloud into world coordinate
    cam_ply_filepath = os.path.join(debug_out_folder, 'est_pano_ply.ply')
    o3d_pointcloud = o3d.io.read_point_cloud(cam_ply_filepath)
    o3d_pointcloud=o3d_pointcloud.transform(cam0_pose)
    o3d.io.write_point_cloud(os.path.join(debug_out_folder, 'est_pano_ply_world.ply'), o3d_pointcloud)
    
    scene_scale = np.max(updated_depth) * 1.05
    print(f'updated scene depth scale: {scene_scale}')
    ref_distance = updated_depth / scene_scale
    
    # ref_depth_filepath = os.path.join(output_dir, f'{scene_name}_ref_distance.npy')
    # np.save(ref_depth_filepath, ref_distance)
    
    unit_sphere = torch.tensor(get_unit_spherical_map(), device=device)
    ref_normal = depth_to_normal(torch.tensor(updated_depth, device=device), unit_sphere)
    # ref_normal_filepath = os.path.join(output_dir, f'{scene_name}_ref_normal.npy')
    # np.save(ref_normal_filepath, ref_normal.detach().cpu().numpy())
    
    return scene_scale, ref_distance, ref_normal.detach().cpu().numpy()
    

def train_perf(scene_pano_img_path: str,
               scene_layout_json_path: str,
               scene_layout_mesh_path: str,
               text_prompt: str,
               scene_scale: float,
               exp_dir: str,
               use_grid_sampler: bool = True):
    if use_grid_sampler:
        config_fn = 'nerf_grid'
    else:
        config_fn = 'nerf'
    command = "CUDA_VISIBLE_DEVICES=1 python ../PeRF/core_exp_runner.py --config-name {} dataset.image_path={} " \
        "dataset.scene_layout_json_file={} " \
        "dataset.scene_layout_mesh_file={} " \
        "dataset.text_prompt=\'{}\' " \
        "dataset.distance_scale={} " \
        "device.base_exp_dir={} ".format(config_fn,
                                         scene_pano_img_path, 
                                         scene_layout_json_path, 
                                         scene_layout_mesh_path, 
                                         text_prompt, 
                                         scene_scale,
                                         exp_dir)
    print(command)
    os.system(command)
    
def render_perf(scene_pano_img_path: str,
               scene_layout_json_path: str,
               scene_layout_mesh_path: str,
               text_prompt: str,
               scene_scale: float,
               exp_dir: str,
               use_grid_sampler: bool = True):
    if use_grid_sampler:
        config_fn = 'nerf_grid'
    else:
        config_fn = 'nerf'
    command = "CUDA_VISIBLE_DEVICES=1 python ../PeRF/core_exp_runner.py --config-name {} dataset.image_path={} " \
        "dataset.scene_layout_json_file={} " \
        "dataset.scene_layout_mesh_file={} " \
        "dataset.text_prompt=\'{}\' " \
        "dataset.distance_scale={} " \
        "device.base_exp_dir={} mode=render_dense is_continue=true ".format(config_fn,
                                                                            scene_pano_img_path, 
                                                                            scene_layout_json_path, 
                                                                            scene_layout_mesh_path, 
                                                                            text_prompt, 
                                                                            scene_scale,
                                                                            exp_dir)
    print(command)
    os.system(command)

def tsdf_perf(scene_pano_img_path: str,
               scene_layout_json_path: str,
               scene_layout_mesh_path: str,
               text_prompt: str,
               scene_scale: float,
               exp_dir: str,
               use_grid_sampler: bool = True):
    if use_grid_sampler:
        config_fn = 'nerf_grid'
    else:
        config_fn = 'nerf'
    command = "CUDA_VISIBLE_DEVICES=1 python ../PeRF/core_exp_runner.py --config-name {} dataset.image_path={} " \
        "dataset.scene_layout_json_file={} " \
        "dataset.scene_layout_mesh_file={} " \
        "dataset.text_prompt=\'{}\' " \
        "dataset.distance_scale={} " \
        "device.base_exp_dir={} mode=tsdf_recon is_continue=true ".format(config_fn,
                                                                        scene_pano_img_path, 
                                                                        scene_layout_json_path, 
                                                                        scene_layout_mesh_path, 
                                                                        text_prompt, 
                                                                        scene_scale,
                                                                        exp_dir)
    print(command)
    os.system(command)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_ckpt_filepath', type=str, default='../ckpts/control_v11p_sd15_seg_livingroom_fullres_40000.ckpt')
    parser.add_argument('--in_scene_pano_img_path', type=str, default='/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/3dv_experiments/bedrooms/0/scene_03113_560_pano.png')
    parser.add_argument('--in_scene_layout_json_filepath', type=str, default='/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/3dv_experiments/bedrooms/0/scene_03113_560.json')
    parser.add_argument('--full_scene_layout_filepath', type=str, default='/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/3dv_experiments/bedrooms/0/scene_03113_560_full.ply')
    parser.add_argument('--text_prompt', type=str, default='The bedroom has six walls. The room has a cabinet \, a window and a bed . There is a lamp above the bed .')
    parser.add_argument('--use_grid_sampler', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default='./3dv_experiments/bedrooms/0/perf')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ckpt_filepath = args.sd_ckpt_filepath
    in_scene_pano_img_path = args.in_scene_pano_img_path
    in_scene_layout_json_filepath = args.in_scene_layout_json_filepath
    full_scene_layout_filepath = args.full_scene_layout_filepath
    text_prompt = args.text_prompt
    use_gridsampler = args.use_grid_sampler
    output_dir = args.output_dir
    


    # complete scene layout with floor and ceiling
    if not os.path.exists(full_scene_layout_filepath):
        complete_scene_layout(scene_layout_json=in_scene_layout_json_filepath,
                            output_scene_mesh_path=full_scene_layout_filepath,
                            room_centroid=ROOM_CENTROID)
    
    img_parent_dir = os.path.dirname(in_scene_pano_img_path)
    scene_name = os.path.basename(in_scene_pano_img_path).replace('.png','')
    ref_depth_filepath = os.path.join(img_parent_dir, f'{scene_name}_ref_distance.npy')
    ref_normal_filepath = os.path.join(img_parent_dir, f'{scene_name}_ref_normal.npy')
    
    # if os.path.exists(ref_depth_filepath) and os.path.exists(ref_normal_filepath):
    #     ic(f'reference depth and normal files exist: {ref_depth_filepath}, {ref_normal_filepath}, skip alignment.......')
    # else:
    # alignn depth of cam0 to scene layout
    scene_scale, aligned_depth, aligned_normal = geometry_align(scene_pano_img_filepath=in_scene_pano_img_path,
                                                                    scene_layout_json_filepath=in_scene_layout_json_filepath,
                                                                    scene_layout_mesh_filepath=full_scene_layout_filepath,
                                                                    output_dir=output_dir,
                                                                    device=device)

    np.save(ref_depth_filepath, aligned_depth)
    np.save(ref_normal_filepath, aligned_normal)    
    
    torch.cuda.empty_cache()
    
    # complete the scene with PeRF
    scene_name = os.path.basename(in_scene_pano_img_path).replace('.png','')
    exp_dir = os.path.join(output_dir, f'{scene_name}')
    os.makedirs(exp_dir, exist_ok=True)
    
    perf_ckpt_filepath = os.path.join(exp_dir, 
                                      f'WildDataset_{scene_name}', 
                                      'nerf_experiment', 
                                      'checkpoints', 
                                      'ckpt.pth')
    if os.path.exists(perf_ckpt_filepath):
        ic(f'perf ckpt file exists: {perf_ckpt_filepath}, skip training.......')
    else:
        train_perf(scene_pano_img_path=in_scene_pano_img_path,
                scene_layout_json_path=in_scene_layout_json_filepath,
                scene_layout_mesh_path=full_scene_layout_filepath,
                text_prompt=text_prompt,
                scene_scale=scene_scale,
                exp_dir=exp_dir,
                use_grid_sampler=use_gridsampler)
    
    # /mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/scripts/perf_exp/3dv_20240801_test_grid_sampler_select_8_points/scene_03110_207_pano/WildDataset_scene_03110_207_pano/nerf_experiment/dense_images_new_pano/video.mp4
    perf_video_filepath = os.path.join(exp_dir,
                                       f'WildDataset_{scene_name}',
                                       'nerf_experiment',
                                       'dense_images_new_pers',
                                       'rgb_geometry.mp4')
    if os.path.exists(perf_video_filepath):
        ic(f'perf video file exists: {perf_video_filepath}, skip rendering.......')
    else:
        render_perf(scene_pano_img_path=in_scene_pano_img_path,
                    scene_layout_json_path=in_scene_layout_json_filepath,
                    scene_layout_mesh_path=full_scene_layout_filepath,
                    text_prompt=text_prompt,
                    scene_scale=scene_scale,
                    exp_dir=exp_dir,
                    use_grid_sampler=use_gridsampler)
        

    tsdf_perf(scene_pano_img_path=in_scene_pano_img_path,
              scene_layout_json_path=in_scene_layout_json_filepath,
              scene_layout_mesh_path=full_scene_layout_filepath,
              text_prompt=text_prompt,
              scene_scale=scene_scale,
              exp_dir=exp_dir,
              use_grid_sampler=use_gridsampler)
