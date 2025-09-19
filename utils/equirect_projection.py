import cv2
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.transform import Rotation
from .utils import euler_angle_to_matrix
from .panostretch import get_cubic_plane_mask


def interpolate_line(p1, p2, num=30):
    t = np.expand_dims(np.linspace(0, 1, num=num, dtype=np.float32), 1)
    points = p1 * (1 - t) + t * p2
    return points


def cam3d2rad(cam3d):
    """
    Transform 3D points in camera coordinate to longitude and latitude.

    Parameters
    ----------
    cam3d: n x 3 numpy array or bdb3d dict

    Returns
    -------
    n x 2 numpy array of longitude and latitude in radiation
    first rotate left-right, then rotate up-down
    longitude: (left) -pi -- 0 --> +pi (right)
    latitude: (up) -pi/2 -- 0 --> +pi/2 (down)
    """
    backend, atan2 = (np, np.arctan2)
    lon = atan2(cam3d[..., 0], cam3d[..., 1])
    # lat = backend.arcsin(cam3d[..., 1] / backend.linalg.norm(cam3d, axis=-1))
    lat = backend.arccos(cam3d[..., 2] / backend.linalg.norm(cam3d, axis=-1)) - np.pi / 2
    return backend.stack([lon, lat], -1)


def camrad2pix(camrad, img_height=512, img_width=1024):
    """
    Transform longitude and latitude of a point to panorama pixel coordinate.

    Parameters
    ----------
    camrad: n x 2 numpy array

    Returns
    -------
    n x 2 numpy array of xy coordinate in pixel
    x: (left) 0 --> (width - 1) (right)
    y: (up) 0 --> (height - 1) (down)
    """
    # if 'K' in self.camera:
    #     raise NotImplementedError
    # if isinstance(camrad, torch.Tensor):
    #     campix = torch.empty_like(camrad, dtype=torch.float32)
    # else:
    campix = np.empty_like(camrad, dtype=np.float32)
    # if isinstance(camrad, torch.Tensor):
    #     width, height = [x.view([-1] + [1] * (camrad.dim() - 2))
    #                      for x in (width, height)]
    campix[..., 0] = camrad[..., 0] * img_width / (2. * np.pi) + img_width / 2. + 0.5
    campix[..., 1] = camrad[..., 1] * img_height / np.pi + img_height / 2. + 0.5
    return campix


def cam3d2pix(cam3d, image):
    """
    Transform 3D points from camera coordinate to pixel coordinate.

    Parameters
    ----------
    cam3d: n x 3 numpy array or bdb3d dict

    Returns
    -------
    for 3D points: n x 2 numpy array of xy in pixel.
    x: (left) 0 --> width - 1 (right)
    y: (up) 0 --> height - 1 (down)
    """
    # if isinstance(cam3d, dict):
    #     campix = self.world2campix(self.cam3d2world(cam3d))
    # else:
    #     if 'K' in self.camera:
    #         campix = self.transform(self.camera['K'], cam3d)
    #     else:
    img_height, img_width = image.shape[:2]
    campix = camrad2pix(cam3d2rad(cam3d), img_height=img_height, img_width=img_width)
    return campix


def obj2frame(point, bdb3d):
    """
    Transform 3D points or Trimesh from normalized object coordinate frame to coordinate frame bdb3d is in.
    object: x-left, y-back, z-up (defined by iGibson)
    world: right-hand coordinate of iGibson (z-up)

    Parameters
    ----------
    point: n x 3 numpy array or Trimesh
    bdb3d: dict, self['objs'][id]['bdb3d']

    Returns
    -------
    n x 3 numpy array or Trimesh
    """
    # rotation = Rotation.from_euler('zyx', bdb3d['angles'], degrees=False).as_matrix()
    rotation = euler_angle_to_matrix(bdb3d['angles'])
    centroid = np.array(bdb3d['center'])
    sizes = np.array(bdb3d['size'])
    return (rotation @ (point * sizes).T).T + centroid


def bdb3d_corners(bdb3d: (dict, np.ndarray)):
    """
    Get ordered corners of given 3D bounding box dict or disordered corners

    Parameters
    ----------
    bdb3d: 3D bounding box dict

    Returns
    -------
    8 x 3 numpy array of bounding box corner points in the following order:
    right-forward-down
    left-forward-down
    right-back-down
    left-back-down
    right-forward-up
    left-forward-up
    right-back-up
    left-back-up
    """
    if isinstance(bdb3d, np.ndarray):
        centroid = np.mean(bdb3d, axis=0)
        z = bdb3d[:, -1]
        surfaces = []
        for surface in (bdb3d[z < centroid[-1]], bdb3d[z >= centroid[-1]]):
            surface_2d = surface[:, :2]
            center_2d = centroid[:2]
            vecters = surface_2d - center_2d
            angles = np.arctan2(vecters[:, 0], vecters[:, 1])
            orders = np.argsort(-angles)
            surfaces.append(surface[orders][(0, 1, 3, 2), :])
        corners = np.concatenate(surfaces)
    else:
        # corners = np.unpackbits(np.arange(8, dtype=np.uint8)[..., np.newaxis],
        #                         axis=1, bitorder='little', count=-5).astype(np.float32)
        corners = np.zeros((8, 3), dtype=np.float32)
        corners[0, :] = np.array([1., 1., 0.])
        corners[1, :] = np.array([0., 1., 0.])
        corners[2, :] = np.array([1., 0., 0.])
        corners[3, :] = np.array([0., 0., 0.])
        corners[4, :] = np.array([1., 1., 1.])
        corners[5, :] = np.array([0., 1., 1.])
        corners[6, :] = np.array([1., 0., 1.])
        corners[7, :] = np.array([0., 0., 1.])
        corners = corners - 0.5
        corners = obj2frame(corners, bdb3d)
    return corners


def bdb3d_corners_no_order(basis: np.array, centroid: np.array, half_sizes: np.array) -> np.array:
    """_summary_

    Args:
        basis (np.array): bounding box orientation
        centroid (np.array): bounding box center coordinate(mm)
        half_sizes (np.array): bounding box radii(mm)

    Returns:
    -------
    8 x 3 numpy array of bounding box corner points in the following order:
    right-forward-down
    left-forward-down
    right-back-down
    left-back-down
    right-forward-up
    left-forward-up
    right-back-up
    left-back-up
    """
    corners = np.zeros((8, 3))
    coeffs = np.abs(half_sizes)

    corners[1, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[0, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]

    corners[5, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[4, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners = corners + np.tile(centroid, (8, 1))
    return corners


def wrapped_line(image, p1, p2, colour, thickness, lineType=cv2.LINE_AA):
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    _p1 = np.array(p1)
    _p2 = np.array(p2)

    dist1 = np.linalg.norm(_p1 - _p2)

    p1b = np.array([p1[0] + image.shape[1], p1[1]])
    p2b = np.array([p2[0] - image.shape[1], p2[1]])

    dist2 = np.linalg.norm(_p1 - p2b)

    if dist1 < dist2:
        cv2.line(image, p1, p2, colour, thickness, lineType=lineType)
    else:
        cv2.line(image, p1, tuple(p2b), colour, thickness, lineType=lineType)
        cv2.line(image, tuple(p1b), p2, colour, thickness, lineType=lineType)


# visualize 3dbbox on panorama
def vis_objs3d(image,
               v_bbox3d,
               camera_position,
               color_to_labels,
               b_show_axes=False,
               b_show_centroid=False,
               b_show_bbox3d=True,
               b_show_info=False,
               b_show_polygen=False,
               thickness=2):

    def draw_line3d(image, p1, p2, color, thickness, quality=30, frame='cam3d'):
        color = (np.ones(3, dtype=np.uint8) * color).tolist()
        if frame != 'cam3d':
            print('input points must be in camera frame')
            raise NotImplementedError
        points = interpolate_line(p1, p2, quality)
        normal_points = normalize(points)
        pix = np.round(cam3d2pix(cam3d=normal_points, image=image)).astype(np.int32)
        for t in range(quality - 1):
            p1, p2 = pix[t], pix[t + 1]
            wrapped_line(image, tuple(p1), tuple(p2), color, thickness, lineType=cv2.LINE_AA)

    def draw_objaxes(image, centroid, sizes, rotation, thickness=2):

        for axis in np.eye(3, dtype=np.float32):
            endpoint = rotation @ ((axis / 2) * sizes) + centroid
            color = axis * 255
            draw_line3d(image, centroid, endpoint, color, thickness, frame='cam3d')

    def draw_centroid(image, centroid, color, thickness=2):
        color = (np.ones(3, dtype=np.uint8) * color).tolist()
        normal_centroid = centroid / np.linalg.norm(centroid)
        center = cam3d2pix(normal_centroid, image)
        cv2.circle(image, tuple(center.astype(np.int32).tolist()), 5, color, thickness=thickness, lineType=cv2.LINE_AA)

    def draw_bdb3d(image, bdb3d, color, thickness=2):
        bbox_frame = 'camera'
        if bbox_frame == 'world':
            centroid = np.array(bdb3d['centroid'])
            sizes = np.array(bdb3d['dimensions'])
            rotation = np.array(bdb3d['rotations'])
            corners = bdb3d_corners_no_order(basis=rotation, centroid=centroid, half_sizes=sizes)
            # print(f'corners in world frame: {corners}')
            corners = (corners - camera_position) * 0.001
        elif bbox_frame == 'camera':
            corners = bdb3d_corners(bdb3d)
        # print(f'corners in camera frame: {corners}')
        corners_box = corners.reshape(2, 2, 2, 3)

        for k in [0, 1]:
            for l in [0, 1]:
                for idx1, idx2 in [((0, k, l), (1, k, l)), ((k, 0, l), (k, 1, l)), ((k, l, 0), (k, l, 1))]:
                    draw_line3d(image, corners_box[idx1], corners_box[idx2], color, thickness=thickness, frame='cam3d')

    def draw_objinfo(image, bdb3d_centeroid_w, obj_cls_name, color):
        """ draw object name on the top of the object bbox

        Args:
            image (np.array): _description_
            bdb3d_centeroid_w (_type_): object bounding box centroid in world frame
            obj_cls_name (_type_): _description_
            color (_type_): _description_
        """
        # bdb3d centroid in camera frame
        # bdb3d_centeroid_c = (bdb3d_centeroid_w - camera_position) * 0.001
        bdb3d_centeroid_c = bdb3d_centeroid_w
        normal_centroid = bdb3d_centeroid_c / np.linalg.norm(bdb3d_centeroid_c)
        bdb3d_pix = cam3d2pix(normal_centroid, image=image)
        bottom_left = bdb3d_pix.astype(np.int32)
        # bottom_left[1] -= 6
        cv2.putText(image,
                    obj_cls_name,
                    tuple(bottom_left.tolist()),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness=1,
                    lineType=cv2.LINE_AA)

    def draw_Poly3d(image, bdb3d, color, object_label=None):
        bbox_frame = 'camera'
        if bbox_frame == 'world':
            centroid = np.array(bdb3d['centroid'])
            sizes = np.array(bdb3d['dimensions'])
            rotation = np.array(bdb3d['rotations'])
            corners = bdb3d_corners_no_order(basis=rotation, centroid=centroid, half_sizes=sizes)
            # print(f'corners in world frame: {corners}')
            corners = (corners - camera_position) * 0.001
        elif bbox_frame == 'camera':
            corners = bdb3d_corners(bdb3d)
        # print(f'corners in camera frame: {corners}')
        corners_box = corners.reshape(-1, 3)

        # get (512,1024) mask for object
        img_h, img_w = image.shape[:2]
        object_mask = get_cubic_plane_mask(corners_box, image_w=img_w, image_h=img_h, bbox_class=object_label)
        image[object_mask > 0] = color

    image = image.copy()
    wall_ids = [i for i, o in enumerate(v_bbox3d) if o['class'] == 'wall']
    if b_show_polygen:
        # ignore curtain to visualize window
        obj_ids = [i for i, o in enumerate(v_bbox3d) if (o['class'] not in ['wall', 'curtain'])]
    else:
        obj_ids = [i for i, o in enumerate(v_bbox3d) if (o['class'] not in ['wall'])]

    dis = {id: np.linalg.norm([v_bbox3d[id]['center']]) for id in obj_ids}
    obj_ids = sorted(obj_ids, key=lambda k: dis[k], reverse=True)

    i_objs = wall_ids + obj_ids
    # draw object from far to near, always draw wall at first
    for i_obj in i_objs:
        bdb3d = v_bbox3d[i_obj]
        obj_label = bdb3d['class']

        if color_to_labels is not None:
            labels_lst = list(color_to_labels.values())
            colors_lst = list(color_to_labels.keys())
            color = colors_lst[labels_lst.index(obj_label)]
        else:
            color = (np.random.random(3) * 255).astype(np.uint8).tolist()

        centroid = np.array(bdb3d['center'])
        sizes = np.array(bdb3d['size'])
        rotation = euler_angle_to_matrix(bdb3d['angles'])

        if b_show_axes:
            draw_objaxes(image, centroid, sizes, rotation, thickness=thickness)
        if b_show_centroid:
            draw_centroid(image, centroid, color, thickness=thickness)
        if b_show_bbox3d:
            draw_bdb3d(image, bdb3d, color, thickness=thickness)
        if b_show_info:
            draw_objinfo(image, centroid, f'{obj_label}_{i_obj}', color)
        if b_show_polygen:
            draw_Poly3d(image, bdb3d, color, object_label=obj_label)
    return image


from .panorama import lineIdxFromCors
from .panostretch import check_3dline_cross_pano

# def vis_floor_ceiling(image: np.array, floor_points: np.array, ceiling_points: np.array, color_to_labels: dict = None):
#     """visualize floor and ceiling of a room, the floor and ceiling points are in camera frame

#     Args:
#         image (np.array): panorama image of the room
#         floor_points (_type_): _description_
#         ceiling_points (_type_): _description_
#         color_to_labels (dict, optional): color for ceiling and floor.

#     Returns:
#         np.array: _description_
#     """
#     image = image.copy()
#     im_h, im_w = image.shape[:2]
#     floor_points = floor_points.reshape(-1, 3)
#     ceiling_points = ceiling_points.reshape(-1, 3)

#     if color_to_labels is not None:
#         labels_lst = list(color_to_labels.values())
#         colors_lst = list(color_to_labels.keys())
#         color = colors_lst[labels_lst.index('floor')]
#     else:
#         color = (np.random.random(3) * 255).astype(np.uint8).tolist()

#     # assume the floor points are sorted in counter-clockwise order
#     quality = 512
#     cross_pix_right = []
#     cross_pix_left = []
#     pix_all = []

#     coords_2d = []
#     # sort the pixel of floor points in counter-clockwise order
#     for i in range(0, (floor_points.shape[0] // 2)):
#         p1 = floor_points[2 * i]
#         p2 = floor_points[2 * i + 1]
#         p1_normalized = p1 / np.linalg.norm(p1)
#         p2_normalized = p2 / np.linalg.norm(p2)
#         p1_pix = np.round(cam3d2pix(cam3d=np.expand_dims(p1_normalized, axis=0), image=image)).astype(np.int32)
#         p2_pix = np.round(cam3d2pix(cam3d=np.expand_dims(p2_normalized, axis=0), image=image)).astype(np.int32)
#         coords_2d.append(p1_pix)
#         coords_2d.append(p2_pix)
#     coords_2d = np.concatenate(coords_2d, axis=0)
#     print(f'coords_2d: {coords_2d}')
#     min_coord_idx = np.argmin(coords_2d[::2, 0])
#     coords_2d = np.roll(coords_2d[:, :2], -2 * min_coord_idx, 0)
#     print(f'rolled coords_2d: {coords_2d}')
#     sorted_floor_points = np.roll(floor_points[:, :3], -2 * min_coord_idx, 0)
#     print(f'rolled_floor_points: {sorted_floor_points}')

#     for i in range(0, (sorted_floor_points.shape[0] // 2)):
#         p1 = sorted_floor_points[2 * i]
#         p2 = sorted_floor_points[2 * i + 1]
#         p1_normalized = p1 / np.linalg.norm(p1)
#         p2_normalized = p2 / np.linalg.norm(p2)
#         p1_pix = np.round(cam3d2pix(cam3d=np.expand_dims(p1_normalized, axis=0), image=image)).astype(np.int32)
#         p2_pix = np.round(cam3d2pix(cam3d=np.expand_dims(p2_normalized, axis=0), image=image)).astype(np.int32)
#         print(f'p1_pix: {p1_pix[0]}, p2_pix: {p2_pix[0]}')

#         points = interpolate_line(p1, p2, num=quality)
#         normal_points = normalize(points)
#         pix = np.round(cam3d2pix(cam3d=normal_points, image=image)).astype(np.int32)
#         if check_3dline_cross_pano(p1, p2, image_w=im_w):
#             print(f'p1: {p1}, p2: {p2} cross pano')
#             # current edge cross the pano
#             for pix_sub in pix:
#                 if (pix_sub[0] > im_w / 2):
#                     cross_pix_right.append(np.expand_dims(pix_sub, 0))
#                 else:
#                     cross_pix_left.append(np.expand_dims(pix_sub, 0))
#         else:
#             # current edge is in the pano
#             pix_all.append(pix)
#     cross_pix_left = sorted(cross_pix_left, key=lambda k: k[0][0])
#     cross_pix_right = sorted(cross_pix_right, key=lambda k: k[0][0])
#     pix_all = cross_pix_left + pix_all + cross_pix_right
#     floor_pixels = np.concatenate(pix_all, axis=0)
#     x_min = np.min(floor_pixels[:, 0], axis=0)
#     x_max = np.max(floor_pixels[:, 0], axis=0)
#     left_pixel = floor_pixels[floor_pixels[:, 0] == x_min][0]
#     right_pixel = floor_pixels[floor_pixels[:, 0] == x_max][0]
#     padding_pixels = np.array([[1023, right_pixel[1]], [1023, 511], [0, 511], [0, left_pixel[1]]])
#     floor_pixels = np.concatenate([floor_pixels, padding_pixels], axis=0)
#     cv2.fillPoly(image, pts=[floor_pixels], color=color)

#     if color_to_labels is not None:
#         labels_lst = list(color_to_labels.values())
#         colors_lst = list(color_to_labels.keys())
#         color = colors_lst[labels_lst.index('ceiling')]
#     else:
#         color = (np.random.random(3) * 255).astype(np.uint8).tolist()

#     cross_pix_right = []
#     cross_pix_left = []
#     pix_all = []
#     coords_2d = []
#     # sort the pixel of ceiling points in counter-clockwise order
#     for i in range(0, (ceiling_points.shape[0] // 2)):
#         p1 = ceiling_points[2 * i]
#         p2 = ceiling_points[2 * i + 1]
#         p1_normalized = p1 / np.linalg.norm(p1)
#         p2_normalized = p2 / np.linalg.norm(p2)
#         p1_pix = np.round(cam3d2pix(cam3d=np.expand_dims(p1_normalized, axis=0), image=image)).astype(np.int32)
#         p2_pix = np.round(cam3d2pix(cam3d=np.expand_dims(p2_normalized, axis=0), image=image)).astype(np.int32)
#         coords_2d.append(p1_pix)
#         coords_2d.append(p2_pix)
#     coords_2d = np.concatenate(coords_2d, axis=0)
#     print(f'ceiling coords_2d: {coords_2d}')
#     min_coord_idx = np.argmin(coords_2d[::2, 0])
#     coords_2d = np.roll(coords_2d[:, :2], -2 * min_coord_idx, 0)
#     print(f'rolled ceiling coords_2d: {coords_2d}')
#     sorted_ceiling_points = np.roll(ceiling_points[:, :3], -2 * min_coord_idx, 0)
#     print(f'rolled_floor_points: {sorted_ceiling_points}')

#     for i in range(0, (sorted_ceiling_points.shape[0] // 2)):
#         p1 = sorted_ceiling_points[2 * i]
#         p2 = sorted_ceiling_points[2 * i + 1]
#         p1_normalized = p1 / np.linalg.norm(p1)
#         p2_normalized = p2 / np.linalg.norm(p2)
#         p1_pix = np.round(cam3d2pix(cam3d=np.expand_dims(p1_normalized, axis=0), image=image)).astype(np.int32)
#         p2_pix = np.round(cam3d2pix(cam3d=np.expand_dims(p2_normalized, axis=0), image=image)).astype(np.int32)
#         print(f'p1_pix: {p1_pix[0]}, p2_pix: {p2_pix[0]}')

#         points = interpolate_line(p1, p2, num=quality)
#         normal_points = normalize(points)
#         pix = np.round(cam3d2pix(cam3d=normal_points, image=image)).astype(np.int32)
#         if check_3dline_cross_pano(p1, p2, image_w=im_w):
#             print(f'p1: {p1}, p2: {p2} cross pano')
#             # current edge cross the pano
#             for pix_sub in pix:
#                 if (pix_sub[0] > im_w / 2):
#                     cross_pix_right.append(np.expand_dims(pix_sub, 0))
#                 else:
#                     cross_pix_left.append(np.expand_dims(pix_sub, 0))
#         else:
#             # current edge is in the pano
#             pix_all.append(pix)
#     cross_pix_left = sorted(cross_pix_left, key=lambda k: k[0][0])
#     cross_pix_right = sorted(cross_pix_right, key=lambda k: k[0][0])
#     pix_all = cross_pix_left + pix_all + cross_pix_right
#     ceiling_pixels = np.concatenate(pix_all, axis=0)
#     x_min = np.min(ceiling_pixels[:, 0], axis=0)
#     x_max = np.max(ceiling_pixels[:, 0], axis=0)
#     left_pixel = ceiling_pixels[ceiling_pixels[:, 0] == x_min][0]
#     right_pixel = ceiling_pixels[ceiling_pixels[:, 0] == x_max][0]
#     padding_pixels = np.array([[1023, right_pixel[1]], [1023, 0], [0, 0], [0, left_pixel[1]]])
#     ceiling_pixels = np.concatenate([ceiling_pixels, padding_pixels], axis=0)
#     cv2.fillPoly(image, pts=[ceiling_pixels], color=color)

#     return image


def vis_floor_ceiling_simple(image: np.array, color_to_labels: dict = None):
    img_h, img_w = image.shape[:2]

    if color_to_labels is not None:
        labels_lst = list(color_to_labels.values())
        colors_lst = list(color_to_labels.keys())
        color = colors_lst[labels_lst.index('ceiling')]
    else:
        color = (np.random.random(3) * 255).astype(np.uint8).tolist()

    ceiling_pixels = np.array([[0, 0], [0, img_h // 2], [img_w - 1, img_h // 2], [img_w - 1, 0]])
    cv2.fillPoly(image, pts=[ceiling_pixels], color=color)

    if color_to_labels is not None:
        labels_lst = list(color_to_labels.values())
        colors_lst = list(color_to_labels.keys())
        color = colors_lst[labels_lst.index('floor')]
    else:
        color = (np.random.random(3) * 255).astype(np.uint8).tolist()

    floor_pixels = np.array([[0, img_h // 2], [0, img_h - 1], [img_w - 1, img_h - 1], [img_w - 1, img_h // 2]])
    cv2.fillPoly(image, pts=[floor_pixels], color=color)

    return image


def vis_floor_ceiling(image: np.array, coords_2d: np.array, color_to_labels: dict = None):
    """
    Visualize the floor and ceiling in the panorama image
    :param image: panorama image
    :param coords_2d_filepath: the file path of the 2D coordinates of floor and ceiling
    :param color_to_labels: color to labels mapping
    :return: panorama image with floor and ceiling
    """
    img_h, img_w = image.shape[:2]
    ceiling_boundary_lst = coords_2d[0, :]
    floor_boundary_lst = coords_2d[1, :]
    # print(f'floor_boundary_lst: {floor_boundary_lst.shape}')
    # print(f'ceiling_boundary_lst: {ceiling_boundary_lst.shape}')
    coords_x = np.arange(0, img_w)
    floor_boundary_coords_y = ((floor_boundary_lst / np.pi + 0.5) * img_h - 0.5).astype(np.int32)
    floor_boundary_pixels = np.stack([coords_x, floor_boundary_coords_y], axis=1)
    # print(f'floor_boundary_pixels: {floor_boundary_pixels}')
    ceiling_boundary_coords_y = ((ceiling_boundary_lst / np.pi + 0.5) * img_h - 0.5).astype(np.int32)
    ceiling_boundary_pixels = np.stack([coords_x, ceiling_boundary_coords_y], axis=1)
    # print(f'ceiling_boundary_pixels: {ceiling_boundary_pixels}')

    if color_to_labels is not None:
        labels_lst = list(color_to_labels.values())
        colors_lst = list(color_to_labels.keys())
        color = colors_lst[labels_lst.index('ceiling')]
    else:
        color = (np.random.random(3) * 255).astype(np.uint8).tolist()

    x_min = np.min(ceiling_boundary_pixels[:, 0], axis=0)
    x_max = np.max(ceiling_boundary_pixels[:, 0], axis=0)
    left_pixel = ceiling_boundary_pixels[ceiling_boundary_pixels[:, 0] == x_min][0]
    right_pixel = ceiling_boundary_pixels[ceiling_boundary_pixels[:, 0] == x_max][0]
    padding_pixels = np.array([[1023, right_pixel[1]], [1023, 0], [0, 0], [0, left_pixel[1]]])
    ceiling_pixels = np.concatenate([ceiling_boundary_pixels, padding_pixels], axis=0)
    cv2.fillPoly(image, pts=[ceiling_pixels], color=color)

    if color_to_labels is not None:
        labels_lst = list(color_to_labels.values())
        colors_lst = list(color_to_labels.keys())
        color = colors_lst[labels_lst.index('floor')]
    else:
        color = (np.random.random(3) * 255).astype(np.uint8).tolist()

    x_min = np.min(floor_boundary_pixels[:, 0], axis=0)
    x_max = np.max(floor_boundary_pixels[:, 0], axis=0)
    left_pixel = floor_boundary_pixels[floor_boundary_pixels[:, 0] == x_min][0]
    right_pixel = floor_boundary_pixels[floor_boundary_pixels[:, 0] == x_max][0]
    padding_pixels = np.array([[1023, right_pixel[1]], [1023, 511], [0, 511], [0, left_pixel[1]]])
    floor_pixels = np.concatenate([floor_boundary_pixels, padding_pixels], axis=0)
    cv2.fillPoly(image, pts=[floor_pixels], color=color)

    return image