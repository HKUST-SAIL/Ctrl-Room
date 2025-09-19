import functools
import numpy as np
import cv2

from scipy.ndimage import map_coordinates


def uv_meshgrid(w, h):
    uv = np.stack(np.meshgrid(range(w), range(h)), axis=-1)
    uv = uv.astype(np.float64)
    uv[..., 0] = ((uv[..., 0] + 0.5) / w - 0.5) * 2 * np.pi
    uv[..., 1] = ((uv[..., 1] + 0.5) / h - 0.5) * np.pi
    return uv


@functools.lru_cache()
def _uv_tri(w, h):
    uv = uv_meshgrid(w, h)
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    tan_v = np.tan(uv[..., 1])
    return sin_u, cos_u, tan_v


def uv_tri(w, h):
    sin_u, cos_u, tan_v = _uv_tri(w, h)
    return sin_u.copy(), cos_u.copy(), tan_v.copy()


def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi


def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi


def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5


def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5


def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y


def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    """connect two image corners into a line

    Args:
        p1 (_type_): _description_
        p2 (_type_): _description_
        z (int, optional): _description_. Defaults to -50.
        w (int, optional): _description_. Defaults to 1024.
        h (int, optional): _description_. Defaults to 512.

    Returns:
        _type_: _description_
    """
    if p1[0] == p2[0]:
        return np.array([p1, p2], np.float32)

    # image coords to uv
    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    # uv to xyz(unit sphere)
    sphere_x1, sphere_y1 = uv2xy(u1, v1, z)
    sphere_x2, sphere_y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)

    # interplation on unit sphere
    coord_img_x_lst = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    # delta x and delta y in unit sphere
    vx = sphere_x2 - sphere_x1
    vy = sphere_y2 - sphere_y1
    u_lst = coorx2u(coord_img_x_lst, w)
    ps = (np.tan(u_lst) * sphere_x1 - sphere_y1) / (vy - np.tan(u_lst) * vx)
    cs = np.sqrt((sphere_x1 + ps * vx)**2 + (sphere_y1 + ps * vy)**2)
    # xyz to uv
    interp_v_lst = np.arctan2(z, cs)
    interp_img_y_lst = v2coory(interp_v_lst, h)

    return np.stack([coord_img_x_lst, interp_img_y_lst], axis=-1)


def pano_stretch(img, corners, kx, ky, order=1):
    '''
    img:     [H, W, C]
    corners: [N, 2] in image coordinate (x, y) format
    kx:      Stretching along front-back direction
    ky:      Stretching along left-right direction
    order:   Interpolation order. 0 for nearest-neighbor. 1 for bilinear.
    '''

    # Process image
    sin_u, cos_u, tan_v = uv_tri(img.shape[1], img.shape[0])
    u0 = np.arctan2(sin_u * kx / ky, cos_u)
    v0 = np.arctan(tan_v * np.sin(u0) / sin_u * ky)

    refx = (u0 / (2 * np.pi) + 0.5) * img.shape[1] - 0.5
    refy = (v0 / np.pi + 0.5) * img.shape[0] - 0.5

    # [TODO]: using opencv remap could probably speedup the process a little
    stretched_img = np.stack(
        [map_coordinates(img[..., i], [refy, refx], order=order, mode='wrap') for i in range(img.shape[-1])], axis=-1)

    # Process corners
    corners_u0 = coorx2u(corners[:, 0], img.shape[1])
    corners_v0 = coory2v(corners[:, 1], img.shape[0])
    corners_u = np.arctan2(np.sin(corners_u0) * ky / kx, np.cos(corners_u0))
    C2 = (np.sin(corners_u0) * ky)**2 + (np.cos(corners_u0) * kx)**2
    corners_v = np.arctan2(np.sin(corners_v0), np.cos(corners_v0) * np.sqrt(C2))

    cornersX = u2coorx(corners_u, img.shape[1])
    cornersY = v2coory(corners_v, img.shape[0])
    stretched_corners = np.stack([cornersX, cornersY], axis=-1)

    return stretched_img, stretched_corners


def visualize_pano_stretch(stretched_img, stretched_cor, title):
    '''
    Helper function for visualizing the effect of pano_stretch
    '''
    thikness = 2
    color = (0, 255, 0)
    for i in range(4):
        xys = pano_connect_points(stretched_cor[i * 2], stretched_cor[(i * 2 + 2) % 8], z=-50)
        xys = xys.astype(int)
        blue_split = np.where((xys[1:, 0] - xys[:-1, 0]) < 0)[0]
        if len(blue_split) == 0:
            cv2.polylines(stretched_img, [xys], False, color, 2)
        else:
            t = blue_split[0] + 1
            cv2.polylines(stretched_img, [xys[:t]], False, color, thikness)
            cv2.polylines(stretched_img, [xys[t:]], False, color, thikness)

    for i in range(4):
        xys = pano_connect_points(stretched_cor[i * 2 + 1], stretched_cor[(i * 2 + 3) % 8], z=50)
        xys = xys.astype(int)
        blue_split = np.where((xys[1:, 0] - xys[:-1, 0]) < 0)[0]
        if len(blue_split) == 0:
            cv2.polylines(stretched_img, [xys], False, color, 2)
        else:
            t = blue_split[0] + 1
            cv2.polylines(stretched_img, [xys[:t]], False, color, thikness)
            cv2.polylines(stretched_img, [xys[t:]], False, color, thikness)

    cv2.putText(stretched_img, title, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return stretched_img.astype(np.uint8)


def unitxyz2uv(xyz, equ_w, equ_h, normlized=False):
    x, z, y = np.split(xyz, 3, axis=-1)
    lon = np.arctan2(x, z)
    c = np.sqrt(x**2 + z**2)
    lat = np.arctan2(y, c)

    # longitude and latitude to equirectangular coordinate
    if normlized:
        u = (lon / (2 * np.pi) + 0.5)
        v = (-lat / np.pi + 0.5)
    else:
        u = (lon / (2 * np.pi) + 0.5) * equ_w - 0.5
        v = (-lat / np.pi + 0.5) * equ_h - 0.5
    # if normlized:
    #     u = (lon / (2 * np.pi) + 0.5)
    #     v = (-lat / np.pi + 0.5)
    # else:
    #     u = (lon / (2 * np.pi) + 0.5) * equ_w + 0.5
    #     v = (lat / np.pi + 0.5) * equ_h + 0.5
    return [u, v]


def interpolate_line(p1, p2, num=30):
    """ interpolate line between two points """
    t = np.expand_dims(np.linspace(0, 1, num=num, dtype=np.float32), 1)
    points = p1 * (1 - t) + t * p2
    return points


def transform_xyz2pix(points: np.ndarray):
    """ transform points in camera coordinate to pixel coordinate

    Args:
        points (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    point_norm = np.expand_dims(np.linalg.norm(points, axis=1), axis=1)
    bbox_points_normlized = points / point_norm
    # calc pixel coordinate
    uv = unitxyz2uv(bbox_points_normlized, equ_w=1024, equ_h=512, normlized=False)
    out = np.concatenate([uv[0], uv[1]], axis=1)
    return out


def check_2dline_cross_pano(p1: np.array, p2: np.array, image_w: int = 1024) -> bool:
    """ check if line cross panorama
    Args:
        p1 (np.array): 2d pixel coordinate
        p2 (np.array): 2d pixel coordinate
        image_w (int): image width
    Returns:
        bool: True if cross
    """
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    _p1 = np.array(p1)
    _p2 = np.array(p2)
    dist1 = np.linalg.norm(_p1 - _p2)
    p1b = np.array([p1[0] + image_w, p1[1]])
    p2b = np.array([p2[0] - image_w, p2[1]])
    dist2 = np.linalg.norm(_p1 - p2b)

    if dist1 > dist2:
        return True
    else:
        return False


def check_3dline_cross_pano(p1: np.array, p2: np.array, image_w: int = 1024) -> bool:
    """ check if line cross panorama
    Args:
        p1 (np.array): 3d camera coordinate
        p2 (np.array): 3d camera coordinate
        image_w (int): image width
    Returns:
        bool: True if cross
    """
    p1_pxl = np.round(transform_xyz2pix(np.expand_dims(p1, 0))).astype(np.int64).squeeze(0)
    p2_pxl = np.round(transform_xyz2pix(np.expand_dims(p2, 0))).astype(np.int64).squeeze(0)
    return check_2dline_cross_pano(p1_pxl, p2_pxl, image_w)


def check_3dplane_cross_pano(corners: np.ndarray, plane_idx: int, imgae_w: int = 1024) -> int:
    """ check if plane cross panorama

    Args:
        corners (np.ndarray): 8 corners of object bbox
        plane_idx (int): _description_

    Returns:
        int: how many edges cross panorama
    """

    # traverse 4 edges of the plane, check how many edges cross panorama,
    # if 2 edges cross panorama, then the plane cross panorama
    # if 1 edge cross panorama, then the plane is on the edge of panorama

    cnt = 0
    for i in range(4):
        p1 = np.round(transform_xyz2pix(np.expand_dims(corners[plane_idx[i][0], :], 0))).astype(np.int64).squeeze(0)
        p2 = np.round(transform_xyz2pix(np.expand_dims(corners[plane_idx[i][1], :], 0))).astype(np.int64).squeeze(0)
        if check_2dline_cross_pano(p1, p2, image_w=imgae_w):
            cnt += 1

    return cnt


def get_cubic_plane_mask(corners: np.ndarray,
                         image_w: int = 1024,
                         image_h: int = 512,
                         bbox_class: str = 'bed') -> np.ndarray:
    """ 
    Args:
        corners (np.ndarray): (8,3) corners of bbox in camera coordinate
    """
    # faces_lst = [[0, 1, 3, 2], [0, 1, 5, 4], [0, 2, 6, 4], [2, 3, 7, 6], [1, 3, 7, 5], [4, 5, 7, 6]]
    # arrange corners in counter-clockwise order
    planes1 = np.array([[0, 1], [1, 3], [3, 2], [2, 0]])
    planes2 = np.array([[0, 1], [1, 5], [5, 4], [4, 0]])
    planes3 = np.array([[0, 2], [2, 6], [6, 4], [4, 0]])
    planes4 = np.array([[2, 3], [3, 7], [7, 6], [6, 2]])
    planes5 = np.array([[1, 3], [3, 7], [7, 5], [5, 1]])
    planes6 = np.array([[4, 5], [5, 7], [7, 6], [6, 4]])
    quality = 1023
    mask = np.zeros((512, 1024))

    planes = [planes1, planes2, planes3, planes4, planes5, planes6]

    for i, plane in enumerate(planes):
        # if the plane cross the panorama
        cross_cnt = check_3dplane_cross_pano(corners, plane)
        # print(f'get_cubic_plane_mask: cross_cnt: {cross_cnt}')
        if cross_cnt > 0:
            pix_all_left = []
            pix_all_right = []
            # traverse 4 edges of the plane
            for i in range(4):
                p1 = corners[plane[i][0], :]
                p2 = corners[plane[i][1], :]
                points = interpolate_line(p1, p2, quality)
                pix = np.round(transform_xyz2pix(points)).astype(np.int32)

                for pix_sub in pix:
                    if (pix_sub[0] > image_w / 2):
                        pix_all_right.append(np.expand_dims(pix_sub, 0))
                    else:
                        pix_all_left.append(np.expand_dims(pix_sub, 0))

            if cross_cnt == 1:
                pix_all = pix_all_left + pix_all_right
                pix_all = sorted(pix_all, key=lambda x: x[0][0])
                pixel_all = np.concatenate(pix_all, axis=0)
                x_min = np.min(pixel_all[:, 0], axis=0)
                # print(f'x_min: {x_min}')
                x_max = np.max(pixel_all[:, 0], axis=0)
                # print(f'x_max: {x_max}')

                # if the bbox is attached to the ceiling
                if np.mean(pixel_all[:, 1]) < int(image_h / 2) and bbox_class != 'bed':
                    left_pixel = pixel_all[pixel_all[:, 0] == x_min][0]
                    # print(f'left_pixel: {left_pixel}')
                    right_pixel = pixel_all[pixel_all[:, 0] == x_max][0]
                    # print(f'right_pixel: {right_pixel}')
                    pixel_padding = np.array([[image_w - 1, right_pixel[1]], [image_w - 1, 0], [0, 0],
                                              [0, left_pixel[1]]])
                    # pixel_padding = padding_pixel_top_or_bottom(left_pixel_y[1], right_pixel_y[1], image_w=image_w, y=0)
                    pixel_all = np.concatenate((pixel_all, pixel_padding), axis=0)
                    # print(f'pixel_all: {pixel_all.shape}')
                elif np.mean(pixel_all[:, 1]) > int(image_h / 2) and bbox_class != 'lamp':
                    # if the bbox is attached to the floor
                    left_pixel = pixel_all[pixel_all[:, 0] == x_min][0]
                    # print(f'left_pixel: {left_pixel}')
                    right_pixel = pixel_all[pixel_all[:, 0] == x_max][0]
                    # print(f'right_pixel: {right_pixel}')
                    pixel_padding = np.array([[image_w - 1, right_pixel[1]], [image_w - 1, image_h - 1],
                                              [0, image_h - 1], [0, left_pixel[1]]])

                    pixel_all = np.concatenate((pixel_all, pixel_padding), axis=0)
                cv2.fillPoly(mask, [pixel_all], 1)

            elif cross_cnt == 2:
                pix_all_right = np.concatenate(pix_all_right, axis=0)
                cv2.fillPoly(mask, [pix_all_right], 1)
                pix_all_left = np.concatenate(pix_all_left, axis=0)
                cv2.fillPoly(mask, [pix_all_left], 1)
            else:
                raise ValueError('cross_cnt should be 1 or 2')

            # pix_all_right = np.concatenate(pix_all_right, axis=0)
            # cv2.fillPoly(mask, [pix_all_right], 1)
            # pix_all_left = np.concatenate(pix_all_left, axis=0)
            # cv2.fillPoly(mask, [pix_all_left], 1)
        else:
            pix_all = []
            for i in range(4):
                p1 = corners[plane[i][0], :]
                p2 = corners[plane[i][1], :]
                points = interpolate_line(p1, p2, quality)
                pix = np.round(transform_xyz2pix(points)).astype(np.int64)
                pix_all.append(pix)
            pix_all = np.concatenate(pix_all, axis=0)
            cv2.fillPoly(mask, [pix_all], 1)

    return mask


if __name__ == '__main__':

    import argparse
    import time
    from PIL import Image
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', default='data/valid/img/pano_abpohapclcyuuz.png')
    parser.add_argument('--i_gt', default='data/valid/label_cor/pano_abpohapclcyuuz.txt')
    parser.add_argument('--o', default='sample_stretched_pano.png')
    parser.add_argument('--kx', default=2, type=float, help='Stretching along front-back direction')
    parser.add_argument('--ky', default=1, type=float, help='Stretching along left-right direction')
    args = parser.parse_args()

    img = np.array(Image.open(args.i), np.float64)
    with open(args.i_gt) as f:
        cor = np.array([line.strip().split() for line in f], np.int32)
    stretched_img, stretched_cor = pano_stretch(img, cor, args.kx, args.ky)

    title = 'kx=%3.2f, ky=%3.2f' % (args.kx, args.ky)
    visual_stretched_img = visualize_pano_stretch(stretched_img, stretched_cor, title)
    Image.fromarray(visual_stretched_img).save(args.o)
