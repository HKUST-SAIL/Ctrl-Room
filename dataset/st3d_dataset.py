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
import json

import trimesh

import torch.utils.data as data
from utils import panostretch
from .metadata import (ST3D_BEDROOM_FURNITURE, ST3D_LIVINGROOM_FURNITURE, ST3D_DININGROOM_FURNITURE,  ST3D_KITCHEN_FURNITURE, ST3D_STUDY_FURNITURE, ST3D_BATHROOM_FURNITURE,\
ST3D_BEDROOM_MAX_LEN, ST3D_DININGROOM_MAX_LEN, ST3D_LIVINGROOM_MAX_LEN, ST3D_KITCHEN_MAX_LEN, ST3D_STUDY_MAX_LEN, ST3D_BATHROOM_MAX_LEN,\
ST3D_BEDROOM_FURNITURE_CNTS, ST3D_DININGROOM_FURNITURE_CNTS, ST3D_LIVINGROOM_FURNITURE_CNTS, ST3D_KITCHEN_FURNITURE_CNTS, ST3D_STUDY_FURNITURE_CNTS, ST3D_BATHROOM_FURNITURE_CNTS,\
    ST3D_BEDROOM_QUAD_WALL_MAX_LEN, ST3D_LIVINGROOM_QUAD_WALL_MAX_LEN, ST3D_KITCHEN_QUAD_WALL_MAX_LEN, ST3D_STUDY_QUAD_WALL_MAX_LEN, ST3D_BATHROOM_QUAD_WALL_MAX_LEN)

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
from dataset.gen_scene_text import get_scene_description
from improved_diffusion.clip_util import FrozenCLIPEmbedder

# ROOM_CLASS_LST = [10, 8, 3, 2, 0, 4, 5, 14, 13, 12, 7, 9, 11, 1, 6, 15]
def get_room_type(room_type_v: int) -> str:
    for k, v in ROOM_TYPE_DICT.items():
        if v == room_type_v:
            return k
    return 'undefined'


def find_occlusion(coor):
    # equirectangular coordinates to sperical image coordinates
    img_x = coor[:, 0]
    img_y = coor[:, 1]
    u = panostretch.coorx2u(img_x)
    v = panostretch.coory2v(img_y)
    # spherical camera coordinates, assume z=-50
    x, y = panostretch.uv2xy(u, v, z=-50)
    occlusion = []
    for i in range(len(x)):
        raycast = LineString([(0, 0), (x[i], y[i])])
        other_layout = []
        for j in range(i + 1, len(x)):
            other_layout.append((x[j], y[j]))
        for j in range(0, i):
            other_layout.append((x[j], y[j]))
        other_layout = LineString(other_layout)
        occlusion.append(raycast.intersects(other_layout))
    return np.array(occlusion)


def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first) * 2 - 1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys


def corners_to_1d_boundary(corner_lst, H, W):
    bound_ceil_x_lst, bound_ceil_y_lst = [], []
    bound_floor_x_lst, bound_floor_y_lst = [], []
    num_corners = len(corner_lst)

    # connect ceiling-wall boundary
    for i in range(num_corners // 2):
        xys = panostretch.pano_connect_points(corner_lst[i * 2], corner_lst[(i * 2 + 2) % num_corners], z=-50, w=W, h=H)
        bound_ceil_x_lst.extend(xys[:, 0])
        bound_ceil_y_lst.extend(xys[:, 1])

    # connect floor-wall boundary
    for i in range(num_corners // 2):
        xys = panostretch.pano_connect_points(corner_lst[i * 2 + 1],
                                              corner_lst[(i * 2 + 3) % num_corners],
                                              z=50,
                                              w=W,
                                              h=H)
        bound_floor_x_lst.extend(xys[:, 0])
        bound_floor_y_lst.extend(xys[:, 1])
    bound_ceil_x_lst, bound_ceil_y_lst = sort_xy_filter_unique(bound_ceil_x_lst, bound_ceil_y_lst, y_small_first=True)
    bound_floor_x_lst, bound_floor_y_lst = sort_xy_filter_unique(bound_floor_x_lst,
                                                                 bound_floor_y_lst,
                                                                 y_small_first=False)
    # ceiling boundary and floor boundary
    boundary_lst = np.zeros((2, W))
    boundary_lst[0] = np.interp(x=np.arange(W), xp=bound_ceil_x_lst, fp=bound_ceil_y_lst, period=W)
    boundary_lst[1] = np.interp(x=np.arange(W), xp=bound_floor_x_lst, fp=bound_floor_y_lst, period=W)
    # scale to [-pi/2, pi/2]
    boundary_lst = ((boundary_lst + 0.5) / H - 0.5) * np.pi
    return boundary_lst


def layout_2_depth(cor_id, h, w, floor_height=1.6, return_mask=False):
    # Convert corners to per-column boundary first
    # Up -pi/2,  Down pi/2
    ceiling_bound_lst, floor_bound_lst = corners_to_1d_boundary(cor_id, h, w)
    ceiling_bound_lst = ceiling_bound_lst[None, :]  # [1, w]
    floor_bound_lst = floor_bound_lst[None, :]  # [1, w]
    assert (ceiling_bound_lst > 0).sum() == 0
    assert (floor_bound_lst < 0).sum() == 0

    # Per-pixel v coordinate (vertical angle)
    v_lst = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    v_lst = np.repeat(v_lst[:, None], w, axis=1)  # [h, w]

    # Floor-plane to depth
    # floor_h = 1.6
    floor_d = np.abs(floor_height / np.sin(v_lst))

    # wall to camera distance on horizontal plane at cross camera center
    cs = floor_height / np.tan(floor_bound_lst)

    # Ceiling-plane to depth
    ceil_height = np.abs(cs * np.tan(ceiling_bound_lst))  # [1, w]
    ceil_d = np.abs(ceil_height / np.sin(v_lst))  # [h, w]

    # Wall to depth
    wall_d = np.abs(cs / np.cos(v_lst))  # [h, w]

    # Recover layout depth
    floor_mask = (v_lst > floor_bound_lst)
    ceil_mask = (v_lst < ceiling_bound_lst)
    wall_mask = (~floor_mask) & (~ceil_mask)
    depth = np.zeros([h, w], np.float32)  # [h, w]
    depth[floor_mask] = floor_d[floor_mask]
    depth[ceil_mask] = ceil_d[ceil_mask]
    depth[wall_mask] = wall_d[wall_mask]

    assert (depth == 0).sum() == 0
    if return_mask:
        return depth, floor_mask, ceil_mask, wall_mask
    return depth


def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * np.pi


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * np.pi


def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u) + floorW / 2 - 0.5
    y = -c * np.cos(u) + floorH / 2 - 0.5
    return np.hstack([x[:, None], y[:, None]])


def np_x_u_solve_y(x, u, floorW=1024, floorH=512):
    c = (x - floorW / 2 + 0.5) / np.sin(u)
    return -c * np.cos(u) + floorH / 2 - 0.5


def np_y_u_solve_x(y, u, floorW=1024, floorH=512):
    c = -(y - floorH / 2 + 0.5) / np.cos(u)
    return c * np.sin(u) + floorW / 2 - 0.5


def np_xy2coor(xy, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    xy: N x 2
    '''
    x = xy[:, 0] - floorW / 2 + 0.5
    y = xy[:, 1] - floorH / 2 + 0.5

    u = np.arctan2(x, -y)
    v = np.arctan(z / np.sqrt(x**2 + y**2))

    coorx = (u / (2 * np.pi) + 0.5) * coorW - 0.5
    coory = (-v / np.pi + 0.5) * coorH - 0.5

    return np.hstack([coorx[:, None], coory[:, None]])


def vote(vec, tol):
    vec = np.sort(vec)
    n = np.arange(len(vec))[::-1]
    n = n[:, None] - n[None, :] + 1.0
    l = squareform(pdist(vec[:, None], 'minkowski', p=1) + 1e-9)

    invalid = (n < len(vec) * 0.4) | (l > tol)
    if (~invalid).sum() == 0 or len(vec) < tol:
        best_fit = np.median(vec)
        p_score = 0
    else:
        l[invalid] = 1e5
        n[invalid] = -1
        score = n
        max_idx = score.argmax()
        max_row = max_idx // len(vec)
        max_col = max_idx % len(vec)
        assert max_col > max_row
        best_fit = vec[max_row:max_col + 1].mean()
        p_score = (max_col - max_row + 1) / len(vec)

    l1_score = np.abs(vec - best_fit).mean()

    return best_fit, p_score, l1_score


def get_gpid(coorx, coorW):
    gpid = np.zeros(coorW)
    gpid[np.round(coorx).astype(int)] = 1
    gpid = np.cumsum(gpid).astype(int)
    gpid[gpid == gpid[-1]] = 0
    return gpid


def gen_ww_cuboid(xy, gpid, tol):
    xy_cor = []
    assert len(np.unique(gpid)) == 4

    # For each part seperated by wall-wall peak, voting for a wall
    for j in range(4):
        now_x = xy[gpid == j, 0]
        now_y = xy[gpid == j, 1]
        new_x, x_score, x_l1 = vote(now_x, tol)
        new_y, y_score, y_l1 = vote(now_y, tol)
        if (x_score, -x_l1) > (y_score, -y_l1):
            xy_cor.append({'type': 0, 'val': new_x, 'score': x_score})
        else:
            xy_cor.append({'type': 1, 'val': new_y, 'score': y_score})

    # Sanity fallback
    scores = [0, 0]
    for j in range(4):
        if xy_cor[j]['type'] == 0:
            scores[j % 2] += xy_cor[j]['score']
        else:
            scores[j % 2] -= xy_cor[j]['score']
    if scores[0] > scores[1]:
        xy_cor[0]['type'] = 0
        xy_cor[1]['type'] = 1
        xy_cor[2]['type'] = 0
        xy_cor[3]['type'] = 1
    else:
        xy_cor[0]['type'] = 1
        xy_cor[1]['type'] = 0
        xy_cor[2]['type'] = 1
        xy_cor[3]['type'] = 0

    return xy_cor


def gen_ww_general(init_coorx, xy, gpid, tol):
    xy_cor = []
    assert len(init_coorx) == len(np.unique(gpid))

    # Candidate for each part seperated by wall-wall boundary
    for j in range(len(init_coorx)):
        now_x = xy[gpid == j, 0]
        now_y = xy[gpid == j, 1]
        # print(f'now_x: {now_x}')
        # print(f'now_y: {now_y}')
        new_x, x_score, x_l1 = vote(now_x, tol)
        new_y, y_score, y_l1 = vote(now_y, tol)
        u0 = np_coorx2u(init_coorx[(j - 1 + len(init_coorx)) % len(init_coorx)])
        u1 = np_coorx2u(init_coorx[j])
        if (x_score, -x_l1) > (y_score, -y_l1):
            xy_cor.append({
                'type': 0,
                'val': new_x,
                'score': x_score,
                'action': 'ori',
                'gpid': j,
                'u0': u0,
                'u1': u1,
                'tbd': True
            })
        else:
            xy_cor.append({
                'type': 1,
                'val': new_y,
                'score': y_score,
                'action': 'ori',
                'gpid': j,
                'u0': u0,
                'u1': u1,
                'tbd': True
            })

    # Construct wall from highest score to lowest
    while True:
        # Finding undetermined wall with highest score
        tbd = -1
        for i in range(len(xy_cor)):
            if xy_cor[i]['tbd'] and (tbd == -1 or xy_cor[i]['score'] > xy_cor[tbd]['score']):
                tbd = i
        if tbd == -1:
            break

        # This wall is determined
        xy_cor[tbd]['tbd'] = False
        p_idx = (tbd - 1 + len(xy_cor)) % len(xy_cor)
        n_idx = (tbd + 1) % len(xy_cor)

        num_tbd_neighbor = xy_cor[p_idx]['tbd'] + xy_cor[n_idx]['tbd']

        # Two adjacency walls are not determined yet => not special case
        if num_tbd_neighbor == 2:
            continue

        # Only one of adjacency two walls is determine => add now or later special case
        if num_tbd_neighbor == 1:
            if (not xy_cor[p_idx]['tbd'] and xy_cor[p_idx]['type'] == xy_cor[tbd]['type']) or\
                    (not xy_cor[n_idx]['tbd'] and xy_cor[n_idx]['type'] == xy_cor[tbd]['type']):
                # Current wall is different from one determined adjacency wall
                if xy_cor[tbd]['score'] >= -1:
                    # Later special case, add current to tbd
                    xy_cor[tbd]['tbd'] = True
                    xy_cor[tbd]['score'] -= 100
                else:
                    # Fallback: forced change the current wall or infinite loop
                    if not xy_cor[p_idx]['tbd']:
                        insert_at = tbd
                        if xy_cor[p_idx]['type'] == 0:
                            new_val = np_x_u_solve_y(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                            new_type = 1
                        else:
                            new_val = np_y_u_solve_x(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                            new_type = 0
                    else:
                        insert_at = n_idx
                        if xy_cor[n_idx]['type'] == 0:
                            new_val = np_x_u_solve_y(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                            new_type = 1
                        else:
                            new_val = np_y_u_solve_x(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                            new_type = 0
                    new_add = {
                        'type': new_type,
                        'val': new_val,
                        'score': 0,
                        'action': 'forced infer',
                        'gpid': -1,
                        'u0': -1,
                        'u1': -1,
                        'tbd': False
                    }
                    xy_cor.insert(insert_at, new_add)
            continue

        # Below checking special case
        if xy_cor[p_idx]['type'] == xy_cor[n_idx]['type']:
            # Two adjacency walls are same type, current wall should be differen type
            if xy_cor[tbd]['type'] == xy_cor[p_idx]['type']:
                # Fallback: three walls with same type => forced change the middle wall
                xy_cor[tbd]['type'] = (xy_cor[tbd]['type'] + 1) % 2
                xy_cor[tbd]['action'] = 'forced change'
                xy_cor[tbd]['val'] = xy[gpid == xy_cor[tbd]['gpid'], xy_cor[tbd]['type']].mean()
        else:
            # Two adjacency walls are different type => add one
            tp0 = xy_cor[n_idx]['type']
            tp1 = xy_cor[p_idx]['type']
            if xy_cor[p_idx]['type'] == 0:
                val0 = np_x_u_solve_y(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                val1 = np_y_u_solve_x(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
            else:
                val0 = np_y_u_solve_x(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                val1 = np_x_u_solve_y(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
            new_add = [
                {
                    'type': tp0,
                    'val': val0,
                    'score': 0,
                    'action': 'forced infer',
                    'gpid': -1,
                    'u0': -1,
                    'u1': -1,
                    'tbd': False
                },
                {
                    'type': tp1,
                    'val': val1,
                    'score': 0,
                    'action': 'forced infer',
                    'gpid': -1,
                    'u0': -1,
                    'u1': -1,
                    'tbd': False
                },
            ]
            xy_cor = xy_cor[:tbd] + new_add + xy_cor[tbd + 1:]

    return xy_cor


def gen_ww(init_coorx_lst: np.array,
           coory_ceil_lst: np.array,
           z_ceil=50,
           coorW=1024,
           coorH=512,
           floorW=1024,
           floorH=512,
           tol=3,
           force_cuboid=True):
    """generate wall-wall from corner_x pixel coordinates and ceiling pixel coordinates

    Args:
        init_coorx_lst (np.array): corners' x pixel coordinates from prediction, 1d
        coory_ceil_lst (np.array): ceiling boundary y pixel coordinates from prediction, 1d
        z_ceil (int, optional): _description_. Defaults to 50.
        coorW (int, optional): _description_. Defaults to 1024.
        coorH (int, optional): _description_. Defaults to 512.
        floorW (int, optional): _description_. Defaults to 1024.
        floorH (int, optional): _description_. Defaults to 512.
        tol (int, optional): _description_. Defaults to 3.
        force_cuboid (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # get corners' indices in width axis
    gpid = get_gpid(init_coorx_lst, coorW)
    print(f'gpid: {gpid}')
    # get ceiling pixel coordinates
    coor_ceil_2d_lst = np.hstack([np.arange(coorW)[:, None], coory_ceil_lst[:, None]])
    # pixel coordinates to unit sphere coordinates
    xy = np_coor2xy(coor_ceil_2d_lst, z_ceil, coorW, coorH, floorW, floorH)
    print(f'xy: {xy}')

    # Generate wall-wall
    if force_cuboid:
        xy_cor = gen_ww_cuboid(xy, gpid, tol)
    else:
        xy_cor = gen_ww_general(init_coorx_lst, xy, gpid, tol)

    # Ceiling view to normal view
    cor = []
    for j in range(len(xy_cor)):
        next_j = (j + 1) % len(xy_cor)
        if xy_cor[j]['type'] == 1:
            cor.append((xy_cor[next_j]['val'], xy_cor[j]['val']))
        else:
            cor.append((xy_cor[j]['val'], xy_cor[next_j]['val']))
    cor = np_xy2coor(np.array(cor), z_ceil, coorW, coorH, floorW, floorH)
    cor = np.roll(cor, -2 * cor[::2, 0].argmin(), axis=0)

    return cor, xy_cor


def infer_coory(coory0, h, z0=50, coorH=512):
    v0 = np_coory2v(coory0, coorH)
    c0 = z0 / np.tan(v0)
    z1 = z0 + h
    v1 = np.arctan2(z1, c0)
    return (-v1 / np.pi + 0.5) * coorH - 0.5


def find_N_peaks(signal, filter_size=29, min_v=0.05, N=None):
    """find N peaks in signal vector

    Args:
        signal (np.array): input vector
        filter_size (int, optional): _description_. Defaults to 29.
        min_v (float, optional): _description_. Defaults to 0.05.
        N (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    max_v = maximum_filter(signal, size=filter_size, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]


def mean_percentile(vec, p1=25, p2=75):
    vmin = np.percentile(a=vec, q=p1)
    vmax = np.percentile(a=vec, q=p2)
    return vec[(vmin <= vec) & (vec <= vmax)].mean()


def refine_boundary_by_fix_floor(coor_y_ceil, coor_y_floor, z_ceil=50, coorH=512):
    '''
    Refine coor_y_ceil by coor_y_floor
    coor_y_floor are assumed on given height z_floor
    '''
    v_ceil = np_coory2v(coor_y_ceil, coorH)
    v_floor = np_coory2v(coor_y_floor, coorH)

    c0 = z_ceil / np.tan(v_ceil)
    z1 = c0 * np.tan(v_floor)
    z1_mean = mean_percentile(z1)
    v1_refine = np.arctan2(z1_mean, c0)
    coory1_refine = (-v1_refine / np.pi + 0.5) * coorH - 0.5
    return coory1_refine, z1_mean

    # c_floor = z_floor / np.tan(v_floor)
    # z_ceil = c_floor * np.tan(v_ceil)
    # z_ceil_mean = mean_percentile(z_ceil)
    # v_ceil_refine = np.arctan2(z_ceil_mean, c_floor)
    # coor_y_ceil_refined = (-v_ceil_refine / np.pi + 0.5) * coorH - 0.5
    # return coor_y_ceil_refined, z_ceil_mean


def get_mesh_from_corners(corners_lst: np.ndarray,
                          H: int,
                          W: int,
                          camera_position: np.array,
                          rgb_img: np.array,
                          b_ignore_floor: bool = False,
                          b_ignore_ceiling: bool = True,
                          b_ignore_wall: bool = False,
                          b_in_world_frame: bool = True) -> Tuple:
    """ generate layout mesh from equirectangular image and corners

    Args:
        corners_lst (np.ndarray): 2d corners in equirectangular image
        H (int): _description_
        W (int): _description_
        camera_position (np.array): _description_
        rgb_img (np.array): rgb panorama
        b_ignore_floor (bool, optional): _description_. Defaults to False.
        b_ignore_ceiling (bool, optional): _description_. Defaults to True.
        b_ignore_wall (bool, optional): _description_. Defaults to False.
        b_in_world_frame (bool, optional): generate mesh in world frame or camera frame. Defaults to False.

    Returns:
        Tuple: _description_
    """

    # Convert corners to layout
    depth_img, floor_mask, ceil_mask, wall_mask = layout_2_depth(corners_lst,
                                                                 H,
                                                                 W,
                                                                 floor_height=camera_position[2],
                                                                 return_mask=True)
    coorx, coory = np.meshgrid(np.arange(W), np.arange(H))
    us = np_coorx2u(coorx, W)
    vs = np_coory2v(coory, H)
    zs = depth_img * np.sin(vs)
    cs = depth_img * np.cos(vs)
    xs = cs * np.sin(us)
    # we align y axis to the panorama image,
    # if we need to flip the y axis, the ys need to be flipped
    ys = cs * np.cos(us)

    # Aggregate mask
    mask = np.ones_like(floor_mask)
    if b_ignore_floor:
        mask &= ~floor_mask
    if b_ignore_ceiling:
        mask &= ~ceil_mask
    if b_ignore_wall:
        mask &= ~wall_mask

    # Prepare ply's points and faces
    xyzrgb = np.concatenate([xs[..., None], ys[..., None], zs[..., None], rgb_img], -1)
    # convert points from camera frame to world frame
    if b_in_world_frame:
        xyzrgb[:, :, :3] = xyzrgb[:, :, :3] + camera_position

    xyzrgb = np.concatenate([xyzrgb, xyzrgb[:, [0]]], 1)
    # print(f' mask: {mask.shape}')
    mask = np.concatenate([mask, mask[:, [0]]], 1)
    # print(f'concatenated mask: {mask.shape}')
    lo_tri_template = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]])
    up_tri_template = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 1]])
    ma_tri_template = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
    lo_mask = (correlate2d(mask, lo_tri_template, mode='same') == 3)
    up_mask = (correlate2d(mask, up_tri_template, mode='same') == 3)
    ma_mask = (correlate2d(mask, ma_tri_template, mode='same') == 3) & (~lo_mask) & (~up_mask)
    ref_mask = (
        lo_mask | (correlate2d(lo_mask, np.flip(lo_tri_template, (0,1)), mode='same') > 0) |\
        up_mask | (correlate2d(up_mask, np.flip(up_tri_template, (0,1)), mode='same') > 0) |\
        ma_mask | (correlate2d(ma_mask, np.flip(ma_tri_template, (0,1)), mode='same') > 0)
    )
    points = xyzrgb[ref_mask]

    ref_id = np.full(ref_mask.shape, -1, np.int32)
    ref_id[ref_mask] = np.arange(ref_mask.sum())
    faces_lo_tri = np.stack([
        ref_id[lo_mask],
        ref_id[shift(lo_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(lo_mask, [1, 1], cval=False, order=0)],
    ], 1)
    faces_up_tri = np.stack([
        ref_id[up_mask],
        ref_id[shift(up_mask, [1, 1], cval=False, order=0)],
        ref_id[shift(up_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces_ma_tri = np.stack([
        ref_id[ma_mask],
        ref_id[shift(ma_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(ma_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces = np.concatenate([faces_lo_tri, faces_up_tri, faces_ma_tri])

    return (points, faces)


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
    elif room_type == ROOM_TYPE_DICT['kitchen']:
        classes = ST3D_KITCHEN_FURNITURE
    elif room_type == ROOM_TYPE_DICT['study']:
        classes = ST3D_STUDY_FURNITURE
    elif room_type == ROOM_TYPE_DICT['bathroom']:
        classes = ST3D_BATHROOM_FURNITURE
    else:
        raise ValueError('The room type is not supported.')

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
    elif room_type == ROOM_TYPE_DICT['kitchen']:
        class_freq_dict = ST3D_KITCHEN_FURNITURE_CNTS
    elif room_type == ROOM_TYPE_DICT['study']:
        class_freq_dict = ST3D_STUDY_FURNITURE_CNTS

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
        max_len = ST3D_BEDROOM_MAX_LEN
        class_num = len(ST3D_BEDROOM_FURNITURE)
    elif room_type == ROOM_TYPE_DICT['living room']:
        max_len = ST3D_LIVINGROOM_MAX_LEN
        class_num = len(ST3D_LIVINGROOM_FURNITURE)
    elif room_type == ROOM_TYPE_DICT['dining room']:
        max_len = ST3D_DININGROOM_MAX_LEN
        class_num = len(ST3D_DININGROOM_FURNITURE)
    elif room_type == ROOM_TYPE_DICT['kitchen']:
        max_len = ST3D_KITCHEN_MAX_LEN
        class_num = len(ST3D_KITCHEN_FURNITURE)
    elif room_type == ROOM_TYPE_DICT['study']:
        max_len = ST3D_STUDY_MAX_LEN
        class_num = len(ST3D_STUDY_FURNITURE)
    elif room_type == ROOM_TYPE_DICT['bathroom']:
        max_len = ST3D_BATHROOM_MAX_LEN
        class_num = len(ST3D_BATHROOM_FURNITURE)
    else:
        raise ValueError('The room type is not supported.')

    # Pad the end label in the end of each sequence, and convert the class labels to -1, 1
    if L < max_len:
        empty_label = np.eye(class_num)[-1] * 2 -1
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
        class_num = len(ST3D_BEDROOM_FURNITURE)
        max_len = ST3D_BEDROOM_QUAD_WALL_MAX_LEN
    elif room_type == ROOM_TYPE_DICT['living room']:
        class_num = len(ST3D_LIVINGROOM_FURNITURE)
        max_len = ST3D_LIVINGROOM_QUAD_WALL_MAX_LEN
    elif room_type == ROOM_TYPE_DICT['dining room']:
        class_num = len(ST3D_DININGROOM_FURNITURE)
        max_len = ST3D_LIVINGROOM_QUAD_WALL_MAX_LEN
    elif room_type == ROOM_TYPE_DICT['kitchen']:
        class_num = len(ST3D_KITCHEN_FURNITURE)
        max_len = ST3D_KITCHEN_QUAD_WALL_MAX_LEN
    elif room_type == ROOM_TYPE_DICT['study']:
        class_num = len(ST3D_STUDY_FURNITURE)
        max_len = ST3D_STUDY_QUAD_WALL_MAX_LEN
    elif room_type == ROOM_TYPE_DICT['bathroom']:
        class_num = len(ST3D_BATHROOM_FURNITURE)
        max_len = ST3D_BATHROOM_QUAD_WALL_MAX_LEN
    else:
        raise ValueError('The room type is not supported.')

    assert L <= max_len, f'The length of the wall bbox list should be less than {max_len}.'

    # Pad the end label in the end of each sequence, and convert the class labels to -1, 1
    empty_label = np.eye(class_num)[-1] * 2 -1
    padding = np.concatenate([empty_label, np.zeros(bbox_dim - class_num, dtype=np.float32)], axis=0)
    wall_bbox_lst = np.vstack([wall_bbox_lst, np.tile(padding, [max_len - L, 1])])

    # print(f'wall_bbox_lst: {wall_bbox_lst}')
    return wall_bbox_lst


def complete_stop_in_sentence(sentence: str) -> str:
    """Implement the stop in the end of the sentence."""
    if sentence[-1] != '.':
        sentence += '.'
    return sentence


class ST3DDataset(data.Dataset):
    '''
    dataset for layout: PanoCoordinatesBoundary
    '''

    def __init__(
            self,
            root_dir,
            flip=False,
            rotate=False,
            gamma=False,
            p_base=0.96,
            max_stretch=2.0,
            normcor=False,
            max_text_sentences=4,  #  max number of text_prompt sentences
            shard=0,  #  support parallel training
            num_shards=1,
            device='cuda',
            return_scene_name=False,
            random_text_desc=True,
            permutation=True,
            use_gpt_text_desc=False,
            train_stats_file=None):
        self.img_dir = os.path.join(root_dir, 'img')
        self.cor_dir = os.path.join(root_dir, 'label_cor')
        self.quad_wall_dir = os.path.join(root_dir, 'quad_walls')
        self.cam_pos_dir = os.path.join(root_dir, 'cam_pos')
        self.room_type_dir = os.path.join(root_dir, 'room_type')
        # object bbox folder
        self.bbox_3d_dir = os.path.join(root_dir, 'bbox_3d')
        # text descritpion folder
        self.text_desc_dir = os.path.join(root_dir, 'text_desc')
        self.text_emb_dir = os.path.join(root_dir, 'text_desc_emb')
        self.train_stats_filepath = train_stats_file

        # total image file names and text file names
        self.img_fnames = sorted(
            [fname for fname in os.listdir(self.img_dir) if fname.endswith('.jpg') or fname.endswith('.png')])
        self.txt_fnames = ['%s.txt' % fname[:-4] for fname in self.img_fnames]
        # self.json_fnames = ['%s_normalized.json' % fname[:-4] for fname in self.img_fnames]
        self.json_fnames = ['%s.json' % fname[:-4] for fname in self.img_fnames]
        self.npy_fnames = ['%s_gpt4.npy' % fname[:-4] for fname in self.img_fnames] if use_gpt_text_desc else ['%s.npy' % fname[:-4] for fname in self.img_fnames] 
        #  image file names and text file names on local_rank machine
        self.local_img_fnames = self.img_fnames[shard::num_shards]
        self.local_txt_fnames = self.txt_fnames[shard::num_shards]
        self.local_json_fnames = self.json_fnames[shard::num_shards]
        self.local_npy_fnames = self.npy_fnames[shard::num_shards]

        self.flip = flip
        self.rotate = rotate
        self.gamma = gamma
        self.p_base = p_base
        self.max_stretch = max_stretch
        self.normcor = normcor
        self.max_text_sentences = max_text_sentences
        self.local_classes = None
        self.return_scene_name = return_scene_name
        self.random_text_desc = random_text_desc
        self.permutation = permutation
        self.use_gpt_text_desc = use_gpt_text_desc
        if random_text_desc:
            self.clip_model = FrozenCLIPEmbedder(device=device)

        # The direction of all camera is always along the negative y-axis.
        self.cam_R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], np.float32)

        self._check_dataset()
        self._parse_train_stats()

    def _check_dataset(self):
        for fname in self.txt_fnames:
            assert os.path.isfile(os.path.join(self.cor_dir, fname)), '%s not found' % os.path.join(self.cor_dir, fname)

    def _parse_train_stats(self):
        with open(self.train_stats_filepath, "r") as f:
            train_stats = json.load(f)
        # min and max centroid, size, angle
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
            # class_label_prob = np.where(descaled_class_labels > 0.5, descaled_class_labels, 0)
            # if np.all(class_label_prob == 0):
            #     continue
            # class_label = self.class_labels[class_label_prob.argmax()]
            # if class_label == 'empty':
            #     continue

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

    def __len__(self):
        # return len(self.img_fnames)
        return len(self.local_img_fnames)

    def __getitem__(self, idx: int) -> Tuple:
        """retrieve scene data

        Args:
            idx (int): panorama/room idx

        Returns:
            List: [Image, [boundary_x:1x1024, boundary_y:1x1024], boundary_wall_probability:1x-024]
        """
        # Read image
        # img_path = os.path.join(self.img_dir, self.local_img_fnames[idx])
        # img = Image.open(img_path).convert('RGB')
        # img = np.array(img, np.float32) / 255.
        # H, W = img.shape[:2]

        # read camera position file
        # cam_pos_lst = []
        # cam_pos_filepath = os.path.join(self.cam_pos_dir, self.local_txt_fnames[idx])
        # with open(cam_pos_filepath) as f:
        #     cam_pos_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)
        # assert len(cam_pos_lst) == 1, cam_pos_filepath

        # read room type file
        room_type = None
        room_type_filepath = os.path.join(self.room_type_dir, self.local_txt_fnames[idx])
        with open(room_type_filepath) as f:
            room_type_str = f.readline().strip()
            assert room_type_str in ROOM_TYPE_DICT.keys(), room_type_filepath
            room_type = ROOM_TYPE_DICT[room_type_str]
        
        # load object bbox file
        object_bbox_filepath = os.path.join(self.bbox_3d_dir, self.local_json_fnames[idx])
        object_bbox_lst = []
        with open(object_bbox_filepath) as f:
            object_bbox_dicts = json.load(f)
            object_bbox_dicts = object_bbox_dicts['objects']
        
        # load wall bbox file
        wall_bbox_filepath = os.path.join(self.quad_wall_dir, self.local_json_fnames[idx])
        wall_bbox_lst = []
        with open(wall_bbox_filepath, 'r') as f:
            wall_bbox_dicts = json.load(f)
            wall_bbox_dicts = wall_bbox_dicts['walls']
            
        if not self.permutation:
            # sort object bbox by class frequency and bbox size
            object_bbox_dicts = ordered_bboxes_with_class_frequencies(room_type=room_type,
                                                                  object_bbox_lst=object_bbox_dicts)
        else:
            objects_num = len(object_bbox_dicts)
            object_bbox_dicts = [object_bbox_dicts[i] for i in np.random.permutation(objects_num)]
        
        # load precomputed text description and text embedding
        if not self.random_text_desc:
            text_desc_lst = []
            if self.use_gpt_text_desc:
                text_desc_filepath = os.path.join(self.text_desc_dir, self.local_txt_fnames[idx][:-4]+'_gpt4.txt')
            else:
                text_desc_filepath = os.path.join(self.text_desc_dir, self.local_txt_fnames[idx])
            with open(text_desc_filepath) as f:
                text_desc = f.readline()
                text_desc_lst = text_desc.strip().split('. ')
                text_desc_lst = [complete_stop_in_sentence(sen) for sen in text_desc_lst if len(sen)]
            # print(f'original text_desc_str: {text_desc}')
            # read text embedding file
            text_emb = np.array([])
            text_emb_filepath = os.path.join(self.text_emb_dir, self.local_npy_fnames[idx])
            text_emb = np.load(text_emb_filepath).astype(np.float32)
        else:
            text_desc_str, text_emb = get_scene_description(room_type=room_type_str,
                                    wall_dict={"walls": wall_bbox_dicts}, 
                                    object_dict={"objects": object_bbox_dicts},
                                    eval=False,
                                    glove_model=self.clip_model,
                                    use_object_ordering=False)
            text_emb = text_emb.astype(np.float32)
            # print(f'randomrized text_desc_str: {text_desc_str}, text_embedding shape: {text_emb.shape}')
            text_desc_lst = text_desc_str.strip().split('. ')
            text_desc_lst = [complete_stop_in_sentence(sen) for sen in text_desc_lst if len(sen)]
        
        # new_bbox_dict_lst = []
        if self.train_stats_filepath is None:
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
        else:
            # normalize object bbox w.r.t. training set statistics
            bbox_onehot_class_labels = []
            bbox_trans = []
            bbox_sizes = []
            bbox_angles = []
            for obj_bbox in object_bbox_dicts:
                # new_obj_bbox = obj_bbox.copy()
                bbox_class_label = obj_bbox['class'].lower()
                bbox_class = ClassLabelsEncode(room_type=room_type, obj_bbox_label=bbox_class_label)
                # scale to [-1, 1]
                bbox_class = bbox_class*2 - 1
                bbox_onehot_class_labels.append(bbox_class)
                
                bbox_centroid = np.array(obj_bbox['center'], np.float32)
                scaled_bbox_centroid = self.scale(bbox_centroid, *self._centroids)
                bbox_trans.append(scaled_bbox_centroid)
                # new_obj_bbox['center'] = self.descale(scaled_bbox_centroid, *self._centroids).tolist()
                
                bbox_size = np.array(obj_bbox['size'], np.float32)
                scaled_bbox_size = self.scale(bbox_size, *self._sizes)
                bbox_sizes.append(scaled_bbox_size)
                # new_obj_bbox['size'] = self.descale(scaled_bbox_size, *self._sizes).tolist()
                
                # only use z_angle
                bbox_angle = np.array(obj_bbox['angles'], np.float32)[-1]
                bbox_cos_sin_angles = np.array([np.cos(bbox_angle), np.sin(bbox_angle)])
                bbox_angles.append(bbox_cos_sin_angles)
                # new_obj_bbox['angles'] = [0,0, np.arctan2(bbox_cos_sin_angles[1], bbox_cos_sin_angles[0])]
                
                # new_bbox_dict_lst.append(new_obj_bbox)
                
            object_bbox_lst = np.concatenate([np.array(bbox_onehot_class_labels), 
                                              np.array(bbox_trans), 
                                              np.array(bbox_sizes), 
                                              np.array(bbox_angles)], axis=-1)
            
            object_bbox_lst = padding_and_reshape_object_bbox(room_type=room_type,
                                                    object_bbox_lst=object_bbox_lst,
                                                    bbox_dim=object_bbox_lst.shape[-1])   
            # print(f'object_bbox_lst: {object_bbox_lst.shape}')
                 
                

        if self.train_stats_filepath is None:
            for wall_bbox in wall_bbox_dicts:
                wall_id = int(wall_bbox['ID'])
                wall_class = ClassLabelsEncode(room_type=room_type, obj_bbox_label='wall')
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
        else:
            # normalize wall bbox w.r.t. training set statistics
            wall_onehot_class_labels = []
            wall_trans = []
            wall_sizes = []
            wall_angles = []
            for wall_bbox in wall_bbox_dicts:
                # new_wall_bbox = wall_bbox.copy()
                wall_class_label = 'wall'
                wall_class = ClassLabelsEncode(room_type=room_type, obj_bbox_label=wall_class_label)
                # scale to [-1, 1]
                wall_class = wall_class*2 - 1
                wall_onehot_class_labels.append(wall_class)
                
                wall_centroid = np.array(wall_bbox['center'], np.float32)
                scaled_wall_centroid = self.scale(wall_centroid, *self._centroids)
                wall_trans.append(scaled_wall_centroid)
                # new_wall_bbox['center'] = self.descale(scaled_wall_centroid, *self._centroids).tolist()
                
                wall_size = np.array([wall_bbox['width'], 0.01, wall_bbox['height']], np.float32)
                scaled_wall_size = self.scale(wall_size, *self._sizes)
                wall_sizes.append(scaled_wall_size)
                # new_wall_bbox['size'] = self.descale(scaled_wall_size, *self._sizes).tolist()
                
                # only use z_angle
                wall_angle = np.array(wall_bbox['angles'], np.float32)[-1]
                wall_cos_sin_angles = np.array([np.cos(wall_angle), np.sin(wall_angle)])
                wall_angles.append(wall_cos_sin_angles)
                # new_wall_bbox['angles'] = [0,0, np.arctan2(wall_cos_sin_angles[1], wall_cos_sin_angles[0])]
                
                # new_bbox_dict_lst.append(new_wall_bbox)
                
            wall_bbox_lst = np.concatenate([wall_onehot_class_labels, wall_trans, wall_sizes, wall_angles], axis=-1)
            
            wall_bbox_lst = padding_and_reshape_wall_bbox(room_type=room_type,
                                                    wall_bbox_lst=wall_bbox_lst,
                                                    bbox_dim=wall_bbox_lst.shape[-1])
            # print(f'wall_bbox_lst: {wall_bbox_lst.shape}')
            
                
        
        # from preprocess.prepare_st3d_dataset import vis_scene_mesh
        # from dataset.metadata import COLOR_TO_ADEK_LABEL
        # scene_bbox_mesh = vis_scene_mesh(room_layout_mesh=None,
        #                                  obj_bbox_lst=new_bbox_dict_lst,
        #                                  color_to_labels=COLOR_TO_ADEK_LABEL)
        # save_mesh_path = os.path.join(self.bbox_3d_dir, self.local_json_fnames[idx][:-5]+'_scaled.ply')
        # scene_bbox_mesh.export(save_mesh_path)
        
        assert wall_bbox_lst.shape[-1] == object_bbox_lst.shape[-1]
        out_lst = np.concatenate([wall_bbox_lst, object_bbox_lst], axis=0)
        out_lst = out_lst.transpose(1, 0)

        cond_dict = {}
        # if room_type is not None:
        #     cond_dict["y"] = np.array(room_type, dtype=np.int64)

        text_desc_len = len(text_desc_lst)
        if text_desc_len:
            if text_desc_len > self.max_text_sentences:
                text_desc_lst = text_desc_lst[:self.max_text_sentences]
            cond_dict["text"] = ''.join(text_desc_lst)
        cond_dict["text_condition"] = np.squeeze(text_emb, axis=0)

        if self.return_scene_name:
            scene_name = self.local_txt_fnames[idx][:-4]
            return out_lst, cond_dict, scene_name
        else:
            return out_lst, cond_dict

    def get_scene(self, scene_name: str) -> Tuple:
        idx = self.local_txt_fnames.index(scene_name+'.txt')
        return self.__getitem__(idx)
    
    def get_gt_layout_mesh(self,
                           idx: int,
                           b_ignore_floor: bool = False,
                           b_ignore_ceiling: bool = True,
                           b_ignore_wall: bool = False) -> Tuple:
        # Read image
        img_path = os.path.join(self.img_dir, self.img_fnames[idx])
        equirect_img = np.array(Image.open(img_path))
        if equirect_img.shape[2] == 4:
            equirect_img = equirect_img[:, :, :3]
        print(f'equirect_img.shape: {equirect_img.shape}')
        H, W = equirect_img.shape[:2]

        # read camera position file
        cam_pos_lst = []
        cam_pos_filepath = os.path.join(self.cam_pos_dir, self.txt_fnames[idx])
        with open(cam_pos_filepath) as f:
            cam_pos_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)
        assert len(cam_pos_lst) == 1, cam_pos_filepath
        # convert the unit into meter
        cam_pos_lst = cam_pos_lst[0] * 0.001
        print(f'cam_pos_lst: {cam_pos_lst}')

        # Read ground truth corners
        corners_lst = []
        with open(os.path.join(self.cor_dir, self.txt_fnames[idx])) as f:
            corners_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)

            # Corner with minimum x should at the beginning
            corners_lst = np.roll(corners_lst[:, :2], -2 * np.argmin(corners_lst[::2, 0]), 0)

        # Prepare 1d wall-wall probability
        # unoccluded_corner_lst = corners_lst[~occlusion]
        # corner_x_lst = corners_lst[~occlusion, 0]
        # dist_o = cdist(corner_x_lst.reshape(-1, 1), np.arange(W).reshape(-1, 1), metric='euclidean', p=1)
        # dist_r = cdist(corner_x_lst.reshape(-1, 1), np.arange(W).reshape(-1, 1) + W, metric='euclidean', p=1)
        # dist_l = cdist(corner_x_lst.reshape(-1, 1), np.arange(W).reshape(-1, 1) - W, metric='euclidean', p=1)
        # dist = np.min([dist_o, dist_r, dist_l], axis=0)
        # nearest_dist = dist.min(0)
        # corner_y_prob_lst = (self.p_base**nearest_dist).reshape(1, -1)

        points, faces = get_mesh_from_corners(corners_lst,
                                              H,
                                              W,
                                              camera_position=cam_pos_lst,
                                              rgb_img=equirect_img,
                                              b_ignore_floor=b_ignore_floor,
                                              b_ignore_ceiling=b_ignore_ceiling,
                                              b_ignore_wall=b_ignore_wall)

        return (points, faces, corners_lst, cam_pos_lst)

    def get_predicted_layout_mesh(self,
                                  room_type: str,
                                  bound_ceil_floor_lst: np.array,
                                  wall_prob_lst: np.array,
                                  obj_bbox_lst: np.array = None,
                                  b_force_raw: bool = False,
                                  b_force_cuboid: bool = False) -> Tuple:
        # random choose a camera position
        # idx = np.random.randint(len(self.local_img_fnames))
        idx = 2
        # Read image
        img_path = os.path.join(self.img_dir, self.local_img_fnames[idx])
        equirect_img = np.array(Image.open(img_path))
        if equirect_img.shape[2] == 4:
            equirect_img = equirect_img[:, :, :3]
        # print(f'equirect_img.shape: {equirect_img.shape}')
        H, W = equirect_img.shape[:2]

        # read camera position file
        cam_pos_lst = []
        cam_pos_filepath = os.path.join(self.cam_pos_dir, self.local_txt_fnames[idx])
        with open(cam_pos_filepath) as f:
            cam_pos_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)
        assert len(cam_pos_lst) == 1, cam_pos_filepath
        # convert the unit into meter
        cam_pos_lst = cam_pos_lst[0] * 0.001
        # print(f'cam_pos_lst: {cam_pos_lst}')

        # convert uv coords to pixel coords
        y_boundary_lst = (bound_ceil_floor_lst / np.pi + 0.5) * H - 0.5
        y_boundary_lst[0] = np.clip(y_boundary_lst[0], 1, H / 2 - 1)
        y_boundary_lst[1] = np.clip(y_boundary_lst[1], H / 2 + 1, H - 2)
        corner_prob_lst = wall_prob_lst

        # Init floor height
        z_ceil = 50
        # z_floor = - cam_pos_lst[2]
        # calculate floor height
        _, z_floor = refine_boundary_by_fix_floor(*y_boundary_lst, z_ceil)
        print(f'z_floor: {z_floor}')

        if b_force_raw:
            # Do not run post-processing, export raw polygon (1024*2 vertices) instead.
            # [TODO] Current post-processing lead to bad results on complex layout.
            # celing pixel coords
            cor = np.stack([np.arange(1024), y_boundary_lst[0]], axis=1)

        else:
            # Detech wall-wall peaks
            min_prob = 0 if b_force_cuboid else 0.05
            filter_size = int(round(W * 0.05 / 2))
            wall_num = 4 if b_force_cuboid else None
            # get corners' x coords from wall probablities
            corner_x_lst, _ = find_N_peaks(corner_prob_lst, filter_size=filter_size, min_v=min_prob, N=wall_num)
            print(f'corner_x_lst: {corner_x_lst.shape}')

            # Generate wall-walls
            cor, xy_cor = gen_ww(corner_x_lst,
                                 y_boundary_lst[0],
                                 z_ceil,
                                 tol=abs(0.16 * z_floor / 1.6),
                                 force_cuboid=b_force_cuboid)
            if not b_force_cuboid:
                # Check valid (for fear self-intersection)
                xy2d = np.zeros((len(xy_cor), 2), np.float32)
                for i in range(len(xy_cor)):
                    xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
                    xy2d[i, xy_cor[i - 1]['type']] = xy_cor[i - 1]['val']
                if not Polygon(xy2d).is_valid:
                    print('Fail to generate valid general layout!! '
                          'Generate cuboid as default.', file=sys.stderr)
                    corner_x_lst, _ = find_N_peaks(corner_prob_lst, filter_size=filter_size, min_v=0, N=4)
                    cor, xy_cor = gen_ww(corner_x_lst,
                                         y_boundary_lst[0],
                                         z_ceil,
                                         tol=abs(0.16 * z_floor / 1.6),
                                         force_cuboid=True)

        # Expand ceiling pixel coords with floor
        coord_floor = infer_coory(cor[:, 1], z_floor - z_ceil, z_ceil)[:, None]
        # pixel coords cor: [x, y_ceil, y_floor]
        cor = np.hstack([cor, coord_floor])

        # Collect corner coords in equirectangular image
        corners_lst = np.zeros((len(cor) * 2, 2), np.float32)
        for j in range(len(cor)):
            corners_lst[j * 2] = cor[j, 0], cor[j, 1]
            corners_lst[j * 2 + 1] = cor[j, 0], cor[j, 2]
        print(f'corners_lst: {corners_lst.shape}')
        # equirect_img = np.random.randint(0, 255, size=(H, W, 3), dtype=np.uint8)
        points, faces = get_mesh_from_corners(corners_lst,
                                              H,
                                              W,
                                              camera_position=cam_pos_lst,
                                              rgb_img=equirect_img,
                                              b_ignore_floor=False,
                                              b_ignore_ceiling=False,
                                              b_ignore_wall=False,
                                              b_in_world_frame=False)
        room_layout_mesh = None
        obj_bbox_dict_list = None
        if obj_bbox_lst is not None:
            room_layout_mesh = trimesh.Trimesh(vertices=points[:, :3], faces=faces)
            room_layout_bbox_min = trimesh.bounds.corners(room_layout_mesh.bounding_box_oriented.bounds).min(axis=0)
            room_layout_bbox_max = trimesh.bounds.corners(room_layout_mesh.bounding_box_oriented.bounds).max(axis=0)
            room_layout_bbox_size = room_layout_bbox_max - room_layout_bbox_min
            print(f'room_layout_bbox_size: {room_layout_bbox_size}')
            layout_bbox = room_layout_mesh.bounding_box_oriented

            if room_type == 'bedroom':
                class_labels_lst = (ST3D_BEDROOM_FURNITURE)
            elif room_type == 'living_room':
                class_labels_lst = (ST3D_LIVINGROOM_FURNITURE)
            elif room_type == 'dining_room':
                class_labels_lst = (ST3D_DININGROOM_FURNITURE)
            else:
                raise NotImplementedError
            class_idx = 0
            centroid_idx = len(class_labels_lst)
            size_idx = 3 + centroid_idx
            angle_idx = 3 + size_idx

            obj_bbox_dict_list = []
            for i in range(len(obj_bbox_lst)):
                # print(f'predict object bbox feature: {obj_bbox_lst[i]}')
                obj_bbox_dict = {}

                # recover class label
                class_label_prob = obj_bbox_lst[i][:centroid_idx]
                # print(f'class_label_prob: {class_label_prob}')
                class_label_prob = np.where(class_label_prob > 0.5, 1, 0)
                if len(class_label_prob) == 0:
                    print(f'object {i} has no class label')
                class_label = class_labels_lst[class_label_prob.argmax()]
                if class_label == 'empty':
                    print(f'object {i} is empty')
                    continue
                obj_bbox_dict['class'] = class_label

                # recover centroid
                centroid = obj_bbox_lst[i][centroid_idx:size_idx]
                centroid = centroid * room_layout_bbox_size
                obj_bbox_dict['center'] = centroid.tolist()
                # recover size
                size = obj_bbox_lst[i][size_idx:angle_idx]
                size = (size + 1) * 0.5
                size = size * room_layout_bbox_size
                obj_bbox_dict['size'] = size.tolist()
                # recover angle
                angle = obj_bbox_lst[i][angle_idx:]
                angle_0 = np.arccos(angle[0])
                angle_1 = np.arcsin(angle[1])
                print(f' object {class_label} centroid: {centroid} size: {size} angle: {angle_0}')
                obj_bbox_dict['angles'] = [0, 0, angle_0]
                obj_bbox_dict_list.append(obj_bbox_dict)

        return (points, faces, corners_lst, cam_pos_lst, room_layout_mesh, obj_bbox_dict_list)
