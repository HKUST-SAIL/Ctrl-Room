import sys

sys.path = [p for p in sys.path if "2.7" not in p]

import os
import json
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Process
from dataset.metadata import INVALID_SCENES_LST, INVALID_ROOMS_LST, COLOR_TO_LABEL

PATH_TO_DATASET = '/data/dataset/Structured3D/Structured3D/'


def flip_towards_viewer(normals, points):
    points = points / np.linalg.norm(points)
    proj = points.dot(normals[:2, :].T)
    flip = np.where(proj > 0)
    normals[flip, :] = -normals[flip, :]
    return normals


def get_corners_of_bb3d(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    # order the basis
    index = np.argsort(np.abs(basis[:, 0]))[::-1]
    # the case that two same value appear the same time
    if index[2] != 2:
        index[1:] = index[1:][::-1]
    basis = basis[index, :]
    coeffs = coeffs[index]
    # Now, we know the basis vectors are orders X, Y, Z. Next, flip the basis vectors towards the viewer
    basis = flip_towards_viewer(basis, centroid)
    coeffs = np.abs(coeffs)
    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners = corners + np.tile(centroid, (8, 1))
    return corners


def get_corners_of_bb3d_no_index(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]

    corners = corners + np.tile(centroid, (8, 1))
    return corners


def complete_bbox_json(path, scene) -> int:
    annotations_file = os.path.join(path, "scene_{}".format(scene), "bbox_3d.json")
    fixed_annotations_file = os.path.join(path, "scene_{}".format(scene), "bbox_3d_fixed.json")

    success_cnt = 0
    try:
        with open(annotations_file) as f:
            annos = json.load(f)
    except FileNotFoundError as e:
        print("Skipping scene {}, because:".format(scene) + str(e))
        return

    id2index = dict()
    for index, object in enumerate(annos):
        id2index[object.get('ID')] = index

    scene_path = os.path.join(path, "scene_{}".format(scene), "2D_rendering")

    for room_id in np.sort(os.listdir(scene_path)):
        room_path = os.path.join(scene_path, room_id, "panorama", "full")
        room_str = "scene_{}".format(scene) + "_" + room_id
        if room_str in INVALID_ROOMS_LST:
            continue
        if not os.path.exists(room_path):
            continue
        semantic_img_filepath = os.path.join(room_path, 'semantic.png')
        instance_img_filepath = os.path.join(room_path, 'instance.png')
        if not os.path.exists(semantic_img_filepath) or not os.path.exists(instance_img_filepath):
            continue

        room_annotations_file = os.path.join(room_path, "bbox_3d.json")
        room_annos_lst = []
        # for position_id in np.sort(os.listdir(room_path)):
        #     position_path = os.path.join(room_path, position_id)

        semantic = cv2.imread(semantic_img_filepath)
        semantic = cv2.cvtColor(semantic, cv2.COLOR_BGR2RGB)
        height, width, _ = semantic.shape

        instance = cv2.imread(instance_img_filepath, cv2.IMREAD_UNCHANGED)
        instances_indexes = np.unique(instance)[:-1]

        for index in instances_indexes:
            # for each instance in current image
            bbox = annos[id2index[index]]

            X, Y = np.where(instance == index)
            labels, occurences = np.unique([COLOR_TO_LABEL[tuple(semantic[x][y])] for x, y in zip(X, Y)],
                                           return_counts=True)

            if 'candidate_labels_and_occurences' not in bbox:
                bbox['candidate_labels_and_occurences'] = dict(zip(labels, [str(occ) for occ in occurences]))
            else:
                new_values = dict(zip(labels, [str(occ) for occ in occurences]))
                bbox['candidate_labels_and_occurences'] = {**bbox['candidate_labels_and_occurences'], **new_values}

            bbox['label'] = labels[max(enumerate(occurences), key=lambda x: x[1])[0]]

            basis = np.array(bbox['basis'])
            coeffs = np.array(bbox['coeffs'])
            centroid = np.array(bbox['centroid'])

            corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)
            corners_with_index = get_corners_of_bb3d(basis, coeffs, centroid)

            if 'corners_no_index' not in bbox:
                bbox['corners_no_index'] = corners.tolist()
            if 'corners_with_index' not in bbox:
                bbox['corners_with_index'] = corners_with_index.tolist()

            room_annos_lst.append(bbox.copy())
        # update bbox_3d.json for current room
        with open(room_annotations_file, 'w') as f:
            json.dump(room_annos_lst, f)

        success_cnt += 1
    # update bbox_3d.json for the whole scene
    with open(fixed_annotations_file, 'w') as f:
        json.dump(annos, f)

    return success_cnt


def complete_bbox_json_for_indexes(path, scenes_indexes, processor_id):
    processor_report_file = os.path.join(PATH_TO_DATASET, "processor_{}_report.txt".format(str(processor_id)))
    success_room_cnt = 0
    for counter, scene_index in enumerate(scenes_indexes):
        success_room_cnt += complete_bbox_json(path, scene_index)
        text = 'Processor {} finished scene {}, that is {}/{} scenes, {} rooms, progress is at: {}%'.format(
            str(processor_id), scene_index, str(counter + 1), str(len(scenes_indexes)), str(success_room_cnt),
            str((counter + 1) / len(scenes_indexes) * 100.))
        print(text)
        with open(processor_report_file, 'w+') as f:
            f.write(text + os.linesep)


if __name__ == "__main__":
    # scene_indexes = [str(index).rjust(5, '0') for index in range(0, 3500)]
    scene_indexes = ['%05d' % i for i in range(1156, 3500)]
    scene_indexes_to_convert = []
    # Skip files that have already been generated
    for scene_index in scene_indexes:
        scene_id_str = "scene_{}".format(scene_index)
        if scene_id_str in INVALID_SCENES_LST:
            continue
        fixed_annotations_file = os.path.join(PATH_TO_DATASET, "scene_{}".format(scene_index), "bbox_3d_fixed.json")
        annotations_file = os.path.join(PATH_TO_DATASET, "scene_{}".format(scene_index), "bbox_3d.json")
        # Skip if fixed annotation file has been done
        # if not os.path.isfile(fixed_annotations_file) and os.path.isfile(annotations_file):
        scene_indexes_to_convert.append(scene_index)

    print(scene_indexes_to_convert)

    nb_processors = 1
    length = len(scene_indexes_to_convert)
    scene_indexes_splits = [
        scene_indexes_to_convert[i * length // nb_processors:(i + 1) * length // nb_processors]
        for i in range(nb_processors)
    ]

    for count, s in enumerate(scene_indexes_splits):
        p = Process(target=complete_bbox_json_for_indexes, args=(PATH_TO_DATASET, s, count))
        p.start()

    # complete_bbox_json(PATH_TO_DATASET, '00000')