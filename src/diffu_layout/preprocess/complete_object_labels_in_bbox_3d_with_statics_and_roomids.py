import sys
sys.path = [p for p in sys.path if "2.7" not in p]

import os
import json
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Process


PATH_TO_DATASET = '/data/dataset/Structured3D/Structured3D/'


def convert_lines_to_vertices(lines):
    """convert line representation to polygon vertices
    """
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons


def verify_normal(corner_i, corner_j, delta_height, plane_normal):
    edge_a = corner_j + delta_height - corner_i
    edge_b = delta_height

    normal = np.cross(edge_a, edge_b)
    normal /= np.linalg.norm(normal, ord=2)

    inner_product = normal.dot(plane_normal)

    if inner_product > 1e-8:
        return False
    else:
        return True


def extract_walls_floor_and_ceiling(path, scene, room):
    # load room annotations
    with open(os.path.join(path, "scene_{}".format(scene), "annotation_3d.json")) as f:
        annos = json.load(f)

    # parse corners
    junctions = np.array([item['coordinate'] for item in annos['junctions']])
    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist())

    lines_holes = np.unique(lines_holes)
    try:  # Workaournd because one scene must have bad annotations that make an error and crashes a process 
        _, vertices_holes = np.where(np.array(annos['lineJunctionMatrix'])[lines_holes])
        vertices_holes = np.unique(vertices_holes)
    except IndexError as e:
        return None, None, None

    # parse annotations
    walls = dict()
    walls_normal = dict()
    for semantic in annos['semantics']:
        if semantic['ID'] != int(room):
            continue

        # find junctions of ceiling and floor
        for planeID in semantic['planeID']:
            plane_anno = annos['planes'][planeID]

            if plane_anno['type'] != 'wall':
                lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0]
                lineIDs = np.setdiff1d(lineIDs, lines_holes)
                junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
                wall = convert_lines_to_vertices(junction_pairs)
                walls[plane_anno['type']] = wall[0]

        # save normal of the vertical walls
        for planeID in semantic['planeID']:
            plane_anno = annos['planes'][planeID]

            if plane_anno['type'] == 'wall':
                lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0]
                lineIDs = np.setdiff1d(lineIDs, lines_holes)
                junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
                wall = convert_lines_to_vertices(junction_pairs)
                walls_normal[tuple(np.intersect1d(wall, walls['floor']))] = plane_anno['normal']

    # we assume that zs of floor equals 0, then the wall height is from the ceiling
    wall_height = np.mean(junctions[walls['ceiling']], axis=0)[-1]
    delta_height = np.array([0, 0, wall_height])

    # list of corner index
    wall_floor = walls['floor']

    walls_corners = []    # 3D coordinate for each wall

    # wall
    for i, j in zip(wall_floor, np.roll(wall_floor, shift=-1)):
        corner_i, corner_j = junctions[i], junctions[j]

        flip = verify_normal(corner_i, corner_j, delta_height, walls_normal[tuple(sorted([i, j]))])

        if flip:
            corner_j, corner_i = corner_i, corner_j

        corner = np.array([corner_i, corner_i + delta_height, corner_j + delta_height, corner_j])

        walls_corners.append(corner)

    # floor and ceiling
    corner_floor = junctions[wall_floor]

    corner_ceiling = np.array([c + delta_height for c in corner_floor])

    return walls_corners, corner_floor, corner_ceiling


def complete_bbox_json(path, scene):
    annotations_file = os.path.join(path, "scene_{}".format(scene), "bbox_3d_fixed.json")
    fixed_annotations_file = os.path.join(path, "scene_{}".format(scene), "bbox_3d_fixed_again.json")

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
        room_path = os.path.join(scene_path, room_id, "perspective", "full")

        walls_corners, corner_floor, corner_ceiling = extract_walls_floor_and_ceiling(path, scene, room_id)

        if walls_corners is not None and corner_floor is not None and corner_ceiling is not None:
            annos.append({
                'ID': len(annos),
                'label': 'floor',
                'corners_no_index': corner_floor.tolist()
            })
            annos.append({
                'ID': len(annos),
                'label': 'ceiling',
                'corners_no_index': corner_ceiling.tolist()
            })
            for wall_corners in walls_corners:
                annos.append({
                    'ID': len(annos),
                    'label': 'wall',
                    'corners_no_index': wall_corners.tolist()
                })

        if not os.path.exists(room_path):
            continue

        for position_id in np.sort(os.listdir(room_path)):
            position_path = os.path.join(room_path, position_id)

            instance = cv2.imread(os.path.join(position_path, 'instance.png'), cv2.IMREAD_UNCHANGED)

            instances_indexes = np.unique(instance)[:-1]

            for index in instances_indexes:
                # for each instance in current image
                bbox = annos[id2index[index]]
                bbox['room_id'] = room_id

    with open(fixed_annotations_file, 'w') as f:
        json.dump(annos, f)


def complete_bbox_json_for_indexes(path, scenes_indexes, processor_id):
    processor_report_file = os.path.join(PATH_TO_DATASET, "processor_{}_report.txt".format(str(processor_id)))
    for counter, scene_index in enumerate(scenes_indexes):
        complete_bbox_json(path, scene_index)
        text = 'Processor {} finished scene {}, that is {}/{} scenes, progress is at: {}%'.format(
        str(processor_id), scene_index, str(counter + 1), str(len(scenes_indexes)), str((counter + 1) / len(scenes_indexes) * 100.))
        print(text)
        with open(processor_report_file, 'a+') as f:
            f.write(text + os.linesep)


if __name__ == "__main__":
    scene_indexes = [str(index).rjust(5, '0') for index in range(0, 3500)]
    scene_indexes_to_convert = []
    # Skip files that have already been generated
    for scene_index in scene_indexes:
        fixed_annotations_file = os.path.join(PATH_TO_DATASET, "scene_{}".format(scene_index), "bbox_3d_fixed.json")
        # Skip if fixed annotation file has been done
        if os.path.isfile(fixed_annotations_file):
            scene_indexes_to_convert.append(scene_index)

    print(scene_indexes_to_convert)

    nb_processors = 16
    length = len(scene_indexes_to_convert)
    scene_indexes_splits = [
        scene_indexes_to_convert[i * length // nb_processors: (i + 1) * length // nb_processors]
        for i in range(nb_processors)
    ]

    for count, s in enumerate(scene_indexes_splits):
        p = Process(target=complete_bbox_json_for_indexes, args=(PATH_TO_DATASET, s, count))
        p.start()

    # complete_bbox_json(PATH_TO_DATASET, '00000')