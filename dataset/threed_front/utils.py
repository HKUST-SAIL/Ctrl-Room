from collections import defaultdict
import os
import csv
import json

import numpy as np
import pickle
import trimesh

from .threed_front_scene import Asset, ModelInfo, Room, ThreedFutureModel, \
    ThreedFutureExtra


def parse_threed_front_scenes(dataset_directory, path_to_model_info, path_to_models, path_to_room_masks_dir=None):
    if os.getenv("PATH_TO_SCENES"):
        print('loading pickled 3d front scenes from :', os.getenv("PATH_TO_SCENES"))
        scenes = pickle.load(open(os.getenv("PATH_TO_SCENES"), "rb"))
    else:
        # Parse the model info
        mf = ModelInfo.from_file(path_to_model_info)
        model_info = mf.model_info

        path_to_scene_layouts = [
            os.path.join(dataset_directory, f) for f in sorted(os.listdir(dataset_directory)) if f.endswith(".json")
        ]
        scenes = []
        unique_room_ids = set()

        mean_window_size = np.array([1.96, 1.6, 0.2])
        mean_door_size = np.array([0.92, 2.15, 0.12])

        transform_matrix = np.eye(4)
        window_box = trimesh.creation.box(mean_window_size, transform_matrix)
        door_box = trimesh.creation.box(mean_door_size, transform_matrix)
        # trimesh.create_box(mean_window_size).export('/tmp/mean_window_size.obj')

        # Start parsing the dataset
        print("Loading dataset ", end="")
        for i, m in enumerate(path_to_scene_layouts):
            with open(m) as f:
                data = json.load(f)

                #  these doors/windows are counted as furniture, but cannot be referred to by the model_info
                doors_in_furniture = defaultdict()
                windows_in_furniture = defaultdict()
                # Parse the furniture of the scene
                furniture_in_scene = defaultdict()
                for ff in data["furniture"]:
                    if "valid" in ff and ff["valid"]:
                        furniture_in_scene[ff["uid"]] = dict(model_uid=ff["uid"],
                                                             model_jid=ff["jid"],
                                                             model_info=model_info[ff["jid"]])
                    if "title" in ff and "window" in ff["title"]:
                        windows_in_furniture[ff["uid"]] = dict(mesh_uid=ff["uid"],
                                                               mesh_jid=ff["jid"],
                                                               mesh_xyz=np.asarray(window_box.vertices).reshape(-1, 3),
                                                               mesh_faces=np.asarray(window_box.faces).reshape(-1, 3),
                                                               mesh_normals=np.zeros_like(window_box.vertices),
                                                               mesh_type="Window")
                    if "title" in ff and "door" in ff["title"]:
                        doors_in_furniture[ff["uid"]] = dict(mesh_uid=ff["uid"],
                                                             mesh_jid=ff["jid"],
                                                             mesh_xyz=np.asarray(door_box.vertices).reshape(-1, 3),
                                                             mesh_faces=np.asarray(door_box.faces).reshape(-1, 3),
                                                             mesh_normals=np.zeros_like(door_box.vertices),
                                                             mesh_type="Door")

                # Parse the extra meshes of the scene e.g walls, doors,
                # windows etc.
                meshes_in_scene = defaultdict()
                # these door/window meshes are not counted as furniture
                doors_in_scene = defaultdict()
                windows_in_scene = defaultdict()
                for mm in data["mesh"]:
                    meshes_in_scene[mm["uid"]] = dict(mesh_uid=mm["uid"],
                                                      mesh_jid=mm["jid"],
                                                      mesh_xyz=np.asarray(mm["xyz"]).reshape(-1, 3),
                                                      mesh_faces=np.asarray(mm["faces"]).reshape(-1, 3),
                                                      mesh_normals=np.asarray(mm["normal"]).reshape(-1, 3),
                                                      mesh_type=mm["type"])
                    if mm["type"] == "Door":
                        doors_in_scene[mm["uid"]] = dict(mesh_uid=mm["uid"],
                                                         mesh_jid=mm["jid"],
                                                         mesh_xyz=np.asarray(mm["xyz"]).reshape(-1, 3),
                                                         mesh_faces=np.asarray(mm["faces"]).reshape(-1, 3),
                                                         mesh_normals=np.asarray(mm["normal"]).reshape(-1, 3),
                                                         mesh_type=mm["type"])
                    elif mm["type"] == "Window":
                        windows_in_scene[mm["uid"]] = dict(mesh_uid=mm["uid"],
                                                           mesh_jid=mm["jid"],
                                                           mesh_xyz=np.asarray(mm["xyz"]).reshape(-1, 3),
                                                           mesh_faces=np.asarray(mm["faces"]).reshape(-1, 3),
                                                           mesh_normals=np.asarray(mm["normal"]).reshape(-1, 3),
                                                           mesh_type=mm["type"])

                # Parse the rooms of the scene
                scene = data["scene"]
                # Keep track of the parsed rooms
                rooms = []
                for rr in scene["room"]:
                    # Keep track of the furniture in the room
                    furniture_in_room = []
                    # Keep track of the extra meshes in the room
                    extra_meshes_in_room = []
                    # Flag to keep track of invalid scenes
                    is_valid_scene = True

                    for cc in rr["children"]:
                        if cc["ref"] in furniture_in_scene:
                            tf = furniture_in_scene[cc["ref"]]
                            # If scale is very small/big ignore this scene
                            if any(si < 1e-5 for si in cc["scale"]):
                                is_valid_scene = False
                                break
                            if any(si > 5 for si in cc["scale"]):
                                is_valid_scene = False
                                break
                            furniture_in_room.append(
                                ThreedFutureModel(tf["model_uid"], tf["model_jid"], tf["model_info"], cc["pos"],
                                                  cc["rot"], cc["scale"], path_to_models))
                        elif cc["ref"] in meshes_in_scene:
                            mf = meshes_in_scene[cc["ref"]]
                            extra_meshes_in_room.append(
                                ThreedFutureExtra(mf["mesh_uid"], mf["mesh_jid"], mf["mesh_xyz"], mf["mesh_faces"],
                                                  mf["mesh_normals"], mf["mesh_type"], cc["pos"], cc["rot"],
                                                  cc["scale"]))
                        elif cc["ref"] in doors_in_furniture:
                            mf = doors_in_furniture[cc["ref"]]
                            extra_meshes_in_room.append(
                                ThreedFutureExtra(mf["mesh_uid"], mf["mesh_jid"], mf["mesh_xyz"], mf["mesh_faces"],
                                                  mf["mesh_normals"], mf["mesh_type"], cc["pos"], cc["rot"],
                                                  cc["scale"]))
                        elif cc["ref"] in windows_in_furniture:
                            mf = windows_in_furniture[cc["ref"]]
                            extra_meshes_in_room.append(
                                ThreedFutureExtra(mf["mesh_uid"], mf["mesh_jid"], mf["mesh_xyz"], mf["mesh_faces"],
                                                  mf["mesh_normals"], mf["mesh_type"], cc["pos"], cc["rot"],
                                                  cc["scale"]))
                        else:
                            continue
                    if len(furniture_in_room) > 1 and is_valid_scene:
                        # Check whether a room with the same instanceid has
                        # already been added to the list of rooms
                        if rr["instanceid"] not in unique_room_ids:
                            unique_room_ids.add(rr["instanceid"])
                            # Add to the list
                            rooms.append(
                                Room(
                                    rr["instanceid"],  # scene_id
                                    rr["type"].lower(),  # scene_type
                                    furniture_in_room,  # bounding boxes
                                    extra_meshes_in_room,  # extras e.g. walls
                                    m.split("/")[-1].split(".")[0],  # json_path
                                    path_to_room_masks_dir,
                                    doors_in_scene,
                                    windows_in_scene))
                            # print(rr["type"].lower())
                scenes.append(rooms)
            s = "{:5d} / {:5d}".format(i, len(path_to_scene_layouts))
            print(s, flush=True, end="\b" * len(s))
        print()

        scenes = sum(scenes, [])
        pickle.dump(scenes,
                    open("/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/threed_front_quad_walls_with_door_window.pkl",
                         "wb"))  #/tmp/threed_front.pkl

    return scenes


def parse_threed_future_models(dataset_directory, path_to_models, path_to_model_info):
    if os.getenv("PATH_TO_3D_FUTURE_OBJECTS"):
        furnitures = pickle.load(open(os.getenv("PATH_TO_3D_FUTURE_OBJECTS"), "rb"))
    else:
        # Parse the model info
        mf = ModelInfo.from_file(path_to_model_info)
        model_info = mf.model_info

        path_to_scene_layouts = [
            os.path.join(dataset_directory, f) for f in sorted(os.listdir(dataset_directory)) if f.endswith(".json")
        ]
        # List to keep track of all available furniture in the dataset
        furnitures = []
        unique_furniture_ids = set()

        # Start parsing the dataset
        print("Loading dataset ", end="")
        for i, m in enumerate(path_to_scene_layouts):
            with open(m) as f:
                data = json.load(f)
                # Parse the furniture of the scene
                furniture_in_scene = defaultdict()
                for ff in data["furniture"]:
                    if "valid" in ff and ff["valid"]:
                        furniture_in_scene[ff["uid"]] = dict(model_uid=ff["uid"],
                                                             model_jid=ff["jid"],
                                                             model_info=model_info[ff["jid"]])
                # Parse the rooms of the scene
                scene = data["scene"]
                for rr in scene["room"]:
                    # Flag to keep track of invalid scenes
                    is_valid_scene = True
                    for cc in rr["children"]:
                        if cc["ref"] in furniture_in_scene:
                            tf = furniture_in_scene[cc["ref"]]
                            # If scale is very small/big ignore this scene
                            if any(si < 1e-5 for si in cc["scale"]):
                                is_valid_scene = False
                                break
                            if any(si > 5 for si in cc["scale"]):
                                is_valid_scene = False
                                break
                            if tf["model_uid"] not in unique_furniture_ids:
                                unique_furniture_ids.add(tf["model_uid"])
                                furnitures.append(
                                    ThreedFutureModel(tf["model_uid"], tf["model_jid"], tf["model_info"], cc["pos"],
                                                      cc["rot"], cc["scale"], path_to_models))
                        else:
                            continue
            s = "{:5d} / {:5d}".format(i, len(path_to_scene_layouts))
            print(s, flush=True, end="\b" * len(s))
        print()

        pickle.dump(furnitures, open("/data/dataset/3D_FRONT_FUTURE/threed_future_model.pkl",
                                     "wb"))  #/tmp/threed_future_model.pkl

    return furnitures


class SplitsBuilder(object):

    def __init__(self, train_test_splits_file):
        self._train_test_splits_file = train_test_splits_file
        self._splits = {}

    def train_split(self):
        return self._splits["train"]

    def test_split(self):
        return self._splits["test"]

    def val_split(self):
        return self._splits["val"]

    def _parse_train_test_splits_file(self):
        with open(self._train_test_splits_file, "r") as f:
            data = [row for row in csv.reader(f)]
        return np.array(data)

    def get_splits(self, keep_splits=["train, val"]):
        if not isinstance(keep_splits, list):
            keep_splits = [keep_splits]
        # Return only the split
        s = []
        for ks in keep_splits:
            s.extend(self._parse_split_file()[ks])
        return s


class CSVSplitsBuilder(SplitsBuilder):

    def _parse_split_file(self):
        if not self._splits:
            data = self._parse_train_test_splits_file()
            for s in ["train", "test", "val"]:
                self._splits[s] = [r[0] for r in data if r[1] == s]
        return self._splits