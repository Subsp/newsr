import json
import os
import random

from arguments import ModelParams
from scene import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration

from .camera_bridge import read_transforms_json_scene


class HybridScene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=None,
        transforms_llffhold=8,
        skip_initial_pcd=False,
    ):
        if resolution_scales is None:
            resolution_scales = [1.0]

        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}

        sparse0 = os.path.join(args.source_path, "sparse", "0")
        has_colmap = (
            os.path.exists(os.path.join(sparse0, "images.bin"))
            or os.path.exists(os.path.join(sparse0, "images.txt"))
        )

        if os.path.exists(os.path.join(args.source_path, "metadata.json")):
            print("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Multi-scale"](
                args.source_path,
                args.white_background,
                args.eval,
                args.load_allres,
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found transforms.json file, assuming Neuralangelo/Instant-NGP style data set!")
            scene_info = read_transforms_json_scene(
                args.source_path,
                args.white_background,
                args.eval,
                llffhold=transforms_llffhold,
            )
        elif has_colmap:
            print("Found sparse/0 with COLMAP model, assuming COLMAP data set!")
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                args.images,
                args.eval,
                llffhold=transforms_llffhold,
            )
        else:
            raise AssertionError(
                "Unsupported dataset format. Expected one of:\n"
                "  - metadata.json (multi-scale Blender)\n"
                "  - transforms_train.json (Blender)\n"
                "  - transforms.json (Neuralangelo/Instant-NGP style)\n"
                "  - sparse/0/images.bin or sparse/0/images.txt (COLMAP)"
            )

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())

            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for idx, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(idx, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w", encoding="utf-8") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    f"iteration_{self.loaded_iter}",
                    "point_cloud.ply",
                )
            )
        elif not skip_initial_pcd:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
