from collections import OrderedDict
from typing import List

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill2_real2sim.utils.common import random_choice
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim import ASSET_DIR

from .base_env import CustomBridgeObjectsInSceneEnv
from .move_near_in_scene import MoveNearInSceneEnv
from .put_on_in_scene import PutOnBridgeInSceneEnv


@register_env("PutUnseenObjOnPlateInScene-v0", max_episode_steps=200)
class PutUnseenObjOnPlateInScene(PutOnBridgeInSceneEnv):
    DEFAULT_MODEL_JSON = "info_bridge_custom_v1.json"

    def __init__(self, obj_name, **kwargs):
        self.available_objects_map = {
            "banana": "011_banana",
            "apple": "apple",
            "coke can": "coke_can",
            "fork": "030_fork",
            "knife": "032_knife",
            "spoon": "031_spoon",
            "orange": "017_orange",
            "orange cup": "065-a_cups",
            "blue cup": "065-b_cups",
            "car": "blender_car",
            "duck": "blender_duck",
            "pitaya": "blender_pitaya",
            "baby bottle": "blender_baby_bottle",
            "croissant": "blender_croissant",
            "cucumber": "blender_cucumber",
            "maize": "blender_maize",
            "shaver": "blender_shaver",
            "tomato": "blender_tomato",
            "rubik's cube": "blender_rubik_cube",
            "teddy bear": "blender_bear",
            "watermelon": "blender_watermelon",
            "grapes": "blender_grapes"
        }
        self.task_obj_name = obj_name
        source_obj_name = self.available_objects_map[obj_name]
        target_obj_name = "bridge_plate_objaverse_larger"

        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
                grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
                + xy_center[None]
        )

        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append(np.array([grid_pos_1, grid_pos_2]))

        quat_configs = [
            np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, -np.pi / 2), [1, 0, 0, 0]]),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        episode_id = obj_init_options.get(
            "episode_id",
            self._episode_rng.randint(len(self._xy_configs) * len(self._quat_configs)),
        )
        xy_config = self._xy_configs[
            (episode_id % (len(self._xy_configs) * len(self._quat_configs)))
            // len(self._quat_configs)
            ]
        quat_config = self._quat_configs[episode_id % len(self._quat_configs)]

        options["model_ids"] = [self._source_obj_name, self._target_obj_name]
        obj_init_options["source_obj_id"] = 0
        obj_init_options["target_obj_id"] = 1
        obj_init_options["init_xys"] = xy_config
        obj_init_options["init_rot_quats"] = quat_config
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"episode_id": episode_id})
        return obs, info

    def get_language_instruction(self, **kwargs):
        return f"put {self.task_obj_name} on plate"

@register_env("PutUnseenObjOnPlateInSceneDebug-v0", max_episode_steps=100)
class PutUnseenObjOnPlateInSceneDebug(PutOnBridgeInSceneEnv):
    DEFAULT_MODEL_JSON = "info_bridge_custom_v1.json"

    def __init__(self, obj_name, **kwargs):

        source_obj_name = obj_name
        target_obj_name = "bridge_plate_objaverse_larger"

        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
                grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
                + xy_center[None]
        )

        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append(np.array([grid_pos_1, grid_pos_2]))

        quat_configs = [
            np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
            np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        episode_id = obj_init_options.get(
            "episode_id",
            self._episode_rng.randint(len(self._xy_configs) * len(self._quat_configs)),
        )
        xy_config = self._xy_configs[
            (episode_id % (len(self._xy_configs) * len(self._quat_configs)))
            // len(self._quat_configs)
            ]
        quat_config = self._quat_configs[episode_id % len(self._quat_configs)]

        options["model_ids"] = [self._source_obj_name, self._target_obj_name]
        obj_init_options["source_obj_id"] = 0
        obj_init_options["target_obj_id"] = 1
        obj_init_options["init_xys"] = xy_config
        obj_init_options["init_rot_quats"] = quat_config
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"episode_id": episode_id})
        return obs, info

    def get_language_instruction(self, **kwargs):
        return f"put on plate"