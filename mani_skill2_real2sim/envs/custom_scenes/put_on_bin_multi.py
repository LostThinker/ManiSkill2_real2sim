import os
from collections import OrderedDict
from typing import List
from pathlib import Path
import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill2_real2sim.utils.common import random_choice
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.object_name_map import name_map
from mani_skill2_real2sim import ASSET_DIR
import random

from .base_env import CustomBridgeObjectsInSceneEnv, CustomBridgeMultiObjectsInSceneEnv
from .move_near_in_scene import MoveNearInSceneEnv
from .put_on_in_scene import PutOnInSceneEnv

from itertools import combinations, permutations
from .put_on_in_scene_multi import PutOnBridgeInSceneEnvMulti


@register_env("PutToyOnBinInSceneMulti-v0", max_episode_steps=300)
class PutToyOnBinInSceneMulti(PutOnBridgeInSceneEnvMulti):
    def __init__(self, **kwargs):
        source_obj_name = "blender_car"
        target_obj_name = "blender_bin1"
        distractor_obj_names = kwargs.pop("distractor_obj_names", ["blender_bear", "blender_duck", "blender_rubik_cube"])

        self.source_obj_ids = [4]
        self.target_obj_ids = [1]
        self.cur_subtask_id = 0
        self.intervals = 10
        self.object_name_map = name_map

        # target_xy = ([-0.14, 0.14],)
        #
        # xy_center = np.array([-0.16, -0.06])
        # half_edge_length_x = 0.075
        # # half_edge_length_y = 0.075
        # half_edge_length_y = 0.05
        # obj_num = len(distractor_obj_names) + 1
        #
        # obj_xy_configs = self._get_pose_config(xy_center, half_edge_length_x, half_edge_length_y, obj_num, grid_size_x=3, grid_size_y=2, remove_x_line=False, threshold=0.04)
        # xy_configs = []
        # for cfg in obj_xy_configs:
        #     cfg = cfg[:1] + target_xy + cfg[1:]
        #     xy_configs.append(cfg)
        obj_num = len(distractor_obj_names) + 1
        xy_configs = self.configure_xy_pose(obj_num)

        quat_configs_1 = [euler2quat(0, 0, np.pi), [1, 0, 0, 0]] + [euler2quat(0, 0, np.pi)] * len(distractor_obj_names)
        quat_configs_2 = [euler2quat(0, 0, -np.pi / 2), [1, 0, 0, 0]] + [euler2quat(0, 0, -np.pi / 2)] * len(distractor_obj_names)
        quat_configs = [
            np.array(quat_configs_1),
            np.array(quat_configs_2),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            distractor_obj_names=distractor_obj_names,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def configure_xy_pose(self, obj_num):
        xy_configs = []

        target_xy = ([-0.14, -0.11],)
        xy_center = np.array([-0.16, 0.09])
        half_edge_length_x = 0.075
        # half_edge_length_y = 0.075
        half_edge_length_y = 0.05

        obj_xy_configs = self._get_pose_config(xy_center, half_edge_length_x, half_edge_length_y, obj_num, grid_size_x=3, grid_size_y=2, remove_x_line=False, threshold=0.04)

        for cfg in obj_xy_configs:
            cfg = cfg[:1] + target_xy + cfg[1:]
            xy_configs.append(cfg)

        target_xy = ([-0.14, 0.14],)
        xy_center = np.array([-0.16, -0.06])
        obj_xy_configs = self._get_pose_config(xy_center, half_edge_length_x, half_edge_length_y, obj_num, grid_size_x=3, grid_size_y=2, remove_x_line=False, threshold=0.04)

        for cfg in obj_xy_configs:
            cfg = cfg[:1] + target_xy + cfg[1:]
            xy_configs.append(cfg)

        return xy_configs

    def evaluate(
            self,
            success_require_src_completely_on_target=True,
            z_flag_required_offset=0.02,
            **kwargs
    ):
        ret_info = {}
        success = []
        for (source_obj_id, target_obj_id) in zip(self.source_obj_ids, self.target_obj_ids):
            if self.episode_objs[source_obj_id].name == "bridge_spoon_generated_modified":
                stats = self.evaluate_pair(source_obj_id, target_obj_id, False, z_flag_required_offset)
            else:
                stats = self.evaluate_pair(source_obj_id, target_obj_id, success_require_src_completely_on_target, z_flag_required_offset)
            source_obj_name = self.episode_objs[source_obj_id].name
            ret_info[f"{source_obj_name}"] = stats
            success.append(stats["success"] and not stats["is_src_obj_grasped"])
        ret_info["success"] = all(success)

        # We don't know which task the robot will complete first.
        # if success[0] is True:
        #     self.cur_subtask_id = 1
        # if success[1] is True:
        #     self.cur_subtask_id = 0
        # if success[0] is True and success[1] is True:
        #     self.cur_subtask_id = 1

        # for i in range(len(success)):
        #     if success[i]:
        #         self.cur_subtask_id = i + 1

        return ret_info

    def get_language_instruction(self, **kwargs):
        return "put rubik's cube on the tray"

    # def get_language_instruction(self, **kwargs):
    #     task_id = min(self.cur_subtask_id, len(self.source_obj_ids) - 1)
    #     source_obj_id = self.source_obj_ids[task_id]
    #     target_obj_id = self.target_obj_ids[task_id]
    #     obj_name = self.episode_objs[source_obj_id].name
    #     obj_name = self.object_name_map[obj_name]
    #     target_obj_name = self.episode_objs[target_obj_id].name
    #     target_obj_name = self.object_name_map[target_obj_name]
    #
    #     return f"put {obj_name} on {target_obj_name}"


@register_env("PutFruitOnBinInSceneMulti-v0", max_episode_steps=300)
class PutFruitOnBinInSceneMulti(PutOnBridgeInSceneEnvMulti):
    def __init__(self, **kwargs):
        source_obj_name = "apple"
        target_obj_name = "blender_bin1"
        distractor_obj_names = kwargs.pop("distractor_obj_names", ["blender_pitaya", "011_banana", "blender_watermelon"])

        self.source_obj_ids = [0, 2, 3, 4]
        self.target_obj_ids = [1, 1, 1, 1]
        self.cur_subtask_id = 0
        self.intervals = 10
        self.object_name_map = name_map
        self.success_tag = []

        # target_xy = ([-0.14, -0.11],)
        #
        # xy_center = np.array([-0.16, 0.09])
        # half_edge_length_x = 0.06
        # # half_edge_length_y = 0.075
        # half_edge_length_y = 0.05
        # obj_num = len(distractor_obj_names) + 1
        #
        # obj_xy_configs = self._get_pose_config(xy_center, half_edge_length_x, half_edge_length_y, obj_num, grid_size_x=2, grid_size_y=2, remove_x_line=False, threshold=0.04)
        # xy_configs = []
        # for cfg in obj_xy_configs:
        #     cfg = cfg[:1] + target_xy + cfg[1:]
        #     xy_configs.append(cfg)
        obj_num = len(distractor_obj_names) + 1
        xy_configs = self.configure_xy_pose(obj_num)

        quat_configs_1 = [euler2quat(0, 0, np.pi), [1, 0, 0, 0]] + [euler2quat(0, 0, np.pi)] * len(distractor_obj_names)
        quat_configs_2 = [euler2quat(0, 0, -np.pi / 2), [1, 0, 0, 0]] + [euler2quat(0, 0, -np.pi / 2)] * len(distractor_obj_names)
        quat_configs = [
            np.array(quat_configs_1),
            np.array(quat_configs_2),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            distractor_obj_names=distractor_obj_names,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def configure_xy_pose(self, obj_num):
        xy_configs = []

        target_xy = ([-0.14, -0.11],)
        xy_center = np.array([-0.16, 0.09])
        half_edge_length_x = 0.06
        # half_edge_length_y = 0.075
        half_edge_length_y = 0.05

        obj_xy_configs = self._get_pose_config(xy_center, half_edge_length_x, half_edge_length_y, obj_num, grid_size_x=2, grid_size_y=2, remove_x_line=False, threshold=0.04)

        for cfg in obj_xy_configs:
            cfg = cfg[:1] + target_xy + cfg[1:]
            xy_configs.append(cfg)

        target_xy = ([-0.14, 0.14],)
        xy_center = np.array([-0.16, -0.06])
        obj_xy_configs = self._get_pose_config(xy_center, half_edge_length_x, half_edge_length_y, obj_num, grid_size_x=2, grid_size_y=2, remove_x_line=False, threshold=0.04)

        for cfg in obj_xy_configs:
            cfg = cfg[:1] + target_xy + cfg[1:]
            xy_configs.append(cfg)

        return xy_configs

    def evaluate(
            self,
            success_require_src_completely_on_target=True,
            z_flag_required_offset=0.02,
            **kwargs
    ):
        ret_info = {}
        success = []
        for (source_obj_id, target_obj_id) in zip(self.source_obj_ids, self.target_obj_ids):
            if self.episode_objs[source_obj_id].name == "bridge_spoon_generated_modified":
                stats = self.evaluate_pair(source_obj_id, target_obj_id, False, z_flag_required_offset)
            else:
                stats = self.evaluate_pair(source_obj_id, target_obj_id, success_require_src_completely_on_target, z_flag_required_offset)
            source_obj_name = self.episode_objs[source_obj_id].name
            ret_info[f"{source_obj_name}"] = stats
            success.append(stats["success"] and not stats["is_src_obj_grasped"])
        ret_info["success"] = all(success)

        self.success_tag = success
        # We don't know which task the robot will complete first.
        if success[self.cur_subtask_id] is True:
            false_indices = [i for i, x in enumerate(success) if not x]
            if len(false_indices) > 0:
                self.cur_subtask_id = false_indices[0]

        return ret_info

    def get_language_instruction(self, **kwargs):
        task_id = min(self.cur_subtask_id, len(self.source_obj_ids) - 1)
        source_obj_id = self.source_obj_ids[task_id]
        obj_name = self.episode_objs[source_obj_id].name
        obj_name = self.object_name_map[obj_name]

        return f"put {obj_name} on the tray"


@register_env("PutFruitOnBinInSceneCorrection-v0", max_episode_steps=300)
class PutFruitOnBinInSceneCorrection(PutFruitOnBinInSceneMulti):

    def evaluate(
            self,
            success_require_src_completely_on_target=True,
            z_flag_required_offset=0.02,
            **kwargs
    ):
        ret_info = {}
        success = []
        for (source_obj_id, target_obj_id) in zip(self.source_obj_ids, self.target_obj_ids):
            if self.episode_objs[source_obj_id].name == "bridge_spoon_generated_modified":
                stats = self.evaluate_pair(source_obj_id, target_obj_id, False, z_flag_required_offset)
            else:
                stats = self.evaluate_pair(source_obj_id, target_obj_id, success_require_src_completely_on_target, z_flag_required_offset)
            source_obj_name = self.episode_objs[source_obj_id].name
            ret_info[f"{source_obj_name}"] = stats
            success.append(stats["success"] and not stats["is_src_obj_grasped"])
        ret_info["success"] = all(success)

        self.success_tag = success
        # We don't know which task the robot will complete first.
        if True in success:
            success_indices = [i for i, x in enumerate(success) if x]
            self.cur_subtask_id = success_indices[0]

        return ret_info

    def get_language_instruction(self, **kwargs):
        if self.cur_subtask_id == 0:
            return "put all fruits on the tray"
        else:
            source_obj_id = self.source_obj_ids[self.cur_subtask_id]
            obj_name = self.episode_objs[source_obj_id].name
            obj_name = self.object_name_map[obj_name]
            return f"take {obj_name} out of the tray"

    def get_correction_prompt(self):
        prompt = "Now you've changed your mind. You want the robot to take everything out of the tray. "

        return prompt