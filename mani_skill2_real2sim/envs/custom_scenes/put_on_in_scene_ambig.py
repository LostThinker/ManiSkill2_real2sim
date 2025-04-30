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
from mani_skill2_real2sim import ASSET_DIR
import random

from .base_env import CustomBridgeObjectsInSceneEnv, CustomBridgeMultiObjectsInSceneEnv
from .move_near_in_scene import MoveNearInSceneEnv
from .put_on_in_scene import PutOnInSceneEnv

from itertools import combinations, permutations
from .put_on_in_scene_distractor import PutOnBridgeInSceneEnv


@register_env("PutVegetableOnPlateInScene-v0", max_episode_steps=100)
class PutVegetableOnPlateInScene(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "bridge_plate_objaverse_larger"
        kwargs.pop("distractor_obj_names", None)
        distractor_obj_names = ["apple", "eggplant"]

        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        # half_edge_length_y = 0.075
        half_edge_length_y = 0.1
        obj_num = len(distractor_obj_names) + 2

        xy_configs = self._get_pose_config(xy_center, half_edge_length_x, half_edge_length_y, obj_num)

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

    def get_language_instruction(self, **kwargs):
        return "put carrot on plate"

    def get_ambig_instruction(self, idx):
        ambig_instruction = [
            "Hello Robot, I'm going to make dinner, do you have any recommendations?",
            "Hello robot. I'm gonna make lunch. Can you help me?",
        ]
        return ambig_instruction[idx]

    def get_prompt(self, idx):
        prompts = [
            "Your intent is to ask the robot for recommendations about cooking and to indicate that you don't like eggplant. Your final goal is to get the robot to put carrot on plate so you can chop them.",
            "Your intention is for the robot to put the ingredients needed to make lunch on a plate. Your final goal is to get the robot to put carrot on plate so you can chop them."
        ]

        return prompts[idx]


@register_env("PutDrinkOnPlateInScene-v0", max_episode_steps=100)
class PutDrinkOnPlateInScene(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        source_obj_name = "coke_can"
        target_obj_name = "bridge_plate_objaverse_larger"
        kwargs.pop("distractor_obj_names", None)
        distractor_obj_names = ["pepsi_can", "blue_plastic_bottle", "7up_can"]

        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        # half_edge_length_y = 0.075
        half_edge_length_y = 0.1
        obj_num = len(distractor_obj_names) + 2

        xy_configs = self._get_pose_config(xy_center, half_edge_length_x, half_edge_length_y, obj_num)

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

    def get_language_instruction(self, **kwargs):
        return "put coke can on plate"

    def get_ambig_instruction(self, idx):
        ambig_instruction = [
            "Hello robot. I just got done playing basketball. I'm so tired.",
            "Hello robot. I'm a little thirsty. What do you have there to drink?",
        ]

        return ambig_instruction[idx]

    def get_prompt(self, idx):
        prompts = [
            "Your intention is to express to the robot that you are thirsty and want something to drink. You need to ask the robot what is available to drink and then choose Coca Cola. Your final goal is to make the robot put the coke can on plate.",
            "Your intention is to express to the robot that you are thirsty and want something to drink. You need to ask the robot what is available to drink and then choose Coca Cola. Your final goal is to make the robot put the coke can on plate."
        ]

        return prompts[idx]


@register_env("PutOnTwoPlateInScene-v0", max_episode_steps=100)
class PutOnTwoPlateInScene(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        source_obj_name = "eggplant"
        target_obj_name = "bridge_plate_objaverse_larger_color"
        kwargs.pop("distractor_obj_names", None)
        distractor_obj_names = ["bridge_plate_objaverse_larger", "apple", "coke_can"]

        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        # half_edge_length_y = 0.075
        half_edge_length_y = 0.1
        obj_num = len(distractor_obj_names) + 2

        xy_configs = self._get_pose_config(xy_center, half_edge_length_x, half_edge_length_y, obj_num)

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

    def get_language_instruction(self, **kwargs):
        return "put eggplant on orange plate"
        # return "put eggplant on plate that is not white"

    def get_ambig_instruction(self, idx):
        ambig_instruction = [
            "Hello robot. Can you help me put some food on the plate for dinner?",
            "Hello robot. Can you help me straighten up the table? That eggplant is falling out of the plate.",
        ]

        return ambig_instruction[idx]

    def get_prompt(self, idx):
        prompts = [
            "Your intention is to have a robot help you prepare for dinner. Instead of wanting apples, you plan to fry eggplant. You need to get the robot to help you put the eggplant on an orange colored plate that is specifically for vegetables.",
            "Your goal is to get the robot to help you organize your desktop. You need to get the robot to help you put the eggplants on the orange plate that is dedicated to vegetables."
        ]

        return prompts[idx]


@register_env("PutFruitOnPlateInScene-v0", max_episode_steps=100)
class PutFruitOnPlateInScene(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        source_obj_name = "apple"
        target_obj_name = "bridge_plate_objaverse_larger"
        kwargs.pop("distractor_obj_names", None)
        distractor_obj_names = ["011_banana", "coke_can"]

        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        # half_edge_length_y = 0.075
        half_edge_length_y = 0.1
        obj_num = len(distractor_obj_names) + 2

        xy_configs = self._get_pose_config(xy_center, half_edge_length_x, half_edge_length_y, obj_num)

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

    def get_language_instruction(self, **kwargs):
        return "put apple on plate"

    def get_ambig_instruction(self, idx):
        ambig_instruction = [
            "Hello robot. I'm kind of thirsty. Could you get me a juicy fruit?",
            "Hello robot. I want some fruit.",
        ]

        return ambig_instruction[idx]

    def get_prompt(self, idx):
        prompts = [
            "Your intention is for the robot to prepare an apple for you on a plate and tell the robot that you don't like bananas.",
            "Your intention is for the robot to prepare an apple for you on a plate and tell the robot that you don't like bananas."
        ]

        return prompts[idx]