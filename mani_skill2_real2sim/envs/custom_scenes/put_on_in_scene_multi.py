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
from .put_on_in_scene_distractor import PutOnBridgeInSceneEnv


class PutOnBridgeInSceneEnvMulti(PutOnBridgeInSceneEnv):
    cur_subtask_id = 0

    def reset(self, seed=None, options=None):
        self.cur_subtask_id = 0
        obs, info = super().reset(seed, options)

        return obs, info

    def evaluate(
            self,
            success_require_src_completely_on_target=True,
            z_flag_required_offset=0.02,
            **kwargs
    ):
        ret_info = {}
        success = []
        for (source_obj_id, target_obj_id) in zip(self.source_obj_ids, self.target_obj_ids):
            stats = self.evaluate_pair(source_obj_id, target_obj_id, success_require_src_completely_on_target, z_flag_required_offset)
            source_obj_name = self.episode_objs[source_obj_id].name
            ret_info[f"{source_obj_name}"] = stats
            success.append(stats["success"])
        ret_info["success"] = all(success)

        for i in range(len(success)):
            if success[i]:
                self.cur_subtask_id = i + 1

        return ret_info

    def evaluate_pair(
            self,
            source_obj_id,
            target_obj_id,
            success_require_src_completely_on_target=True,
            z_flag_required_offset=0.02,
            **kwargs
    ):
        source_obj = self.episode_objs[source_obj_id]
        target_obj = self.episode_objs[target_obj_id]
        source_obj_pose = self.get_obj_pose(source_obj_id)
        target_obj_pose = self.get_obj_pose(target_obj_id)

        # whether moved the correct object
        source_obj_xy_move_dist = np.linalg.norm(
            self.episode_obj_xyzs_after_settle[source_obj_id][:2] - source_obj_pose.p[:2]
        )
        other_obj_xy_move_dist = []
        for obj, obj_xyz_after_settle in zip(
                self.episode_objs, self.episode_obj_xyzs_after_settle
        ):
            if obj.name == source_obj.name:
                continue
            other_obj_xy_move_dist.append(
                np.linalg.norm(obj_xyz_after_settle[:2] - obj.pose.p[:2])
            )
        moved_correct_obj = (source_obj_xy_move_dist > 0.03) and (
            all([x < source_obj_xy_move_dist for x in other_obj_xy_move_dist])
        )
        moved_wrong_obj = any([x > 0.03 for x in other_obj_xy_move_dist]) and any(
            [x > source_obj_xy_move_dist for x in other_obj_xy_move_dist]
        )

        # whether the source object is grasped
        is_src_obj_grasped = self.agent.check_grasp(source_obj)
        if is_src_obj_grasped:
            self.consecutive_grasp += 1
        else:
            self.consecutive_grasp = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        # whether the source object is on the target object based on bounding box position
        tgt_obj_half_length_bbox = (
                self.episode_obj_bbox_world[target_obj_id] / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_half_length_bbox = self.episode_obj_bbox_world[source_obj_id] / 2

        pos_src = source_obj_pose.p
        pos_tgt = target_obj_pose.p
        offset = pos_src - pos_tgt
        xy_flag = (
                np.linalg.norm(offset[:2])
                <= np.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.003
        )
        z_flag = (offset[2] > 0) and (
                offset[2] - tgt_obj_half_length_bbox[2] - src_obj_half_length_bbox[2]
                <= z_flag_required_offset
        )
        src_on_target = xy_flag and z_flag

        if success_require_src_completely_on_target:
            # whether the source object is on the target object based on contact information
            contacts = self._scene.get_contacts()
            flag = True
            robot_link_names = [x.name for x in self.agent.robot.get_links()]
            tgt_obj_name = target_obj.name
            ignore_actor_names = [tgt_obj_name] + robot_link_names
            for contact in contacts:
                actor_0, actor_1 = contact.actor0, contact.actor1
                other_obj_contact_actor_name = None
                if actor_0.name == source_obj.name:
                    other_obj_contact_actor_name = actor_1.name
                elif actor_1.name == source_obj.name:
                    other_obj_contact_actor_name = actor_0.name
                if other_obj_contact_actor_name is not None:
                    # the object is in contact with an actor
                    contact_impulse = np.sum(
                        [point.impulse for point in contact.points], axis=0
                    )
                    if (other_obj_contact_actor_name not in ignore_actor_names) and (
                            np.linalg.norm(contact_impulse) > 1e-6
                    ):
                        # the object has contact with an actor other than the robot link or the target object, so the object is not yet put on the target object
                        flag = False
                        break
            src_on_target = src_on_target and flag

        success = src_on_target

        # self.episode_stats["moved_correct_obj"] = moved_correct_obj
        # self.episode_stats["moved_wrong_obj"] = moved_wrong_obj
        # self.episode_stats["src_on_target"] = src_on_target
        # self.episode_stats["is_src_obj_grasped"] = (
        #         self.episode_stats["is_src_obj_grasped"] or is_src_obj_grasped
        # )
        # self.episode_stats["consecutive_grasp"] = (
        #         self.episode_stats["consecutive_grasp"] or consecutive_grasp
        # )

        return dict(
            moved_correct_obj=moved_correct_obj,
            moved_wrong_obj=moved_wrong_obj,
            is_src_obj_grasped=is_src_obj_grasped,
            consecutive_grasp=consecutive_grasp,
            src_on_target=src_on_target,
            episode_stats={},
            success=success,
        )


@register_env("PutOnPlateInSceneMulti-v0", max_episode_steps=300)
class PutOnPlateInSceneMulti(PutOnBridgeInSceneEnvMulti):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "bridge_plate_objaverse_larger"
        distractor_obj_names = kwargs.pop("distractor_obj_names", ["apple", "pepsi_can", "eggplant"])

        self.source_obj_ids = [0, 2]
        self.target_obj_ids = [1, 1]
        self.cur_subtask_id = 0

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

    def evaluate(
            self,
            success_require_src_completely_on_target=True,
            z_flag_required_offset=0.02,
            **kwargs
    ):
        ret_info = {}
        success = []
        for (source_obj_id, target_obj_id) in zip(self.source_obj_ids, self.target_obj_ids):
            stats = self.evaluate_pair(source_obj_id, target_obj_id, success_require_src_completely_on_target, z_flag_required_offset)
            source_obj_name = self.episode_objs[source_obj_id].name
            ret_info[f"{source_obj_name}"] = stats
            success.append(stats["success"] and not stats["is_src_obj_grasped"])
        ret_info["success"] = all(success)

        for i in range(len(success)):
            if success[i]:
                self.cur_subtask_id = i + 1

        return ret_info

    def get_language_instruction(self, **kwargs):
        task_id = min(self.cur_subtask_id, len(self.source_obj_ids) - 1)
        source_obj_id = self.source_obj_ids[task_id]
        obj_name = self.episode_objs[source_obj_id].name
        obj_name = self.name_map[obj_name]
        return f"put {obj_name} on plate"

    # def get_dialog_instruction(self, **kwargs):
    #     choices = [
    #         "Hi robot, I'm ready to cook. So help me put the vegetables on the plate."
    #         "Hello, robot. I'm going to start plating and cooking. Can you help me?"
    #     ]
    #
    #     dial = random.choice(choices)
    #
    #     return dial


@register_env("PutOnPlateInSceneSequence-v0", max_episode_steps=300)
class PutOnPlateInSceneSequence(PutOnBridgeInSceneEnvMulti):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "bridge_plate_objaverse_larger"
        distractor_obj_names = kwargs.pop("distractor_obj_names", ["bridge_spoon_generated_modified", "table_cloth_generated_shorter", "apple"])

        self.source_obj_ids = [0, 2]
        self.target_obj_ids = [1, 3]
        self.cur_subtask_id = 0

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
        if success[0] is True:
            self.cur_subtask_id = 1
        if success[1] is True:
            self.cur_subtask_id = 0
        if success[0] is True and success[1] is True:
            self.cur_subtask_id = 1

        # for i in range(len(success)):
        #     if success[i]:
        #         self.cur_subtask_id = i + 1

        return ret_info

    def get_language_instruction(self, **kwargs):
        task_id = min(self.cur_subtask_id, len(self.source_obj_ids) - 1)
        source_obj_id = self.source_obj_ids[task_id]
        target_obj_id = self.target_obj_ids[task_id]
        obj_name = self.episode_objs[source_obj_id].name
        obj_name = self.name_map[obj_name]
        target_obj_name = self.episode_objs[target_obj_id].name
        target_obj_name = self.name_map[target_obj_name]

        return f"put {obj_name} on {target_obj_name}"


@register_env("PutOnPlateInSceneComb-v0", max_episode_steps=300)
class PutOnPlateInSceneComb(PutOnPlateInSceneSequence):
    def get_language_instruction(self, **kwargs):
        instr_list = []
        for (source_obj_id, target_obj_id) in zip(self.source_obj_ids, self.target_obj_ids):
            obj_name = self.episode_objs[source_obj_id].name
            obj_name = self.name_map[obj_name]
            target_obj_name = self.episode_objs[target_obj_id].name
            target_obj_name = self.name_map[target_obj_name]
            instr = f"put {obj_name} on {target_obj_name}"
            instr_list.append(instr)
        instruction = " and ".join(instr_list)

        return instruction
