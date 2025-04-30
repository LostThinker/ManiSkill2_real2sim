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


class PutOnBridgeInSceneEnv(PutOnInSceneEnv, CustomBridgeMultiObjectsInSceneEnv):
    def __init__(
            self,
            source_obj_name: str = None,
            target_obj_name: str = None,
            distractor_obj_names: List[str] = None,
            xy_configs: List[np.ndarray] = None,
            quat_configs: List[np.ndarray] = None,
            **kwargs,
    ):
        self._source_obj_name = source_obj_name
        self._target_obj_name = target_obj_name
        self._distractor_obj_names = distractor_obj_names
        self._xy_configs = xy_configs
        self._quat_configs = quat_configs
        self.episode_id = 0
        super().__init__(**kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "widowx"
        ret["control_freq"] = 5
        ret["sim_freq"] = 500
        ret["control_mode"] = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        ret["scene_name"] = "bridge_table_1_v1"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/bridge_real_eval_1.png"
        )
        ret["rgb_overlay_cameras"] = ["3rd_view_camera"]

        return ret

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)
        # self._episode_rng.choice(choices, size=2)

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        episode_id = obj_init_options.get(
            "episode_id",
            self._episode_rng.randint(len(self._xy_configs) * len(self._quat_configs)),
        )
        self.episode_id = episode_id
        xy_config = self._xy_configs[
            (episode_id % (len(self._xy_configs) * len(self._quat_configs)))
            // len(self._quat_configs)
            ]
        # quat_config = self._quat_configs[episode_id % len(self._quat_configs)]
        quat_config = []
        for j in range(len(self._quat_configs[0])):
            opts = [self._quat_configs[i][j] for i in range(len(self._quat_configs))]
            if j < 2:
                # hard control for grasped object's quat
                idx = episode_id % len(opts)
            else:
                idx = self._episode_rng.choice([i for i in range(len(opts))])
            quat_config.append(opts[idx])
        quat_config = np.array(quat_config)

        options["model_ids"] = [self._source_obj_name, self._target_obj_name, *self._distractor_obj_names]
        obj_init_options["source_obj_id"] = 0
        obj_init_options["target_obj_id"] = 1
        obj_init_options["init_xys"] = xy_config
        obj_init_options["init_rot_quats"] = quat_config
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"episode_id": episode_id})
        return obs, info

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.147, 0.028],
            "init_rot_quat": [0, 0, 0, 1],
        }
        return False

    def _load_model(self):
        self.episode_objs = []
        for (model_id, model_scale) in zip(
                self.episode_model_ids, self.episode_model_scales
        ):
            density = self.model_db[model_id].get("density", 1000)

            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = model_id
            self.episode_objs.append(obj)

    def _get_pose_config(self, xy_center, half_edge_length_x=0.075, half_edge_length_y=0.1, obj_num=2):
        # 生成网格点的相对位置
        grid_size = 3  # 这里可以根据需要调整网格大小
        grid_positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                grid_positions.append([i, j])
        grid_pos = np.array(grid_positions) * 2 / (grid_size - 1) - 1
        grid_pos_new = []

        for pos in grid_pos:
            if pos[0] == 0 and pos[1] != 0:
                pass
            else:
                grid_pos_new.append(pos)
        grid_pos = np.stack(grid_pos_new, axis=0)

        # 计算网格点的实际坐标
        grid_pos = (
                grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
                + xy_center[None]
        )

        # generate for source and target object first
        xy_configs = []
        valid_sets = []
        for comb in combinations(grid_pos, 2):
            is_valid = True
            # 检查组合中任意两点之间的距离是否都不小于最小间隔
            for p1, p2 in combinations(comb, 2):
                distance = np.linalg.norm(p1 - p2)
                if distance <= 0.075:
                    is_valid = False
                    break
            if is_valid:
                valid_sets.append(np.array(comb))

        for valid_set in valid_sets:
            perms = list(permutations(valid_set))
            max_set = min(len(perms), 5)
            xy_configs.extend(perms[::int(len(perms) / max_set)])

        if obj_num == 2:
            return xy_configs

        xy_configs_full = []
        for fix_ps in xy_configs:
            # find remain points
            distractor_xy_options = []
            for p in grid_pos:
                in_tag = False
                for array in fix_ps:
                    if np.array_equal(p, array):
                        in_tag = True
                if not in_tag:
                    distractor_xy_options.append(p)

            # find valid distractor points
            select_distractor_xy = []
            for comb in combinations(distractor_xy_options, obj_num - 2):
                selected_xy = list(fix_ps) + list(comb)
                is_valid = True
                for p1, p2 in combinations(selected_xy, 2):
                    distance = np.linalg.norm(p1 - p2)
                    if distance <= 0.075:
                        is_valid = False
                        break
                if is_valid:
                    select_distractor_xy.append(comb)

            # random choose one distractor pose comb
            idx = np.random.choice([j for j in range(len(select_distractor_xy))])

            cfg = fix_ps + select_distractor_xy[idx]
            xy_configs_full.append(cfg)

        return xy_configs_full

    @property
    def name_map(self):
        name_map = dict(
            bridge_carrot_generated_modified="carrot",
            bridge_plate_objaverse_larger="plate",
            pepsi_can="pepsi can",
            apple="apple",
            eggplant="eggplant",
            table_cloth_generated_shorter="towel",
            bridge_spoon_generated_modified="spoon"
        )
        return name_map


@register_env("PutCarrotOnPlateInSceneDistract-v0", max_episode_steps=100)
class PutCarrotOnPlateInSceneDistract(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "bridge_plate_objaverse_larger"
        distractor_obj_names = kwargs.pop("distractor_obj_names", ["apple", "pepsi_can", "eggplant"])

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

    # def get_dialog_instruction(self, **kwargs):
    #     choices = [
    #         "Hi robot, I'm ready to cook. So help me put the vegetables on the plate."
    #         "Hello, robot. I'm going to start plating and cooking. Can you help me?"
    #     ]
    #
    #     dial = random.choice(choices)
    #
    #     return dial


@register_env("StackGreenCubeOnYellowCubeInSceneDistract-v0", max_episode_steps=100)
class StackGreenCubeOnYellowCubeInSceneDistract(PutOnBridgeInSceneEnv):
    def __init__(
            self,
            source_obj_name="green_cube_3cm",
            target_obj_name="yellow_cube_3cm",
            **kwargs,
    ):
        distractor_obj_names = kwargs.pop("distractor_obj_names", ["apple", "pepsi_can", "eggplant"])

        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        # half_edge_length_y = 0.075
        half_edge_length_y = 0.1
        obj_num = len(distractor_obj_names) + 2

        xy_configs = self._get_pose_config(xy_center, half_edge_length_x, half_edge_length_y, obj_num)

        quat_configs_1 = [[1, 0, 0, 0], [1, 0, 0, 0]] + [euler2quat(0, 0, np.pi)] * len(distractor_obj_names)
        quat_configs_2 = [[1, 0, 0, 0], [1, 0, 0, 0]] + [euler2quat(0, 0, -np.pi / 2)] * len(distractor_obj_names)
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
        return "stack the green block on the yellow block"


@register_env("PutSpoonOnTableClothInSceneDistract-v0", max_episode_steps=100)
class PutSpoonOnTableClothInSceneDistract(PutOnBridgeInSceneEnv):
    def __init__(
            self,
            source_obj_name="bridge_spoon_generated_modified",
            target_obj_name="table_cloth_generated_shorter",
            **kwargs,
    ):
        distractor_obj_names = kwargs.pop("distractor_obj_names", ["apple", "pepsi_can", "eggplant"])
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        # half_edge_length_y = 0.075
        half_edge_length_y = 0.1
        obj_num = len(distractor_obj_names) + 2

        xy_configs = self._get_pose_config(xy_center, half_edge_length_x, half_edge_length_y, obj_num)

        quat_configs_1 = [[1, 0, 0, 0], [1, 0, 0, 0]] + [euler2quat(0, 0, np.pi)] * len(distractor_obj_names)
        quat_configs_2 = [[1, 0, 0, 0], [1, 0, 0, 0]] + [euler2quat(0, 0, -np.pi / 2)] * len(distractor_obj_names)
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

    def evaluate(self, success_require_src_completely_on_target=False, **kwargs):
        # this environment allows spoons to be partially on the table cloth to be considered successful
        return super().evaluate(success_require_src_completely_on_target, **kwargs)

    def get_language_instruction(self, **kwargs):
        return "put the spoon on the towel"


@register_env("PutEggplantInBasketSceneDistract-v0", max_episode_steps=120)
class PutEggplantInBasketSceneDistract(PutOnBridgeInSceneEnv):
    def __init__(
            self,
            **kwargs,
    ):
        source_obj_name = "eggplant"
        target_obj_name = "dummy_sink_target_plane"  # invisible
        kwargs.pop("distractor_obj_names")
        distractor_obj_names = ["apple"]

        target_xy = np.array([-0.125, 0.025])
        xy_center = [-0.105, 0.206]

        half_span_x = 0.018
        half_span_y = 0.02
        num_x = 2
        num_y = 2

        grid_pos = []
        for x in np.linspace(-half_span_x, half_span_x, num_x):
            for y in np.linspace(-half_span_y, half_span_y, num_y):
                grid_pos.append(np.array([x + xy_center[0], y + xy_center[1]]))

        valid_sets = []
        xy_configs = []
        for comb in combinations(grid_pos, len(distractor_obj_names) + 1):
            is_valid = True
            # 检查组合中任意两点之间的距离是否都不小于最小间隔
            for p1, p2 in combinations(comb, 2):
                distance = np.linalg.norm(p1 - p2)
                if distance <= half_span_x:
                    is_valid = False
                    break
            if is_valid:
                valid_sets.append(comb)

        for valid_set in valid_sets:
            perms = list(permutations(valid_set))
            max_set = min(len(perms), 3)
            xy_configs.extend(perms[::int(len(perms) / max_set)])

        for i in range(len(xy_configs)):
            xy_configs[i] = xy_configs[i][:1] + (target_xy,) + xy_configs[i][1:]

        quat_config_1 = [euler2quat(0, 0, 0, 'sxyz'), [1, 0, 0, 0]] + [[1, 0, 0, 0]] * len(distractor_obj_names)
        quat_config_2 = [euler2quat(0, 0, 1 * np.pi / 4, 'sxyz'), [1, 0, 0, 0]] + [[1, 0, 0, 0]] * len(distractor_obj_names)
        quat_config_3 = [euler2quat(0, 0, -1 * np.pi / 4, 'sxyz'), [1, 0, 0, 0]] + [[1, 0, 0, 0]] * len(distractor_obj_names)

        quat_configs = [
            np.array(quat_config_1),
            np.array(quat_config_2),
            np.array(quat_config_3),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            distractor_obj_names=distractor_obj_names,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            rgb_always_overlay_objects=['sink', 'dummy_sink_target_plane'],
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "put eggplant into yellow basket"

    def _load_model(self):
        super()._load_model()
        self.sink_id = 'sink'
        self.sink = self._build_actor_helper(
            self.sink_id,
            self._scene,
            density=self.model_db[self.sink_id].get("density", 1000),
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction, dynamic_friction=self.obj_dynamic_friction, restitution=0.0
            ),
            root_dir=self.asset_root,
        )
        self.sink.name = self.sink_id

    def _initialize_actors(self):
        # Move the robot far away to avoid collision
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        self.sink.set_pose(sapien.Pose(
            [-0.16, 0.13, 0.88],
            [1, 0, 0, 0]
        ))
        self.sink.lock_motion()

        super()._initialize_actors()

    def evaluate(self, *args, **kwargs):
        return super().evaluate(success_require_src_completely_on_target=False,
                                z_flag_required_offset=0.06,
                                *args, **kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = super()._setup_prepackaged_env_init_config()
        ret["robot"] = "widowx_sink_camera_setup"
        ret["scene_name"] = "bridge_table_1_v2"
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/bridge_sink.png"
        )
        return ret

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.127, 0.06],
            "init_rot_quat": [0, 0, 0, 1],
        }
        return False  # in env reset options, no need to reconfigure the environment

    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow

        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        self._scene.add_directional_light(
            [0, 0, -1],
            [0.3, 0.3, 0.3],
            position=[0, 0, 1],
            shadow=shadow,
            scale=5,
            shadow_map_size=2048,
        )


@register_env("PutCarrotOnPlateInSceneLangDistract-v0", max_episode_steps=100)
class PutCarrotOnPlateInSceneLangDistract(PutCarrotOnPlateInSceneDistract):

    def get_language_instruction(self, **kwargs):
        instr_options = [
            "grasp the carrot and place it on the plate",
            "grasp carrot and put on plate",
            "pick carrot then place on plate"
        ]

        idx = self._episode_rng.choice([i for i in range(len(instr_options))])
        # idx = self.episode_id
        return instr_options[idx]


@register_env("StackGreenCubeOnYellowCubeInSceneLangDistract-v0", max_episode_steps=100)
class StackGreenCubeOnYellowCubeInSceneLangDistract(StackGreenCubeOnYellowCubeInSceneDistract):

    def get_language_instruction(self, **kwargs):
        instr_options = [
            "pick the green block and stack on the yellow block",
            "put the green block on top of the yellow block",
            "grab the green block and put it on top of the yellow block"
        ]

        idx = self._episode_rng.choice([i for i in range(len(instr_options))])
        # idx = self.episode_id
        return instr_options[idx]


@register_env("PutSpoonOnTableClothInSceneLangDistract-v0", max_episode_steps=100)
class PutSpoonOnTableClothInSceneLangDistract(PutSpoonOnTableClothInSceneDistract):

    def get_language_instruction(self, **kwargs):
        instr_options = [
            "move the spoon down on the towel",
            "place the spoon over the towel",
            "grab the spoon and put it on the towel"
        ]

        idx = self._episode_rng.choice([i for i in range(len(instr_options))])
        # idx = self.episode_id
        return instr_options[idx]

# @register_env("PutDrinkOnPlateInSceneDistract-v0", max_episode_steps=60)
# class PutDrinkOnPlateInSceneDistract(PutOnBridgeInSceneEnv):
#     def __init__(self, **kwargs):
#         source_obj_name = "coke_can"
#         target_obj_name = "bridge_plate_objaverse_larger"
#         distractor_obj_names = ["pepsi_can", "fanta_can", "blue_plastic_bottle", "024_bowl"]
#
#         xy_center = np.array([-0.16, 0.00])
#         half_edge_length_x = 0.075
#         # half_edge_length_y = 0.075
#         half_edge_length_y = 0.1
#
#         # 生成网格点的相对位置
#         grid_size = 3  # 这里可以根据需要调整网格大小
#         grid_positions = []
#         for i in range(grid_size):
#             for j in range(grid_size):
#                 grid_positions.append([i, j])
#         grid_pos = np.array(grid_positions) * 2 / (grid_size - 1) - 1
#
#         # 计算网格点的实际坐标
#         grid_pos = (
#                 grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
#                 + xy_center[None]
#         )
#
#         k = len(distractor_obj_names) + 2  # 这里可以根据需要修改 k 的值
#
#         # # 生成 k 个点的组合
#         # xy_configs = []
#         # for comb in combinations(grid_pos, k):
#         #     xy_configs.append(np.array(comb))
#
#         xy_configs = []
#         for perm in permutations(grid_pos, k):
#             is_valid = True
#             # 检查组合中任意两点之间的距离是否都不小于最小间隔
#             for i in range(k):
#                 for j in range(i + 1, k):
#                     distance = np.linalg.norm(perm[i] - perm[j])
#                     if distance < 0.09:
#                         is_valid = False
#                         break
#                 if not is_valid:
#                     break
#             if is_valid:
#                 xy_configs.append(np.array(perm))
#
#         [euler2quat(np.pi / 2, 0, 0)]
#         [euler2quat(0, 0, -np.pi / 2)]
#         [euler2quat(0, 0, np.pi)]
#
#         quat_configs_1 = [euler2quat(np.pi / 2, 0, 0), [1, 0, 0, 0]] + [euler2quat(np.pi / 2, 0, 0)] * len(distractor_obj_names)
#         quat_configs_2 = [euler2quat(0, 0, -np.pi / 2), [1, 0, 0, 0]] + [euler2quat(0, 0, -np.pi / 2)] * len(distractor_obj_names)
#         quat_configs_3 = [euler2quat(0, 0, np.pi), [1, 0, 0, 0]] + [euler2quat(0, 0, np.pi)] * len(distractor_obj_names)
#         quat_configs = [
#             np.array(quat_configs_1),
#             np.array(quat_configs_2),
#             np.array(quat_configs_3),
#         ]
#
#         super().__init__(
#             source_obj_name=source_obj_name,
#             target_obj_name=target_obj_name,
#             distractor_obj_names=distractor_obj_names,
#             xy_configs=xy_configs,
#             quat_configs=quat_configs,
#             **kwargs,
#         )
#
#     def get_language_instruction(self, **kwargs):
#         return "put carrot on plate"
#
#     def get_dialog_instruction(self, **kwargs):
#         choices = [
#             "Hi robot, I'm ready to cook. So help me put the vegetables on the plate."
#             "Hello, robot. I'm going to start plating and cooking. Can you help me?"
#         ]
#
#         dial = random.choice(choices)
#
#         return dial
