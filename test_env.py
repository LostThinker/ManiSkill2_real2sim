import gymnasium as gym
import mani_skill2_real2sim.envs
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np


def test_env(task_name):
    env_kwargs = {}
    env_kwargs['obj_name'] = "blender_grapes"
    env_kwargs["obs_mode"] = "rgbd",
    env_kwargs["prepackaged_config"] = True

    env = gym.make(task_name, **env_kwargs)

    env_reset_options = {}
    env_reset_options["obj_init_options"] = {
        # "episode_id": random.randint(0, 100),  # this determines the obj inits in bridge
        "episode_id": 13,  # this determines the obj inits in bridge
    }
    obs = env.reset(options=env_reset_options)
    instr = env.get_language_instruction()
    print(instr)
    image = obs[0]['image']["3rd_view_camera"]["rgb"]  # 3rd_view_camera
    image = Image.fromarray(image)
    instr = instr.replace(" ", "_")
    image.save(f"./images/{task_name}.png")
    # image.save(f"./images/{instr}.png")
    plt.imshow(image)
    plt.show()


def check_episode(task_name):
    env_kwargs = {}
    env_kwargs['obj_name'] = "orange"
    # env_kwargs["distractor_obj_names"] = []
    env_kwargs["obs_mode"] = "rgbd",
    env_kwargs["prepackaged_config"] = True

    # 假设每行显示 10 个小图像
    rows = 10
    cols = 10

    env = gym.make(task_name, **env_kwargs)
    images = []
    for i in range(rows*cols):
        env_reset_options = {}
        env_reset_options["obj_init_options"] = {
            "episode_id": i,  # this determines the obj inits in bridge
        }
        obs = env.reset(options=env_reset_options)
        image = obs[0]['image']["3rd_view_camera"]["rgb"]  # 3rd_view_camera
        pil_image = Image.fromarray(np.uint8(image))
        images.append(pil_image)



    # 获取小图像的尺寸
    width, height = images[0].size

    # 计算大图像的尺寸
    big_image_width = cols * width
    big_image_height = rows * height

    # 创建空白大图像
    big_image = Image.new('RGB', (big_image_width, big_image_height))

    # 依次粘贴小图像
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(images):
                big_image.paste(images[index], (j * width, i * height))

    # 显示大图像
    plt.imshow(big_image)
    plt.show()


def change_color(hex_color):
    def hex_to_rgb(hex_color):
        """
        将十六进制颜色代码转换为 RGB 元组
        """
        hex_color = hex_color.lstrip('#')
        length = len(hex_color)
        return tuple(int(hex_color[i:i + length // 3], 16) for i in range(0, length, length // 3))

    img_path = "/data/user/qianlong/remote-ws/embodied-ai/vla/RoboVLMs-dev/SimplerEnv-dev/ManiSkill2_real2sim/data/custom/models/bridge_plate_objaverse_larger_color/plate_texture.png"
    image = Image.open(img_path)
    rgb_color = hex_to_rgb(hex_color)
    new_image = Image.new('RGB', image.size, rgb_color)
    new_image.save(img_path)


if __name__ == '__main__':
    # change_color("#E59623")
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    # test_env("PutUnseenObjOnPlateInSceneDebug-v0")
    test_env("PutFruitOnBinInSceneMulti-v0")
    # test_env("PutDrinkOnPlateInSceneDistract-v0")
    # test_env("PutUnseenObjOnPlateInScene-v0")
    # check_episode("PutToyOnBinInSceneMulti-v0")
