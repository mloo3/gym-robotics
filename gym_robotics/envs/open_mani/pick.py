import os
from gym import utils
from gym.envs.robotics import fetch_env
from gym_robotics.envs import open_mani_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'open_mani', 'pick.xml')

class PickEnv(open_mani_env.OpenManiEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
                # xyz positions
            'robot0:slide0': 1,
            'robot0:slide1': 0.75,
            'robot0:slide2': 0.2,
            'object0:joint': [2, 0, 0, 1., 0., 0., 0.],
        }
        open_mani_env.OpenManiEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
