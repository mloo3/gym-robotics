import os
from gym import utils
from gym_robotics.envs import play_demo


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'fetch', 'pick_and_place.xml')


class FetchPickAndPlaceEnv(play_demo.PlayDemo, utils.EzPickle):
    def __init__(self, reward_type='sparse', goal_loc=None, obj_loc=None):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        play_demo.PlayDemo.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, goal_loc=goal_loc,
            obj_loc=obj_loc,)
        utils.EzPickle.__init__(self)
