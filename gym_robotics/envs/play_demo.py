import numpy as np
from gym.envs.robotics import rotations, robot_env, utils, fetch_env

class PlayDemo(fetch_env.FetchEnv):
    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, goal_loc=None, obj_loc=None,
    ):
        self.goal_loc = goal_loc
        self.obj_loc = obj_loc
        super(PlayDemo, self).__init__(
            model_path, n_substeps, gripper_extra_height, block_gripper,
            has_object, target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type)

    def _sample_goal(self):
        if self.goal_loc is None:
            return super(PlayDemo, self)._sample_goal()
        else:
            return self.goal_loc.copy()

    def _reset_sim(self):
        if self.obj_loc is None:
            return super(PlayDemo, self)._reset_sim()
        else:
            self.sim.set_state(self.initial_state)

            if self.has_object:
                object_qpos = self.sim.data.get_joint_qpos('object0:joint')
                assert object_qpos.shape == (7,)
                object_qpos[:2] = self.obj_loc
                self.sim.data.set_joint_qpos('object0:joint', object_qpos)

            self.sim.forward()
            return True

    def set_goal_loc(self, goal_loc):
        self.goal_loc = goal_loc

    def set_obj_loc(self, obj_loc):
        self.obj_loc = obj_loc



