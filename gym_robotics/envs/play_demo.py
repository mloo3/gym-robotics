import numpy as np
from gym.envs.robotics import rotations, robot_env, utils, fetch_env
from gym import error

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500

class PlayDemo(fetch_env.FetchEnv):
    def __init__(
        self,
        model_path,
        n_substeps,
        gripper_extra_height,
        block_gripper,
        has_object,
        target_in_the_air,
        target_offset,
        obj_range,
        target_range,
        distance_threshold,
        initial_qpos,
        reward_type,
        goal_loc=None,
        obj_loc=None,
    ):
        self.goal_loc = goal_loc
        self.obj_loc = obj_loc
        self.mode = None
        self.actions = []
        super(PlayDemo, self).__init__(
            model_path,
            n_substeps,
            gripper_extra_height,
            block_gripper,
            has_object,
            target_in_the_air,
            target_offset,
            obj_range,
            target_range,
            distance_threshold,
            initial_qpos,
            reward_type,
        )
        self.metadata['render.modes'].append('controller')
        self.metadata['render.modes'].append('keyboard')

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
                object_qpos = self.sim.data.get_joint_qpos("object0:joint")
                assert object_qpos.shape == (7,)
                object_qpos[:2] = self.obj_loc
                self.sim.data.set_joint_qpos("object0:joint", object_qpos)

            self.actions = []

            self.sim.forward()
            return True

    def set_goal_loc(self, goal_loc):
        self.goal_loc = goal_loc

    def set_obj_loc(self, obj_loc):
        self.obj_loc = obj_loc

    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self.mode = mode
        if mode == 'controller':
            self._render_callback()
            return self._get_viewer(mode).render()
        else:
            return super().render(mode, width, height)

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'keyboard':
                self.viewer = mujoco_py.MjViewerKeyboard(self.sim)
            elif mode == 'controller':
                self.viewer = mujoco_py.MjViewerController(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def _set_action(self, action):
        assert action.shape == (4,)
        if self.mode == "controller":
            self.viewer.joystick_callback(0)
            action = self.viewer.action.copy() # get action from user
        action = action.copy()  # ensure that we don't change the action outside of this scope
        if self.mode == "controller":
            self.actions.append(action.reshape(1,4).copy()) # hardcoded reshape
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)
