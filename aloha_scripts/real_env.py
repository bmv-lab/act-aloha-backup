import sys
directory_to_add = "~/act_dec2023"
sys.path.append(directory_to_add)


import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env

from aloha_scripts.constants import DT, HOME_POSE, MASTER_IP, FOLLOWER_IP
from aloha_scripts.robot_utils import Recorder, ImageRecorder
from aloha_scripts.teleop import Follower, Master, generateTrajectory
# from interbotix_xs_modules.arm import InterbotixManipulatorXS
# from interbotix_xs_msgs.msg import JointSingleCommand

# import IPython
# e = IPython.embed


class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self):
        self.puppet = Follower(FOLLOWER_IP)

        # self.recorder = Recorder('left', init_node=False)
        self.image_recorder = ImageRecorder(init_node=True)


    def get_qpos(self):
        qpos = self.puppet.getJointAngles()
        return np.array(qpos)

    def get_qvel(self):
        qvel = self.puppet.getJointVelocity()
        return np.array(qvel)

    def get_effort(self):
        qeff = self.puppet.getJointEffort()
        return np.array(qeff)
    
    def get_images(self):
        return self.image_recorder.get_images()

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            pass
            # # Reboot puppet robot gripper motors,                                           look into this
            # self.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
            # self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
            # self._reset_joints()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def step(self, action):
        self.puppet.operate(action)
        # time.sleep(DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())


def get_action(master):
    action = np.zeros(6) # 6 joint + 1 gripper, for two arms
    # Arm actions
    action = master.getJointAngles()

    return action


def make_real_env():
    env = RealEnv()
    return env
