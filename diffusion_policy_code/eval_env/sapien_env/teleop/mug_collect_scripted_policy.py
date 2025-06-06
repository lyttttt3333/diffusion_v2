from sapien_env.teleop.teleop_robot import TeleopRobot
from sapien_env.gui.teleop_gui_trossen import GUIBase, DEFAULT_TABLE_TOP_CAMERAS
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.rl_env.mug_collect_env import MugCollectRLEnv
import sys
import cv2
import numpy as np

# from pyquaternion import Quaternion
import IPython
import transforms3d

e = IPython.embed
# import os
# import h5py
# import torch
# sys.path.append('/home/bing4090/yixuan_old_branch/general_dp/sapien_env')

MIN_VALUE = 0.0  # Minimum value of the contact sensor data
MAX_VALUE = 0.05  # Maximum value of the contact sensor data
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300


class SingleArmPolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint["xyz"]
        curr_quat = curr_waypoint["quat"]
        curr_grip = curr_waypoint["gripper"]
        next_xyz = next_waypoint["xyz"]
        next_quat = next_waypoint["quat"]
        next_grip = next_waypoint["gripper"]
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def single_trajectory(self, env, ee_link_pose, mode="straight"):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(env, ee_link_pose, mode)

        if self.trajectory[0]["t"] == self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)

        if len(self.trajectory) == 0:
            quit = True
            return None, self.curr_waypoint, quit
            # return None, quit
        else:
            quit = False

        next_waypoint = self.trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        xyz, quat, gripper = self.interpolate(
            self.curr_waypoint, next_waypoint, self.step_count
        )

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            xyz = xyz + np.random.uniform(-scale, scale, xyz.shape)

        self.step_count += 1
        cartisen_action_dim = 6
        grip_dim = 1
        cartisen_action = np.zeros(cartisen_action_dim + grip_dim)
        eluer = transforms3d.euler.quat2euler(quat, axes="sxyz")
        cartisen_action[0:3] = xyz
        cartisen_action[3:6] = eluer
        cartisen_action[6] = gripper

        # waypoint = next_waypoint['xyz'].tolist()
        # waypoint.append(next_waypoint['gripper'])

        return cartisen_action, next_waypoint, quit
        # return cartisen_action, quit

    def generate_trajectory(self, env: MugCollectRLEnv, ee_link_pose, mode="straight"):
        init_position = env.manipulated_object.get_pose().p
        goal_position = env.box_pos

        if env.manip_obj_name == "diet_soda":
            grasp_h = 0.13
            place_h = 0.20
        elif env.manip_obj_name == "drpepper":
            grasp_h = 0.14
            place_h = 0.25
        elif env.manip_obj_name == "white_pepsi":
            grasp_h = 0.14
            place_h = 0.23  # placeholder, this parameter is not accurate
        elif env.manip_obj_name == "pepsi":
            grasp_h = 0.16
            place_h = 0.23
        elif env.manip_obj_name == "cola":
            grasp_h = 0.16
            place_h = 0.17
        else:
            raise NotImplementedError

        if mode == "straight":
            self.trajectory = [
                {
                    "t": 0,
                    "xyz": ee_link_pose.p,
                    "quat": ee_link_pose.q,
                    "gripper": 0.09,
                },
                {
                    "t": 30,
                    "xyz": init_position + np.array([0, 0, grasp_h + 0.1]),
                    "quat": ee_link_pose.q,
                    "gripper": 0.09,
                },
                {
                    "t": 60,
                    "xyz": init_position + np.array([0, 0, grasp_h]),
                    "quat": ee_link_pose.q,
                    "gripper": 0.09,
                },
                {
                    "t": 70,
                    "xyz": init_position + np.array([0, 0, grasp_h]),
                    "quat": ee_link_pose.q,
                    "gripper": 0.01,
                },
                {
                    "t": 90,
                    "xyz": init_position + np.array([0, 0, 0.3]),
                    "quat": ee_link_pose.q,
                    "gripper": 0.01,
                },
                {
                    "t": 120,
                    "xyz": goal_position + np.array([0, 0, 0.3]),
                    "quat": ee_link_pose.q,
                    "gripper": 0.01,
                },
                {
                    "t": 140,
                    "xyz": goal_position + np.array([0, 0, place_h]),
                    "quat": ee_link_pose.q,
                    "gripper": 0.01,
                },
                {
                    "t": 150,
                    "xyz": goal_position + np.array([0, 0, place_h]),
                    "quat": ee_link_pose.q,
                    "gripper": 0.09,
                },
            ]
        else:
            raise NotImplementedError


def plot_contact_color_map(contact_data, sensor_number_dim):
    contact_data_left = contact_data[:sensor_number_dim].reshape((4, 4))
    contact_data_right = contact_data[sensor_number_dim:].reshape((4, 4))

    contact_data_left_norm = (contact_data_left - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)
    contact_data_right_norm = (contact_data_right - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)

    contact_data_left_norm_scaled = (contact_data_left_norm * 255).astype(np.uint8)
    contact_data_right_norm_scaled = (contact_data_right_norm * 255).astype(np.uint8)

    colormap_left = cv2.applyColorMap(
        contact_data_left_norm_scaled, cv2.COLORMAP_VIRIDIS
    )
    colormap_right = cv2.applyColorMap(
        contact_data_right_norm_scaled, cv2.COLORMAP_VIRIDIS
    )

    separator = np.ones((colormap_left.shape[0], 1, 3), dtype=np.uint8) * 255

    combined_colormap = np.concatenate(
        (colormap_left, separator, colormap_right), axis=1
    )
    cv2.imshow("Left finger and Right finger Contact Data", combined_colormap)
    cv2.waitKey(1)


def main_env():
    teleop = TeleopRobot("panda")
    max_timesteps = 800
    num_episodes = 3
    onscreen = True
    for episode_idx in range(num_episodes):
        env = MugCollectRLEnv(
            use_gui=True, robot_name="panda", frame_skip=10, use_visual_obs=False
        )
        env.seed(episode_idx)
        env.reset()
        # viewer = Viewer(base_env.renderer)
        # viewer.set_scene(base_env.scene)
        # base_env.viewer = viewer

        # Setup viewer and camera
        add_default_scene_light(env.scene, env.renderer)
        gui = GUIBase(env.scene, env.renderer)
        for cam_name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
            gui.create_camera(**params)
        gui.viewer.set_camera_rpy(0, -0.7, 0.01)
        gui.viewer.set_camera_xyz(-0.4, 0, 0.45)
        scene = env.scene
        # steps = 0
        scene.step()

        for link in env.robot.get_links():
            print(link.get_name(), link.get_pose())
        for joint in env.robot.get_active_joints():
            print(joint.get_name())

        env.seed(episode_idx)
        # env.reset_env()
        scene.step()
        scripted_policy = SingleArmPolicy()
        cartisen_action = teleop.init_cartisen_action(env.robot.get_qpos()[:])
        action = np.zeros(7)
        arm_dof = env.arm_dof
        # Set data dir

        for _ in range(max_timesteps):
            cartisen_action = scripted_policy.single_trajectory(
                env, teleop.ee_link_pose
            )
            action[:arm_dof] = teleop.ik_panda(env.robot.get_qpos()[:], cartisen_action)
            action[arm_dof:] = cartisen_action[6]
            obs, reward, done, _ = env.step(action[:7])
            if onscreen:
                gui.render()
        gui.viewer.close()


if __name__ == "__main__":
    main_env()
