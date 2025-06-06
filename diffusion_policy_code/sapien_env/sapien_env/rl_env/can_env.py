from functools import cached_property
from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.sim_env.can_env import CanEnv
from sapien_env.rl_env.para import ARM_INIT
from sapien_env.utils.common_robot_utils import (
    generate_free_robot_hand_info,
    generate_arm_robot_hand_info,
    generate_panda_info,
)


class CanRLEnv(CanEnv, BaseRLEnv):
    def __init__(
        self,
        use_gui=False,
        frame_skip=5,
        robot_name="panda",
        constant_object_state=False,
        object_scale=1.0,
        randomness_scale=1,
        friction=1,
        object_pose_noise=0.01,
        manip_obj="cola",
        extra_manip_obj="pepsi",
        randomness_level="full",
        task_level_multimodality=False,
        **renderer_kwargs,
    ):
        super().__init__(
            use_gui,
            frame_skip,
            object_scale,
            randomness_scale,
            friction,
            manip_obj=manip_obj,
            extra_manip_obj=extra_manip_obj,
            randomness_level=randomness_level,
            task_level_multimodality=task_level_multimodality,
            **renderer_kwargs,
        )
        self.setup(robot_name)

        self.constant_object_state = constant_object_state
        self.object_pose_noise = object_pose_noise

        # Parse link name
        if self.is_robot_free:
            info = generate_free_robot_hand_info()[robot_name]
        elif self.is_xarm:
            info = generate_arm_robot_hand_info()[robot_name]
        elif self.is_panda:
            info = generate_panda_info()[robot_name]
        else:
            raise NotImplementedError
        self.palm_link_name = info.palm_name
        self.palm_link = [
            link
            for link in self.robot.get_links()
            if link.get_name() == self.palm_link_name
        ][0]

        # Finger tip: thumb, index, middle, ring
        finger_tip_names = ["panda_leftfinger", "panda_rightfinger"]

        robot_link_names = [link.get_name() for link in self.robot.get_links()]
        self.finger_tip_links = [
            self.robot.get_links()[robot_link_names.index(name)]
            for name in finger_tip_names
        ]

        # Object init pose
        self.object_episode_init_pose = sapien.Pose()
        self.task_level_multimodality = task_level_multimodality

    def get_oracle_state(self):
        # TODO: include multipe objects situation
        robot_qpos_vec = self.robot.get_qpos()
        object_pose = (
            self.object_episode_init_pose
            if self.constant_object_state
            else self.manip_obj.get_pose()
        )
        object_pose_vec = np.concatenate([object_pose.p, object_pose.q])
        z_axis = object_pose.to_transformation_matrix()[:3, 2]
        theta_cos = np.sum(np.array([0, 0, 1]) * z_axis)
        palm_pose = self.palm_link.get_pose()
        object_in_palm = object_pose.p - palm_pose.p
        v = self.manip_obj.get_velocity()
        w = self.manip_obj.get_angular_velocity()
        return np.concatenate(
            [
                robot_qpos_vec,
                object_pose_vec,
                v,
                w,
                object_in_palm,
                np.array([theta_cos]),
            ]
        )

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate([robot_qpos_vec, palm_pose.p])

    def get_reward(self, action=None):
        # 1.0 if success, 0.0 otherwise
        obj_pose = self.manip_obj.get_pose()
        box_pose = self.box_ls.get_pose()

        obj_pos = obj_pose.p
        box_pos = box_pose.p
        obj_box_dist_thresh = 0.03

        obj_z_axis = obj_pose.to_transformation_matrix()[:3, 1]
        obj_align_thresh = 0.99

        reward = 0.0
        #if self.task_level_multimodality:
        if False:
            extra_obj_pose = self.extra_manip_obj.get_pose()
            extra_box_pose = self.extra_box_ls.get_pose()

            extra_obj_pos = extra_obj_pose.p
            extra_box_pos = extra_box_pose.p
            obj_box_dist = min(
                np.linalg.norm(obj_pos[:2] - box_pos[:2]),
                np.linalg.norm(obj_pos[:2] - extra_box_pos[:2]),
            )
            extra_obj_box_dist = min(
                np.linalg.norm(extra_obj_pos[:2] - box_pos[:2]),
                np.linalg.norm(extra_obj_pos[:2] - extra_box_pos[:2]),
            )

            extra_obj_z_axis = extra_obj_pose.to_transformation_matrix()[:3, 1]
            obj_align = np.sum(np.array([0, 0, 1]) * obj_z_axis)
            extra_obj_align = np.sum(np.array([0, 0, 1]) * extra_obj_z_axis)

            if obj_box_dist < obj_box_dist_thresh and obj_align > obj_align_thresh:
                reward += 0.5
            if extra_obj_box_dist < obj_box_dist_thresh and extra_obj_align > obj_align_thresh:
                reward += 0.5
        else:
            obj_box_dist = np.linalg.norm(obj_pos[:2] - box_pos[:2])  # ignore z axis
            obj_align = np.sum(np.array([0, 0, 1]) * obj_z_axis)
            if obj_box_dist < obj_box_dist_thresh and obj_align > obj_align_thresh:
                reward += 1.0
        return reward

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=seed)
        if self.is_xarm:
            qpos = np.zeros(self.robot.dof)
            arm_qpos = self.robot_info.arm_init_qpos
            qpos[: self.arm_dof] = arm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = ARM_INIT + self.robot_info.root_offset
            init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        elif self.is_trossen_arm:
            qpos = np.zeros(self.robot.dof)
            qpos[self.arm_dof :] = [0.021, -0.021]
            arm_qpos = self.robot_info.arm_init_qpos
            qpos[: self.arm_dof] = arm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = ARM_INIT + self.robot_info.root_offset
            init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        elif self.is_panda:
            qpos = self.robot_info.arm_init_qpos.copy()
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = np.array([0.0, -0.5, 0.0])
            init_ori = transforms3d.euler.euler2quat(0, 0, np.pi / 2)
            init_pose = sapien.Pose(init_pos, init_ori)
        else:
            init_pose = sapien.Pose(
                np.array([-0.4, 0, 0.2]), transforms3d.euler.euler2quat(0, np.pi / 2, 0)
            )
        self.robot.set_pose(init_pose)
        self.reset_internal()
        for _ in range(100):
            self.robot.set_qf(
                self.robot.compute_passive_force(
                    external=False, coriolis_and_centrifugal=False
                )
            )
            self.scene.step()

        self.object_episode_init_pose = self.manip_obj.get_pose()
        random_quat = transforms3d.euler.euler2quat(
            *(self.np_random.randn(3) * self.object_pose_noise * 10)
        )
        random_pos = self.np_random.randn(3) * self.object_pose_noise
        self.object_episode_init_pose *= sapien.Pose(
            random_pos, random_quat
        )

        if self.task_level_multimodality:
            self.extra_object_episode_init_pose = self.extra_manip_obj.get_pose()
            extra_random_quat = transforms3d.euler.euler2quat(
                *(self.np_random.randn(3) * self.object_pose_noise * 10)
            )
            extra_random_pos = self.np_random.randn(3) * self.object_pose_noise
            self.extra_object_episode_init_pose *= sapien.Pose(
                extra_random_pos, extra_random_quat
            )

        return self.get_observation()

    def set_init(self, init_states):
        init_pose = sapien.Pose.from_transformation_matrix(init_states[0])
        self.manip_obj.set_pose(init_pose)
        init_box_pose = sapien.Pose.from_transformation_matrix(init_states[1])
        self.box_ls.set_pose(init_box_pose)
        if self.task_level_multimodality:
            extra_init_pose = sapien.Pose.from_transformation_matrix(init_states[2])
            self.extra_manip_obj.set_pose(extra_init_pose)
            extra_box_pose = sapien.Pose.from_transformation_matrix(init_states[3])
            self.extra_box_ls.set_pose(extra_box_pose)

    '''    def add_traj(self,action):
        action_seq=[]
        for i in range(10):
            delta_action = np.zeros_like(action)
            delta_action[2]=-0.05
            action=action+delta_action
            action_seq.append(action)
        try:
            for i in range(len(self.wp)):
                actor = self.wp[i]
                self.scene.remove_actor(actor=actor)
        except:
            pass
        self.wp=[]
        blue = np.array([0., 0., 1.], dtype=np.float32)
        red = np.array([1., 0., 0.], dtype=np.float32)
        for i in range(len(action_seq)):
            action = action_seq[i]
            pose = sapien.Pose(p=action[:3],q=action[3:])
            builder = self.scene.create_actor_builder()
            color = blue * i/10 + red * (1-i/10)
            builder.add_box_visual(pose=pose,
                                half_size=0.0005*np.array([1., 1., 1.], dtype=np.float32),
                                color=color)
            wp = builder.build_static(name=f"wp_{i}")
            self.wp.append(wp)
        return None'''

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return self.robot.dof + 7 + 6 + 3 + 1
        else:
            return len(self.get_robot_state())

    def is_done(self):
        return self.current_step >= self.horizon

    @cached_property
    def horizon(self):
        return 10000


def main_env():
    task_level_multimodality = True

    obj_list = np.array(["cola", "mtndew", "pepsi"])
    obj_list = np.random.choice(obj_list, 2, replace=False)

    env = CanRLEnv(
        use_gui=True,
        robot_name="panda",
        frame_skip=10,
        use_visual_obs=False,
        manip_obj=obj_list[0],
        extra_manip_obj=obj_list[1],
        task_level_multimodality=task_level_multimodality,
    )
    base_env = env
    robot_dof = env.arm_dof + 1
    env.reset()
    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi / 2)
    base_env.viewer = viewer

    viewer.toggle_pause(True)
    for i in range(5000):
        action = np.zeros(robot_dof)
        action[2] = 0.01
        obs, reward, done, _ = env.step(action)
        env.render()

    while not viewer.closed:
        env.render()


if __name__ == "__main__":
    main_env()
