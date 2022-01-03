from typing import Any, Dict, List, Union

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Reach(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d


class ObstructedReach(Reach):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
        obstacles_f="",
    ) -> None:
        self.obstacles = {}
        self.obstacles_f = obstacles_f
        super().__init__(sim, get_ee_position, reward_type, distance_threshold, goal_range)
        self.base_position = self.get_ee_position()

    def _create_scene(self) -> None:
        super()._create_scene()
        self.create_obstacle('obstacle1', np.array([0., 0., 0.]), np.array([0.05, 0.05, 0.05]))

    def create_obstacle(
        self,
        obstacle_name: str,
        obstacle_pos: np.ndarray,
        obstacle_size: np.ndarray
    ) -> None:
        # BUG: Setting the location of an obstacle within the arm causes wild behaviour. Ensure this doesn't happen!
        self.sim.create_box(
            body_name=obstacle_name,
            half_extents=obstacle_size / 2,
            mass=0.0,
            position=obstacle_pos,
            specular_color=np.zeros(3),
            rgba_color=np.array([0.0, 0.0, 0.95, 1]),
        )
        self.obstacles[obstacle_name] = obstacle_pos

    def set_obstacle_pose(self):
        # For now, place an obstacle that is at the midpoint of path between the start and end goal
        obstacle_offset = (self.goal - self.base_position) / 2.0
        obstacle_cp = self.base_position + obstacle_offset
        # print(f'obstacle_cp: {obstacle_cp}, obstacle_offset: {obstacle_offset}')
        self.sim.set_base_pose("obstacle1", obstacle_cp, np.array([0.0, 0.0, 0.0, 1.0]))
        self.obstacles["obstacle1"] = obstacle_cp

    def reset(self) -> None:
        super().reset()
        self.base_position = self.get_ee_position()
        self.set_obstacle_pose()

    def get_obs(self) -> np.ndarray:
        obs = np.array([])
        try:
            obs = np.concatenate(list(self.obstacles.values()))
        except ValueError as e:
            obs = np.array([-1., -1., -1.])
        return obs

    # def get_obs(self, get_ee: bool=True, get_joint: bool=False) -> np.ndarray:
    #     # TODO: Allow getting joint positions in the observation (place somewhere else?)
    #     obs = np.array([], dtype=np.float32)

    #     # end-effector position and velocity
    #     if get_ee:
    #         ee_position = np.array(self.get_ee_position())
    #         ee_velocity = np.array(self.get_ee_velocity())
    #         obs = np.concatenate((obs, ee_position, ee_velocity))

    #     # joint angles and velocity
    #     if get_joint:
    #         joint_angle = np.array(self.get_joint_angle())
    #         joint_velocity = np.array(self.get_joint_velocity())
    #         obs = np.concatenate((obs, joint_angle, joint_velocity))

    #     # fingers opening
    #     if not self.block_gripper:
    #         fingers_width = self.get_fingers_width()
    #         obs = np.concatenate((obs, [fingers_width]))

    #     return obs
