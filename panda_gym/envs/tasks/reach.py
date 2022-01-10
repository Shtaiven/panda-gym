from typing import Any, Dict, List, Union
from collections import deque, OrderedDict

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
        get_ee_velocity,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
        obstacles_f="",
    ) -> None:
        self.obstacles = OrderedDict()
        self.obstacles_f = obstacles_f
        super().__init__(sim, get_ee_position, reward_type, distance_threshold, goal_range)
        self.get_ee_velocity = get_ee_velocity
        self.start_ee_position = self.get_ee_position()
        self.prev_distances = deque(maxlen=20)

    def _create_scene(self) -> None:
        super()._create_scene()
        for idx in range(6):
            self.create_obstacle(f'obstacle{idx}', np.array([-2., -2., -2.]), np.array([0.05, 0.05, 0.05]))

    def create_obstacle(
        self,
        obstacle_name: str,
        obstacle_pos: np.ndarray,
        obstacle_size: np.ndarray
    ) -> None:
        self.sim.create_box(
            body_name=obstacle_name,
            half_extents=obstacle_size / 2,
            mass=0.0,
            position=obstacle_pos,
            specular_color=np.zeros(3),
            rgba_color=np.array([0.0, 0.0, 0.95, 1]),
        )
        self.obstacles[obstacle_name] = obstacle_pos

    def reset_obstacle_pose(self):
        reset_pos = np.array([-2., -2., -2.])
        reset_rot = np.array([0.0, 0.0, 0.0, 1.0])
        for idx in range(6):
            obstacle_name = f"obstacle{idx}"
            self.sim.set_base_pose(obstacle_name, reset_pos, reset_rot)
            self.obstacles[obstacle_name] = reset_pos

    def _create_obstacle_L(self, idx1, idx2, thickness, arm1, arm2, position=None):
        # TODO: use this function to construct L obstacles
        pass

    def _create_obstacle_plates(self, idx, thickness, position=None):
        # TODO: use this fuction to construct plate obstacles
        pass

    def set_obstacle_pose(self, placement: str="inline"):
        self.reset_obstacle_pose()
        if placement == "inline":
            # For now, place an obstacle that is at the midpoint of path between the start and end goal
            obstacle_offset = (self.goal - self.start_ee_position) / 2.0
            obstacle_cp = self.start_ee_position + obstacle_offset
            self.sim.set_base_pose("obstacle0", obstacle_cp, np.array([0.0, 0.0, 0.0, 1.0]))
            self.obstacles["obstacle0"] = obstacle_cp
        elif placement == "L":
            # Place L-shaped objects
            self._create_obstacle_L(0, 1, 0.05, 0.1, 0.1)
            self._create_obstacle_L(2, 3, 0.05, 0.1, 0.1)
            self._create_obstacle_L(4, 5, 0.05, 0.1, 0.1)
        elif placement == "plates":
            # Place plates obstructing the end goal
            self._create_obstacle_plates(0, 0.05)
            self._create_obstacle_plates(1, 0.05)

    def reset(self) -> None:
        super().reset()
        self.start_ee_position = self.get_ee_position()
        self.set_obstacle_pose()
        self.prev_distances.clear()

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

    #     return obs

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        # PID-like action in the reward function
        # TODO: PID multipliers (must be tuned)
        P = 1.0
        I = 1.0
        D = 1.0

        # PID components
        d = distance(achieved_goal, desired_goal)  # euclidean distance to goal
        v3 = self.get_ee_velocity()  # velocity as a vector
        v = distance(v3, np.zeros_like(v3))  # scalar velocity
        self.prev_distances.append(d)
        integrator = np.sum(self.prev_distances)

        pid_reward = -(P*d + I*integrator + D*v)

        # Sparse reward
        sparse_reward = -np.array(d > self.distance_threshold, dtype=np.float32)

        # Velocity reward
        velocity_reward = -v

        return np.sum([pid_reward, sparse_reward])
