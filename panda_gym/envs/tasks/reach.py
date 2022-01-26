from typing import Any, Dict, Tuple, Union
from collections import deque, OrderedDict
import copy

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
            self.sim.place_visualizer(
                target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30
            )

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

    def is_success(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray
    ) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(
        self, achieved_goal, desired_goal, info: Dict[str, Any]
    ) -> Union[np.ndarray, float]:
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
        obstacle_type="inline",
        max_obstacles=6,
        reward_weights=(1.0, 1.0, 1.0),
    ) -> None:
        # These parameters must be places before super().__init__ because they
        # are used in _create_scene, which is called by super().__init__
        self.obstacles = OrderedDict()
        self.obstacle_type = obstacle_type
        self.max_obstacles = max_obstacles

        super().__init__(
            sim, get_ee_position, reward_type, distance_threshold, goal_range
        )

        # These parameters may come after super().__init__
        self.get_ee_velocity = get_ee_velocity
        self.start_ee_position = self.get_ee_position()
        self.prev_distances = deque(maxlen=20)
        self.reward_weights = reward_weights

    def _create_scene(self) -> None:
        super()._create_scene()
        # Construct obstacles based on obstacle type
        obstacle_sizes = np.full((self.max_obstacles, 3), 0.05)
        if self.obstacle_type == "L":
            obstacle_sizes = self._create_obstacle_L()
        elif self.obstacle_type == "planes":
            obstacle_sizes = self._create_obstacle_planes()
        elif self.obstacle_type == "bin":
            obstacle_sizes = self._create_obstacle_bin()

        for idx in range(self.max_obstacles):
            self._generate_obstacle(
                f"obstacle{idx}", np.array([-2.0, -2.0, -2.0]), obstacle_sizes[idx]
            )

    def _generate_obstacle(
        self, obstacle_name: str, obstacle_pos: np.ndarray, obstacle_size: np.ndarray
    ) -> None:
        self.sim.create_box(
            body_name=obstacle_name,
            half_extents=obstacle_size / 2,
            mass=0.0,
            position=obstacle_pos,
            specular_color=np.zeros(3),
            rgba_color=np.array([0.0, 0.0, 0.95, 1]),
        )
        self.obstacles[obstacle_name] = np.concatenate([obstacle_pos, obstacle_size])

    def _remove_obstacle(self, obstacle_name: str) -> None:
        self.sim.remove_body(obstacle_name)
        del self.obstacles[obstacle_name]

    def reset_obstacle_pose(self, obstacle_name=None):
        reset_pos = np.array([-2.0, -2.0, -2.0])
        reset_rot = np.array([0.0, 0.0, 0.0, 1.0])

        if obstacle_name is not None:
            obs_size = self.obstacles[obstacle_name][-3:]
            self.sim.set_base_pose(obstacle_name, reset_pos, reset_rot)
            self.obstacles[obstacle_name] = np.concatenate([reset_pos, obs_size])
            return

        for idx in range(6):
            obstacle_name = f"obstacle{idx}"
            obs_size = self.obstacles[obstacle_name][-3:]
            self.sim.set_base_pose(obstacle_name, reset_pos, reset_rot)
            self.obstacles[obstacle_name] = np.concatenate([reset_pos, obs_size])

    def _create_obstacle_L(self):
        obstacle_sizes = np.zeros((self.max_obstacles, 3), dtype=np.float32)
        # Even obstacles are tall, odd obstacles are long in the y direction
        for idx in range(self.max_obstacles):
            if idx % 3 == 0:
                obstacle_sizes[idx] = np.array([0.3, 0.1, 0.1])
            elif idx % 3 == 1:
                obstacle_sizes[idx] = np.array([0.1, 0.3, 0.1])
            else:
                obstacle_sizes[idx] = np.array([0.1, 0.1, 0.3])
        return obstacle_sizes

    def _create_obstacle_planes(self):
        obstacle_sizes = np.zeros((self.max_obstacles, 3), dtype=np.float32)
        for idx in range(self.max_obstacles):
            if idx < 2:  # Planes 0,1 tangent to x
                obstacle_sizes[idx] = np.array([0.1, 0.4, 0.4])
            elif idx < 4:  # 2,3 tangent to y
                obstacle_sizes[idx] = np.array([0.4, 0.1, 0.4])
            else:  # 5,6 tangent to z
                obstacle_sizes[idx] = np.array([0.4, 0.4, 0.1])
        return obstacle_sizes

    def _create_obstacle_bin(self):
        obstacle_sizes = np.zeros((self.max_obstacles, 3), dtype=np.float32)
        # TODO: finish this
        return obstacle_sizes

    def _move_obstacle_common(
        self, obstacle_name: str, position: np.ndarray, orientation: np.ndarray
    ) -> None:
        self.sim.set_base_pose(obstacle_name, position, orientation)
        self.obstacles[obstacle_name][:3] = np.array(position)

    def _move_obstacle_inline(self, idx: int) -> None:
        # For now, place an obstacle that is at the midpoint of path between the start and end goal
        obstacle_offset = (self.goal - self.start_ee_position) / 2.0
        obstacle_cp = self.start_ee_position + obstacle_offset
        self._move_obstacle_common(
            f"obstacle{idx}", obstacle_cp, np.array([0.0, 0.0, 0.0, 1.0])
        )

    def _move_obstacle_L(
        self, idx1: int, idx2: int, position=None, orientation=None
    ) -> None:
        r"""
        Creates an L-shaped obstacle out of 2 blocks.
        Blocks are constructed as follows:

                        _____
                       |\_____\
                       ||idx1 |_____
                       ||  *  |______\
                       ||     | idx2 |
                       \|_____|______|

        position and orientation are taken relative to the center of the idx1 block
        shown (marked with an apostrophe).

        Parameters:
            idx1 (int): index of the first block to use
            idx2 (int): index of the second block to use
            position (list(float)[3]): position of the L (default random, based on goal_range)
            orientation (list(float)[3]): orientation in quaternion or Euler angles (default random)
        """
        obs1 = f"obstacle{idx1}"
        obs2 = f"obstacle{idx2}"
        position1 = self.obstacles[obs1][:3]
        position2 = self.obstacles[obs2][:3]
        size1 = self.obstacles[obs1][-3:]
        size2 = self.obstacles[obs2][-3:]

        # randomize position if not "random", or keep same if None
        if position == "random":
            goal_range_diff = self.goal_range_high - self.goal_range_low
            position = np.random.rand(3) * goal_range_diff - self.goal_range_low

        # randomize orientation if "random", or keep same if None
        if orientation == "random":
            orientation = np.random.rand(3) * np.full((3,), 4 * np.pi) - np.full(
                (3,), 2 * np.pi
            )

        # calculation of positions of individual obstacles
        if position is not None:
            # TODO
            x1, y1, z1 = position
            l1, w1, h1 = size1
            l2, w2, h2 = size2
            leg_pos = np.array([x1, y1 + (w1 + w2) / 2, z1 + (-h1 + h2) / 2])
            position1 = position
            position2 = position + leg_pos

        # calculation of orientations of individual obstacles
        if orientation is not None:
            orientation1 = orientation
            # TODO: Determine actual orientation of idx2
            orientation2 = orientation

        # move the obstacles to the correct position
        self._move_obstacle_common(obs1, position1, orientation1)
        self._move_obstacle_common(obs2, position2, orientation2)

    def _move_obstacle_planes(self, idx, normal, position=None):
        # TODO: construct planes obstacles
        pass

    def _move_obstacle_bin(self, position=None):
        # TODO: construct a bin obstacle that the arm can reach into
        pass

    def set_obstacle_pose(self, placement: str = "inline"):
        self.reset_obstacle_pose()
        if placement == "inline":
            self._move_obstacle_inline(0)
        elif placement == "L":
            # Place L-shaped objects
            self._move_obstacle_L(0, 1, position="random", orientation="random")
            self._move_obstacle_L(2, 3, position="random", orientation="random")
            self._move_obstacle_L(4, 5, position="random", orientation="random")
        elif placement == "planes":
            # Place planes obstructing the end goal
            self._move_obstacle_planes(0, 0.05)
            self._move_obstacle_planes(1, 0.05)

    def reset(self) -> None:
        super().reset()
        self.start_ee_position = self.get_ee_position()
        self.set_obstacle_pose()
        self.prev_distances.clear()

    def get_obs(self) -> np.ndarray:
        obs = np.array([])
        try:
            # TODO: sort the dictionary items by position (x, then y, then z) then by size
            obs = np.concatenate(list(self.obstacles.values()), dtype=np.float32)
        except ValueError as e:
            obs = np.array(
                [-2.0, -2.0, -2.0, 0.0, 0.0, 0.0] * self.max_obstacles, dtype=np.float32
            )

        return obs

    def compute_reward(
        self, achieved_goal, desired_goal, info: Dict[str, Any]
    ) -> Union[np.ndarray, float]:
        # PID-like action in the reward function
        # TODO: PID multipliers (must be tuned)
        P, I, D = self.reward_weights

        # PID components
        d = distance(achieved_goal, desired_goal)  # euclidean distance to goal
        v3 = self.get_ee_velocity()  # velocity as a vector
        v = distance(v3, np.zeros_like(v3))  # scalar velocity
        self.prev_distances.append(d)
        integrator = np.sum(self.prev_distances)

        pid_reward = -(P * d + I * integrator + D * v)

        # Sparse reward
        sparse_reward = -np.array(d > self.distance_threshold, dtype=np.float32)

        reward = sparse_reward
        if self.reward_type == "dense":
            reward = pid_reward
        return float(reward)
