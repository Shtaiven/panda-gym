from binascii import a2b_base64
from typing import Any, Dict, Tuple, Union
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
        return np.array([])  # no task-specific observation

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


class OrientationParam(object):
    """
    Definitions for obstacle orientation for use by ObstructedReach.

    Args:
        axis (str): The primary axis of the obstacle (vertical leg of L, normal plane for planes).
        direction (int): The direction to place the horizontal leg of the L.
            Direction is determined be the lowest bit of the direction number:
                0 for next axis (e.g. x->y, y->z, z->x)
                1 for prev axis (e.g. x->z, y->x, z->y)
            bit 1 determines the sign:
                0 for positive (e.g. x)
                1 for negative (e.g. -x)
            for example:
                axis = "y"
                direction = 2
                2 in binary is 10
                bit 0 is 0, so we take the next axis after "y", which is "z"
                bit 1 is 1, so we take the negative direction, so the leg will
                    go in the negative y direction
        flip (bool): Whether to flip the L along its primary axis.
            False will be closer to the bottom of the leg in "axis"
            True will be closer to the top of the leg in "axis"
    """

    def __init__(
        self,
        axis: str = "z",
        direction: int = 0,
        flip: bool = False,
    ) -> None:
        self.axis = axis
        self.direction = direction
        self.flip = flip
        self._axis_position = 3
        self._axis_mask = 0b11
        self._direction_position = 1
        self._direction_mask = 0b11
        self._flip_position = 0
        self._flip_mask = 0b1

    def from_bits(self, bits):
        axis_bits = (bits >> self._axis_position) & self._axis_mask
        self.axis = None
        if axis_bits == 0b00:
            self.axis = "x"
        elif axis_bits == 0b01:
            self.axis = "y"
        elif axis_bits == 0b10:
            self.axis = "z"

        self.direction = (bits >> self._direction_position) & self._direction_mask
        self.flip = bool((bits >> self._flip_position) & self._flip_mask)

        return self

    def to_bits(self):
        bits = 0
        if self.axis == "x":
            bits |= 0b00 << self._axis_position
        elif self.axis == "y":
            bits |= 0b01 << self._axis_position
        elif self.axis == "z":
            bits |= 0b10 << self._axis_position

        bits |= self.direction << self._direction_position
        bits |= int(self.flip) << self._flip_position

        return bits

    def to_idxs(self, obs_type="L", obs_id=0):
        # Get the indices of the obstacles needed for the given orientation.
        idxs = [
            0,
        ]
        axis_dict = {"x": 0, "y": 1, "z": 2}
        if obs_type == "L":
            # This get the the lowest bit of the direction, which defines which
            # axis the next obstacle will be on by adding the the base axis
            direction_mod = (self.direction % 2) + 1

            # Obtain the axis {x, y, z} as a number from 0-2
            axis1 = axis_dict[self.axis]

            # Get the number of the axis of the small leg
            axis2 = (axis1 + direction_mod) % 3

            # Get the indices of the appropriate obstacles from the axis numbers
            # Obstacles are arranged in x-aligned, y-aligned, z-aligned, x-aligned...
            # when they are created by _create_obstacle_L
            #
            # obs_id gives the set of obstacles to use, e.g. 0-2 is obs_id 0, 3-5 is obs_id 1, etc.
            idx1 = axis1 + obs_id * 3
            idx2 = axis2 + obs_id * 3
            idxs = [idx1, idx2]
        elif obs_type == "planes":
            # Obtain the axis {x, y, z} as a number from 0-2
            axis1 = axis_dict[self.axis]

            # Get the indices of the approprite obstacles from the axis numbers
            # Obstacles are arranged in x-aligned, y-aligned, z-aligned, x-aligned...
            # when they are created by _create_obstacle_planes
            #
            # obs_id gives the set of obstacles to use, e.g. 0-2 is obs_id 0, 3-5 is obs_id 1, etc.
            idx1 = axis1 + obs_id * 3
            idx2 = axis1 + (obs_id + 1) * 3
            idxs = [idx1, idx2]
        elif obs_type == "bin":
            # Obtain the axis {x, y, z} as a number from 0-2
            axis1 = axis_dict[self.axis]

            # Get the index of the side of the bin to ignore (1 of the planes along the axis)
            idx_ignore = axis1 + 3  # y-case, z-case
            if self.axis == "x":
                idx_ignore = 0

            idxs = [idx for idx in range(6) if idx != idx_ignore]
        else:
            raise ValueError("Unknown obstacle type: {}".format(obs_type))

        return idxs

    def randomize(self):
        """Randomize the orientation of the obstacle."""
        self.axis = np.random.choice(["x", "y", "z"])
        self.direction = np.random.randint(0, 4)
        self.flip = np.random.choice([True, False])

        return self


class ObstructedReach(Reach):
    def __init__(
        self,
        sim,
        get_ee_position,
        get_ee_velocity,
        reward_type="pid",
        distance_threshold=0.01,
        goal_range=0.3,
        goal_range_low=None,
        goal_range_high=None,
        obstacle_type="inline",
        max_obstacles=6,
        reward_weights=(5.0, 5.0, 1.0),
        visual_debug=False,
        prev_distance_len=None,
    ) -> None:
        # These parameters must be places before super().__init__ because they
        # are used in _create_scene, which is called by super().__init__
        super(Reach, self).__init__(sim)
        self.obstacles = OrderedDict()
        self.obstacle_type = obstacle_type
        self.max_obstacles = max_obstacles
        self.visual_debug = visual_debug
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.get_ee_velocity = get_ee_velocity
        self.start_ee_position = self.get_ee_position()
        self.prev_distances = deque(maxlen=prev_distance_len)
        self.reward_weights = reward_weights

        # TODO: Extend the goal range to accommodate the new short table
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, -goal_range])
        if goal_range_low is not None:
            self.goal_range_low = goal_range_low
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        if goal_range_high is not None:
            self.goal_range_high = goal_range_high

        # Create the scene
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(
                target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30
            )

    def _create_scene(self) -> None:
        # draw the goal range if visual debugging is enabled
        if self.visual_debug:
            half_extents = (self.goal_range_high - self.goal_range_low) / 2
            self.sim.create_box(
                body_name="goal_range",
                half_extents=half_extents,
                position=self.goal_range_low + half_extents,
                rgba_color=np.array([1, 0, 0, 0.3]),
                ghost=True,
                mass=0.0,
            )

        # The plane represents the floor
        self.sim.create_plane(z_offset=-0.4)

        # Table edge must reach -0.85 to be under the arm
        self.sim.create_table(length=0.4, width=0.4, height=0.4, x_offset=-0.65)

        # Create the end goal sphere
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

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
        """Generate and obstacle in the pybullet scene and add it to the obstacle list."""
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
        """Remove an obstacle from the pybullet scene and remove it from the obstacle list."""
        self.sim.remove_body(obstacle_name)
        del self.obstacles[obstacle_name]

    def reset_obstacle_pose(self, obstacle_name=None):
        """Reset obstacle positions. By default, reset all obstacles."""
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
        """Create the components of an L-shaped obstacle."""
        obstacle_sizes = np.zeros((self.max_obstacles, 3))
        leg_length = 0.3
        leg_thickness = 0.1

        # Even obstacles are tall, odd obstacles are long in the y direction
        for idx in range(self.max_obstacles):
            if idx % 3 == 0:  # Obstacle along x axis
                obstacle_sizes[idx] = np.array(
                    [leg_length, leg_thickness, leg_thickness]
                )
            elif idx % 3 == 1:  # Obstacle along y axis
                obstacle_sizes[idx] = np.array(
                    [leg_thickness, leg_length, leg_thickness]
                )
            else:  # Obstacle along z axis
                obstacle_sizes[idx] = np.array(
                    [leg_thickness, leg_thickness, leg_length]
                )
        return obstacle_sizes

    def _create_obstacle_planes(self):
        """Create the components for a set of planes."""
        obstacle_sizes = np.zeros((self.max_obstacles, 3))
        plane_sides = 0.4
        plane_thickness = 0.025
        for idx in range(self.max_obstacles):
            if idx % 3 == 0:  # Planes 0,3 tangent to x
                obstacle_sizes[idx] = np.array(
                    [plane_thickness, plane_sides, plane_sides]
                )
            elif idx % 3 == 1:  # 1,4 tangent to y
                obstacle_sizes[idx] = np.array(
                    [plane_sides, plane_thickness, plane_sides]
                )
            else:  # 2,5 tangent to z
                obstacle_sizes[idx] = np.array(
                    [plane_sides, plane_sides, plane_thickness]
                )
        return obstacle_sizes

    def _create_obstacle_bin(self):
        """Create the components for a bin."""
        bin_length, bin_width, bin_height = 0.6, 0.4, 0.325
        bin_thickness = 0.02
        obstacle_sizes = np.zeros((self.max_obstacles, 3))
        for idx in range(self.max_obstacles):
            if idx % 3 == 0:  # Planes 0,3 tangent to x
                obstacle_sizes[idx] = np.array([bin_thickness, bin_width, bin_height])
            elif idx % 3 == 1:  # 1,4 tangent to y
                obstacle_sizes[idx] = np.array([bin_length, bin_thickness, bin_height])
            else:  # 2,5 tangent to z
                obstacle_sizes[idx] = np.array([bin_length, bin_width, bin_thickness])
        return obstacle_sizes

    def _random_goal_position(self):
        goal_range_diff = self.goal_range_high - self.goal_range_low
        position = (np.random.rand(3) * goal_range_diff) + self.goal_range_low

        return position

    def _parse_orientation_params(self, orientation):
        """Parse a value into a possible OrientationParam."""
        if isinstance(orientation, OrientationParam):
            if orientation.axis is None:
                raise TypeError("OrientationParam.axis must not be None")
        elif isinstance(orientation, int):
            orientation = OrientationParam().from_bits(orientation)
        else:
            orientation = OrientationParam()

        return orientation

    def _move_obstacle_common(
        self, obstacle_name: str, position: np.ndarray, orientation: np.ndarray = None
    ) -> None:
        """Move an obstacle in the pybullet scene."""
        if orientation is None:
            orientation = np.array([0, 0, 0, 1])
        self.sim.set_base_pose(obstacle_name, position, orientation)
        self.obstacles[obstacle_name][:3] = np.array(position)

    def _move_obstacle_inline(self, idx: int) -> None:
        """Move an obstacle in the pybullet scene in between the start position and the end goal."""
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
                         ^  ||idx1 |_____
        parallel to axis |  ||  *  |______\
                            ||     | idx2 | -> direction 0
                            \|_____|______|

        position and orientation are taken relative to the center of the idx1 block
        shown (marked with an apostrophe).

        Parameters:
            idx1 (int): index of the first block to use
            idx2 (int): index of the second block to use
            position (list(float)[3]): position of the L (default random, based on goal_range)
            orientation (list(float)[3]): orientation in quaternion or Euler angles (default random)
        """
        # FIXME: This only works in the axis="z" and flip=False case
        obs1 = f"obstacle{idx1}"
        obs2 = f"obstacle{idx2}"
        position1 = self.obstacles[obs1][:3]
        position2 = self.obstacles[obs2][:3]
        size1 = self.obstacles[obs1][-3:]
        size2 = self.obstacles[obs2][-3:]

        # randomize position if None
        if position is None:
            position = self._random_goal_position()

        # Obstacle Enumerations
        #
        # The L obstacles can have a total of 24 configurations, 8 in each axis.
        # 14 configurations can be stored in 5 bits, as follows:
        #
        # MSB || axis (4-3) | direction (2-1) | flip (0) || LSB
        #
        # where axis can take the following values:
        #   axis | bits
        #   ============
        #   x     | 00
        #   y     | 01
        #   z     | 10
        #
        # and direction can take the following values:
        #   direction | bits
        #   =================
        #   prev axis | 00
        #   next axis | 01
        #   prev axis reversed | 10
        #   next axis reversed | 11
        #
        # and flip can take the following values:
        #   flip | bits
        #   =============
        #   no flip | 0
        #   flip    | 1
        #
        # where flip mirrors along the primary axis
        #
        # For example:
        # The bits 01000 (8 in decimal) represents the L which extends its
        # vertical line in the y-direction and its leg in the
        #
        # Alternative, use the OrientationParam object
        # e.g. `orientation = OrientationParam("y", 1, False)`

        # Parse orientation into an OrientationParam object if needed
        orientation = self._parse_orientation_params(orientation)

        # Use the equation p2 = p1 + D.dot(theta)/2 to calculate the position of the second obstacle
        # where p1 and p2 are points 1 and 2, respectively
        # D is the dimensions matrix, and theta is the parameter vector
        #
        # if expanded, the equation is:
        #   x2 = x1 + (theta[0]*size1[0] + theta[1]*size2[0])/2
        #   y2 = y1 + (theta[2]*size1[1] + theta[3]*size2[1])/2
        #   z2 = z1 + (theta[4]*size1[2] + theta[5]*size2[2])/2
        #
        # theta values may only be one of [-1, 0, 1]
        l1, w1, h1 = size1
        l2, w2, h2 = size2
        dimensions = np.array(
            [[l1, l2, 0, 0, 0, 0], [0, 0, w1, w2, 0, 0], [0, 0, 0, 0, h1, h2]],
        )
        parameters = np.zeros((6, 1))
        flip_vector = np.ones((6, 1))

        # define values for constructing the parameters
        # direction_axis_switch selects between the next axis or previous axis
        # negative_axis_switch selects between positive direction or negative direction
        direction_axis_switch = orientation.direction & 1
        negative_axis_switch = (orientation.direction >> 1) & 1
        a1 = (not direction_axis_switch) * (-1 if negative_axis_switch else 1)
        b1 = (not direction_axis_switch) * (-1 if negative_axis_switch else 1)
        a2 = (direction_axis_switch) * (-1 if negative_axis_switch else 1)
        b2 = (direction_axis_switch) * (-1 if negative_axis_switch else 1)
        if orientation.axis == "x":
            parameters = np.array([-1, 1, a1, b1, a2, b2]).reshape((6, 1))
            if orientation.flip:
                flip_vector[0:1] = -1
        elif orientation.axis == "y":
            parameters = np.array([a2, b2, -1, 1, a1, b1]).reshape((6, 1))
            if orientation.flip:
                flip_vector[2:3] = -1
        elif orientation.axis == "z":
            parameters = np.array([a1, b1, a2, b2, -1, 1]).reshape((6, 1))
            if orientation.flip:
                flip_vector[4:5] = -1

        position1 = position.reshape((3, 1))
        position2 = position1 + (dimensions.dot(parameters * flip_vector) / 2)

        # move the obstacles to the correct position
        self._move_obstacle_common(obs1, position1.flatten())
        self._move_obstacle_common(obs2, position2.flatten())

    def _move_obstacle_planes(
        self, idx1, idx2, position=None, orientation=None, spacing=None
    ):
        """Move a set of plane obstacles."""
        obs1 = f"obstacle{idx1}"
        obs2 = f"obstacle{idx2}"

        # randomize position if None
        if position is None:
            position = self._random_goal_position()

        # Parse orientation into an OrientationParam object if needed
        orientation = self._parse_orientation_params(orientation)

        # Set default spacing if it is None
        if spacing is None:
            spacing = 0.2

        # Calculate positions for the planes
        axis_dict = {"x": 0, "y": 1, "z": 2}
        axis_index = (axis_dict[orientation.axis] + 1 + int(orientation.flip)) % 3
        obs1_dim = self.obstacles[obs1][axis_index + 3]
        obs2_dim = self.obstacles[obs2][axis_index + 3]
        obs1_offset = np.zeros(3)
        obs1_offset[axis_index] = obs1_dim + spacing
        obs2_offset = np.zeros(3)
        obs2_offset[axis_index] = obs2_dim + spacing
        position1 = position + obs1_offset / 2
        position2 = position - obs2_offset / 2

        # Place a plane obstacle at the specified position
        self._move_obstacle_common(obs1, position1)
        self._move_obstacle_common(obs2, position2)

    def _move_obstacle_bin(self, idxs, position=None, orientation=None):
        """Create a bin-shaped obstacle by moving its constituent parts."""
        # randomize position if None
        if position is None:
            position = self._random_goal_position()

        # Parse orientation into an OrientationParam object if needed
        orientation = self._parse_orientation_params(orientation)

        # Get the dimensions of the bin
        # The bottom panel of the bin can be used to get both length (x-dim) and width (y-dim). The height can be obtained from any of the side panels (x-dim in this case)
        bin_length, bin_width, bin_thick = self.obstacles["obstacle2"][3:6]
        bin_height = self.obstacles["obstacle0"][5]
        bin_dim = [bin_length, bin_width, bin_height]

        # Calculate the offset of all of the bin's size
        # and move them to the correct position
        for obs_idx in idxs:
            obs_normal_axis = (
                obs_idx % 3
            )  # 0, 1, or 2 depending on the short side of the obstacle as defined in _create_obstacle_bin
            obs_offset = np.zeros(3)
            obs_offset[obs_normal_axis] = (bin_dim[obs_normal_axis] + bin_thick) / 2

            # Add or subtract the offset depending on the index for either side of the obstacle
            obs_name = f"obstacle{obs_idx}"
            if obs_idx < 3:
                self._move_obstacle_common(obs_name, position - obs_offset)
            else:
                self._move_obstacle_common(obs_name, position + obs_offset)

    def set_obstacle_pose(self, obs_type: str = "inline"):
        self.reset_obstacle_pose()
        params = OrientationParam().randomize()
        if obs_type == "inline":
            self._move_obstacle_inline(0)
        elif obs_type == "L":
            # Place L-shaped objects
            # FIXME: Fix offset when flip == True
            params.flip = False
            idx1, idx2 = params.to_idxs("L")
            self._move_obstacle_L(idx1, idx2, orientation=params)
        elif obs_type == "planes":
            # Place planes obstructing the end goal
            params.axis = np.random.choice(["x", "z"])
            params.flip = np.random.choice([True, False])
            idx1, idx2 = params.to_idxs("planes")
            self._move_obstacle_planes(idx1, idx2, orientation=params)
        elif obs_type == "bin":
            # Place bin-shaped obstacle
            params.axis = np.random.choice(["x", "z"])
            idxs = params.to_idxs("bin")
            self._move_obstacle_bin(idxs, orientation=params)

    def reset(self) -> None:
        super().reset()
        self.start_ee_position = self.get_ee_position()
        self.set_obstacle_pose(obs_type=self.obstacle_type)
        self.prev_distances.clear()

    def _sort_obstacles(self):
        """Sort the obstacles by their position and dimension."""
        obstacles = list(self.obstacles.values())
        n_sort = len(obstacles[0])
        for i in reversed(range(n_sort)):
            obstacles.sort(key=lambda x: x[i])

        return obstacles

    def get_obs(self) -> np.ndarray:
        obs = self._sort_obstacles()
        try:
            obs = np.concatenate(obs)
        except ValueError as e:
            obs = np.array([-2.0, -2.0, -2.0, 0.0, 0.0, 0.0] * self.max_obstacles)

        obs = np.concatenate([obs, self.goal], dtype=np.float32)

        return obs

    def compute_reward(
        self, achieved_goal, desired_goal, info: Dict[str, Any]
    ) -> Union[np.ndarray, float]:
        # PID-like action in the reward function
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
        if self.reward_type == "pid":
            reward = pid_reward
        elif self.reward_type == "dense":
            reward = -d

        return float(reward)
