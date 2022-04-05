from typing import Tuple
import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.reach import Reach, ObstructedReach
from panda_gym.pybullet import PyBullet


class PandaReachEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(
        self,
        render: bool = False,
        reward_type: str = "sparse",
        control_type: str = "ee",
        obstacle_type: str = "inline",
        n_substeps: int = 20,
        reward_weights: Tuple[float] = (1.0, 1.0, 1.0),
        visual_debug: bool = False,
        prev_distance_len: int = 10,
        aggregate_prev_distances=True,
    ) -> None:
        sim = PyBullet(render=render, n_substeps=n_substeps)
        robot = Panda(
            sim,
            block_gripper=True,
            base_position=np.array([-0.6, 0.0, 0.0]),
            control_type=control_type,
        )
        task = ObstructedReach(
            sim,
            reward_type=reward_type,
            get_ee_position=robot.get_ee_position,
            get_ee_velocity=robot.get_ee_velocity,
            obstacle_type=obstacle_type,
            reward_weights=reward_weights,
            prev_distance_len=prev_distance_len,
            visual_debug=visual_debug,
            aggregate_prev_distances=aggregate_prev_distances,
        )
        super().__init__(robot, task)
