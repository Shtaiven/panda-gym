from typing import Any, Dict, Tuple
import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.reach import Reach, ObstructedReach
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


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
        reward_type: str = "pid",
        control_type: str = "joint",
        obstacle_type: str = "bin",
        n_substeps: int = 20,
        reward_weights: Tuple[float] = (5.0, 5.0, 1.0),
        sparse_term: float = 0.0,
        visual_debug: bool = False,
        prev_distance_len: int = 10,
        aggregate_prev_distances=True,
        print_distances=False,
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
            sparse_term=sparse_term,
        )
        self.print_distances = print_distances
        super().__init__(robot, task)

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        obs = self._get_obs()
        info = {
            "is_success": self.task.is_success(
                obs["achieved_goal"], self.task.get_goal()
            )
        }
        done = (
            info["is_success"] > 0.0
        )  # this makes sure the task ends as soon as the goal is reached
        reward = self.task.compute_reward(
            obs["achieved_goal"], self.task.get_goal(), info
        )
        assert isinstance(reward, float)  # needed for pytype cheking

        # Use this for debugging
        if self.print_distances:
            print(
                f"distance to goal: {distance(obs['achieved_goal'], self.task.get_goal())}"
            )

        return obs, reward, done, info
