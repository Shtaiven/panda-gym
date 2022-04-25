import time
import gym
import panda_gym


def main(terminal_debug=False, visual_debug=False):
    env = gym.make(
        "PandaReach-v2",
        render=True,
        control_type="joints",
        obstacle_type="bin",
        reward_type="pid",
        init_pose_type="neutral",
        goal_pose_type="fixed",
        visual_debug=visual_debug,
    )

    for _ in range(10):
        obs, done = env.reset(), False

        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            distance_to_goal = panda_gym.utils.distance(obs["desired_goal"], obs["achieved_goal"])
            if terminal_debug:
                print(f"distance to goal: {distance_to_goal}")
            time.sleep(0.1)


if __name__ == "__main__":
    main()
