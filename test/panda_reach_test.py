import time
import gym
import panda_gym


def main():
    env = gym.make(
        "PandaReach-v2",
        render=True,
        control_type="joints",
        obstacle_type="planes",
        reward_type="pid",
        visual_debug=True,
    )

    for _ in range(10):
        obs, done = env.reset(), False

        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            time.sleep(0.1)


if __name__ == "__main__":
    main()
