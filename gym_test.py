import gymnasium as gym
import imageio
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    # 单环境，支持帧捕获
    # env = gym.make("CartPole-v1", render_mode="rgb_array")
    # env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = gym.make("Acrobot-v1", render_mode="rgb_array")
    model = PPO("MlpPolicy", env, device="cpu", verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_acrobot")
    del model

    model = PPO.load("ppo_acrobot", device="cpu")
    obs, _ = env.reset()
    frames = []
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        frame = env.render()
        frames.append(frame)

    # 保存视频
    os.makedirs("ppo_videos", exist_ok=True)
    video_path = "ppo_videos/ppo_acrobot_test.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    print(f"测试过程视频已保存到: {video_path}")

    env.close()


# # 用 if __name__ == "__main__": 包裹后，只有主进程会执行主逻辑，子进程不会，这样才能安全地使用多进程。
# if __name__ == "__main__":
#     # Parallel environments
#     vec_env = make_vec_env("CartPole-v1", n_envs=4, vec_env_cls=SubprocVecEnv)

#     model = PPO("MlpPolicy", vec_env, device="cpu", verbose=1)
#     model.learn(total_timesteps=25000)
#     model.save("ppo_cartpole")

#     del model # remove to demonstrate saving and loading

#     model = PPO.load("ppo_cartpole", device="cpu")
#     obs = vec_env.reset()
#     while True:
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = vec_env.step(action)
#         # vec_env.render("human")