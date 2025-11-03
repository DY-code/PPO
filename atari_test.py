import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

import imageio
import os

# 重要：注册ALE环境
# 注册ALE环境的目的是让 Gymnasium 能够识别和使用 Atari 游戏环境。
# ale_py 是 Atari Learning Environment 的 Python 实现，包含了所有 Atari 游戏的环境定义，但这些环境默认不会自动注册到 Gymnasium 的环境注册表中。
gym.register_envs(ale_py)

# 创建Atari环境（使用新的环境ID）
env = make_atari_env('ALE/Pong-v5', n_envs=4, seed=0)
# 帧堆叠
env = VecFrameStack(env, n_stack=4)

# # 创建PPO模型
# model = PPO(
#     "CnnPolicy",  # 使用CNN策略网络
#     env,
#     verbose=1,
#     learning_rate=2.5e-4,
#     n_steps=128,
#     batch_size=256,
#     n_epochs=4,
#     gamma=0.99,
#     gae_lambda=0.95,
#     clip_range=0.1,
#     ent_coef=0.01
# )

# # 训练模型
# model.learn(total_timesteps=100_000)
# # 保存模型
# model.save("ppo_pong")
# del model


# 测试模型
model = PPO.load("ppo_pong", device="cpu")
obs = env.reset()
frames = []
done = False

for _ in range(1000):  # 测试1000步，可根据需要调整
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    # 捕获第一个环境的帧（VecFrameStack下env.envs[0]为单个环境）
    # frame = env.envs[0].render(mode="rgb_array")
    frame = env.render()
    frames.append(frame)
    # 如果任一环境结束，可以break（可选）
    if dones[0]:
        break

# 保存视频
os.makedirs("ppo_videos", exist_ok=True)
video_path = "ppo_videos/ppo_pong_test.mp4"
imageio.mimsave(video_path, frames, fps=30)
print(f"测试过程视频已保存到: {video_path}")

env.close()
