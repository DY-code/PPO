import gymnasium as gym

# 重要：显式导入ale_py来注册Atari环境
try:
    import ale_py
    gym.register_envs(ale_py)
    print("✓ ALE环境已注册")
except ImportError:
    print("✗ ale_py未安装")
    exit(1)

# 列出所有环境
all_envs = list(gym.envs.registry.keys())
print(f"\n总共有 {len(all_envs)} 个环境")

# 查找Atari环境
atari_envs = [env for env in all_envs if 'ALE/' in env]
print(f"\n找到 {len(atari_envs)} 个Atari环境")
print("前10个Atari环境:", atari_envs[:10])

# 测试创建环境
if atari_envs:
    try:
        env = gym.make(atari_envs[0])
        print(f"\n✓ 成功创建环境: {atari_envs[0]}")
        env.close()
    except Exception as e:
        print(f"\n✗ 创建环境失败: {e}")
else:
    print("\n✗ 未找到任何Atari环境")