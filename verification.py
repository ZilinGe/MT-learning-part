import numpy as np
import matplotlib.pyplot as plt
from environment import CellFreeMiMoCSIEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

# 设置 AP 和 UE 数量
N_AP = 16
N_UE = 8

# 模型主目录
save_dir = "ppo_checkpoints_v4_ratio/"

# 找到最新时间戳的子目录
all_subdirs = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
latest_subdir = sorted(all_subdirs)[-1] if all_subdirs else None

if latest_subdir:
    subdir_path = os.path.join(save_dir, latest_subdir)

    # 找到该子目录下最新的 .zip 模型文件
    zip_models = [f for f in os.listdir(subdir_path) if f.endswith(".zip")]
    latest_model = sorted(zip_models)[-1] if zip_models else None

    if latest_model:
        model_path = os.path.join(subdir_path, latest_model)
        print(f"✅ 加载模型: {model_path}")
        model = PPO.load(model_path, device="cuda")
    else:
        print("❌ 没有找到 .zip 模型文件，请确认是否已保存模型。")
        exit()
else:
    print("❌ 没有找到任何模型子目录，请先训练模型。")
    exit()

# 2. 创建环境
############################################
N_AP = 16
N_UE = 4
env = CellFreeMiMoCSIEnv(N_AP, N_UE)

############################################
# 3. 评估循环：自定义跑若干episodes，收集SE/Ptot信息
############################################
num_eval_episodes = 5

all_rewards = []
all_SEs = []
all_Ptots = []

for ep in range(num_eval_episodes):
    obs, info = env.reset()  # gymnasium新API：reset()返回 (obs, info)
    done = False

    ep_reward = 0.0
    ep_SE_values = []
    ep_Ptot_values = []

    while not done:
        # 使用训练好的模型决策
        action, _states = model.predict(obs, deterministic=True)
        # 交互一步
        obs, reward, terminated, truncated, info = env.step(action)

        ep_reward += reward
        ep_SE_values.append(info["SE"])  # 从info字典中获取SE
        ep_Ptot_values.append(info["Ptot"])  # 从info字典中获取Ptot

        # Gymnasium 的新终止判断方式
        done = terminated or truncated

    # 一个episode结束
    all_rewards.append(ep_reward)
    all_SEs.append(np.mean(ep_SE_values))
    all_Ptots.append(np.mean(ep_Ptot_values))

    print(f"Episode {ep + 1} finished.")
    print(f"  Total reward = {ep_reward:.3f}")
    print(f"  Mean SE      = {np.mean(ep_SE_values):.3f}")
    print(f"  Mean Ptot    = {np.mean(ep_Ptot_values):.3f}")
    print("--------------------------------------------------")

############################################
# 4. 验证结果汇总
############################################
print("========== 验证结果汇总 ==========")
print(f"平均奖励: {np.mean(all_rewards):.3f}")
print(f"平均SE:   {np.mean(all_SEs):.3f}")
print(f"平均Ptot: {np.mean(all_Ptots):.3f}")

# 如果需要最后关闭 MATLAB 引擎
env.close()
