import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from environment import CellFreeMiMoCSIEnv  # 确保你的 environment.py 在同一目录
from datetime import datetime


# Tensorborad-------------------------------------------------------------------
# tensorboard --logdir=./tensorboard_logs/PPO_Training --reload_interval 5

# ============================== #
# 定义训练进度显示的 Callback
# ============================== #
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, print_interval=100, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_interval = print_interval
        self.start_time = time.time()

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step % self.print_interval == 0 or step == self.total_timesteps:
            percent = 100 * step / self.total_timesteps
            elapsed = time.time() - self.start_time
            print(f"📊 Progress: {step}/{self.total_timesteps} ({percent:.2f}%) - Elapsed: {elapsed:.1f} sec")
        return True

# ============================== #
# 训练参数配置
# ============================== #
# 固定50个种子
seed_list = [i for i in range(1000, 1050)]

# 网络设置
N_AP = 16
N_UE = 4

# 每个epoch训练5000步
steps_per_epoch = 5000
total_epochs = len(seed_list)

# 保存目录
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
save_dir = f"ppo_checkpoints_v4_ratio/{current_time}/"
tb_base_dir = f"./tensorboard_logs/PPO_Training/{current_time}/"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(tb_base_dir, exist_ok=True)

# ============================== #
# 开始训练
# ============================== #
for epoch_idx, seed in enumerate(seed_list):
    print(f"\n=== 🚀 Epoch {epoch_idx+1}/{total_epochs} 使用 Seed={seed} 开始训练 ===")

    # 创建环境
    env = make_vec_env(lambda: CellFreeMiMoCSIEnv(N_AP=N_AP, N_UE=N_UE, seed=seed), n_envs=1)

    # 第一次创建新模型，后面加载上一个epoch的模型继续训练
    if epoch_idx == 0:
        model = PPO("MlpPolicy", env, learning_rate=0.01, verbose=1,
                    tensorboard_log=tb_base_dir, device="cpu")  # 注意用cpu
    else:
        model = PPO.load(os.path.join(save_dir, f"ppo_epoch_{epoch_idx}.zip"), env=env, device="cpu")

    # 学习，添加进度回调
    model.learn(
        total_timesteps=steps_per_epoch,
        callback=ProgressCallback(total_timesteps=steps_per_epoch)
    )

    # 保存本次epoch的模型
    model.save(os.path.join(save_dir, f"ppo_epoch_{epoch_idx+1}.zip"))
    print(f"✅ 保存 epoch {epoch_idx+1} 模型成功！")

print("\n🎉 所有种子训练完成！")