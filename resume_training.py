import os
import re
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from environment import CellFreeMiMoCSIEnv

# ============ 参数 ============
total_target_steps = 300_000
ckpt_dir = Path("./ppo_ckpt")   # 如果检查点也在别处，改这里
tb_logdir = r"C:\Users\asus\Desktop\master\MasterThesis_learningpart\tensorboard_logs\PPO_run\PPO_13"
device = "cpu"                  # 或 "cuda"

# ============ 环境 ============
env = CellFreeMiMoCSIEnv(N_AP=16, N_UE=4, max_steps=256, se_threshold=3.0)

# ============ 回调 ============
class SaveEveryCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            fname = os.path.join(self.save_path, f"ppo_step_{self.num_timesteps}.zip")
            self.model.save(fname)
            if self.verbose:
                print(f"💾 model saved to {fname}")

        if self.num_timesteps % 1000 == 0:
            print(f"🧮 Current timesteps: {self.num_timesteps}")
        return True

save_cb = SaveEveryCallback(save_freq=10_000, save_path=str(ckpt_dir), verbose=1)

# ============ 找最新检查点 ============
ckpt_files = list(ckpt_dir.glob("ppo_step_*.zip"))
if not ckpt_files:
    raise FileNotFoundError("⚠️  没有找到任何检查点文件，无法续训。")

latest_ckpt = max(ckpt_files, key=lambda p: int(re.search(r"ppo_step_(\d+)\.zip", p.name).group(1)))
trained_steps = int(re.search(r"ppo_step_(\d+)\.zip", latest_ckpt.name).group(1))
print(f"✅ 最新检查点: {latest_ckpt}（已训练 {trained_steps} 步）")

remaining_steps = total_target_steps - trained_steps
if remaining_steps <= 0:
    print("🎯 目标步数已达到或超过，无需续训。")
    exit(0)
print(f"ℹ️  还需训练 {remaining_steps} 步。")

# ============ 加载模型 & 重挂 logger ============
model = PPO.load(latest_ckpt, env=env, device=device)

# 保持同一 TensorBoard 目录，曲线连续
new_logger = configure(tb_logdir, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# ============ 继续训练 ============
model.learn(
    total_timesteps=remaining_steps,
    callback=save_cb,
    reset_num_timesteps=False
)

# ============ 保存最终模型 ============
model.save("ppo_final.zip")
print("🎉 Training resumed, completed, and model saved to ppo_final.zip")
