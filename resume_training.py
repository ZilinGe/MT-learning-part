from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from environment import CellFreeMiMoCSIEnv
import os

# ---------- 1. 复制之前的自定义 callback ---------- #
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
        return True

# ---------- 2. 实例化 callback ---------- #
save_cb = SaveEveryCallback(save_freq=50_000,
                            save_path="./ppo_ckpt",
                            verbose=1)

# ---------- 3. 加载模型并续训 ---------- #
env   = CellFreeMiMoCSIEnv(N_AP=16, N_UE=4, max_steps=256, se_threshold=3.0)
model = PPO.load("./ppo_ckpt/ppo_step_250000.zip", env=env, device="cpu")

model.learn(total_timesteps=150_000,
            callback=save_cb,
            reset_num_timesteps=False)
