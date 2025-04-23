import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from environment import CellFreeMiMoCSIEnv

# ---------- 环境实例，用 check_env 快速 sanity‑check ---------- #
env = CellFreeMiMoCSIEnv(N_AP=16, N_UE=4, max_steps=256, se_threshold=3.0)
check_env(env, warn=True)

tb_logdir = "./tensorboard_logs/PPO_run"
os.makedirs(tb_logdir, exist_ok=True)

model = PPO(
    "MlpPolicy",  # fully connected NN
    env,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=256,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=tb_logdir,
    device="cpu",
)


# ---------- 训练回调：每 10 万步存一次模型 ---------- #
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


save_cb = SaveEveryCallback(save_freq=50_000, save_path="./ppo_ckpt", verbose=1)

# ---------- 开始训练 ---------- #
model.learn(total_timesteps=400_000, callback=save_cb)

# 训练完，保存最终模型
model.save("ppo_final.zip")
print("🎉 Training finished & model saved.")
